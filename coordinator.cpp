#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <random>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// coordinator accepts client requests, picks a worker using one of 4 strategies,
// and forwards the request. if a worker fails we mark it dead and try another.

enum Strategy { RR, LEAST_CONN, RESP_TIME, RANDOM_LB };

Strategy parse(const std::string& s) {
    if (s == "round_robin") return RR;
    if (s == "least_connections") return LEAST_CONN;
    if (s == "response_time") return RESP_TIME;
    if (s == "random") return RANDOM_LB;
    std::cerr << "unknown strategy '" << s << "', defaulting to round_robin\n";
    return RR;
}

struct Worker {
    std::string host;
    int port;
    std::atomic<bool> alive{true};
    std::atomic<int>  active{0};
    std::mutex        rt_mu;
    double            avg_rt_us = 1.0;  // EMA of response time
};

static std::vector<Worker*> g_workers;
static std::mutex           g_wmu;
static std::atomic<int>     g_rr{0};
static Strategy             g_strat;

int dial(Worker* w) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    // timeout so dead workers fail fast
    struct timeval tv{3, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(w->port);
    inet_pton(AF_INET, w->host.c_str(), &addr.sin_addr);

    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

Worker* pick() {
    std::lock_guard<std::mutex> g(g_wmu);

    std::vector<Worker*> live;
    for (auto* w : g_workers) if (w->alive) live.push_back(w);
    if (live.empty()) return nullptr;

    switch (g_strat) {
        case RR: {
            int i = g_rr.fetch_add(1) % (int)live.size();
            return live[i];
        }
        case LEAST_CONN: {
            return *std::min_element(live.begin(), live.end(),
                [](Worker* a, Worker* b) { return a->active < b->active; });
        }
        case RESP_TIME: {
            return *std::min_element(live.begin(), live.end(),
                [](Worker* a, Worker* b) { return a->avg_rt_us < b->avg_rt_us; });
        }
        case RANDOM_LB: {
            static std::mt19937 rng(42);
            std::uniform_int_distribution<int> d(0, (int)live.size() - 1);
            return live[d(rng)];
        }
    }
    return live[0];
}

// send img to w, return prediction or -1 on failure
int forward(Worker* w, const std::vector<float>& img) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int fd = dial(w);
    if (fd < 0) {
        std::cerr << "worker " << w->host << ":" << w->port << " down\n";
        w->alive = false;
        return -1;
    }
    w->active++;

    const char* b = (const char*)img.data();
    int total = 0, need = 784 * sizeof(float);
    while (total < need) {
        int n = send(fd, b + total, need - total, 0);
        if (n <= 0) {
            w->alive = false;
            w->active--;
            close(fd);
            return -1;
        }
        total += n;
    }

    int pred = -1;
    int n = recv(fd, &pred, sizeof(int), MSG_WAITALL);
    close(fd);
    w->active--;

    if (n != sizeof(int)) {
        w->alive = false;
        return -1;
    }

    // update EMA of response time
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    {
        std::lock_guard<std::mutex> g(w->rt_mu);
        w->avg_rt_us = 0.8 * w->avg_rt_us + 0.2 * us;
    }
    return pred;
}

void handle_client(int fd) {
    std::vector<float> img(784);
    int total = 0, need = 784 * sizeof(float);
    char* b = (char*)img.data();
    while (total < need) {
        int n = recv(fd, b + total, need - total, 0);
        if (n <= 0) { close(fd); return; }
        total += n;
    }

    // try workers until one succeeds
    int pred = -1;
    int tries = (int)g_workers.size();
    for (int i = 0; i < tries && pred == -1; i++) {
        Worker* w = pick();
        if (!w) break;
        pred = forward(w, img);
    }

    send(fd, &pred, sizeof(int), 0);
    close(fd);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " port strategy worker_host:port [worker_host:port ...]\n";
        std::cerr << "strategies: round_robin, least_connections, response_time, random\n";
        return 1;
    }

    int port = std::stoi(argv[1]);
    g_strat = parse(argv[2]);

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        size_t c = a.rfind(':');
        if (c == std::string::npos) {
            std::cerr << "bad address: " << a << "\n";
            return 1;
        }
        Worker* w = new Worker();
        w->host = a.substr(0, c);
        w->port = std::stoi(a.substr(c + 1));
        g_workers.push_back(w);
        std::cout << "worker: " << w->host << ":" << w->port << "\n";
    }

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(srv, 256) < 0) { perror("listen"); return 1; }

    std::cout << "coordinator on " << port << ", strategy=" << argv[2] << "\n";

    while (true) {
        sockaddr_in ca{};
        socklen_t len = sizeof(ca);
        int cfd = accept(srv, (sockaddr*)&ca, &len);
        if (cfd < 0) { perror("accept"); continue; }
        std::thread(handle_client, cfd).detach();
    }
    close(srv);
    return 0;
}