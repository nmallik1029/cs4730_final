#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

//  Load balancing strategies
enum Strategy { ROUND_ROBIN, LEAST_CONNECTIONS, RESPONSE_TIME, RANDOM_LB };

Strategy parse_strategy(const std::string& s) {
    if (s == "round_robin") return ROUND_ROBIN;
    if (s == "least_connections") return LEAST_CONNECTIONS;
    if (s == "response_time") return RESPONSE_TIME;
    if (s == "random") return RANDOM_LB;
    std::cerr << "Unknown strategy '" << s << "', defaulting to round_robin\n";
    return ROUND_ROBIN;
}

// Worker info
struct Worker {
    std::string host;
    int port;
    std::atomic<bool> alive{true};
    std::atomic<int>  active_connections{0};
    std::mutex rt_mutex;
    double avg_response_us = 1.0;
};

static std::vector<Worker*> g_workers;
static std::mutex g_workers_mutex;
static std::atomic<int> g_rr_index{0};   // round robin counter
static Strategy g_strategy;

// Connect to a worker 
int connect_to_worker(Worker* w) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    // 3 second timeout for failure detection
    struct timeval tv{3, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(w->port);
    inet_pton(AF_INET, w->host.c_str(), &addr.sin_addr);

    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// Pick a worker based on strategy 
Worker* pick_worker() {
    std::lock_guard<std::mutex> lock(g_workers_mutex);

    std::vector<Worker*> alive;
    for (auto* w : g_workers)
        if (w->alive) alive.push_back(w);

    if (alive.empty()) return nullptr;

    switch (g_strategy) {
        case ROUND_ROBIN: {
            int idx = g_rr_index.fetch_add(1) % (int)alive.size();
            return alive[idx];
        }
        case LEAST_CONNECTIONS: {
            return *std::min_element(alive.begin(), alive.end(),
                [](Worker* a, Worker* b) {
                    return a->active_connections < b->active_connections;
                });
        }
        case RESPONSE_TIME: {
            return *std::min_element(alive.begin(), alive.end(),
                [](Worker* a, Worker* b) {
                    return a->avg_response_us < b->avg_response_us;
                });
        }
        case RANDOM_LB: {
            static std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(0, (int)alive.size() - 1);
            return alive[dist(rng)];
        }
    }
    return alive[0];
}

// Forward one request to a worker 
// Returns prediction, or -1 on failure
int forward_to_worker(Worker* w, const std::vector<float>& input) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int fd = connect_to_worker(w);
    if (fd < 0) {
        std::cerr << "Worker " << w->host << ":" << w->port << " unreachable\n";
        w->alive = false;
        return -1;
    }

    w->active_connections++;

    // Send 784 floats
    const char* buf = reinterpret_cast<const char*>(input.data());
    int total = 0, needed = 784 * sizeof(float);
    while (total < needed) {
        int n = send(fd, buf + total, needed - total, 0);
        if (n <= 0) {
            std::cerr << "Worker " << w->host << ":" << w->port << " send failed\n";
            w->alive = false;
            w->active_connections--;
            close(fd);
            return -1;
        }
        total += n;
    }

    // prediction
    int prediction = -1;
    int n = recv(fd, &prediction, sizeof(int), MSG_WAITALL);
    if (n != sizeof(int)) {
        std::cerr << "Worker " << w->host << ":" << w->port << " recv failed\n";
        w->alive = false;
        w->active_connections--;
        close(fd);
        return -1;
    }

    close(fd);
    w->active_connections--;

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    {
        std::lock_guard<std::mutex> lock(w->rt_mutex);
        w->avg_response_us = 0.8 * w->avg_response_us + 0.2 * elapsed;
    }

    return prediction;
}

// Handle one client connection 
void handle_client(int client_fd) {
    std::vector<float> input(784);
    int total = 0, needed = 784 * sizeof(float);
    char* buf = reinterpret_cast<char*>(input.data());

    while (total < needed) {
        int n = recv(client_fd, buf + total, needed - total, 0);
        if (n <= 0) {
            std::cerr << "Coordinator: client disconnected during read\n";
            close(client_fd);
            return;
        }
        total += n;
    }

    // Try workers until one succeeds (handles failures)
    int prediction = -1;
    int attempts = (int)g_workers.size();
    for (int i = 0; i < attempts && prediction == -1; i++) {
        Worker* w = pick_worker();
        if (!w) { std::cerr << "No alive workers!\n"; break; }
        prediction = forward_to_worker(w, input);
    }

    // Send result back to client
    send(client_fd, &prediction, sizeof(int), 0);
    close(client_fd);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <port> <strategy> <worker_host:port> [<worker_host:port> ...]\n";
        std::cerr << "Strategies: round_robin, least_connections, response_time, random\n";
        std::cerr << "Example: ./coordinator 5000 round_robin localhost:5001 localhost:5002\n";
        return 1;
    }

    int coord_port = std::stoi(argv[1]);
    g_strategy = parse_strategy(argv[2]);

    for (int i = 3; i < argc; i++) {
        std::string addr = argv[i];
        size_t colon = addr.rfind(':');
        if (colon == std::string::npos) {
            std::cerr << "Bad worker address: " << addr << "\n";
            return 1;
        }
        Worker* w = new Worker();
        w->host = addr.substr(0, colon);
        w->port = std::stoi(addr.substr(colon + 1));
        g_workers.push_back(w);
        std::cout << "Registered worker: " << w->host << ":" << w->port << "\n";
    }

    // Start coordinator server
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(coord_port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(server_fd, 256) < 0) { perror("listen"); return 1; }

    std::cout << "Coordinator listening on port " << coord_port
              << "strategy: " << argv[2] << "\n";

    while (true) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);
        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &len);
        if (client_fd < 0) { perror("accept"); continue; }
        std::thread(handle_client, client_fd).detach();
    }

    close(server_fd);
    return 0;
}
