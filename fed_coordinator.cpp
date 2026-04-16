// federated averaging coordinator
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

constexpr int MSG_REGISTER = 1;
constexpr int MSG_WEIGHTS  = 2;
constexpr int MSG_AVG      = 3;

// same 784-128-64-10 MLP as the inference side
constexpr int NFLOATS =
    (128 * 784) + 128 +
    (64  * 128) + 64  +
    (10  * 64)  + 10;

bool recv_all(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t g = 0;
    while (g < n) {
        ssize_t r = recv(fd, p + g, n - g, 0);
        if (r <= 0) return false;
        g += r;
    }
    return true;
}
bool send_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    size_t s = 0;
    while (s < n) {
        ssize_t k = send(fd, p + s, n - s, 0);
        if (k <= 0) return false;
        s += k;
    }
    return true;
}

// per-round barrier state
struct State {
    std::mutex mu;
    std::condition_variable done_cv;
    std::vector<std::vector<float>> subs;  // weights submitted this round
    std::vector<float> avg;
    int round = 0;
    bool ready = false;
};

static int    g_n_workers;
static int    g_rounds;
static State  g_st;

// one thread per worker
void worker_loop(int fd, int slot) {
    // read registration
    int32_t hdr[2];
    if (!recv_all(fd, hdr, sizeof(hdr))) { close(fd); return; }
    if (hdr[0] != MSG_REGISTER) { close(fd); return; }

    std::string id(hdr[1], '\0');
    if (!recv_all(fd, id.data(), hdr[1])) { close(fd); return; }
    std::cout << "[" << id << "] connected (slot " << slot << ")\n";

    for (int r = 1; r <= g_rounds; r++) {
        // get weights
        int32_t wh[2];
        if (!recv_all(fd, wh, sizeof(wh))) { close(fd); return; }
        if (wh[0] != MSG_WEIGHTS || wh[1] != NFLOATS) { close(fd); return; }

        std::vector<float> w(NFLOATS);
        if (!recv_all(fd, w.data(), NFLOATS * sizeof(float))) { close(fd); return; }

        // put into shared state; last one in does the averaging
        {
            std::unique_lock<std::mutex> lk(g_st.mu);
            g_st.subs[slot] = std::move(w);

            int got = 0;
            for (auto& v : g_st.subs) if (!v.empty()) got++;

            if (got == g_n_workers) {
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<float> a(NFLOATS, 0.0f);
                for (auto& v : g_st.subs)
                    for (int i = 0; i < NFLOATS; i++) a[i] += v[i];
                for (int i = 0; i < NFLOATS; i++) a[i] /= g_n_workers;
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

                g_st.avg = std::move(a);
                g_st.ready = true;
                g_st.round = r;
                std::cout << "round " << r << " averaged in " << ms << " ms\n";
                g_st.done_cv.notify_all();
            } else {
                g_st.done_cv.wait(lk, [r] {
                    return g_st.ready && g_st.round == r;
                });
            }
        }

        // send averaged weights back
        int32_t ah[2] = { MSG_AVG, NFLOATS };
        if (!send_all(fd, ah, sizeof(ah)) ||
            !send_all(fd, g_st.avg.data(), NFLOATS * sizeof(float))) {
            close(fd); return;
        }

        // clear this slot. last to clear resets for next round
        {
            std::unique_lock<std::mutex> lk(g_st.mu);
            g_st.subs[slot].clear();
            int cleared = 0;
            for (auto& v : g_st.subs) if (v.empty()) cleared++;
            if (cleared == g_n_workers) g_st.ready = false;
        }
    }

    std::cout << "[" << id << "] done\n";
    close(fd);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " port num_workers rounds\n";
        return 1;
    }
    int port = std::stoi(argv[1]);
    g_n_workers = std::stoi(argv[2]);
    g_rounds = std::stoi(argv[3]);
    g_st.subs.resize(g_n_workers);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(srv, 16) < 0) { perror("listen"); return 1; }

    std::cout << "fed coord on " << port << ", "
              << g_n_workers << " workers, " << g_rounds << " rounds\n";

    std::vector<std::thread> ts;
    for (int i = 0; i < g_n_workers; i++) {
        sockaddr_in ca{};
        socklen_t len = sizeof(ca);
        int cfd = accept(srv, (sockaddr*)&ca, &len);
        if (cfd < 0) { perror("accept"); return 1; }
        ts.emplace_back(worker_loop, cfd, i);
    }
    for (auto& t : ts) t.join();

    std::cout << "all done\n";
    close(srv);
    return 0;
}
