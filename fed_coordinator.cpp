// fed_coordinator.cpp
// Federated averaging coordinator.
// Waits for N workers to register, then for R rounds:
//   1. Collects weights from every worker
//   2. Averages them element-wise (FedAvg)
//   3. Broadcasts averaged weights back to every worker
//
// Usage:
//   ./fed_coordinator <port> <num_workers> <rounds>
// Example:
//   ./fed_coordinator 6000 2 20

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

// Architecture: 784 -> 128 -> 64 -> 10  (same as model.cpp / weights_export.py)
constexpr int NUM_FLOATS =
    (128 * 784) + 128 +
    (64  * 128) + 64  +
    (10  * 64)  + 10;

// ----- Helpers -----
bool recv_exact(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, 0);
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

bool send_exact(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < n) {
        ssize_t s = send(fd, p + sent, n - sent, 0);
        if (s <= 0) return false;
        sent += s;
    }
    return true;
}

// ----- Per-round sync state -----
struct RoundState {
    std::mutex mu;
    std::condition_variable cv_all_in;   // workers -> averager
    std::condition_variable cv_avg_done; // averager -> workers
    std::vector<std::vector<float>> received; // one weight vec per worker
    std::vector<float> averaged;
    int round_num = 0;                   // incremented each round
    bool avg_ready = false;
};

static int               g_num_workers;
static int               g_rounds;
static RoundState        g_state;

// Per-connection handler: one thread per worker
void handle_worker(int fd, int worker_idx) {
    // Read registration
    int32_t hdr[2];
    if (!recv_exact(fd, hdr, sizeof(hdr))) { std::cerr << "register hdr failed\n"; close(fd); return; }
    int mtype = hdr[0], id_len = hdr[1];
    if (mtype != MSG_REGISTER) { std::cerr << "expected REGISTER, got " << mtype << "\n"; close(fd); return; }

    std::string wid(id_len, '\0');
    if (!recv_exact(fd, wid.data(), id_len)) { std::cerr << "register id failed\n"; close(fd); return; }
    std::cout << "[coord] worker " << wid << " registered (slot " << worker_idx << ")\n";

    // Main round loop
    for (int r = 1; r <= g_rounds; r++) {
        // --- Receive weights from this worker ---
        int32_t whdr[2];
        if (!recv_exact(fd, whdr, sizeof(whdr))) {
            std::cerr << "[coord] worker " << wid << " disconnected at round " << r << "\n";
            close(fd); return;
        }
        if (whdr[0] != MSG_WEIGHTS || whdr[1] != NUM_FLOATS) {
            std::cerr << "[coord] bad weights header from " << wid << "\n";
            close(fd); return;
        }

        std::vector<float> weights(NUM_FLOATS);
        if (!recv_exact(fd, weights.data(), NUM_FLOATS * sizeof(float))) {
            std::cerr << "[coord] failed reading weights from " << wid << "\n";
            close(fd); return;
        }

        // --- Submit weights to shared state ---
        {
            std::unique_lock<std::mutex> lk(g_state.mu);
            g_state.received[worker_idx] = std::move(weights);

            // Am I the last one in?
            int received_count = 0;
            for (auto& v : g_state.received) if (!v.empty()) received_count++;

            if (received_count == g_num_workers) {
                // Do the averaging
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<float> avg(NUM_FLOATS, 0.0f);
                for (auto& wv : g_state.received)
                    for (int i = 0; i < NUM_FLOATS; i++)
                        avg[i] += wv[i];
                for (int i = 0; i < NUM_FLOATS; i++)
                    avg[i] /= g_num_workers;
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

                g_state.averaged = std::move(avg);
                g_state.avg_ready = true;
                g_state.round_num = r;

                std::cout << "[coord] round " << r
                          << " averaged " << g_num_workers
                          << " workers in " << ms << " ms\n";

                g_state.cv_avg_done.notify_all();
            } else {
                // Wait until averager broadcasts
                g_state.cv_avg_done.wait(lk, [r] {
                    return g_state.avg_ready && g_state.round_num == r;
                });
            }
        }

        // --- Send averaged weights back to this worker ---
        int32_t ahdr[2] = { MSG_AVG, NUM_FLOATS };
        if (!send_exact(fd, ahdr, sizeof(ahdr)) ||
            !send_exact(fd, g_state.averaged.data(), NUM_FLOATS * sizeof(float))) {
            std::cerr << "[coord] send avg failed for " << wid << "\n";
            close(fd); return;
        }

        // --- Barrier: last worker to finish clears state for next round ---
        {
            std::unique_lock<std::mutex> lk(g_state.mu);
            g_state.received[worker_idx].clear();

            int cleared = 0;
            for (auto& v : g_state.received) if (v.empty()) cleared++;
            if (cleared == g_num_workers) {
                g_state.avg_ready = false;  // ready for next round
            }
        }
    }

    std::cout << "[coord] worker " << wid << " finished all " << g_rounds << " rounds\n";
    close(fd);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <port> <num_workers> <rounds>\n";
        std::cerr << "Example: " << argv[0] << " 6000 2 20\n";
        return 1;
    }
    int port      = std::stoi(argv[1]);
    g_num_workers = std::stoi(argv[2]);
    g_rounds      = std::stoi(argv[3]);

    g_state.received.resize(g_num_workers);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(server_fd, 16) < 0) { perror("listen"); return 1; }

    std::cout << "[coord] waiting for " << g_num_workers
              << " workers on port " << port
              << " (will run " << g_rounds << " rounds)\n";

    std::vector<std::thread> workers;
    for (int i = 0; i < g_num_workers; i++) {
        sockaddr_in caddr{};
        socklen_t len = sizeof(caddr);
        int cfd = accept(server_fd, (sockaddr*)&caddr, &len);
        if (cfd < 0) { perror("accept"); return 1; }
        std::cout << "[coord] accepted connection " << (i+1) << "/" << g_num_workers << "\n";
        workers.emplace_back(handle_worker, cfd, i);
    }

    for (auto& t : workers) t.join();

    std::cout << "[coord] all rounds complete\n";
    close(server_fd);
    return 0;
}
