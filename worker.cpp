#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "model.hpp"

// Protocol:
// Client -> Worker: 784 floats (3136 bytes)
// Worker -> Client: 1 int (4 bytes) = predicted class

static Model g_model;
static std::atomic<int> g_active_requests(0);
static int g_slow_ms = 0;  // artificial per-request delay (for heterogeneous worker experiments)

void handle_client(int client_fd) {
    g_active_requests++;

    // Read 784 floats
    std::vector<float> input(784);
    int total = 0;
    int needed = 784 * sizeof(float);
    char* buf = reinterpret_cast<char*>(input.data());

    while (total < needed) {
        int n = recv(client_fd, buf + total, needed - total, 0);
        if (n <= 0) {
            std::cerr << "Worker: connection lost while reading input\n";
            close(client_fd);
            g_active_requests--;
            return;
        }
        total += n;
    }

    // Run inference
    int prediction = g_model.predict(input);

    // Optional artificial delay (simulates slow worker for load-balancing experiments)
    if (g_slow_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(g_slow_ms));
    }

    // Send prediction back
    send(client_fd, &prediction, sizeof(int), 0);

    close(client_fd);
    g_active_requests--;
}

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <port> <weights_file> [slow_ms]\n";
        std::cerr << "  slow_ms (optional): artificial per-request delay in milliseconds\n";
        return 1;
    }

    int port = std::stoi(argv[1]);
    std::string weights_path = argv[2];
    if (argc == 4) {
        g_slow_ms = std::stoi(argv[3]);
    }

    if (!g_model.load_weights(weights_path)) {
        std::cerr << "Failed to load weights\n";
        return 1;
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(server_fd, 128) < 0) {
        perror("listen"); return 1;
    }

    std::cout << "Worker listening on port " << port
              << " | weights: " << weights_path
              << " | slow_ms: " << g_slow_ms << "\n";

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