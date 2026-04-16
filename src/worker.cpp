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

// worker reads 784 floats (an image), runs the model, sends back an int label

static Model g_model;
static std::atomic<int> g_active(0);
static int g_slow_ms = 0;  // added delay, used to fake a slow worker in experiments

void handle_client(int fd) {
    g_active++;

    std::vector<float> in(784);
    int total = 0;
    int need = 784 * sizeof(float);
    char* buf = (char*)in.data();

    while (total < need) {
        int n = recv(fd, buf + total, need - total, 0);
        if (n <= 0) {
            close(fd);
            g_active--;
            return;
        }
        total += n;
    }

    int pred = g_model.predict(in);

    if (g_slow_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(g_slow_ms));
    }

    send(fd, &pred, sizeof(int), 0);
    close(fd);
    g_active--;
}

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: " << argv[0] << " port weights [slow_ms]\n";
        return 1;
    }

    int port = std::stoi(argv[1]);
    std::string wpath = argv[2];
    if (argc == 4) g_slow_ms = std::stoi(argv[3]);

    if (!g_model.load_weights(wpath)) {
        std::cerr << "couldn't load weights\n";
        return 1;
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
    if (listen(srv, 128) < 0) { perror("listen"); return 1; }

    std::cout << "worker up on " << port;
    if (g_slow_ms > 0) std::cout << " (slow=" << g_slow_ms << "ms)";
    std::cout << "\n";

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
