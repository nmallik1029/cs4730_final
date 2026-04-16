#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

struct MnistSample {
    std::vector<float> pixels; 
    int label;
};

bool load_mnist(const std::string& path, std::vector<MnistSample>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    int n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (!f || n <= 0) return false;

    out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i].pixels.resize(784);
        f.read(reinterpret_cast<char*>(out[i].pixels.data()), 784 * sizeof(float));
        f.read(reinterpret_cast<char*>(&out[i].label), sizeof(int));
        if (!f) return false;
    }
    return true;
}

long long send_request(const std::string& host, int port,
                       const std::vector<float>& image, int* pred) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    const char* buf = reinterpret_cast<const char*>(image.data());
    int total = 0, needed = 784 * sizeof(float);
    while (total < needed) {
        int n = send(fd, buf + total, needed - total, 0);
        if (n <= 0) { close(fd); return -1; }
        total += n;
    }

    int prediction = -1;
    int n = recv(fd, &prediction, sizeof(int), MSG_WAITALL);

    auto t1 = std::chrono::high_resolution_clock::now();
    close(fd);

    if (n != sizeof(int)) return -1;

    *pred = prediction;
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// Worker thread: grabs next index from shared counter and sends requests 
void worker_thread(const std::string& host, int port,
                   const std::vector<MnistSample>* samples,
                   std::atomic<int>* next_idx,
                   int total_requests,
                   std::vector<long long>* latencies,
                   std::mutex* lat_mutex,
                   std::atomic<int>* errors,
                   std::atomic<int>* correct) {
    while (true) {
        int i = next_idx->fetch_add(1);
        if (i >= total_requests) break;

        const auto& s = (*samples)[i % samples->size()];
        int pred = -1;
        long long lat = send_request(host, port, s.pixels, &pred);

        if (lat < 0) {
            errors->fetch_add(1);
        } else {
            {
                std::lock_guard<std::mutex> lock(*lat_mutex);
                latencies->push_back(lat);
            }
            if (pred == s.label) correct->fetch_add(1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <coordinator_host> <coordinator_port> <num_requests> "
                     "<concurrency> <mnist_test.bin>\n";
        std::cerr << "Example: ./client localhost 5000 1000 8 mnist_test.bin\n";
        std::cerr << "  concurrency = 1 for serial, >1 for concurrent requests\n";
        return 1;
    }

    std::string host = argv[1];
    int port = std::stoi(argv[2]);
    int num_requests = std::stoi(argv[3]);
    int concurrency = std::stoi(argv[4]);
    std::string mnist_path = argv[5];

    std::vector<MnistSample> samples;
    if (!load_mnist(mnist_path, samples)) {
        std::cerr << "Failed to load " << mnist_path << "\n";
        return 1;
    }
    std::cout << "Loaded " << samples.size() << " MNIST test samples\n";

    std::cout << "Sending " << num_requests << " requests to "
              << host << ":" << port
              << " with concurrency=" << concurrency << "\n";

    std::vector<long long> latencies;
    latencies.reserve(num_requests);
    std::mutex lat_mutex;
    std::atomic<int> next_idx{0};
    std::atomic<int> errors{0};
    std::atomic<int> correct{0};

    auto wall_start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < concurrency; t++) {
        threads.emplace_back(worker_thread, host, port, &samples,
                             &next_idx, num_requests, &latencies, &lat_mutex,
                             &errors, &correct);
    }
    for (auto& th : threads) th.join();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_sec = std::chrono::duration_cast<std::chrono::microseconds>(
                          wall_end - wall_start).count() / 1e6;

    // stats
    std::sort(latencies.begin(), latencies.end());
    double mean_us = latencies.empty() ? 0 :
        std::accumulate(latencies.begin(), latencies.end(), 0LL) / (double)latencies.size();
    long long p50 = latencies.empty() ? 0 : latencies[latencies.size() * 50 / 100];
    long long p95 = latencies.empty() ? 0 : latencies[latencies.size() * 95 / 100];
    long long p99 = latencies.empty() ? 0 : latencies[latencies.size() * 99 / 100];
    double throughput = (num_requests - errors) / wall_sec;
    int completed = num_requests - errors;
    double accuracy = completed == 0 ? 0.0 : (100.0 * correct) / completed;

    std::cout << "\n=== Results ===\n";
    std::cout << "Requests: " << num_requests << "\n";
    std::cout << "Concurrency: " << concurrency << "\n";
    std::cout << "Errors: " << errors.load() << "\n";
    std::cout << "Wall time: " << wall_sec << " s\n";
    std::cout << "Throughput: " << throughput << " req/s\n";
    std::cout << "Mean latency: " << mean_us / 1000.0 << " ms\n";
    std::cout << "P50 latency: " << p50 / 1000.0 << " ms\n";
    std::cout << "P95 latency: " << p95 / 1000.0 << " ms\n";
    std::cout << "P99 latency: " << p99 / 1000.0 << " ms\n";
    std::cout << "Accuracy: " << accuracy << "% ("
              << correct.load() << "/" << completed << ")\n";

    // csv line
    std::cout << "\nCSV," << num_requests << "," << concurrency << ","
              << errors.load() << "," << wall_sec << "," << throughput << ","
              << mean_us / 1000.0 << "," << p50 / 1000.0 << ","
              << p95 / 1000.0 << "," << p99 / 1000.0 << "," << accuracy << "\n";

    return 0;
}
