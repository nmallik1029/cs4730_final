#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// client: sends MNIST images to coordinator and times responses

struct Sample {
    std::vector<float> px;
    int label;
};

bool load_mnist(const std::string& path, std::vector<Sample>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int n = 0;
    f.read((char*)&n, sizeof(int));
    if (!f || n <= 0) return false;
    out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i].px.resize(784);
        f.read((char*)out[i].px.data(), 784 * sizeof(float));
        f.read((char*)&out[i].label, sizeof(int));
        if (!f) return false;
    }
    return true;
}

// send one image to coord, return latency in us (or -1 on error)
long long one_request(const std::string& host, int port,
                      const std::vector<float>& img, int* pred) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd); return -1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    const char* b = (const char*)img.data();
    int total = 0, need = 784 * sizeof(float);
    while (total < need) {
        int n = send(fd, b + total, need - total, 0);
        if (n <= 0) { close(fd); return -1; }
        total += n;
    }

    int p = -1;
    int n = recv(fd, &p, sizeof(int), MSG_WAITALL);
    auto t1 = std::chrono::high_resolution_clock::now();
    close(fd);
    if (n != sizeof(int)) return -1;
    *pred = p;
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// worker thread: grabs next index, sends request, repeats
void run_thread(const std::string& host, int port,
                const std::vector<Sample>* samples,
                std::atomic<int>* next, int total,
                std::vector<long long>* lats, std::mutex* lm,
                std::atomic<int>* errs, std::atomic<int>* ok) {
    while (true) {
        int i = next->fetch_add(1);
        if (i >= total) break;
        const auto& s = (*samples)[i % samples->size()];
        int pred = -1;
        long long lat = one_request(host, port, s.px, &pred);
        if (lat < 0) {
            errs->fetch_add(1);
        } else {
            { std::lock_guard<std::mutex> g(*lm); lats->push_back(lat); }
            if (pred == s.label) ok->fetch_add(1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " host port num_requests concurrency mnist.bin\n";
        return 1;
    }
    std::string host = argv[1];
    int port = std::stoi(argv[2]);
    int num = std::stoi(argv[3]);
    int conc = std::stoi(argv[4]);
    std::string mpath = argv[5];

    std::vector<Sample> samples;
    if (!load_mnist(mpath, samples)) {
        std::cerr << "couldn't load " << mpath << "\n";
        return 1;
    }
    std::cout << "loaded " << samples.size() << " samples\n";
    std::cout << "sending " << num << " reqs, concurrency=" << conc << "\n";

    std::vector<long long> lats;
    lats.reserve(num);
    std::mutex lm;
    std::atomic<int> next{0}, errs{0}, ok{0};

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> ts;
    for (int t = 0; t < conc; t++) {
        ts.emplace_back(run_thread, host, port, &samples,
                        &next, num, &lats, &lm, &errs, &ok);
    }
    for (auto& th : ts) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

    std::sort(lats.begin(), lats.end());
    double mean = lats.empty() ? 0 :
        std::accumulate(lats.begin(), lats.end(), 0LL) / (double)lats.size();
    long long p50 = lats.empty() ? 0 : lats[lats.size() * 50 / 100];
    long long p95 = lats.empty() ? 0 : lats[lats.size() * 95 / 100];
    long long p99 = lats.empty() ? 0 : lats[lats.size() * 99 / 100];
    double tput = (num - errs) / secs;
    int done = num - errs;
    double acc = done == 0 ? 0.0 : (100.0 * ok) / done;

    std::cout << "\n-- results --\n";
    std::cout << "requests: " << num << "\n";
    std::cout << "errors: " << errs.load() << "\n";
    std::cout << "time: " << secs << " s\n";
    std::cout << "throughput: " << tput << " req/s\n";
    std::cout << "mean lat: " << mean / 1000.0 << " ms\n";
    std::cout << "p50: " << p50 / 1000.0 << " ms\n";
    std::cout << "p95: " << p95 / 1000.0 << " ms\n";
    std::cout << "p99: " << p99 / 1000.0 << " ms\n";
    std::cout << "accuracy:  " << acc << "% (" << ok.load() << "/" << done << ")\n";

    // one-line CSV for scripting
    std::cout << "\nCSV," << num << "," << conc << ","
              << errs.load() << "," << secs << "," << tput << ","
              << mean / 1000.0 << "," << p50 / 1000.0 << ","
              << p95 / 1000.0 << "," << p99 / 1000.0 << "," << acc << "\n";
    return 0;
}
