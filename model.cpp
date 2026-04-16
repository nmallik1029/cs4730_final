#include "model.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>

Model::Model() {}

bool Model::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "can't open " << path << "\n";
        return false;
    }

    auto read = [&](std::vector<float>& v, int n) {
        v.resize(n);
        f.read((char*)v.data(), n * sizeof(float));
    };

    // order must match weights_export.py / train_worker.py
    read(W1, 784 * 128);
    read(b1, 128);
    read(W2, 128 * 64);
    read(b2, 64);
    read(W3, 64 * 10);
    read(b3, 10);

    if (!f) {
        std::cerr << "weights file too short\n";
        return false;
    }
    return true;
}

std::vector<float> Model::relu(const std::vector<float>& x) {
    std::vector<float> o(x.size());
    for (size_t i = 0; i < x.size(); i++) o[i] = x[i] > 0 ? x[i] : 0;
    return o;
}

// y = W*x + b, where W is stored as (out_dim, in_dim)
std::vector<float> Model::linear(const std::vector<float>& W,
                                  const std::vector<float>& b,
                                  const std::vector<float>& x,
                                  int in_dim, int out_dim) {
    std::vector<float> y(out_dim);
    for (int i = 0; i < out_dim; i++) {
        float s = b[i];
        for (int j = 0; j < in_dim; j++) s += W[i * in_dim + j] * x[j];
        y[i] = s;
    }
    return y;
}

int Model::predict(const std::vector<float>& input) {
    if ((int)input.size() != 784) return -1;
    auto h1 = relu(linear(W1, b1, input, 784, 128));
    auto h2 = relu(linear(W2, b2, h1,    128, 64));
    auto out = linear(W3, b3, h2, 64, 10);
    return (int)(std::max_element(out.begin(), out.end()) - out.begin());
}
