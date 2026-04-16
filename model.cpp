#include "model.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

Model::Model() {}

bool Model::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open weights file: " << path << std::endl;
        return false;
    }

    auto read_floats = [&](std::vector<float>& vec, int n) {
        vec.resize(n);
        f.read(reinterpret_cast<char*>(vec.data()), n * sizeof(float));
    };

    read_floats(W1, 784 * 128);
    read_floats(b1, 128);
    read_floats(W2, 128 * 64);
    read_floats(b2, 64);
    read_floats(W3, 64 * 10);
    read_floats(b3, 10);

    if (!f) {
        std::cerr << "Error reading weights file (too short?)" << std::endl;
        return false;
    }

    std::cout << "Weights loaded from " << path << std::endl;
    return true;
}

std::vector<float> Model::relu(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++)
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return out;
}

std::vector<float> Model::matmul_add(const std::vector<float>& W,
                                      const std::vector<float>& b,
                                      const std::vector<float>& x,
                                      int in_dim, int out_dim) {
    std::vector<float> out(out_dim, 0.0f);
    for (int i = 0; i < out_dim; i++) {
        out[i] = b[i];
        for (int j = 0; j < in_dim; j++)
            out[i] += W[i * in_dim + j] * x[j];
    }
    return out;
}

int Model::predict(const std::vector<float>& input) {
    if ((int)input.size() != 784) {
        std::cerr << "Expected 784 inputs, got " << input.size() << std::endl;
        return -1;
    }

    auto h1 = relu(matmul_add(W1, b1, input, 784, 128));
    auto h2 = relu(matmul_add(W2, b2, h1, 128, 64));
    auto out = matmul_add(W3, b3, h2, 64, 10);

    return (int)(std::max_element(out.begin(), out.end()) - out.begin());
}
