#pragma once
#include <vector>
#include <string>

// 3-layer MLP: 784 -> 128 -> 64 -> 10
class Model {
public:
    Model();
    bool load_weights(const std::string& path);
    int predict(const std::vector<float>& input); // input: 784 floats, returns class 0-9

private:
    std::vector<float> W1, b1; // 784x128, 128
    std::vector<float> W2, b2; // 128x64, 64
    std::vector<float> W3, b3; // 64x10, 10

    std::vector<float> relu(const std::vector<float>& x);
    std::vector<float> matmul_add(const std::vector<float>& W,
                                   const std::vector<float>& b,
                                   const std::vector<float>& x,
                                   int in_dim, int out_dim);
};
