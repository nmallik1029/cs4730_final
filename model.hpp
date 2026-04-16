#pragma once
#include <vector>
#include <string>

// 3-layer MLP: 784 -> 128 -> 64 -> 10
class Model {
public:
    Model();
    bool load_weights(const std::string& path);
    int predict(const std::vector<float>& input);

private:
    std::vector<float> W1, b1;
    std::vector<float> W2, b2;
    std::vector<float> W3, b3;

    std::vector<float> relu(const std::vector<float>& x);
    std::vector<float> linear(const std::vector<float>& W,
                              const std::vector<float>& b,
                              const std::vector<float>& x,
                              int in_dim, int out_dim);
};
