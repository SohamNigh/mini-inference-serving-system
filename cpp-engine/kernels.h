#pragma once
#include <vector>

namespace kernels {

// A is batch x k (row-major), 
// B is k x n (row-major). 
// Returns batch x n. --> matrix multiplication of A and B
std::vector<float> matmul(const std::vector<float>& a, int batch, int k,
                           const std::vector<float>& b, int n);

// x is batch x n (row-major, modified in place). bias has length n and is
// added to every row.
void add_bias(std::vector<float>& x, int batch, int n, const std::vector<float>& bias);

// In-place elementwise ReLU
void relu(std::vector<float>& x);

// x is batch x n (row-major, modified in place). Softmax is applied independently to each row.
void softmax(std::vector<float>& x, int batch, int n);

}