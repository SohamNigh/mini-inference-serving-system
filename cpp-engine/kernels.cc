#include "kernels.h"
#include <algorithm>
#include <cmath>

namespace kernels {

std::vector<float> matmul(const std::vector<float>& a, int batch, int k,
                           const std::vector<float>& b, int n) {
  std::vector<float> out(static_cast<size_t>(batch) * n, 0.0f);
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        sum += a[static_cast<size_t>(i) * k + p] * b[static_cast<size_t>(p) * n + j];
      }
      out[static_cast<size_t>(i) * n + j] = sum;
    }
  }
  return out;
}

void add_bias(std::vector<float>& x, int batch, int n,
              const std::vector<float>& bias) {
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < n; ++j) {
      x[static_cast<size_t>(i) * n + j] += bias[j];
    }
  }
}

void relu(std::vector<float>& x) {
  for (auto& v : x) {
    v = std::max(0.0f, v);
  }
}

void softmax(std::vector<float>& x, int batch, int n) {
  for (int i = 0; i < batch; ++i) {
    size_t offset = static_cast<size_t>(i) * n;
    float max_val = x[offset];
    for (int j = 1; j < n; ++j) {
      max_val = std::max(max_val, x[offset + j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
      float e = std::exp(x[offset + j] - max_val);
      x[offset + j] = e;
      sum += e;
    }
    for (int j = 0; j < n; ++j) {
      x[offset + j] /= sum;
    }
  }
}

}