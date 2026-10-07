#include "model.h"
#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

// Decode explicitly: the file is little-endian regardless of host byte order.
static uint32_t read_u32(std::istream& in) {
  unsigned char bytes[4];
  if (!in.read(reinterpret_cast<char*>(bytes), 4)) throw std::runtime_error("truncated weights");
  uint32_t value = 0;
  for (int index = 0; index < 4; ++index) {
    // Each successive byte contributes eight more bits to the integer.
    value |= uint32_t(bytes[index]) << (8 * index);
  }
  return value;
}
static std::vector<float> read_floats(std::istream& in, size_t count) {
  static_assert(sizeof(float) == 4, "requires float32");
  std::vector<float> values(count);
  for (auto& value : values) {
    auto bits = read_u32(in);
    std::memcpy(&value, &bits, 4);
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite weight");
  }
  return values;
}
Model::Model(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4];
  if (!in.read(magic, 4) || std::memcmp(magic, "MIN1", 4)) throw std::runtime_error("invalid weights header");
  input_ = read_u32(in);
  hidden_ = read_u32(in);
  output_ = read_u32(in);
  // Bound allocations before accepting dimensions from an untrusted file.
  if (input_ < 1 || input_ > 4096 || hidden_ < 1 || hidden_ > 4096 || output_ < 1 || output_ > 4096)
    throw std::runtime_error("invalid model dimensions");
  w1_ = read_floats(in, size_t(input_) * hidden_);
  b1_ = read_floats(in, hidden_);
  w2_ = read_floats(in, size_t(hidden_) * output_);
  b2_ = read_floats(in, output_);
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing weights data");
}
std::vector<Prediction> Model::predict(const std::vector<float>& pixels, int batch) const {
  if (batch < 1 || batch > 8 || pixels.size() != size_t(batch) * input_)
    throw std::invalid_argument("expected 1..8 images of input_dim pixels");
  for (float pixel : pixels) {
    if (!std::isfinite(pixel) || pixel < 0 || pixel > 1) {
      throw std::invalid_argument("pixels must be finite and in [0,1]");
    }
  }
  auto hidden = kernels::matmul(pixels, batch, input_, w1_, hidden_);
  kernels::add_bias(hidden, batch, hidden_, b1_);
  kernels::relu(hidden);
  auto probabilities = kernels::matmul(hidden, batch, hidden_, w2_, output_);
  kernels::add_bias(probabilities, batch, output_, b2_);
  kernels::softmax(probabilities, batch, output_);
  std::vector<Prediction> predictions;
  for (int row = 0; row < batch; ++row) {
    int row_start = row * output_;
    int predicted_class = 0;
    float confidence = probabilities[row_start];
    for (int column = 1; column < output_; ++column) {
      float probability = probabilities[row_start + column];
      if (probability > confidence) {
        predicted_class = column;
        confidence = probability;
      }
    }
    if (!std::isfinite(confidence)) {
      throw std::runtime_error("nonfinite model output");
    }
    predictions.push_back({predicted_class, confidence});
  }
  return predictions;
}
