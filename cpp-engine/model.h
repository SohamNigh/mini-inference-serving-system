#pragma once
#include <string>
#include <vector>
struct Prediction { int predicted_class; float confidence; };
class Model {
 public:
  explicit Model(const std::string& path);
  std::vector<Prediction> predict(const std::vector<float>& pixels, int batch) const;
  int input_dim() const { return input_; }
 private:
  int input_, hidden_, output_;
  std::vector<float> w1_, b1_, w2_, b2_;
};
