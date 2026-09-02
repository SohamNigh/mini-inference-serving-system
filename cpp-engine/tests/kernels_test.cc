#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "kernels.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("matmul multiplies row-major matricies", "[kernels]") {
  std::vector<float> a = {1, 2, 3, 4, 5, 6}; // 2x3
  std::vector<float> b = {7, 8, 9, 10, 11, 12}; // 3x2
  std::vector<float> expected = {58, 64, 139, 154}; // 2x2

  auto result = kernels::matmul(a, 2, 3, b, 2);
  REQUIRE(result.size() == expected.size());
  for(size_t i = 0; i < result.size(); ++i) {
    REQUIRE_THAT(result[i], WithinAbs(expected[i], 1e-5));
  }
}

TEST_CASE("add_bias adds bias to every row", "[kernels]") {
  std::vector<float> x = {1, 2, 3, 4};  // 2x2
  std::vector<float> bias = {10, 20};
  kernels::add_bias(x, 2, 2, bias);
  CHECK_THAT(x[0], WithinAbs(11.0f, 1e-4f));
  CHECK_THAT(x[1], WithinAbs(22.0f, 1e-4f));
  CHECK_THAT(x[2], WithinAbs(13.0f, 1e-4f));
  CHECK_THAT(x[3], WithinAbs(24.0f, 1e-4f));
}

TEST_CASE("relu zeroes negatives and passes positives through", "[kernels]") {
  std::vector<float> x = {-1, 0, 2, -3, 5};
  kernels::relu(x);
  std::vector<float> expected = {0, 0, 2, 0, 5};
  for (size_t i = 0; i < x.size(); ++i) {
    CHECK_THAT(x[i], WithinAbs(expected[i], 1e-4f));
  }
}

TEST_CASE("softmax produces a probability distribution per row", "[kernels]") {
  std::vector<float> x = {1, 2, 3};  // single row, n=3
  kernels::softmax(x, 1, 3);
  CHECK_THAT(x[0], WithinAbs(0.09003f, 1e-4f));
  CHECK_THAT(x[1], WithinAbs(0.24473f, 1e-4f));
  CHECK_THAT(x[2], WithinAbs(0.66524f, 1e-4f));
  CHECK_THAT(x[0] + x[1] + x[2], WithinAbs(1.0f, 1e-4f));
}