# Mini Inference Serving System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-process inference serving system — a Go REST server that queues and dynamically batches requests, and a C++ gRPC engine that runs a hand-rolled neural network — for an MNIST digit classifier.

**Architecture:** `Client --HTTP/JSON--> [Go server: handler -> batcher -> gRPC client] --gRPC--> [C++ engine: gRPC service -> Model -> kernels]`. Both processes are independent binaries sharing a protobuf contract. The model is trained offline in Python and loaded from a binary weights file at startup by the C++ engine.

**Tech Stack:** Go 1.22+ (net/http, google.golang.org/grpc, prometheus client_golang), C++17 (gRPC C++, Protobuf, CMake, Catch2 for tests), Python 3 + numpy (offline training only).

**Spec:** `docs/superpowers/specs/2026-08-25-mini-inference-serving-system-design.md`

## Global Constraints

- No ML framework (no ONNX Runtime, no libtorch) in the C++ engine — only hand-rolled kernels.
- Single fixed model (2-layer FC net, 784→128→10) — no multi-model support in v1.
- External API is REST/HTTP+JSON; internal Go↔C++ link is gRPC.
- Go and C++ communicate only via the `proto/inference.proto` contract.
- Weight file format must be dimension-generic (dims read from the file header), not hardcoded to 784/128/10, so small fixtures can be used in unit tests.

---

## Prerequisites (one-time host setup)

Before Task 1, make sure these are installed (macOS/Homebrew commands shown):

```bash
brew install go cmake protobuf grpc python3
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install numpy
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

Make sure `$(go env GOPATH)/bin` is on your `PATH` so `protoc-gen-go` and `protoc-gen-go-grpc` are found by `protoc`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `go-server/go.mod`
- Create: `go-server/main.go`
- Create: `cpp-engine/.gitkeep`, `proto/.gitkeep`, `training/.gitkeep`, `testdata/.gitkeep`, `scripts/.gitkeep`

**Interfaces:**
- Produces: repo directory layout (`go-server/`, `cpp-engine/`, `proto/`, `training/`, `testdata/`, `scripts/`) that every later task assumes exists.

- [ ] **Step 1: Create the directory layout and .gitignore**

```bash
mkdir -p go-server cpp-engine/tests cpp-engine/tests/fixtures proto training/data testdata scripts
touch cpp-engine/.gitkeep proto/.gitkeep testdata/.gitkeep scripts/.gitkeep
```

Write `.gitignore`:

```
# Go
go-server/inference-server
go-server/*.test

# C++ build output
cpp-engine/build/

# Python
training/.venv/
training/data/
__pycache__/
*.pyc

# generated weights (produced by training, not hand-edited)
training/weights.bin
```

- [ ] **Step 2: Write a one-paragraph README stub**

Write `README.md`:

```markdown
# Mini Inference Serving System

A learning project: a Go REST server that dynamically batches inference
requests and forwards them over gRPC to a C++ engine running a hand-rolled
neural network (MNIST digit classifier).

See `docs/superpowers/specs/2026-08-25-mini-inference-serving-system-design.md`
for the design and `docs/superpowers/plans/2026-08-25-mini-inference-serving-system.md`
for the build plan.
```

- [ ] **Step 3: Initialize the Go module with a placeholder main**

```bash
cd go-server
go mod init mini-inference-serving-system/go-server
```

Write `go-server/main.go`:

```go
package main

import "fmt"

func main() {
	fmt.Println("go-server placeholder")
}
```

- [ ] **Step 4: Verify the Go module builds**

Run: `cd go-server && go build ./... && ./go-server` (or `go run .`)
Expected: prints `go-server placeholder`, exits 0.

- [ ] **Step 5: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add .gitignore README.md go-server cpp-engine proto training testdata scripts
git commit -m "Scaffold project directory layout"
```

---

### Task 2: Proto contract and generated stubs

**Files:**
- Create: `proto/inference.proto`
- Create: `scripts/gen-proto.sh`
- Create (generated, committed): `go-server/internal/pb/inference.pb.go`, `go-server/internal/pb/inference_grpc.pb.go`
- Create (generated, committed): `cpp-engine/generated/inference.pb.h`, `cpp-engine/generated/inference.pb.cc`, `cpp-engine/generated/inference.grpc.pb.h`, `cpp-engine/generated/inference.grpc.pb.cc`

**Interfaces:**
- Produces: Go package `mini-inference-serving-system/go-server/internal/pb` exposing `pb.InferenceEngineClient`, `pb.InferenceEngineServer`, `pb.Image{Pixels []float32}`, `pb.PredictBatchRequest{Images []*pb.Image}`, `pb.Prediction{PredictedClass int32, Confidence float32}`, `pb.PredictBatchResponse{Predictions []*pb.Prediction}`.
- Produces: C++ namespace `inference` exposing `inference::InferenceEngine::Service`, `inference::InferenceEngine::Stub`, `inference::Image`, `inference::PredictBatchRequest`, `inference::Prediction`, `inference::PredictBatchResponse` (mirroring the same field names as generated by protoc's C++ plugin).

- [ ] **Step 1: Write the proto contract**

Write `proto/inference.proto`:

```protobuf
syntax = "proto3";
package inference;

option go_package = "mini-inference-serving-system/go-server/internal/pb;pb";

service InferenceEngine {
  rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);
}

message Image {
  repeated float pixels = 1; // length must equal the model's input_dim
}

message PredictBatchRequest {
  repeated Image images = 1;
}

message Prediction {
  int32 predicted_class = 1;
  float confidence = 2;
}

message PredictBatchResponse {
  repeated Prediction predictions = 1; // same order/length as the request
}
```

- [ ] **Step 2: Write the codegen script**

Write `scripts/gen-proto.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p go-server/internal/pb
protoc \
  --proto_path=proto \
  --go_out=go-server/internal/pb --go_opt=paths=source_relative \
  --go-grpc_out=go-server/internal/pb --go-grpc_opt=paths=source_relative \
  inference.proto

mkdir -p cpp-engine/generated
protoc \
  --proto_path=proto \
  --cpp_out=cpp-engine/generated \
  --grpc_out=cpp-engine/generated \
  --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
  inference.proto

echo "Generated Go stubs in go-server/internal/pb/ and C++ stubs in cpp-engine/generated/"
```

- [ ] **Step 3: Run codegen and verify output**

```bash
chmod +x scripts/gen-proto.sh
./scripts/gen-proto.sh
ls go-server/internal/pb/
ls cpp-engine/generated/
```

Expected: `go-server/internal/pb/` contains `inference.pb.go` and `inference_grpc.pb.go`; `cpp-engine/generated/` contains `inference.pb.h`, `inference.pb.cc`, `inference.grpc.pb.h`, `inference.grpc.pb.cc`.

- [ ] **Step 4: Verify the Go stubs compile**

```bash
cd go-server
go build ./...
```

Expected: succeeds with no errors (generated package compiles; nothing references it yet).

- [ ] **Step 5: Add the Go gRPC/protobuf dependencies to go.mod**

```bash
cd go-server
go get google.golang.org/grpc@latest
go get google.golang.org/protobuf@latest
go build ./...
```

Expected: `go.mod`/`go.sum` updated, build still succeeds.

- [ ] **Step 6: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add proto scripts/gen-proto.sh go-server/internal/pb go-server/go.mod go-server/go.sum
git commit -m "Add inference.proto contract and generated Go/C++ stubs"
```

---

### Task 3: C++ math kernels with unit tests

**Files:**
- Create: `cpp-engine/kernels.h`
- Create: `cpp-engine/kernels.cc`
- Create: `cpp-engine/tests/kernels_test.cc`
- Create: `cpp-engine/CMakeLists.txt`

**Interfaces:**
- Produces: `kernels::matmul(a, batch, k, b, n) -> std::vector<float>`, `kernels::add_bias(x, batch, n, bias)` (in-place), `kernels::relu(x)` (in-place), `kernels::softmax(x, batch, n)` (in-place, per-row). All matrices are row-major `std::vector<float>` with explicit dimensions passed alongside — this is the shape every later kernel/model consumer relies on.

- [ ] **Step 1: Write the kernel interface**

Write `cpp-engine/kernels.h`:

```cpp
#pragma once
#include <vector>

namespace kernels {

// A is batch x k (row-major), B is k x n (row-major). Returns batch x n.
std::vector<float> matmul(const std::vector<float>& a, int batch, int k,
                           const std::vector<float>& b, int n);

// x is batch x n (row-major, modified in place). bias has length n and is
// added to every row.
void add_bias(std::vector<float>& x, int batch, int n,
              const std::vector<float>& bias);

// In-place elementwise ReLU.
void relu(std::vector<float>& x);

// x is batch x n (row-major, modified in place). Softmax is applied
// independently to each row.
void softmax(std::vector<float>& x, int batch, int n);

}  // namespace kernels
```

- [ ] **Step 2: Write the failing tests**

Write `cpp-engine/tests/kernels_test.cc`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "kernels.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("matmul multiplies row-major matrices", "[kernels]") {
  std::vector<float> a = {1, 2, 3, 4};  // 2x2
  std::vector<float> b = {5, 6, 7, 8};  // 2x2
  auto c = kernels::matmul(a, 2, 2, b, 2);
  REQUIRE(c.size() == 4);
  CHECK_THAT(c[0], WithinAbs(19.0f, 1e-4f));
  CHECK_THAT(c[1], WithinAbs(22.0f, 1e-4f));
  CHECK_THAT(c[2], WithinAbs(43.0f, 1e-4f));
  CHECK_THAT(c[3], WithinAbs(50.0f, 1e-4f));
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
```

- [ ] **Step 3: Write the CMake build**

Write `cpp-engine/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(catch2)

add_library(inference_core
  kernels.cc
)
target_include_directories(inference_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_executable(inference_tests
  tests/kernels_test.cc
)
target_link_libraries(inference_tests PRIVATE inference_core Catch2::Catch2WithMain)
add_test(NAME inference_tests COMMAND inference_tests)
```

- [ ] **Step 4: Confirm the tests fail to build (no implementation yet)**

```bash
cd cpp-engine
cmake -S . -B build
cmake --build build
```

Expected: build fails with an "undefined reference" / unresolved symbol error for `kernels::matmul` etc. (declared in the header, not yet defined).

- [ ] **Step 5: Implement the kernels**

Write `cpp-engine/kernels.cc`:

```cpp
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

}  // namespace kernels
```

- [ ] **Step 6: Build and run the tests**

```bash
cd cpp-engine
cmake --build build
./build/inference_tests
```

Expected: all 4 test cases pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add cpp-engine/kernels.h cpp-engine/kernels.cc cpp-engine/tests/kernels_test.cc cpp-engine/CMakeLists.txt
git commit -m "Add hand-rolled matmul/add_bias/relu/softmax kernels with tests"
```

---

### Task 4: Weight file format and Model class

**Files:**
- Create: `training/weights_io.py`
- Create: `cpp-engine/tests/fixtures/make_tiny_fixture.py`
- Create: `cpp-engine/model.h`
- Create: `cpp-engine/model.cc`
- Create: `cpp-engine/tests/model_test.cc`
- Modify: `cpp-engine/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks except the repo layout.
- Produces: `Model(weights_path)` constructor that throws `std::runtime_error` if the file is malformed; `Model::Predict(const std::vector<std::vector<float>>& batch) const -> std::vector<Prediction>` where `struct Prediction { int predicted_class; float confidence; };`. This is the exact type later consumed by the gRPC service in Task 5. Weight file binary layout (little-endian): `char magic[4] = "MINF"`, `uint32 version = 1`, `uint32 input_dim`, `uint32 hidden_dim`, `uint32 output_dim`, then `float32[input_dim*hidden_dim] W1`, `float32[hidden_dim] b1`, `float32[hidden_dim*output_dim] W2`, `float32[output_dim] b2`, all row-major.

- [ ] **Step 1: Write the shared Python weight writer**

Write `training/weights_io.py`:

```python
import struct

MAGIC = b"MINF"
VERSION = 1


def write_weights(path, w1, b1, w2, b2, input_dim, hidden_dim, output_dim):
    """w1: input_dim x hidden_dim, w2: hidden_dim x output_dim, row-major flat lists."""
    assert len(w1) == input_dim * hidden_dim
    assert len(b1) == hidden_dim
    assert len(w2) == hidden_dim * output_dim
    assert len(b2) == output_dim

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", input_dim))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", output_dim))
        f.write(struct.pack(f"<{len(w1)}f", *w1))
        f.write(struct.pack(f"<{len(b1)}f", *b1))
        f.write(struct.pack(f"<{len(w2)}f", *w2))
        f.write(struct.pack(f"<{len(b2)}f", *b2))
```

- [ ] **Step 2: Generate a tiny deterministic fixture for C++ tests**

Write `cpp-engine/tests/fixtures/make_tiny_fixture.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "training"))
from weights_io import write_weights  # noqa: E402

# input_dim=4, hidden_dim=3, output_dim=2, chosen so the forward pass for
# input [1,0,0,0] can be hand-computed:
#   hidden = relu([0.1,0.2,0.3] + [0,0,0]) = [0.1,0.2,0.3]
#   logits = hidden @ W2 + b2 = [0.4, 0.5]
#   softmax([0.4,0.5]) ~= [0.4750, 0.5250] -> predicted_class=1
w1 = [
    0.1, 0.2, 0.3,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
]
b1 = [0.0, 0.0, 0.0]
w2 = [
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
]
b2 = [0.0, 0.0]

out_path = os.path.join(os.path.dirname(__file__), "tiny_weights.bin")
write_weights(out_path, w1, b1, w2, b2, input_dim=4, hidden_dim=3, output_dim=2)
print(f"wrote {out_path}")
```

Run it:

```bash
cd cpp-engine/tests/fixtures
python3 make_tiny_fixture.py
```

Expected: prints `wrote .../tiny_weights.bin`; the file is 4+4+4+4+4 + (12+3+6+2)*4 = 20 + 92 = 112 bytes.

- [ ] **Step 3: Write the Model interface**

Write `cpp-engine/model.h`:

```cpp
#pragma once
#include <string>
#include <vector>

struct Prediction {
  int predicted_class;
  float confidence;
};

class Model {
 public:
  explicit Model(const std::string& weights_path);

  std::vector<Prediction> Predict(
      const std::vector<std::vector<float>>& batch) const;

  int input_dim() const { return input_dim_; }

 private:
  int input_dim_;
  int hidden_dim_;
  int output_dim_;
  std::vector<float> w1_, b1_, w2_, b2_;
};
```

- [ ] **Step 4: Write the failing model test**

Write `cpp-engine/tests/model_test.cc`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "model.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("Model loads a fixture and predicts the expected class", "[model]") {
  Model model("tests/fixtures/tiny_weights.bin");
  REQUIRE(model.input_dim() == 4);

  std::vector<std::vector<float>> batch = {{1.0f, 0.0f, 0.0f, 0.0f}};
  auto predictions = model.Predict(batch);

  REQUIRE(predictions.size() == 1);
  CHECK(predictions[0].predicted_class == 1);
  CHECK_THAT(predictions[0].confidence, WithinAbs(0.5250f, 0.01f));
}

TEST_CASE("Model throws on a missing weights file", "[model]") {
  CHECK_THROWS_AS(Model("does_not_exist.bin"), std::runtime_error);
}
```

- [ ] **Step 5: Update CMakeLists to build model + its test**

Modify `cpp-engine/CMakeLists.txt`: change the `inference_core` library and test executable to include the new files:

```cmake
add_library(inference_core
  kernels.cc
  model.cc
)
target_include_directories(inference_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_executable(inference_tests
  tests/kernels_test.cc
  tests/model_test.cc
)
target_link_libraries(inference_tests PRIVATE inference_core Catch2::Catch2WithMain)
add_test(NAME inference_tests COMMAND inference_tests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
```

(`WORKING_DIRECTORY` is set so the test's relative path `tests/fixtures/tiny_weights.bin` resolves correctly when run via `ctest` or the binary directly from `cpp-engine/`.)

- [ ] **Step 6: Confirm it fails to build (model.cc missing)**

```bash
cd cpp-engine
cmake -S . -B build
cmake --build build
```

Expected: link error — `model.cc` declared in CMakeLists but doesn't exist yet, or undefined reference to `Model::Model`/`Model::Predict`.

- [ ] **Step 7: Implement the Model class**

Write `cpp-engine/model.cc`:

```cpp
#include "model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "kernels.h"

namespace {

uint32_t read_u32(std::ifstream& f) {
  uint32_t v;
  f.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!f) throw std::runtime_error("unexpected end of weights file");
  return v;
}

std::vector<float> read_floats(std::ifstream& f, size_t count) {
  std::vector<float> v(count);
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
  if (!f) throw std::runtime_error("unexpected end of weights file");
  return v;
}

}  // namespace

Model::Model(const std::string& weights_path) {
  std::ifstream f(weights_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("could not open weights file: " + weights_path);
  }

  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, "MINF", 4) != 0) {
    throw std::runtime_error("bad magic in weights file: " + weights_path);
  }

  uint32_t version = read_u32(f);
  if (version != 1) {
    throw std::runtime_error("unsupported weights file version");
  }

  input_dim_ = static_cast<int>(read_u32(f));
  hidden_dim_ = static_cast<int>(read_u32(f));
  output_dim_ = static_cast<int>(read_u32(f));

  w1_ = read_floats(f, static_cast<size_t>(input_dim_) * hidden_dim_);
  b1_ = read_floats(f, hidden_dim_);
  w2_ = read_floats(f, static_cast<size_t>(hidden_dim_) * output_dim_);
  b2_ = read_floats(f, output_dim_);
}

std::vector<Prediction> Model::Predict(
    const std::vector<std::vector<float>>& batch) const {
  int batch_size = static_cast<int>(batch.size());

  std::vector<float> input(static_cast<size_t>(batch_size) * input_dim_);
  for (int i = 0; i < batch_size; ++i) {
    std::copy(batch[i].begin(), batch[i].end(),
              input.begin() + static_cast<size_t>(i) * input_dim_);
  }

  auto hidden = kernels::matmul(input, batch_size, input_dim_, w1_, hidden_dim_);
  kernels::add_bias(hidden, batch_size, hidden_dim_, b1_);
  kernels::relu(hidden);

  auto logits = kernels::matmul(hidden, batch_size, hidden_dim_, w2_, output_dim_);
  kernels::add_bias(logits, batch_size, output_dim_, b2_);
  kernels::softmax(logits, batch_size, output_dim_);

  std::vector<Prediction> predictions(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    size_t offset = static_cast<size_t>(i) * output_dim_;
    int best_class = 0;
    float best_prob = logits[offset];
    for (int j = 1; j < output_dim_; ++j) {
      if (logits[offset + j] > best_prob) {
        best_prob = logits[offset + j];
        best_class = j;
      }
    }
    predictions[i] = {best_class, best_prob};
  }
  return predictions;
}
```

- [ ] **Step 8: Build and run the tests**

```bash
cd cpp-engine
cmake --build build
./build/inference_tests
```

Expected: all tests pass, including the two new `[model]` cases.

- [ ] **Step 9: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add training/weights_io.py cpp-engine/tests/fixtures cpp-engine/model.h cpp-engine/model.cc cpp-engine/tests/model_test.cc cpp-engine/CMakeLists.txt
git commit -m "Add weight file format and Model forward pass with fixture test"
```

---

### Task 5: C++ gRPC service and engine binary

**Files:**
- Create: `cpp-engine/service.h`
- Create: `cpp-engine/service.cc`
- Create: `cpp-engine/main.cc`
- Modify: `cpp-engine/CMakeLists.txt`

**Interfaces:**
- Consumes: `Model` and `Prediction` from Task 4 (`model.h`); generated `inference::InferenceEngine::Service`, `inference::PredictBatchRequest`, `inference::PredictBatchResponse` from Task 2 (`cpp-engine/generated/inference.grpc.pb.h`).
- Produces: a runnable `inference_engine` binary listening on `0.0.0.0:50051`, consumed by Go integration in Task 8 and the e2e script in Task 11.

- [ ] **Step 1: Write the service adapter**

Write `cpp-engine/service.h`:

```cpp
#pragma once
#include <grpcpp/grpcpp.h>

#include "generated/inference.grpc.pb.h"
#include "model.h"

class InferenceEngineServiceImpl final : public inference::InferenceEngine::Service {
 public:
  explicit InferenceEngineServiceImpl(const Model& model);

  grpc::Status PredictBatch(grpc::ServerContext* context,
                             const inference::PredictBatchRequest* request,
                             inference::PredictBatchResponse* response) override;

 private:
  const Model& model_;
};
```

Write `cpp-engine/service.cc`:

```cpp
#include "service.h"

InferenceEngineServiceImpl::InferenceEngineServiceImpl(const Model& model)
    : model_(model) {}

grpc::Status InferenceEngineServiceImpl::PredictBatch(
    grpc::ServerContext* /*context*/,
    const inference::PredictBatchRequest* request,
    inference::PredictBatchResponse* response) {
  std::vector<std::vector<float>> batch;
  batch.reserve(request->images_size());
  for (const auto& image : request->images()) {
    batch.emplace_back(image.pixels().begin(), image.pixels().end());
  }

  if (batch.empty()) {
    return grpc::Status::OK;
  }

  auto predictions = model_.Predict(batch);
  for (const auto& p : predictions) {
    auto* out = response->add_predictions();
    out->set_predicted_class(p.predicted_class);
    out->set_confidence(p.confidence);
  }
  return grpc::Status::OK;
}
```

- [ ] **Step 2: Write the server entrypoint**

Write `cpp-engine/main.cc`:

```cpp
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

#include "model.h"
#include "service.h"

int main(int argc, char** argv) {
  std::string weights_path = argc > 1 ? argv[1] : "../training/weights.bin";
  std::string server_address = "0.0.0.0:50051";

  Model model(weights_path);
  InferenceEngineServiceImpl service(model);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "C++ inference engine listening on " << server_address
            << " (model input_dim=" << model.input_dim() << ")" << std::endl;
  server->Wait();
  return 0;
}
```

- [ ] **Step 3: Update CMakeLists to link gRPC/Protobuf and build the engine binary**

Modify `cpp-engine/CMakeLists.txt` to its full new contents:

```cmake
cmake_minimum_required(VERSION 3.16)
project(inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(catch2)

add_library(inference_core
  kernels.cc
  model.cc
  service.cc
  generated/inference.pb.cc
  generated/inference.grpc.pb.cc
)
target_include_directories(inference_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
)
target_link_libraries(inference_core PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_executable(inference_engine main.cc)
target_link_libraries(inference_engine PRIVATE inference_core)

enable_testing()
add_executable(inference_tests
  tests/kernels_test.cc
  tests/model_test.cc
)
target_link_libraries(inference_tests PRIVATE inference_core Catch2::Catch2WithMain)
add_test(NAME inference_tests COMMAND inference_tests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
```

- [ ] **Step 4: Build**

```bash
cd cpp-engine
cmake -S . -B build
cmake --build build
```

Expected: `build/inference_engine` and `build/inference_tests` both build successfully.

- [ ] **Step 5: Run the unit tests (still pass, unaffected by the service layer)**

```bash
./build/inference_tests
```

Expected: all tests still pass.

- [ ] **Step 6: Manually verify the engine starts using the tiny fixture**

```bash
cd cpp-engine
./build/inference_engine tests/fixtures/tiny_weights.bin
```

Expected: prints `C++ inference engine listening on 0.0.0.0:50051 (model input_dim=4)` and blocks (Ctrl+C to stop). This confirms the binary runs end-to-end; full request/response verification happens in Task 8 once the Go client exists.

- [ ] **Step 7: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add cpp-engine/service.h cpp-engine/service.cc cpp-engine/main.cc cpp-engine/CMakeLists.txt cpp-engine/generated
git commit -m "Add gRPC service adapter and engine entrypoint"
```

---

### Task 6: Offline training pipeline

**Files:**
- Create: `training/requirements.txt`
- Create: `training/mnist_loader.py`
- Create: `training/train.py`
- Create: `testdata/sample_digit.json` (generated output, committed)

**Interfaces:**
- Consumes: `training/weights_io.write_weights(...)` from Task 4.
- Produces: `training/weights.bin` (real MNIST-trained weights, input_dim=784, hidden_dim=128, output_dim=10) consumed by the C++ engine in Task 11's end-to-end run; `testdata/sample_digit.json` (`{"pixels": [784 floats], "label": int}`) consumed by the e2e script in Task 11.

- [ ] **Step 1: Write requirements**

Write `training/requirements.txt`:

```
numpy>=1.24
```

- [ ] **Step 2: Write the MNIST loader (downloads and parses idx files)**

Write `training/mnist_loader.py`:

```python
import gzip
import os
import struct
import urllib.request

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _download(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, FILES[name])
    if not os.path.exists(dest):
        print(f"downloading {FILES[name]}...")
        urllib.request.urlretrieve(BASE_URL + FILES[name], dest)
    return dest


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic for images file: {magic}"
        buf = f.read(count * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(count, rows * cols)
        return data.astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic for labels file: {magic}"
        buf = f.read(count)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist():
    train_images = _read_images(_download("train_images"))
    train_labels = _read_labels(_download("train_labels"))
    test_images = _read_images(_download("test_images"))
    test_labels = _read_labels(_download("test_labels"))
    return train_images, train_labels, test_images, test_labels
```

- [ ] **Step 3: Write the training script**

Write `training/train.py`:

```python
import json
import os

import numpy as np

from mnist_loader import load_mnist
from weights_io import write_weights

INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.1


def one_hot(labels, num_classes):
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def train():
    train_images, train_labels, test_images, test_labels = load_mnist()

    rng = np.random.default_rng(seed=0)
    w1 = rng.normal(0, 0.1, size=(INPUT_DIM, HIDDEN_DIM)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    w2 = rng.normal(0, 0.1, size=(HIDDEN_DIM, OUTPUT_DIM)).astype(np.float32)
    b2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

    n = train_images.shape[0]
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = train_images[idx]
            y = one_hot(train_labels[idx], OUTPUT_DIM)
            bs = x.shape[0]

            z1 = x @ w1 + b1
            h = np.maximum(0, z1)
            z2 = h @ w2 + b2
            probs = softmax(z2)

            dz2 = (probs - y) / bs
            dw2 = h.T @ dz2
            db2 = dz2.sum(axis=0)
            dh = dz2 @ w2.T
            dz1 = dh * (z1 > 0)
            dw1 = x.T @ dz1
            db1 = dz1.sum(axis=0)

            w1 -= LEARNING_RATE * dw1
            b1 -= LEARNING_RATE * db1
            w2 -= LEARNING_RATE * dw2
            b2 -= LEARNING_RATE * db2

        test_h = np.maximum(0, test_images @ w1 + b1)
        test_probs = softmax(test_h @ w2 + b2)
        acc = (np.argmax(test_probs, axis=1) == test_labels).mean()
        print(f"epoch {epoch + 1}/{EPOCHS} test_accuracy={acc:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "weights.bin")
    write_weights(
        out_path,
        w1.flatten().tolist(), b1.tolist(),
        w2.flatten().tolist(), b2.tolist(),
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
    )
    print(f"wrote {out_path}")

    sample_idx = 0
    sample = {
        "pixels": test_images[sample_idx].tolist(),
        "label": int(test_labels[sample_idx]),
    }
    testdata_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "sample_digit.json")
    with open(testdata_path, "w") as f:
        json.dump(sample, f)
    print(f"wrote {testdata_path}")


if __name__ == "__main__":
    train()
```

- [ ] **Step 4: Run training**

```bash
cd training
source .venv/bin/activate  # from Prerequisites
pip install -r requirements.txt
python3 train.py
```

Expected: downloads MNIST on first run, prints per-epoch test accuracy (should reach roughly 90%+ by epoch 5 for this simple architecture), writes `training/weights.bin` and `testdata/sample_digit.json`.

- [ ] **Step 5: Sanity-check the trained model against the C++ engine**

```bash
cd cpp-engine
cmake --build build
./build/inference_engine ../training/weights.bin
```

Expected: prints `C++ inference engine listening on 0.0.0.0:50051 (model input_dim=784)`. Ctrl+C to stop; full request round trip is verified in Task 11.

- [ ] **Step 6: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add training/requirements.txt training/mnist_loader.py training/train.py testdata/sample_digit.json
git commit -m "Add MNIST training pipeline producing weights.bin and a sample test image"
```

(`training/weights.bin` itself stays untracked per `.gitignore` — it's a generated artifact, regenerated by re-running `train.py`.)

---

### Task 7: Go batcher core with fake-client tests

**Files:**
- Create: `go-server/batcher.go`
- Create: `go-server/batcher_test.go`

**Interfaces:**
- Produces: `type Prediction struct { Class int32; Confidence float32 }`; `type EngineClient interface { PredictBatch(ctx context.Context, images [][]float32) ([]Prediction, error) }`; `type Batcher struct{...}` with `NewBatcher(client EngineClient, maxBatch int, maxWait time.Duration) *Batcher` and `func (b *Batcher) Submit(ctx context.Context, pixels []float32) (Prediction, error)`. `Batcher` satisfies the `Submitter` interface consumed by the HTTP handler in Task 9.

- [ ] **Step 1: Write the failing tests**

Write `go-server/batcher_test.go`:

```go
package main

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type fakeEngineClient struct {
	mu        sync.Mutex
	calls     [][][]float32
	predictFn func(images [][]float32) ([]Prediction, error)
}

func (f *fakeEngineClient) PredictBatch(_ context.Context, images [][]float32) ([]Prediction, error) {
	f.mu.Lock()
	f.calls = append(f.calls, images)
	f.mu.Unlock()
	return f.predictFn(images)
}

func (f *fakeEngineClient) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func echoClassZero(images [][]float32) ([]Prediction, error) {
	preds := make([]Prediction, len(images))
	for i := range images {
		preds[i] = Prediction{Class: 0, Confidence: 0.9}
	}
	return preds, nil
}

func TestBatchFlushesAtMaxSize(t *testing.T) {
	client := &fakeEngineClient{predictFn: echoClassZero}
	b := NewBatcher(client, 2, time.Second) // large wait so size triggers first

	var wg sync.WaitGroup
	results := make([]Prediction, 2)
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], errs[i] = b.Submit(context.Background(), []float32{0, 0, 0, 0})
		}()
	}
	wg.Wait()

	for i := range results {
		if errs[i] != nil {
			t.Fatalf("unexpected error: %v", errs[i])
		}
		if results[i].Class != 0 {
			t.Fatalf("expected class 0, got %d", results[i].Class)
		}
	}
	if client.callCount() != 1 {
		t.Fatalf("expected exactly 1 batch call, got %d", client.callCount())
	}
}

func TestBatchFlushesAtMaxWait(t *testing.T) {
	client := &fakeEngineClient{predictFn: echoClassZero}
	b := NewBatcher(client, 100, 20*time.Millisecond) // large size so timeout triggers

	result, err := b.Submit(context.Background(), []float32{0, 0, 0, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Class != 0 {
		t.Fatalf("expected class 0, got %d", result.Class)
	}
	if client.callCount() != 1 {
		t.Fatalf("expected exactly 1 batch call, got %d", client.callCount())
	}
}

func TestExpiredJobExcludedFromBatch(t *testing.T) {
	client := &fakeEngineClient{predictFn: echoClassZero}
	b := NewBatcher(client, 100, 200*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()
	time.Sleep(5 * time.Millisecond) // ensure ctx is already expired before Submit

	_, err := b.Submit(ctx, []float32{0, 0, 0, 0})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
}

func TestBackendErrorFailsWholeBatch(t *testing.T) {
	wantErr := errors.New("engine unavailable")
	client := &fakeEngineClient{predictFn: func(images [][]float32) ([]Prediction, error) {
		return nil, wantErr
	}}
	b := NewBatcher(client, 2, time.Second)

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, errs[i] = b.Submit(context.Background(), []float32{0, 0, 0, 0})
		}()
	}
	wg.Wait()

	for i := range errs {
		if !errors.Is(errs[i], wantErr) {
			t.Fatalf("expected %v, got %v", wantErr, errs[i])
		}
	}
}
```

- [ ] **Step 2: Confirm the tests fail to compile (no implementation yet)**

```bash
cd go-server
go test ./...
```

Expected: compile error — `Prediction`, `EngineClient`, `Batcher`, `NewBatcher` undefined.

- [ ] **Step 3: Implement the batcher**

Write `go-server/batcher.go`:

```go
package main

import (
	"context"
	"time"
)

type Prediction struct {
	Class      int32
	Confidence float32
}

type EngineClient interface {
	PredictBatch(ctx context.Context, images [][]float32) ([]Prediction, error)
}

type job struct {
	ctx      context.Context
	pixels   []float32
	resultCh chan jobResult
}

type jobResult struct {
	prediction Prediction
	err        error
}

type Batcher struct {
	client   EngineClient
	maxBatch int
	maxWait  time.Duration
	queue    chan job
}

func NewBatcher(client EngineClient, maxBatch int, maxWait time.Duration) *Batcher {
	b := &Batcher{
		client:   client,
		maxBatch: maxBatch,
		maxWait:  maxWait,
		queue:    make(chan job, 1024),
	}
	go b.run()
	return b
}

// Submit enqueues a request and blocks until a result is available or ctx
// is done, whichever happens first.
func (b *Batcher) Submit(ctx context.Context, pixels []float32) (Prediction, error) {
	j := job{ctx: ctx, pixels: pixels, resultCh: make(chan jobResult, 1)}
	b.queue <- j
	select {
	case r := <-j.resultCh:
		return r.prediction, r.err
	case <-ctx.Done():
		return Prediction{}, ctx.Err()
	}
}

func (b *Batcher) run() {
	for {
		batch := b.collectBatch()
		if len(batch) > 0 {
			b.dispatch(batch)
		}
	}
}

// collectBatch blocks for at least one job, then keeps adding jobs until
// maxBatch is reached or maxWait elapses since the batch started forming.
// Jobs whose context already expired are dropped and completed immediately
// with context.DeadlineExceeded instead of consuming a batch slot.
func (b *Batcher) collectBatch() []job {
	var batch []job

	first := <-b.queue
	if isExpired(first) {
		first.resultCh <- jobResult{err: context.DeadlineExceeded}
	} else {
		batch = append(batch, first)
	}

	timer := time.NewTimer(b.maxWait)
	defer timer.Stop()

	for len(batch) < b.maxBatch {
		select {
		case j := <-b.queue:
			if isExpired(j) {
				j.resultCh <- jobResult{err: context.DeadlineExceeded}
				continue
			}
			batch = append(batch, j)
		case <-timer.C:
			return batch
		}
	}
	return batch
}

func isExpired(j job) bool {
	select {
	case <-j.ctx.Done():
		return true
	default:
		return false
	}
}

func (b *Batcher) dispatch(batch []job) {
	images := make([][]float32, len(batch))
	for i, j := range batch {
		images[i] = j.pixels
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	predictions, err := b.client.PredictBatch(ctx, images)
	if err != nil {
		for _, j := range batch {
			j.resultCh <- jobResult{err: err}
		}
		return
	}
	for i, j := range batch {
		j.resultCh <- jobResult{prediction: predictions[i]}
	}
}
```

- [ ] **Step 4: Run the tests**

```bash
cd go-server
go test ./... -run TestBatch -v
go test ./... -run TestExpired -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add go-server/batcher.go go-server/batcher_test.go
git commit -m "Add request batcher with size/timeout/deadline handling"
```

---

### Task 8: Go gRPC client wired to the real C++ engine

**Files:**
- Create: `go-server/grpc_client.go`
- Create: `go-server/grpc_client_integration_test.go`

**Interfaces:**
- Consumes: `pb.InferenceEngineClient`, `pb.Image`, `pb.PredictBatchRequest`, `pb.Prediction` from Task 2; `EngineClient` interface from Task 7.
- Produces: `NewGRPCEngineClient(target string) (*GRPCEngineClient, func() error, error)` where `*GRPCEngineClient` implements `EngineClient`, consumed by `main.go` wiring in Task 11.

- [ ] **Step 1: Add the grpc dependency for insecure credentials**

```bash
cd go-server
go get google.golang.org/grpc/credentials/insecure@latest
```

- [ ] **Step 2: Write the client**

Write `go-server/grpc_client.go`:

```go
package main

import (
	"context"

	pb "mini-inference-serving-system/go-server/internal/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type GRPCEngineClient struct {
	client pb.InferenceEngineClient
}

// NewGRPCEngineClient dials the C++ engine at target (e.g. "localhost:50051").
// The returned close function must be called to release the connection.
func NewGRPCEngineClient(target string) (*GRPCEngineClient, func() error, error) {
	conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, err
	}
	return &GRPCEngineClient{client: pb.NewInferenceEngineClient(conn)}, conn.Close, nil
}

func (c *GRPCEngineClient) PredictBatch(ctx context.Context, images [][]float32) ([]Prediction, error) {
	req := &pb.PredictBatchRequest{Images: make([]*pb.Image, len(images))}
	for i, img := range images {
		req.Images[i] = &pb.Image{Pixels: img}
	}

	resp, err := c.client.PredictBatch(ctx, req)
	if err != nil {
		return nil, err
	}

	predictions := make([]Prediction, len(resp.Predictions))
	for i, p := range resp.Predictions {
		predictions[i] = Prediction{Class: p.PredictedClass, Confidence: p.Confidence}
	}
	return predictions, nil
}
```

- [ ] **Step 3: Write an integration test gated behind a running engine**

Write `go-server/grpc_client_integration_test.go`:

```go
//go:build integration

package main

import (
	"context"
	"testing"
	"time"
)

// Run with: go test -tags=integration -run TestGRPCClientAgainstRealEngine ./...
// Requires ./cpp-engine/build/inference_engine tests/fixtures/tiny_weights.bin
// running on localhost:50051 first (see cpp-engine Task 5/Step 6).
func TestGRPCClientAgainstRealEngine(t *testing.T) {
	client, closeFn, err := NewGRPCEngineClient("localhost:50051")
	if err != nil {
		t.Fatalf("failed to dial engine: %v", err)
	}
	defer closeFn()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	predictions, err := client.PredictBatch(ctx, [][]float32{{1, 0, 0, 0}})
	if err != nil {
		t.Fatalf("PredictBatch failed: %v", err)
	}
	if len(predictions) != 1 {
		t.Fatalf("expected 1 prediction, got %d", len(predictions))
	}
	if predictions[0].Class != 1 {
		t.Fatalf("expected class 1 (per tiny_weights.bin fixture), got %d", predictions[0].Class)
	}
}
```

- [ ] **Step 4: Verify the package still builds with the default (non-integration) tag**

```bash
cd go-server
go build ./...
go vet ./...
```

Expected: succeeds — the integration test file is excluded by the `integration` build tag.

- [ ] **Step 5: Run the integration test against the real engine**

Terminal 1:
```bash
cd cpp-engine
./build/inference_engine tests/fixtures/tiny_weights.bin
```

Terminal 2:
```bash
cd go-server
go test -tags=integration -run TestGRPCClientAgainstRealEngine ./... -v
```

Expected: `PASS` — confirms the Go client and C++ engine actually interoperate over gRPC using the real generated stubs on both sides. Stop the engine (Ctrl+C in Terminal 1) afterward.

- [ ] **Step 6: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add go-server/grpc_client.go go-server/grpc_client_integration_test.go go-server/go.mod go-server/go.sum
git commit -m "Add gRPC client to the C++ engine with a real-engine integration test"
```

---

### Task 9: Go HTTP handler with tests

**Files:**
- Create: `go-server/handler.go`
- Create: `go-server/handler_test.go`

**Interfaces:**
- Consumes: `Prediction` from Task 7. Defines `type Submitter interface { Submit(ctx context.Context, pixels []float32) (Prediction, error) }` — `*Batcher` (Task 7) satisfies this implicitly.
- Produces: `NewPredictHandler(submitter Submitter, logger *slog.Logger) http.HandlerFunc`, mounted at `POST /v1/predict` in `main.go` (Task 11).

- [ ] **Step 1: Write the failing tests**

Write `go-server/handler_test.go`:

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeSubmitter struct {
	submitFn func(ctx context.Context, pixels []float32) (Prediction, error)
}

func (f *fakeSubmitter) Submit(ctx context.Context, pixels []float32) (Prediction, error) {
	return f.submitFn(ctx, pixels)
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func makePixels(n int, val float32) []float32 {
	p := make([]float32, n)
	for i := range p {
		p[i] = val
	}
	return p
}

func TestPredictHandlerRejectsInvalidJSON(t *testing.T) {
	h := NewPredictHandler(&fakeSubmitter{}, testLogger())
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewBufferString("not json"))
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestPredictHandlerRejectsWrongLength(t *testing.T) {
	h := NewPredictHandler(&fakeSubmitter{}, testLogger())
	body, _ := json.Marshal(map[string]any{"pixels": []float32{0.1, 0.2}})
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewBuffer(body))
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestPredictHandlerRejectsOutOfRangePixels(t *testing.T) {
	h := NewPredictHandler(&fakeSubmitter{}, testLogger())
	pixels := makePixels(784, 0.5)
	pixels[0] = 1.5
	body, _ := json.Marshal(map[string]any{"pixels": pixels})
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewBuffer(body))
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestPredictHandlerSuccess(t *testing.T) {
	sub := &fakeSubmitter{submitFn: func(_ context.Context, pixels []float32) (Prediction, error) {
		return Prediction{Class: 7, Confidence: 0.99}, nil
	}}
	h := NewPredictHandler(sub, testLogger())
	body, _ := json.Marshal(map[string]any{"pixels": makePixels(784, 0.5)})
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewBuffer(body))
	rec := httptest.NewRecorder()
	h(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp predictResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Class != 7 || resp.Confidence != 0.99 {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestPredictHandlerTimeout(t *testing.T) {
	sub := &fakeSubmitter{submitFn: func(_ context.Context, _ []float32) (Prediction, error) {
		return Prediction{}, context.DeadlineExceeded
	}}
	h := NewPredictHandler(sub, testLogger())
	body, _ := json.Marshal(map[string]any{"pixels": makePixels(784, 0.5)})
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewBuffer(body))
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("expected 504, got %d", rec.Code)
	}
}

func TestPredictHandlerBackendError(t *testing.T) {
	sub := &fakeSubmitter{submitFn: func(_ context.Context, _ []float32) (Prediction, error) {
		return Prediction{}, errors.New("engine down")
	}}
	h := NewPredictHandler(sub, testLogger())
	body, _ := json.Marshal(map[string]any{"pixels": makePixels(784, 0.5)})
	req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewBuffer(body))
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", rec.Code)
	}
}
```

- [ ] **Step 2: Confirm the tests fail to compile**

```bash
cd go-server
go test ./... -run TestPredictHandler
```

Expected: compile error — `NewPredictHandler`, `predictResponse`, `Submitter` undefined.

- [ ] **Step 3: Implement the handler**

Write `go-server/handler.go`:

```go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"
)

type Submitter interface {
	Submit(ctx context.Context, pixels []float32) (Prediction, error)
}

type predictRequest struct {
	Pixels []float32 `json:"pixels"`
}

type predictResponse struct {
	Class      int32   `json:"class"`
	Confidence float32 `json:"confidence"`
}

const (
	pixelCount     = 784
	requestTimeout = 2 * time.Second
)

func NewPredictHandler(submitter Submitter, logger *slog.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		var req predictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON body", http.StatusBadRequest)
			return
		}
		if len(req.Pixels) != pixelCount {
			http.Error(w, "pixels must have length 784", http.StatusBadRequest)
			return
		}
		for _, p := range req.Pixels {
			if p < 0 || p > 1 {
				http.Error(w, "pixel values must be in [0, 1]", http.StatusBadRequest)
				return
			}
		}

		ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
		defer cancel()

		prediction, err := submitter.Submit(ctx, req.Pixels)
		latency := time.Since(start)

		if err != nil {
			status := http.StatusServiceUnavailable
			outcome := "error"
			if errors.Is(err, context.DeadlineExceeded) {
				status = http.StatusGatewayTimeout
				outcome = "timeout"
			}
			logger.Info("predict request", "latency_ms", latency.Milliseconds(), "outcome", outcome, "error", err.Error())
			recordRequestMetric(outcome, latency)
			http.Error(w, err.Error(), status)
			return
		}

		logger.Info("predict request", "latency_ms", latency.Milliseconds(), "outcome", "success")
		recordRequestMetric("success", latency)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(predictResponse{
			Class:      prediction.Class,
			Confidence: prediction.Confidence,
		})
	}
}
```

Note: `recordRequestMetric` is defined in Task 10 — this task will not compile until Task 10 adds it. To keep this task independently testable, add a temporary no-op in this task and remove it in Task 10:

Write it as part of `handler.go` for now:

```go
func recordRequestMetric(outcome string, latency time.Duration) {
	// Wired up to real Prometheus metrics in Task 10.
	_ = outcome
	_ = latency
}
```

(Place this function at the bottom of `handler.go`.)

- [ ] **Step 4: Run the tests**

```bash
cd go-server
go test ./... -run TestPredictHandler -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add go-server/handler.go go-server/handler_test.go
git commit -m "Add HTTP predict handler with validation and error mapping"
```

---

### Task 10: Metrics and structured logging wiring

**Files:**
- Create: `go-server/metrics.go`
- Modify: `go-server/handler.go` (remove the temporary no-op `recordRequestMetric`)
- Modify: `go-server/batcher.go` (record batch size / queue depth on dispatch)
- Create: `go-server/metrics_test.go`

**Interfaces:**
- Produces: `MetricsHandler() http.Handler` mounted at `GET /metrics` in `main.go` (Task 11); `recordRequestMetric(outcome string, latency time.Duration)` (used by `handler.go`); `recordBatchMetric(size int, queueDepth int)` (used by `batcher.go`).

- [ ] **Step 1: Add the Prometheus client dependency**

```bash
cd go-server
go get github.com/prometheus/client_golang/prometheus@latest
go get github.com/prometheus/client_golang/prometheus/promhttp@latest
```

- [ ] **Step 2: Write the metrics module**

Write `go-server/metrics.go`:

```go
package main

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	requestCounter = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "predict_requests_total",
		Help: "Total predict requests by outcome",
	}, []string{"outcome"})

	requestLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "predict_request_latency_seconds",
		Help:    "Predict request latency in seconds",
		Buckets: prometheus.DefBuckets,
	})

	batchSizeHistogram = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "batch_size",
		Help:    "Size of batches dispatched to the inference engine",
		Buckets: []float64{1, 2, 4, 8, 16, 32},
	})

	queueDepthGauge = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "queue_depth",
		Help: "Batch queue depth sampled at dispatch time",
	})
)

func init() {
	prometheus.MustRegister(requestCounter, requestLatency, batchSizeHistogram, queueDepthGauge)
}

func recordRequestMetric(outcome string, latency time.Duration) {
	requestCounter.WithLabelValues(outcome).Inc()
	requestLatency.Observe(latency.Seconds())
}

func recordBatchMetric(size int, queueDepth int) {
	batchSizeHistogram.Observe(float64(size))
	queueDepthGauge.Set(float64(queueDepth))
}

func MetricsHandler() http.Handler {
	return promhttp.Handler()
}
```

- [ ] **Step 3: Remove the temporary no-op from handler.go**

Modify `go-server/handler.go`: delete the temporary `recordRequestMetric` function added at the end of Task 9 (it's now defined in `metrics.go`).

- [ ] **Step 4: Wire batch metrics into the batcher's dispatch**

Modify `go-server/batcher.go`, in `func (b *Batcher) dispatch(batch []job)`, add a metrics call right after building `images`:

```go
func (b *Batcher) dispatch(batch []job) {
	images := make([][]float32, len(batch))
	for i, j := range batch {
		images[i] = j.pixels
	}
	recordBatchMetric(len(batch), len(b.queue))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// ... unchanged below
```

- [ ] **Step 5: Write a metrics smoke test**

Write `go-server/metrics_test.go`:

```go
package main

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestMetricsHandlerExposesRegisteredMetrics(t *testing.T) {
	recordRequestMetric("success", 5*time.Millisecond)
	recordBatchMetric(3, 1)

	req := httptest.NewRequest("GET", "/metrics", nil)
	rec := httptest.NewRecorder()
	MetricsHandler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{"predict_requests_total", "predict_request_latency_seconds", "batch_size", "queue_depth"} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected metrics output to contain %q, got:\n%s", want, body)
		}
	}
}
```

- [ ] **Step 6: Run all Go tests**

```bash
cd go-server
go build ./...
go test ./... -v
```

Expected: everything from Tasks 7, 9, and 10 passes (integration test from Task 8 is skipped by default since it requires the `integration` build tag).

- [ ] **Step 7: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add go-server/metrics.go go-server/metrics_test.go go-server/handler.go go-server/batcher.go go-server/go.mod go-server/go.sum
git commit -m "Add Prometheus metrics and wire them into the handler and batcher"
```

---

### Task 11: Wire the server binary, dev scripts, and end-to-end verification

**Files:**
- Modify: `go-server/main.go`
- Create: `scripts/run-dev.sh`
- Create: `scripts/e2e.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: `NewGRPCEngineClient` (Task 8), `NewBatcher` (Task 7), `NewPredictHandler` (Task 9), `MetricsHandler` (Task 10).
- Produces: the final `go-server` binary serving `POST /v1/predict` and `GET /metrics` on `:8080`.

- [ ] **Step 1: Rewrite main.go to wire everything together**

Write `go-server/main.go`:

```go
package main

import (
	"log/slog"
	"net/http"
	"os"
	"time"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	engineAddr := os.Getenv("ENGINE_ADDR")
	if engineAddr == "" {
		engineAddr = "localhost:50051"
	}

	client, closeFn, err := NewGRPCEngineClient(engineAddr)
	if err != nil {
		logger.Error("failed to connect to inference engine", "addr", engineAddr, "error", err)
		os.Exit(1)
	}
	defer closeFn()

	batcher := NewBatcher(client, 8, 10*time.Millisecond)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/predict", NewPredictHandler(batcher, logger))
	mux.Handle("GET /metrics", MetricsHandler())

	addr := ":8080"
	logger.Info("go-server listening", "addr", addr, "engine_addr", engineAddr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 2: Verify it builds**

```bash
cd go-server
go build ./...
```

Expected: succeeds.

- [ ] **Step 3: Write the dev-run script**

Write `scripts/run-dev.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Building C++ engine..."
cmake -S cpp-engine -B cpp-engine/build >/dev/null
cmake --build cpp-engine/build >/dev/null

if [ ! -f training/weights.bin ]; then
  echo "training/weights.bin not found — run training/train.py first." >&2
  exit 1
fi

echo "Starting C++ engine on :50051..."
(cd cpp-engine && ./build/inference_engine ../training/weights.bin) &
ENGINE_PID=$!
trap 'kill $ENGINE_PID' EXIT

sleep 1

echo "Starting Go server on :8080..."
(cd go-server && go run .)
```

- [ ] **Step 4: Write the end-to-end verification script**

Write `scripts/e2e.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f training/weights.bin ]; then
  echo "training/weights.bin not found — run training/train.py first." >&2
  exit 1
fi

echo "Starting C++ engine..."
(cd cpp-engine && ./build/inference_engine ../training/weights.bin) &
ENGINE_PID=$!

echo "Starting Go server..."
(cd go-server && go run .) &
SERVER_PID=$!

trap 'kill $ENGINE_PID $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for server to be ready..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8080/metrics >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "Sending sample digit for prediction..."
PIXELS=$(python3 -c "import json; print(json.dumps(json.load(open('testdata/sample_digit.json'))['pixels']))")
LABEL=$(python3 -c "import json; print(json.load(open('testdata/sample_digit.json'))['label'])")

RESPONSE=$(curl -sf -X POST http://localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d "{\"pixels\": $PIXELS}")

echo "True label: $LABEL"
echo "Response: $RESPONSE"

PREDICTED=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['class'])" "$RESPONSE")
if [ "$PREDICTED" != "$LABEL" ]; then
  echo "WARNING: predicted class ($PREDICTED) does not match true label ($LABEL) — model accuracy is not 100%, this is not necessarily a bug." >&2
fi

echo "e2e check complete."
```

- [ ] **Step 5: Run the end-to-end script**

```bash
chmod +x scripts/run-dev.sh scripts/e2e.sh
./scripts/e2e.sh
```

Expected: both processes start, the script prints the true label and the JSON response `{"class": ..., "confidence": ...}`, and exits 0 (a class mismatch only prints a warning — the model isn't 100% accurate, that's expected and fine).

- [ ] **Step 6: Manually verify with curl (optional sanity check)**

```bash
curl -s http://localhost:8080/metrics | grep predict_requests_total
```

Expected (while `run-dev.sh` is running in another terminal): shows the `predict_requests_total` metric with a nonzero count after sending at least one request.

- [ ] **Step 7: Finish the README**

Modify `README.md` to add setup and run instructions:

```markdown
## Setup

See "Prerequisites" in `docs/superpowers/plans/2026-08-25-mini-inference-serving-system.md`
for one-time host setup (Go, CMake, protobuf, gRPC, Python).

```bash
./scripts/gen-proto.sh                 # generate Go/C++ protobuf stubs
cmake -S cpp-engine -B cpp-engine/build && cmake --build cpp-engine/build
source training/.venv/bin/activate && pip install -r training/requirements.txt
python3 training/train.py              # trains the model, writes training/weights.bin
```

## Run

```bash
./scripts/run-dev.sh
```

Then, in another terminal:

```bash
curl -X POST http://localhost:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d @testdata/sample_digit.json
```

(Note: `sample_digit.json` also contains a `label` field the server ignores —
only `pixels` is read.)

## Tests

```bash
cd cpp-engine && cmake --build build && ./build/inference_tests
cd go-server && go test ./...
```
```

- [ ] **Step 8: Commit**

```bash
cd /Users/sohamn/Desktop/mini-inference-serving-system
git add go-server/main.go scripts/run-dev.sh scripts/e2e.sh README.md
git commit -m "Wire server binary, add dev/e2e scripts, finish README"
```

---

## Self-Review Notes

- **Spec coverage:** architecture (Tasks 1,5,8,11), model (Tasks 4,6), proto contract (Task 2), C++ engine + kernels (Tasks 3,4,5), Go server + batching (Tasks 7,9), gRPC boundary (Tasks 2,8), observability (Task 10), testing strategy — kernel/model/batcher/handler unit tests plus e2e (Tasks 3,4,7,9,11), repo layout (Task 1) — all covered.
- **Placeholder scan:** no TBD/TODO markers; the one temporary stub (`recordRequestMetric` no-op in Task 9, replaced in Task 10) is intentional and explicitly called out with the exact removal step, not a placeholder left unresolved.
- **Type consistency:** `Prediction{Class int32, Confidence float32}` (Task 7) matches its use in `grpc_client.go` (Task 8), `handler.go`/`predictResponse` (Task 9), and test fakes throughout. `EngineClient`/`Submitter` interface method signatures match between definition and implementers (`Batcher`, `GRPCEngineClient`, fakes). C++ `Prediction{predicted_class, confidence}` (Task 4) matches its use in `service.cc` (Task 5). Weight file field order/types match between `weights_io.py` (Task 4/6) and `model.cc`'s reader (Task 4).
