# Mini Inference Serving System — Design Spec

## Purpose

A learning + portfolio project that combines Go and C++ to build a small
but architecturally realistic model-inference-serving system, modeled
loosely on how systems like NVIDIA Triton, TorchServe, and TGI split a
client-facing serving layer from a backend compute engine. The primary
goals, in order:

1. Learn Go (concurrency, HTTP services, gRPC clients) and C++ (memory
   layout, manual numerical kernels, gRPC servers) through a real,
   working system rather than isolated exercises.
2. Produce something resume-legible: dynamic batching, a Go↔C++ service
   boundary over gRPC, and basic production observability are the
   distinctive features that make this "an inference server" rather
   than a CRUD app.
3. Lay a foundation that could later be extended toward real use (larger
   models, multiple models, GPU execution) — not required for v1, but
   the architecture should not preclude it.

Explicitly out of scope for v1 (documented here so it isn't rediscovered
as a gap later): multi-model serving, model versioning/registry, GPU
execution, authentication/authorization, TLS, horizontal scaling of the
C++ engine, model formats other than the hand-rolled one (no ONNX/
libtorch).

## Architecture

Two separate OS processes, communicating over gRPC on localhost:

```
Client --HTTP/JSON--> [Go server] --gRPC--> [C++ engine]
                         |
                         +-- REST handler (net/http)
                         +-- batching queue (goroutine + channels)
                         +-- gRPC client (calls C++ engine)

[C++ engine]
  +-- gRPC server (generated from proto/inference.proto)
  +-- Model (loads weights.bin at startup)
  +-- Kernels (matmul, ReLU, softmax — hand-rolled, no ML framework)
```

The Go server is the only process clients talk to. The C++ engine is an
internal service; it never talks to the client directly. This mirrors
real systems where a lightweight, high-concurrency front layer (Go is
well suited to this) shields a heavier, compute-bound backend (C++ is
well suited to this) — and gives the project two distinct languages
each doing the job they're actually good at, rather than gluing them
together arbitrarily.

## Model

A hand-rolled 2-layer fully-connected neural network for MNIST digit
classification:

- Input: 784 floats (28×28 grayscale image, flattened, normalized to
  [0, 1]).
- Layer 1: Linear(784 → 128) + ReLU.
- Layer 2: Linear(128 → 10) + softmax.
- Output: a probability distribution over 10 digit classes.

Training happens **offline**, outside the serving system, via a small
Python + numpy script (`training/train.py`). No ML framework
dependency (no PyTorch/TensorFlow required) — a basic gradient-descent
training loop is sufficient since the model is intentionally tiny. The
script trains on the standard MNIST dataset and exports the two weight
matrices and two bias vectors to a single binary file
(`training/weights.bin`) using a simple fixed binary layout (documented
in the training script and read by the C++ loader). The serving system
(Go and C++) never trains — it only loads pre-trained weights at
startup.

## Components

### `proto/inference.proto`

The shared contract between Go and C++. Defines a single batched RPC:

```protobuf
syntax = "proto3";
package inference;

service InferenceEngine {
  rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);
}

message Image {
  repeated float pixels = 1; // length 784
}

message PredictBatchRequest {
  repeated Image images = 1;
}

message Prediction {
  int32 predicted_class = 1;
  float confidence = 2;
}

message PredictBatchResponse {
  repeated Prediction predictions = 1; // same order/length as request
}
```

Generated Go and C++ stubs live alongside their respective services
(standard protoc-gen-go / protoc-gen-go-grpc and protoc's C++ gRPC
plugin output), not committed as hand-written code.

### C++ engine (`cpp-engine/`)

- `Model` class: loads `weights.bin` at construction; exposes
  `Predict(const std::vector<std::vector<float>>& batch) ->
  std::vector<Prediction>` where `Prediction` is `{int predicted_class;
  float confidence;}`.
- Kernel functions (free functions, unit-testable in isolation):
  `matmul`, `add_bias`, `relu`, `softmax`. Operate on `std::vector<float>`
  buffers with explicit dimensions (no framework tensor type).
- gRPC service implementation (`InferenceEngineServiceImpl`) that
  deserializes the request into the batch format `Model::Predict`
  expects, calls it, and serializes the response. This class contains
  no numerical logic itself — it is purely the adapter between gRPC
  messages and `Model`.
- `main.cc`: loads the model, starts the gRPC server on a fixed local
  port (e.g. `50051`).

### Go server (`go-server/`)

- HTTP handler (`POST /v1/predict`): parses `{"pixels": [784]float64}`,
  validates length and value range, wraps the request into a `job`
  struct containing the input and a buffered response channel, pushes
  it onto the batching queue with a request-scoped `context` deadline
  (e.g. 2s).
- Batcher (a single goroutine owning the queue): drains queued jobs into
  a batch, flushing when either the batch reaches a configured max size
  (e.g. 8) or a configured max wait elapses since the first job in the
  batch arrived (e.g. 10ms) — whichever happens first. Any job whose
  context has already expired when the batcher looks at it is dropped
  from the batch and completed immediately with a deadline-exceeded
  result, rather than blocking or wasting a batch slot.
- gRPC client: on each flush, the batcher makes one `PredictBatch` call
  to the C++ engine and distributes each response element back to the
  corresponding job's channel, in order.
- The HTTP handler blocks on the job's channel (or the request context)
  and writes the JSON response once a result (or error) arrives.

### Observability

- Structured logging via `log/slog`: one log line per request with
  latency, assigned batch size, and outcome (success/timeout/error).
- Prometheus `/metrics` endpoint (`promhttp` handler) exposing:
  - request counter, labeled by outcome
  - request latency histogram
  - batch size histogram
  - queue depth gauge (sampled at flush time)

## Data flow & error handling

1. Client `POST /v1/predict` with `{"pixels": [784]float64}`.
2. Go validates: wrong length or out-of-range values → `400 Bad
   Request` immediately, no queueing.
3. Valid request becomes a job on the batch queue with a 2s context
   deadline.
4. Batcher flushes the batch (by size or timeout) and calls
   `PredictBatch` on the C++ engine.
   - If the gRPC call errors (engine down, RPC error) or times out, every
     job in that batch receives `503 Service Unavailable`.
   - If a job's context expires before its batch is flushed, it is
     removed from the queue and receives `504 Gateway Timeout` without
     affecting the rest of the batch.
5. On success, each job's HTTP handler receives its `Prediction` and
   responds `200 OK` with `{"class": int, "confidence": float}`.

## Testing strategy

- **C++ kernel tests**: unit tests (Catch2 or doctest — chosen in the
  implementation plan) verifying `matmul`, `relu`, `softmax` against
  hand-computed values on small fixed inputs, independent of the model
  or gRPC.
- **C++ model test**: loads a small fixture weights file and checks
  `Model::Predict` produces the expected shape and a valid probability
  distribution (sums to ~1, all values in [0, 1]).
- **Go batcher tests**: unit tests against a fake `InferenceEngine`
  client (satisfying the same interface as the generated gRPC client)
  covering: batch flushes at max size, batch flushes at max wait, a job
  past its deadline is excluded from the batch and completed with a
  timeout result independent of the rest of the batch, a backend error
  fails every job in the batch.
- **Go HTTP handler tests**: request validation (400 cases), successful
  round trip using the fake engine.
- **End-to-end script**: `scripts/e2e.sh` (or equivalent) starts both
  processes, POSTs a real sample image from `testdata/`, and checks the
  response is a plausible prediction. Used manually / in CI, not a unit
  test.

## Repo layout

```
mini-inference-serving-system/
  go-server/
    main.go
    handler.go       # HTTP handler
    batcher.go        # queueing + batching + gRPC client calls
    metrics.go
    *_test.go
  cpp-engine/
    CMakeLists.txt
    main.cc
    model.h / model.cc
    kernels.h / kernels.cc
    service.h / service.cc   # gRPC service adapter
    tests/
  proto/
    inference.proto
  training/
    train.py
    weights.bin           # generated, not hand-edited
  testdata/
    sample_digit.json     # or similar, for e2e test
  scripts/
    run-dev.sh
    e2e.sh
```
