# Mini Inference Serving System

A small model-inference server, split the way production systems (Triton,
TorchServe) split it: a concurrent front end and a compute-bound back end.

```
Client --HTTP/JSON--> [Go server] --gRPC (batched)--> [C++ engine]
                       handler -> queue -> batcher      service -> model -> kernels
```

## Purpose

The focus is the **serving infrastructure**, not the model. The model is a
tiny MNIST classifier (784 -> 128 -> 10) used only as a realistic workload.
The interesting parts are:

- **Dynamic batching (Go):** requests are queued and flushed as one gRPC call
  when a size or time limit is hit; results are routed back to each caller.
- **Concurrency (Go + C++):** many in-flight requests, per-request deadlines
  and cancellation, read-only shared weights, no shared mutable scratch state.
- **Service boundary:** Go and C++ are separate processes sharing only
  `proto/inference.proto`.
- **Observability and shutdown (planned):** metrics, graceful drain.
- **Hand-written C++ kernels** (matmul, bias, ReLU, softmax): no ML framework.

## Layout

| Path | What |
| --- | --- |
| `proto/` | gRPC/Protobuf contract (`PredictBatch`) |
| `cpp-engine/` | gRPC service, weight loader, kernels, tests |
| `go-server/` | HTTP front end, queue, batcher (in progress) |
| `training/` | Offline NumPy training and weight export (not the focus) |
| `scripts/` | Proto code generation |

## Run

```
# engine (C++), then server (Go)
./cpp-engine/build/inference_engine weights.bin 127.0.0.1:50051
cd go-server && go run . -engine 127.0.0.1:50051 -listen :8080

curl -XPOST -H 'Content-Type: application/json' -d '{"pixels":[...784 floats in [0,1]...]}' localhost:8080/predict
curl localhost:8080/metrics
```

Flags (`go run . -h`): `-max-batch`, `-max-wait`, `-queue-size`, `-max-in-flight`, `-request-timeout`, `-engine-timeout`, `-shutdown-timeout`, `-drain-delay`. Tests: `cd go-server && go test -race ./...`.

## Status

- Go server complete and race-tested: batcher, gRPC client, HTTP layer, Prometheus metrics, wiring and graceful shutdown.
- C++ kernels (matmul, bias, ReLU, softmax) with unit tests; the model loader and gRPC service are being redone.
- Not done: load test and performance numbers, a real training run.

Full write-up (design, edge cases, trade-offs, interview Q&A): `docs/PROJECT_DEEP_DIVE.md`.
Build plan: `docs/superpowers/plans/BUILD_PLAN.md`.
