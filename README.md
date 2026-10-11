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

## Status

- Done: proto contract, C++ kernels/model/gRPC service with unit tests, training/export script, Go batcher (`go-server/internal/batcher`) with race-tested edge cases.
- Next: Go gRPC client and HTTP handler, Prometheus metrics, end-to-end and load tests.

Full write-up (design, edge cases, trade-offs, interview Q&A): `docs/PROJECT_DEEP_DIVE.md`.
Build plan: `docs/superpowers/plans/BUILD_PLAN.md`.
