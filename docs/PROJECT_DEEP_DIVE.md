# Project Deep Dive: Mini Inference Serving System

This is the long-form briefing for the project. The README is the short version. The focus is the serving infrastructure (batching, concurrency, deadlines, backpressure, shutdown, observability). The model is just a workload.

## 1. What it is

Two separate processes, joined by one gRPC contract (`proto/inference.proto`):

```
Client --HTTP/JSON--> [Go server]                          [C++ engine]
                      handler -> bounded queue -> collector --gRPC PredictBatch--> service -> Model -> kernels
                                       \-> up to MaxInFlight engine calls
```

- **Go server** is the only thing clients talk to. It has many concurrent connections and cheap goroutines, which suit the front-end work: admission, queueing, batching, deadlines, shedding load.
- **C++ engine** is compute-bound. It loads a weight file once, then runs a 784 -> 128 -> 10 network with hand-written kernels (matmul, bias, ReLU, softmax). It uses no ML framework.
- **The model is not the point.** It exists so the batching has a realistic backend.

This mirrors production inference servers like Triton and TorchServe: a lightweight front layer shielding a heavy back layer.

## 2. Why batch at all

Each engine call has fixed overhead: serialization, the RPC, and a small matmul that runs inefficiently. Grouping N requests into one call pays that cost once and gives the kernel a bigger matrix. The cost is **added latency**, because early requests wait for the batch to form. Two knobs control the trade-off:

| Knob | Raise it and... |
| --- | --- |
| `MaxBatch` | throughput improves, but a batch takes longer to fill and to compute |
| `MaxWait` | batches fill more often, but every request can wait up to this long |

A batch flushes when it reaches `MaxBatch` **or** `MaxWait` after its first request, whichever comes first. Under light load the timer fires (the latency floor is about `MaxWait`). Under heavy load the size trigger fires and the timer rarely matters.

## 3. The Go batcher (`go-server/internal/batcher`)

### Structure

- `Submit(ctx, pixels)` is called by each request handler goroutine. It enqueues a job and blocks on that job's private buffered result channel or on the caller's `ctx`.
- **One collector goroutine** owns batch formation. Because it is the only reader of the queue, batch assembly needs no locking.
- **Dispatch goroutines**, bounded by a semaphore (`MaxInFlight`), make the actual engine calls. Results go back through each job's `resultCh`. The channel has capacity 1, so the sender never blocks even if the caller has already left.

### Edge cases and how each is handled

Each one has a test in `batcher_test.go`.

| Edge case | Behavior | Why |
| --- | --- | --- |
| **Queue is full** | `Submit` returns `ErrQueueFull` immediately (non-blocking send). The HTTP layer maps this to 429/503. | An unbounded or blocking queue turns an overload into unbounded memory and latency. Shedding load fast keeps the requests you do accept healthy. |
| **Caller context already expired** | Rejected before enqueueing, and nothing reaches the engine. | There's no point queueing work nobody wants. |
| **Caller cancels or times out while queued** | The job is dropped when the collector dequeues it, or pruned again just before dispatch. `Submit` itself returns at the caller's deadline. | The engine never computes dead work. There are two checks because a caller can give up while the batch waits for an engine slot. |
| **Every job in a batch is dead** | No engine call is made. | A flush can be a no-op. |
| **Caller gives up while the engine is working** | The caller returns at its own deadline. The result is dropped into the buffered channel, so nothing blocks and nothing leaks. | One slow batch can't hold a caller past its deadline. |
| **Mixed deadlines in one batch** | The engine call is bounded by the **latest** caller deadline (capped by `EngineTimeout`). If any caller has no deadline, `EngineTimeout` applies. | Using the earliest deadline would let one impatient caller abort work the others still need. Using the latest cancels the call once no one could still use the answer. |
| **Hung engine** | `EngineTimeout` cuts the call off. | A stuck backend must not pin dispatch slots forever. |
| **Engine returns an error** | Every job in that batch gets the error. Later batches are unaffected. | A batch fails or succeeds as a unit, so the caller sees one clear error. |
| **Engine returns the wrong number of predictions** | `ErrBadResponse` to all jobs, with no index-out-of-range panic. | Never trust a remote peer's response shape. A mismatch would otherwise hand a caller someone else's result. |
| **Engine client panics** | `recover` turns it into `ErrEnginePanic` for that batch, and the batcher keeps serving. | A panic in a dispatch goroutine would otherwise crash the whole process. |
| **Engine saturated** | The collector blocks on the semaphore, so the queue fills and `Submit` starts shedding. | Backpressure propagates from the slowest stage back to admission. |
| **Concurrent `Submit` and `Shutdown`** | `Submit` holds a read lock while it sends, and `Shutdown` takes the write lock before closing the queue. | This prevents "send on closed channel" panics, and `-race` verifies it. |
| **Graceful shutdown** | Stop accepting (`ErrClosed`), flush queued jobs **immediately** (don't wait out `MaxWait`), wait for in-flight calls, honor the caller's shutdown deadline. It is idempotent. | Queued requests get answers instead of dropped connections, and shutdown stays bounded. |
| **Result routing** | Each job has its own result channel, and a 300-goroutine test checks every caller gets its own answer. | The classic batching bug is returning results to the wrong request. |

### Known limitations (good to volunteer)

- **Pixel buffers are borrowed.** The caller mustn't mutate them until `Submit` returns a result. After a cancel they may still be read by an in-flight batch. The handler allocates a fresh slice per request, which satisfies this.
- **One slow batch can fail all its jobs.** There's no per-item error in the proto yet.
- **Engine calls aren't cancelled when all callers cancel** before the latest deadline. The call runs to its deadline. Fixing this means tracking live waiters per batch.
- **No priority or fairness.** It's FIFO.
- **Single collector.** It is not a bottleneck at this scale. With very large queues or tiny batches it would be the first thing to profile.
- **No adaptive batching.** `MaxWait` is static. Production systems tune it from queue depth or recent latency.

## 4. Observability

`Batcher.Stats()` exposes counters that map directly onto Prometheus metrics (wiring is a later task):

| Stat | Answers |
| --- | --- |
| `Submitted`, `Rejected` | what the offered load is and how much is shed |
| `Expired` | how many requests died waiting (queue wait vs. client deadlines) |
| `Batches`, `Items` | the average batch size (`Items/Batches`), the number that shows whether batching is working |
| `EngineErrors` | backend health |
| `QueueDepth` | the leading indicator of saturation |

Planned additions: queue-wait and engine-latency histograms, and a batch-size histogram. The average alone hides bimodal behavior.

## 5. The C++ engine

- **gRPC adapter** takes a `PredictBatch` request, validates each image's length, calls `Model::predict`, and returns predictions in order. The Go side relies on the same-length, same-order guarantee.
- **Model** loads a dimension-generic binary file (header with dimensions, then weights and biases), so small fixtures can be used in tests.
- **Concurrency:** weights are read-only after load, and each call allocates its own temporaries. There's no shared mutable scratch space, so concurrent calls need no locks. Ownership is `main` -> `Model` -> service holding a reference, so the model outlives the server.
- **Kernels** are plain loops over contiguous row-major buffers. They are tested against small hand-computed cases.

## 6. Testing strategy

- Batcher tests use a **fake engine client** with hooks to block, fail, panic or delay it, so each failure mode is deterministic.
- State is synchronized with channels and an `eventually` poll on `Stats()`, not fixed sleeps, wherever possible.
- Everything runs under `go test -race`.
- I mutation-checked the tests. Removing the dispatch-time prune, making the enqueue blocking, and removing the panic recovery each make tests fail.
- The C++ side has unit tests for the kernels and the model loader.

## 7. Interview questions to be ready for

1. **"What's the latency cost of batching?"** Up to `MaxWait` at low load. At high load the size trigger dominates and the cost is mostly batch compute time.
2. **"What if the queue grows without bound?"** It can't. The queue is bounded and `Submit` sheds with `ErrQueueFull`.
3. **"One request in a batch times out. What happens to the rest?"** Only that caller returns early, and the engine deadline follows the latest caller.
4. **"How do you avoid sending results to the wrong caller?"** Each job has its own channel, and the engine's ordered response is zipped back by index after a length check.
5. **"How would you shut down without dropping requests?"** Close admission, flush the queue immediately, wait for in-flight calls under a deadline.
6. **"What breaks first under load?"** The queue fills and rejections rise. `QueueDepth` and `Rejected` show it before latency does.
7. **"Why Go for the front and C++ for the back?"** Go for cheap concurrency and I/O, C++ for control of memory layout and compute.
8. **"What would you do next?"** Per-item errors in the proto, adaptive `MaxWait`, cancelling an engine call when every waiter is gone, and histograms.

## 8. Status

| Piece | State |
| --- | --- |
| Proto contract, generated stubs | done |
| C++ kernels, model loader, gRPC service | done, unit-tested |
| NumPy training/export script | written; no real training run yet |
| Go batcher (`internal/batcher`) | done, 18 tests, passes under `-race` |
| Go gRPC client to the C++ engine | next |
| Go HTTP handler (validation, status mapping) | next |
| Prometheus metrics, structured logs | next |
| End-to-end run and load test | next |

Don't cite performance numbers until the load test has produced them.
