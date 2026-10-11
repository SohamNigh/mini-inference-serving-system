# Project Deep Dive: Mini Inference Serving System

This is the full explanation of the project as it exists **right now**. The README is the short version.

Read order if you're learning it: section 1, then 3 (one request, start to finish), then 4–6 (each component), then 9 (edge cases). Section 11 is the interview prep.

Contents:
1. What this is and why
2. What's built, and what isn't
3. Life of one request
4. The Go batcher
5. The Go gRPC client
6. The Go HTTP layer
7. The C++ engine
8. The model and training (kept small on purpose)
9. Every edge case in one table
10. How it's tested
11. Interview prep
12. Known limitations and next steps
13. Go concepts used here, in plain English

---

## 1. What this is and why

An **inference server** is a program that answers "run this input through a model and tell me the result" over the network. Real ones (Triton, TorchServe) split into two layers, and so does this project:

```
Client --HTTP/JSON--> [Go server]                         [C++ engine]
                      validate -> queue -> batch  --gRPC--> kernels -> result
```

- **Go server (front).** Clients only talk to this. Its job is traffic handling: validating input, queueing, grouping requests into batches, enforcing deadlines, shedding load when overloaded, and reporting health. Go is good at this because goroutines make "thousands of waiting requests" cheap.
- **C++ engine (back).** Its job is computation: load model weights once, run the math fast and predictably. C++ gives control over memory layout and no garbage-collector pauses.
- **gRPC** is the wire between them. Both sides are generated from one file, `proto/inference.proto`, so the contract can't silently drift.

**The point of the project is the infrastructure**, not the model. The model is a tiny digit classifier (MNIST, 28×28 images) used as a realistic workload. The things worth talking about are batching, backpressure, deadlines, cancellation, failure handling, concurrency and shutdown.

### Why batch at all?

Every engine call has a fixed cost: serializing, the network hop, scheduling. The math itself is also more efficient on one big matrix than on many tiny ones. So grouping N requests into one call pays the fixed cost once.

The price is **latency**: the first request in a batch waits for the batch to fill. Two knobs control that trade-off:

| Knob | Raising it means |
| --- | --- |
| `MaxBatch` | more throughput per call, but bigger batches take longer to fill and compute |
| `MaxWait` | batches fill more often, but every request may wait up to this long |

A batch is sent when it reaches `MaxBatch` **or** when `MaxWait` has passed since its first request, whichever happens first. At low traffic the timer fires (so latency ≈ `MaxWait`). At high traffic the size limit fires first and the timer rarely matters.

---

## 2. What's built, and what isn't

| Piece | State |
| --- | --- |
| `proto/inference.proto` + generated Go and C++ code | done |
| C++ math kernels (matmul, bias, ReLU, softmax) | done, unit-tested |
| C++ model loader and forward pass | done, unit-tested |
| C++ gRPC service (`inference_engine` binary) | done, runs, verified from Go |
| NumPy training + export script | written, **never run on real MNIST data** (no dataset in `training/data`) |
| Go batcher (`internal/batcher`) | done, 23 tests, race-tested |
| Go gRPC client (`internal/engineclient`) | done, 11 in-process tests + 2 integration tests against the real C++ engine |
| Go HTTP layer (`internal/httpapi`) | done, ~40 test cases |
| Go metrics (`internal/metrics`) | done: Prometheus histograms + counters, 8 tests |
| Go wiring + graceful shutdown (`internal/app`, `main.go`) | done: flags, signals, ordered shutdown, 7 tests |
| Manual end-to-end smoke run: curl -> Go server -> real C++ engine (random weights) | done once: 200s, 400 on bad input, 503 when the engine was killed, clean SIGTERM shutdown. Not automated. |
| Load test / any performance numbers | **not done. Don't quote numbers.** |

Honest one-liner for an interview today: *"The Go server is complete and tested: batching, gRPC client, HTTP layer, metrics and graceful shutdown. I've run it against the C++ engine; what's left is the engine side and a proper load test."*

### Repo map

```
proto/inference.proto            the contract (one RPC: PredictBatch)
cpp-engine/
  kernels.{h,cc}                 matmul, add_bias, relu, softmax
  model.{h,cc}                   weight loading + forward pass
  main.cc                        gRPC service + process entry point
  generated/                     protobuf/gRPC C++ code (generated, don't read)
  tests/                         Catch2 tests
go-server/
  main.go                        flags + signal handling, calls app.Run
  internal/batcher/              queue + dynamic batching + shutdown
  internal/engineclient/         gRPC client to the C++ engine
  internal/httpapi/              HTTP endpoint, validation, status codes
  internal/metrics/              Prometheus metrics
  internal/app/                  wiring everything together + graceful shutdown
  internal/pb/                   generated Go protobuf/gRPC code
training/                        offline NumPy training + weight export
scripts/gen-proto.sh             regenerates the generated code
```

---

## 3. Life of one request

Say a client POSTs `{"pixels":[...784 numbers...]}` to `/predict`.

1. **net/http** starts a goroutine for this connection and calls the HTTP layer's handler.
2. **Middleware** starts a timer for metrics and installs a panic guard.
3. **Cheap checks first:** method must be POST (else 405), Content-Type must be JSON (else 415).
4. **Body decode and validation:** the body is read through a size limiter (413 if too big); JSON must parse, have no unknown fields and no trailing junk; there must be exactly 784 numbers, each within [0, 1] (else 400). Nothing past this point sees invalid data.
5. **A deadline is attached:** `context.WithTimeout(request context, RequestTimeout)`. This single budget covers queueing *and* the engine call. The request context also fires if the client disconnects.
6. **`Batcher.Submit(ctx, pixels)`** (this is where the request enters the batching system):
   - already-expired context -> return immediately;
   - batcher shutting down -> `ErrClosed`;
   - queue full -> `ErrQueueFull` **immediately** (never blocks);
   - otherwise the job goes into the bounded queue, and the handler goroutine blocks, waiting for either its result or its own context to end.
7. **The collector goroutine** pulls jobs off the queue. The first live job starts a `MaxWait` timer. It keeps adding jobs until the batch has `MaxBatch` jobs or the timer fires. Jobs whose caller already gave up are discarded rather than added.
8. **The batch waits for an engine slot** (a semaphore of size `MaxInFlight`). If the engine is saturated, the collector blocks here, the queue fills, and step 6 starts returning `ErrQueueFull`. That chain is backpressure.
9. **A dispatch goroutine** takes over: it re-checks for callers who gave up while waiting for the slot, drops them, then builds an engine context whose deadline is the *latest* remaining caller deadline (capped by `EngineTimeout`).
10. **The gRPC client** sends one `PredictBatch` request. (If the batch is bigger than the engine's limit of 8, it's split into sequential calls and reassembled in order.)
11. **The C++ engine** validates the batch (1–8 images, each exactly 784 pixels), runs the model, returns one prediction per image in the same order.
12. **Results fan out:** the dispatch goroutine checks the prediction count matches, then sends each prediction down the private result channel of the job it belongs to.
13. **Each handler goroutine wakes up,** turns the prediction into JSON (`{"class":7,"confidence":0.91}`) and responds 200. If anything failed, the error is translated to a status code (section 6).

### A worked batching example

`MaxBatch=4`, `MaxWait=10ms`. Requests A, B, C arrive at t=0, 2ms, 3ms:

- t=0: A arrives, timer starts (fires at t=10ms). Batch = [A].
- t=2, t=3: B and C join. Batch = [A, B, C].
- t=8: C's client disconnects. C is now dead.
- t=10: timer fires. The dispatcher prunes C and sends [A, B] as one engine call.
- The engine returns 2 predictions. A gets the first, B gets the second.

If D had arrived at t=4, the batch would have hit 4 and flushed at t=4 without waiting for the timer.

---

## 4. The Go batcher (`go-server/internal/batcher`)

### The goroutines involved

```
many handler goroutines  --Submit-->  [bounded queue]  -->  1 collector  -->  up to MaxInFlight dispatch goroutines  --> engine
        ^                                                                              |
        +------------------ private result channel per job <--------------------------+
```

- **Handler goroutines** (one per HTTP request) call `Submit` and wait.
- **One collector goroutine** forms batches. Being the only reader of the queue means batch formation needs no lock.
- **Dispatch goroutines**, at most `MaxInFlight` at once, make the engine calls.
- **Each job has a private result channel with capacity 1.** The sender never blocks even if the caller has already left, so nothing leaks and one departed caller can't stall a batch.

### Config (all values required; no hidden defaults yet)

| Field | Meaning |
| --- | --- |
| `MaxBatch` | flush when this many jobs are collected |
| `MaxWait` | flush this long after the first job of a batch |
| `QueueSize` | queue capacity; full means reject |
| `EngineTimeout` | hard cap on one engine call |
| `MaxInFlight` | simultaneous engine calls |

### Design decisions worth explaining

- **Non-blocking enqueue.** `Submit` uses a `select` with a `default`, so it never waits for queue space. Blocking would let overload turn into unbounded latency and memory.
- **The batching window starts at the first *live* job.** A dead job never starts a timer, and an idle batcher doesn't spin.
- **Two liveness checks.** Dead jobs are dropped when taken off the queue and again just before the engine call, since callers can give up while the batch waits for a free engine slot.
- **Engine deadline = latest caller deadline** (capped by `EngineTimeout`; if any caller has no deadline, `EngineTimeout` applies). Using the *earliest* would let one impatient caller cancel work that everyone else is still waiting for. Using the latest means the call stops once nobody could still use the answer. Each caller still returns at *its own* deadline because `Submit` waits on its own context.
- **Shutdown vs. Submit race.** `Submit` holds a read lock while sending to the queue; `Shutdown` takes the write lock before closing it. This makes "send on closed channel" panics impossible. The race detector verifies it.
- **Shutdown flushes instead of waiting.** On close, queued jobs are sent immediately without waiting out `MaxWait`, then in-flight calls are awaited, bounded by the caller's context. It's idempotent.
- **Failure isolation.** An engine error, wrong-length response or panic fails *that batch only*; later batches work. The panic is caught with `recover` and becomes `ErrEnginePanic`.

### The optional `Observer`

`Config.Observer` is a two-method interface (`ObserveQueueWait`, `ObserveBatch`) that the batcher calls for metrics. It exists so the batcher doesn't import Prometheus. Calls are wrapped in `recover`, so a bug in a metrics hook can never break serving. **Queue wait** is measured from enqueue until the engine call starts (so it includes the batching window and any wait for an engine slot).

### What `Stats()` reports

`Submitted`, `Rejected` (queue full), `Expired` (dropped before dispatch), `Batches`, `Items` (so average batch size = Items / Batches), `EngineErrors`, `QueueDepth`.

---

## 5. The Go gRPC client (`go-server/internal/engineclient`)

The adapter between the batcher and the C++ process.

- **Conversion:** `[][]float32` -> protobuf `Image` messages; protobuf predictions -> `batcher.Prediction`.
- **Splitting:** the C++ engine accepts at most 8 images per call. The client splits larger batches into chunks, keeps order, **fails the whole call if any chunk fails**, and stops sending further chunks. No partial results.
- **Fail fast:** no "wait for ready". If the engine is down, the call returns `ErrUnavailable` immediately instead of queueing behind a reconnect.
- **Error translation** (so callers can use `errors.Is`):
  - `INVALID_ARGUMENT` -> `ErrInvalidRequest` (our bug or bad input; retrying won't help)
  - `UNAVAILABLE` -> `ErrUnavailable`
  - deadline/cancel codes -> `context.DeadlineExceeded` / `context.Canceled`
  - if the caller's own context has ended, that wins over any transport error
- **Deadline propagation:** the context deadline goes over the wire, so the engine can see it.
- **Response check:** prediction count must equal image count, else `ErrBadResponse`.

---

## 6. The Go HTTP layer (`go-server/internal/httpapi`)

### Endpoints

| Route | Behavior |
| --- | --- |
| `POST /predict` | the inference endpoint |
| `GET /healthz` | liveness: always 200 |
| `GET /readyz` | readiness: 200, or 503 once `SetDraining()` is called |

Liveness and readiness differ on purpose. When draining, readiness fails so the load balancer stops sending traffic, but liveness stays OK so the orchestrator doesn't kill the process while it's finishing in-flight work.

### Validation (before the batcher, so bad input costs nothing)

| Problem | Status |
| --- | --- |
| wrong method | 405 + `Allow: POST` |
| Content-Type not JSON | 415 |
| body too large | 413 |
| bad JSON, unknown field, wrong type, missing/null/empty pixels, wrong count, value outside [0,1], float overflow (`1e999`), trailing data | 400 |

### Error -> status mapping

| What happened | Status | Why |
| --- | --- | --- |
| queue full | 429 + `Retry-After` | overload; the client should back off |
| shutting down, or engine unreachable | 503 + `Retry-After` | temporary |
| deadline exceeded | 504 | our time budget ran out |
| client disconnected | 499 (recorded only) | not a server fault; stays out of error-rate alerts |
| engine rejected input the handler already validated | 500 | the two sides disagree on the contract: our bug |
| bad engine response, engine panic, anything else | 502 | upstream misbehaved |

5xx responses carry a generic message; detail goes to the log, never to the client.

### Other behavior

- One `RequestTimeout` covers queue wait *and* engine time.
- A panic anywhere in the request is caught and becomes a 500; the server keeps running.
- A NaN/Inf confidence from the engine becomes a 502 rather than a broken JSON body.
- A `Recorder(status, latency)` callback fires once per request (all outcomes), ready to feed Prometheus.

---

## 6b. Metrics (`go-server/internal/metrics`)

Served at `GET /metrics` in Prometheus text format. A private registry is used (no global state), so tests and multiple instances can't collide.

| Metric | Type | What it tells you |
| --- | --- | --- |
| `inference_http_requests_total{code}` | counter | traffic by status code; 429s, 503s and 504s are the overload story |
| `inference_http_request_duration_seconds` | histogram | what clients actually experience |
| `inference_queue_wait_seconds` | histogram | time spent waiting for a batch to form and for an engine slot |
| `inference_engine_latency_seconds` | histogram | the C++ call alone |
| `inference_batch_size` | histogram | whether batching works; has **one bucket per size 1..8** because the engine's limit is 8 |
| `inference_batcher_submitted_total`, `_rejected_total`, `_expired_total`, `_batches_total`, `_items_total`, `_engine_errors_total` | counters | read from `Batcher.Stats()` **at scrape time**, so they're always current |
| `inference_batcher_queue_depth` | gauge | the leading indicator of saturation |

How to read them: client latency is roughly queue wait + engine latency + a little overhead. If queue wait dominates, the batching window or the engine slots are the bottleneck. If engine latency dominates, it's the C++ side. Average batch size = `items_total / batches_total`.

Label cardinality is kept low on purpose: the only label is the status code. Per-path or per-client labels would blow up the number of time series.

## 6c. Wiring and graceful shutdown (`go-server/internal/app`, `main.go`)

`main.go` parses flags (all have defaults), listens for SIGINT/SIGTERM via `signal.NotifyContext`, and calls `app.Run`. `Run` validates the config, binds the port (failing fast if it's taken), builds the engine client, metrics, batcher and HTTP server, and serves.

**Config checks that catch misconfiguration at startup:** every field must be set; `RequestTimeout` must exceed `MaxWait` (otherwise every request would time out waiting for its batch); `DrainDelay` must be less than `ShutdownTimeout`.

**Shutdown order (this is the part to be able to explain):**

1. **Mark not-ready** (`/readyz` -> 503) and wait `DrainDelay`. The listener stays open, so requests still arriving are served while the load balancer notices. `/healthz` stays 200, otherwise the orchestrator would kill the process mid-drain.
2. **Stop the HTTP server** (`http.Server.Shutdown`): stop accepting and wait for in-flight requests to finish. In-flight requests are waiting on the batcher, so the batcher and engine must still be alive.
3. **Shut down the batcher.** HTTP is stopped, so nothing can enqueue anymore. Anything left is flushed immediately, and in-flight engine calls are awaited.
4. **Close the engine connection.**

If the order were reversed (batcher first), requests that arrive during the drain would get errors instead of answers.

The whole sequence shares one `ShutdownTimeout` budget, created from a *fresh* context. The signal context is already cancelled by then, and using it would abort the shutdown immediately. If the budget runs out (for example, the engine is stuck), it force-closes the remaining connections and `Run` returns an error saying so, rather than hanging forever.

**Tested with a fake engine over real TCP:** end to end request + metrics; 64 concurrent requests, each routed to the right caller, in far fewer batches than requests; engine killed -> 503 within a second; a request mid-flight when shutdown starts still completes with 200 while `/readyz` already shows 503; the shutdown timeout is enforced when the engine hangs; the server fails fast when the port is taken.

**Manual smoke run** against the real C++ binary (random weights, so predictions are meaningless): valid request -> 200; wrong length -> 400; 40 concurrent requests -> all served, with 33 images grouped into 5 engine calls at the time I scraped; engine killed -> 503; SIGTERM -> readiness 503 during the drain, then a clean exit. This was a one-off sanity check, **not a benchmark**.

## 7. The C++ engine (`cpp-engine`)

### Process layout

`main` builds a `Model` (loads the weight file), builds a gRPC `Engine` service holding a **reference** to the model, starts a server and blocks in `Wait()`. The model lives in `main`'s scope, so it outlives the server. Run it with `inference_engine weights.bin [address]` (default `127.0.0.1:50051`).

### The `PredictBatch` handler

1. If the call was already cancelled -> `CANCELLED`.
2. Batch size not in 1..8 -> `INVALID_ARGUMENT`.
3. Any image not exactly `input_dim` (784) pixels -> `INVALID_ARGUMENT`.
4. Copy all images into one contiguous buffer and call `Model::predict`.
5. `invalid_argument` exceptions -> `INVALID_ARGUMENT`; any other exception -> `INTERNAL`.

### Model and file format

Weight file (all little-endian, decoded explicitly so host byte order doesn't matter):

```
"MIN1"  | input u32 | hidden u32 | output u32 | W1 | b1 | W2 | b2   (float32, row-major)
```

The loader is defensive because the file is untrusted input: bad magic, truncated data, dimensions outside 1..4096 (to bound allocation), non-finite weights, and trailing bytes all throw.

### The forward pass (`Model::predict`)

Given `batch` images flattened into one array:

1. Reject any pixel that isn't finite and within [0, 1].
2. `hidden = pixels × W1`  (matmul: batch×784 times 784×128)
3. add bias `b1`, then **ReLU** (negatives -> 0)
4. `scores = hidden × W2`  (batch×128 times 128×10), add bias `b2`
5. **softmax** per row: turns 10 scores into 10 probabilities that sum to 1 (it subtracts the row maximum first so `exp` can't overflow)
6. For each row, the highest-probability column is the predicted class; its probability is the confidence.

Matrices are flat arrays in row-major order: element (row i, column j) of an `m×n` matrix is at index `i*n + j`. The kernels are plain triple loops, with no SIMD or blocking, because correctness and clarity matter more here than speed.

### Concurrency in the engine

The gRPC server handles requests on a thread pool, so several `PredictBatch` calls can run at once. That's safe because the model is read-only after loading (`predict` is `const`), and each call allocates its own temporary vectors. There's no shared mutable state, so no locks.

---

## 8. The model and training

- Model: 784 inputs -> 128 hidden (ReLU) -> 10 outputs (softmax).
- `training/train.py` is a NumPy-only training loop (no PyTorch). It reads MNIST files, trains with gradient descent, and exports the binary weight file via `weights_io.py`.
- **It has never been run on real data here**, and `training/data` is empty. To test the Go↔C++ link I generated a **randomly weighted** file (outside the repo), which is enough to prove the plumbing but means predictions are meaningless. Don't claim accuracy.

---

## 9. Every edge case in one table

| Edge case | Where | Behavior |
| --- | --- | --- |
| Queue full | batcher | immediate `ErrQueueFull` -> HTTP 429 |
| Context already expired | batcher | rejected before enqueue |
| Caller cancels while queued | batcher | dropped; engine never sees it |
| Caller gives up while batch waits for an engine slot | batcher | pruned again at dispatch |
| Whole batch is dead | batcher | no engine call at all |
| Caller deadline hits during engine call | batcher | caller returns at its own deadline; the result is discarded safely |
| Mixed deadlines in a batch | batcher | engine deadline = latest, capped by `EngineTimeout` |
| Hung engine | batcher | `EngineTimeout` cuts it off |
| Engine error | batcher | whole batch fails; later batches fine |
| Wrong prediction count | batcher + client | `ErrBadResponse`, never a panic, never a wrong result |
| Engine client panics | batcher | recovered; batch fails; batcher survives |
| Engine saturated | batcher | collector blocks -> queue fills -> 429 (backpressure) |
| Submit races Shutdown | batcher | read/write lock; no send-on-closed-channel |
| Shutdown with queued jobs | batcher | flushed immediately, not after `MaxWait` |
| Shutdown with a stuck engine | batcher | returns the caller's context error |
| Batch > 8 images | client | split, ordered, fail-as-a-unit |
| Engine down | client | fails fast as `ErrUnavailable` -> 503 |
| Engine rejects input | client | `ErrInvalidRequest` -> 500 (our bug) |
| Oversized body | HTTP | 413 |
| Malformed JSON, trailing data, out-of-range values | HTTP | 400, never reaches batcher |
| Client disconnects | HTTP | request context cancels; batcher drops the job; recorded as 499 |
| Panic in a request | HTTP | 500, server keeps serving |
| NaN confidence from engine | HTTP | 502, not a corrupt body |
| Bad or truncated weight file | C++ | engine refuses to start |

---

## 10. How it's tested

- **Batcher (23 tests):** a fake engine client that can block, fail, panic or return wrong-sized answers, so each failure is deterministic. Timing is coordinated with channels and a polling `eventually` helper on `Stats()` rather than fixed sleeps. Everything runs with `go test -race`.
- **gRPC client (11 unit + 2 integration tests):** a fake gRPC server running in memory (`bufconn`), so the real client code and real protobuf run without a network. Includes a 200-request batcher+client run, and a check that an `INVALID_ARGUMENT` from the server reaches the original caller as `ErrInvalidRequest`.
- **Integration tests (build tag `integration`):** run the client against the real C++ binary:
  ```
  ENGINE_ADDR=127.0.0.1:50051 go test -tags=integration ./internal/engineclient/
  ```
  They check the split of 20 images into 3 calls and that a wrong-length image comes back as `ErrInvalidRequest`. They need a running engine, so they're skipped in a normal `go test`.
- **HTTP layer (~40 cases):** table tests for every malformed input (asserting the batcher is never reached), every error-to-status mapping (asserting no internal detail leaks), timeouts, disconnect, panic recovery, readiness, recording, and a saturation test with the real batcher producing a 429.
- **Metrics (8 tests):** read values back out of the registry; check the batch-size buckets resolve sizes 1..8, that counters are read at scrape time, and that duplicate registration errors instead of panicking.
- **App (7 tests):** end to end over real TCP with a fake engine, including the graceful-shutdown scenarios in 6c.
- **C++ (5 Catch2 test cases, 27 assertions):** kernels on small hand-computed matrices, and the model loader on a tiny fixture.
- **Mutation checks:** I deliberately broke the code (no dispatch-time prune, blocking enqueue, no panic recovery, no body limit, no range check, no trailing-data check, unmapped 429, no 499, never marking draining, shutting down with the cancelled context, unbounded batcher shutdown). One surviving mutation: removing the forced `srv.Close()` after a shutdown timeout isn't caught, because the test only checks that `Run` returns an error in time and confirmed tests fail. Passing tests only mean something if they can fail.

Run everything:
```
cd go-server && go test -race ./...
cd cpp-engine/build && ./inference_tests
```

---

## 11. Interview prep

**30-second pitch.** "A small inference server split like Triton: a Go front end that validates, queues and dynamically batches requests, and a C++ gRPC engine that runs a hand-written neural network. I focused on the serving problems: backpressure, deadlines, cancellation, failure isolation and graceful shutdown, and I tested each failure mode deterministically with the race detector on."

**Likely questions:**

1. **What does batching cost?** Up to `MaxWait` of added latency at low load; at high load the size trigger dominates and the cost is just batch compute time.
2. **What stops the queue growing forever?** It's bounded, and `Submit` never blocks; a full queue returns `ErrQueueFull` -> 429.
3. **One request in a batch times out. What happens to the others?** Only that caller returns early. The engine call keeps going until the latest deadline, so the others still get answers.
4. **How do you avoid returning a result to the wrong caller?** Each job has its own channel; the engine's ordered response is matched by index *after* checking the length; a 300-goroutine test checks it.
5. **Engine is down?** The client fails fast with `ErrUnavailable`; the batcher fails that batch immediately; callers get 503 with `Retry-After`. Nothing piles up waiting on a reconnect.
6. **How do you shut down without dropping requests?** Mark draining (readiness fails), stop accepting, flush queued jobs immediately, wait for in-flight engine calls under a deadline. `app.Run` does exactly this, and the order matters: HTTP stops before the batcher so in-flight requests still have a live backend.
7. **Why 429 vs 503 vs 504?** 429: you're sending too much, back off. 503: temporarily unavailable, retry elsewhere. 504: I accepted it but ran out of time.
8. **Why is a client disconnect 499, not an error?** It isn't the server's fault, and counting it as an error would pollute alerting.
9. **Where could there be a data race?** The shared `closed` flag versus sends on the queue (guarded by the lock), the `Stats` counters (atomics), and the model weights (read-only). The race detector runs on all of it.
10. **Why earliest vs latest deadline for the engine call?** See section 4: latest avoids one impatient caller destroying everyone's work.
11. **What would break first under load?** The queue fills and rejections rise; `QueueDepth` and `Rejected` show it before latency does.
12. **Why two languages?** Go for concurrent I/O, C++ for control of memory and compute.
13. **A cross-language bug you found?** The engine silently capped batches at 8 and validates pixel range on its own, constraints that only existed in the C++ code. The Go client now enforces the split, and the HTTP layer validates the same range so the two stay consistent.

**Be upfront about:** no real training run, the end-to-end run was a manual smoke test with random weights (not automated), and there are no performance numbers.

---

## 12. Known limitations and next steps

**Limitations (worth volunteering; they show you understand the trade-offs):**

- Pixel buffers are borrowed from the caller: a cancelled request's buffer may still be read by an in-flight batch. The HTTP layer allocates a fresh slice per request, which satisfies this.
- One failing request can fail its whole batch (the proto has no per-item error).
- An engine call is not cancelled when *every* waiter has left but the latest deadline hasn't passed; the C++ engine only checks cancellation at the start of a call.
- FIFO only: no priorities or fairness.
- Single collector goroutine: fine at this scale, the first thing to profile at very high rates.
- `MaxWait` is static, whereas production systems adapt it from load.
- No retries: inference is idempotent so a retry on `UNAVAILABLE` would be safe, but it's not implemented.
- Matmul is a naive triple loop.

**Next steps:**

1. Finish the engine side. Per the current plan, only the kernels (matmul, ReLU, softmax) stay as finished C++; the model loader and gRPC service are to be redone later.
2. A load test with failure injection (kill the engine mid-load) to produce real numbers and real stories.
3. Run against real trained weights.

---

## 13. Go concepts used here, in plain English

- **Goroutine:** a very cheap thread managed by Go's runtime. `go f()` runs `f` concurrently. net/http runs one per request.
- **Channel:** a typed pipe between goroutines. A buffered channel (`make(chan T, n)`) holds up to n items before senders block. The queue is a buffered channel of jobs; each job's result channel has capacity 1.
- **`select`:** waits on several channel operations and runs whichever is ready first. With a `default` branch it never blocks, which is how `Submit` rejects on a full queue.
- **`context.Context`:** carries a deadline and a cancellation signal across function and goroutine boundaries. `ctx.Done()` is a channel that closes when time runs out or someone cancels. It's how deadlines and client disconnects propagate down to the engine call.
- **Mutex / RWMutex:** a lock. `RLock` lets many holders in at once; `Lock` is exclusive. Here many `Submit` calls share the read lock while `Shutdown` takes the exclusive one.
- **Semaphore (a buffered channel used as a counter):** acquiring = send into the channel, releasing = receive. Capacity = max concurrent holders. `MaxInFlight` is one.
- **`sync.WaitGroup`:** counts running goroutines so you can wait for them all to finish.
- **`atomic`:** lock-free counters safe to use from many goroutines.
- **`defer` / `recover`:** `defer` runs code when a function exits; `recover` inside a deferred function catches a panic, so one bad call doesn't kill the process.
- **`errors.Is`:** checks whether an error (even if wrapped with extra context) matches a known sentinel such as `ErrQueueFull`. It's how the HTTP layer picks a status code without string matching.
- **Race detector (`-race`):** instruments the program and reports unsynchronized memory access at runtime. It only catches races that actually execute, which is why the tests deliberately create heavy concurrency.
