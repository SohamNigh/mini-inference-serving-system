package batcher

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ---- test helpers -------------------------------------------------------

type fakeClient struct {
	mu          sync.Mutex
	calls       [][][]float32
	deadlines   []time.Time
	inflight    int32
	maxInflight int32
	fn          func(ctx context.Context, images [][]float32) ([]Prediction, error)
}

func (f *fakeClient) PredictBatch(ctx context.Context, images [][]float32) ([]Prediction, error) {
	n := atomic.AddInt32(&f.inflight, 1)
	defer atomic.AddInt32(&f.inflight, -1)
	for {
		m := atomic.LoadInt32(&f.maxInflight)
		if n <= m || atomic.CompareAndSwapInt32(&f.maxInflight, m, n) {
			break
		}
	}
	f.mu.Lock()
	f.calls = append(f.calls, images)
	d, _ := ctx.Deadline()
	f.deadlines = append(f.deadlines, d)
	f.mu.Unlock()
	if f.fn == nil {
		return classFromFirstPixel(ctx, images)
	}
	return f.fn(ctx, images)
}

func (f *fakeClient) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func (f *fakeClient) batchSizes() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	sizes := make([]int, len(f.calls))
	for i, c := range f.calls {
		sizes[i] = len(c)
	}
	return sizes
}

// classFromFirstPixel makes each prediction identifiable: class == pixels[0].
func classFromFirstPixel(_ context.Context, images [][]float32) ([]Prediction, error) {
	out := make([]Prediction, len(images))
	for i, img := range images {
		out[i] = Prediction{Class: int32(img[0]), Confidence: 1}
	}
	return out, nil
}

func testConfig() Config {
	return Config{
		MaxBatch:      8,
		MaxWait:       20 * time.Millisecond,
		QueueSize:     64,
		EngineTimeout: 5 * time.Second,
		MaxInFlight:   1,
	}
}

func newBatcher(t *testing.T, c EngineClient, cfg Config) *Batcher {
	t.Helper()
	b, err := New(c, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = b.Shutdown(ctx)
	})
	return b
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for: %s", what)
}

// submit uses a bounded context so a broken batcher fails the test instead of hanging it.
func submit(b *Batcher, v float32) (Prediction, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	return b.Submit(ctx, []float32{v})
}

// ---- configuration ------------------------------------------------------

func TestNewRejectsInvalidConfig(t *testing.T) {
	bad := []func(*Config){
		func(c *Config) { c.MaxBatch = 0 },
		func(c *Config) { c.MaxWait = 0 },
		func(c *Config) { c.QueueSize = 0 },
		func(c *Config) { c.EngineTimeout = 0 },
		func(c *Config) { c.MaxInFlight = 0 },
	}
	for i, mutate := range bad {
		cfg := testConfig()
		mutate(&cfg)
		if _, err := New(&fakeClient{}, cfg); err == nil {
			t.Errorf("case %d: expected config error, got nil", i)
		}
	}
	if _, err := New(nil, testConfig()); err == nil {
		t.Error("expected error for nil client")
	}
}

// ---- flush triggers -----------------------------------------------------

func TestFlushesAtMaxBatchSizeWithoutWaitingForTimer(t *testing.T) {
	cfg := testConfig()
	cfg.MaxBatch = 2
	cfg.MaxWait = time.Hour // only the size trigger can fire
	client := &fakeClient{}
	b := newBatcher(t, client, cfg)

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := submit(b, 1); err != nil {
				t.Errorf("submit: %v", err)
			}
		}()
	}
	wg.Wait()

	if got := client.batchSizes(); len(got) != 1 || got[0] != 2 {
		t.Fatalf("expected one batch of 2, got %v", got)
	}
}

func TestFlushesPartialBatchAtMaxWait(t *testing.T) {
	cfg := testConfig()
	cfg.MaxBatch = 100
	cfg.MaxWait = 30 * time.Millisecond
	client := &fakeClient{}
	b := newBatcher(t, client, cfg)

	start := time.Now()
	if _, err := submit(b, 1); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if elapsed := time.Since(start); elapsed < 25*time.Millisecond {
		t.Fatalf("flushed after %v, before the %v batching window", elapsed, cfg.MaxWait)
	}
	if got := client.batchSizes(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("expected one batch of 1, got %v", got)
	}
}

// ---- correctness under concurrency --------------------------------------

func TestResultsRoutedToCorrectCallerUnderConcurrency(t *testing.T) {
	cfg := testConfig()
	cfg.MaxBatch = 16
	cfg.MaxWait = 5 * time.Millisecond
	cfg.QueueSize = 512
	cfg.MaxInFlight = 4
	client := &fakeClient{}
	b := newBatcher(t, client, cfg)

	const n = 300
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			p, err := submit(b, float32(i))
			if err != nil {
				t.Errorf("submit %d: %v", i, err)
				return
			}
			if int(p.Class) != i {
				t.Errorf("caller %d received result for %d", i, p.Class)
			}
		}()
	}
	wg.Wait()

	total := 0
	for _, s := range client.batchSizes() {
		if s > cfg.MaxBatch {
			t.Errorf("batch of %d exceeds MaxBatch %d", s, cfg.MaxBatch)
		}
		total += s
	}
	if total != n {
		t.Fatalf("engine saw %d images, want %d", total, n)
	}
}

// ---- deadlines and cancellation -----------------------------------------

func TestAlreadyCancelledContextIsRejectedWithoutEnqueueing(t *testing.T) {
	client := &fakeClient{}
	b := newBatcher(t, client, testConfig())

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := b.Submit(ctx, []float32{1})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
	if s := b.Stats(); s.Submitted != 0 {
		t.Fatalf("cancelled request was enqueued: %+v", s)
	}
}

func TestJobCancelledWhileWaitingForBatchIsNotSentToEngine(t *testing.T) {
	cfg := testConfig()
	cfg.MaxWait = 300 * time.Millisecond
	client := &fakeClient{}
	b := newBatcher(t, client, cfg)

	ctxA, cancelA := context.WithCancel(context.Background())
	errA := make(chan error, 1)
	go func() {
		_, err := b.Submit(ctxA, []float32{111})
		errA <- err
	}()
	eventually(t, "A enqueued", func() bool { return b.Stats().Submitted == 1 })
	cancelA()
	if err := <-errA; !errors.Is(err, context.Canceled) {
		t.Fatalf("A: want context.Canceled, got %v", err)
	}

	p, err := submit(b, 7)
	if err != nil || p.Class != 7 {
		t.Fatalf("B: got %+v, %v", p, err)
	}

	client.mu.Lock()
	defer client.mu.Unlock()
	if len(client.calls) != 1 || len(client.calls[0]) != 1 || client.calls[0][0][0] != 7 {
		t.Fatalf("engine should see only B's image, saw %v", client.calls)
	}
}

func TestEngineNotCalledWhenEveryJobInBatchIsDead(t *testing.T) {
	client := &fakeClient{}
	b := newBatcher(t, client, testConfig())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _, _ = b.Submit(ctx, []float32{1}); close(done) }()
	eventually(t, "enqueued", func() bool { return b.Stats().Submitted == 1 })
	cancel()
	<-done

	eventually(t, "job counted expired", func() bool { return b.Stats().Expired == 1 })
	if n := client.callCount(); n != 0 {
		t.Fatalf("engine called %d times for a batch with no live jobs", n)
	}
}

func TestCallerReturnsAtItsOwnDeadlineWhileEngineIsStillWorking(t *testing.T) {
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		time.Sleep(150 * time.Millisecond)
		return classFromFirstPixel(ctx, images)
	}}
	cfg := testConfig()
	cfg.MaxBatch = 2
	b := newBatcher(t, client, cfg)

	shortCtx, c1 := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer c1()
	longCtx, c2 := context.WithTimeout(context.Background(), 2*time.Second)
	defer c2()

	type res struct {
		p   Prediction
		err error
		at  time.Duration
	}
	start := time.Now()
	out := make(chan res, 2)
	go func() { p, err := b.Submit(shortCtx, []float32{1}); out <- res{p, err, time.Since(start)} }()
	go func() { p, err := b.Submit(longCtx, []float32{2}); out <- res{p, err, time.Since(start)} }()

	var short, long res
	for i := 0; i < 2; i++ {
		r := <-out
		if errors.Is(r.err, context.DeadlineExceeded) {
			short = r
		} else {
			long = r
		}
	}
	if !errors.Is(short.err, context.DeadlineExceeded) || short.at > 120*time.Millisecond {
		t.Fatalf("short-deadline caller: err=%v after %v", short.err, short.at)
	}
	if long.err != nil || long.p.Class != 2 {
		t.Fatalf("long-deadline caller should still succeed: %+v", long)
	}
}

func TestEngineContextDeadlineIsLatestCallerDeadline(t *testing.T) {
	client := &fakeClient{}
	cfg := testConfig()
	cfg.MaxBatch = 2
	cfg.EngineTimeout = 5 * time.Minute // must not cap the caller deadlines below
	b := newBatcher(t, client, cfg)

	early, c1 := context.WithTimeout(context.Background(), 30*time.Second)
	defer c1()
	late, c2 := context.WithTimeout(context.Background(), 60*time.Second)
	defer c2()
	lateDeadline, _ := late.Deadline()

	var wg sync.WaitGroup
	for _, ctx := range []context.Context{early, late} {
		ctx := ctx
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = b.Submit(ctx, []float32{1}) }()
	}
	wg.Wait()

	client.mu.Lock()
	defer client.mu.Unlock()
	if len(client.deadlines) != 1 || !client.deadlines[0].Equal(lateDeadline) {
		t.Fatalf("engine deadline %v, want the latest caller deadline %v", client.deadlines, lateDeadline)
	}
}

func TestEngineTimeoutCapsEngineContext(t *testing.T) {
	client := &fakeClient{fn: func(ctx context.Context, _ [][]float32) ([]Prediction, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	cfg := testConfig()
	cfg.EngineTimeout = 50 * time.Millisecond
	b := newBatcher(t, client, cfg)

	ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	start := time.Now()
	_, err := b.Submit(ctx, []float32{1})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want DeadlineExceeded from engine timeout, got %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("hung engine was not cut off by EngineTimeout")
	}
}

// ---- engine failure modes -----------------------------------------------

func TestEngineErrorFailsEveryJobInBatchButNotLaterBatches(t *testing.T) {
	var fail atomic.Bool
	fail.Store(true)
	wantErr := errors.New("engine unavailable")
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		if fail.Load() {
			return nil, wantErr
		}
		return classFromFirstPixel(ctx, images)
	}}
	cfg := testConfig()
	cfg.MaxBatch = 2
	b := newBatcher(t, client, cfg)

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := submit(b, 1); !errors.Is(err, wantErr) {
				t.Errorf("want %v, got %v", wantErr, err)
			}
		}()
	}
	wg.Wait()

	fail.Store(false)
	if p, err := submit(b, 5); err != nil || p.Class != 5 {
		t.Fatalf("batcher did not recover after engine error: %+v %v", p, err)
	}
}

func TestResponseLengthMismatchIsReportedNotPanic(t *testing.T) {
	client := &fakeClient{fn: func(context.Context, [][]float32) ([]Prediction, error) {
		return []Prediction{}, nil // engine bug: fewer predictions than images
	}}
	b := newBatcher(t, client, testConfig())

	_, err := submit(b, 1)
	if !errors.Is(err, ErrBadResponse) {
		t.Fatalf("want ErrBadResponse, got %v", err)
	}
}

func TestEnginePanicFailsBatchAndBatcherKeepsServing(t *testing.T) {
	var first atomic.Bool
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		if first.CompareAndSwap(false, true) {
			panic("boom")
		}
		return classFromFirstPixel(ctx, images)
	}}
	b := newBatcher(t, client, testConfig())

	if _, err := submit(b, 1); !errors.Is(err, ErrEnginePanic) {
		t.Fatalf("want ErrEnginePanic, got %v", err)
	}
	if p, err := submit(b, 9); err != nil || p.Class != 9 {
		t.Fatalf("batcher died after panic: %+v %v", p, err)
	}
}

// ---- backpressure and concurrency limits --------------------------------

func TestFullQueueRejectsImmediatelyInsteadOfBlocking(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{}, 8)
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		entered <- struct{}{}
		<-release
		return classFromFirstPixel(ctx, images)
	}}
	cfg := testConfig()
	cfg.MaxBatch = 1
	cfg.QueueSize = 1
	cfg.MaxInFlight = 1
	b := newBatcher(t, client, cfg)
	defer close(release)

	var wg sync.WaitGroup
	launch := func() {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = submit(b, 1) }()
	}
	launch() // #1 occupies the single in-flight slot
	<-entered
	launch() // #2 is held by the collector, waiting for a slot
	eventually(t, "#2 taken by collector", func() bool { s := b.Stats(); return s.Submitted == 2 && s.QueueDepth == 0 })
	launch() // #3 fills the queue
	eventually(t, "#3 queued", func() bool { s := b.Stats(); return s.Submitted == 3 && s.QueueDepth == 1 })

	start := time.Now()
	_, err := b.Submit(context.Background(), []float32{1})
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("want ErrQueueFull, got %v", err)
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatalf("rejection took %v; Submit must not block on a full queue", time.Since(start))
	}
	if s := b.Stats(); s.Rejected != 1 {
		t.Fatalf("Rejected = %d, want 1", s.Rejected)
	}
	release <- struct{}{}
	release <- struct{}{}
	release <- struct{}{}
	wg.Wait()
}

func TestMaxInFlightLimitsConcurrentEngineCalls(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{}, 16)
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		entered <- struct{}{}
		<-release
		return classFromFirstPixel(ctx, images)
	}}
	cfg := testConfig()
	cfg.MaxBatch = 1
	cfg.MaxInFlight = 2
	b := newBatcher(t, client, cfg)

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = submit(b, 1) }()
	}
	<-entered
	<-entered
	select {
	case <-entered:
		t.Fatal("third engine call started while MaxInFlight=2")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	wg.Wait()
	if m := atomic.LoadInt32(&client.maxInflight); m != 2 {
		t.Fatalf("max concurrent engine calls = %d, want 2", m)
	}
}

// ---- shutdown ------------------------------------------------------------

func TestShutdownFlushesQueuedJobsImmediatelyAndRejectsNewOnes(t *testing.T) {
	cfg := testConfig()
	cfg.MaxBatch = 100
	cfg.MaxWait = time.Hour
	client := &fakeClient{}
	b := newBatcher(t, client, cfg)

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			p, err := submit(b, float32(i))
			if err != nil || int(p.Class) != i {
				t.Errorf("queued job %d lost during shutdown: %+v %v", i, p, err)
			}
		}()
	}
	eventually(t, "3 queued", func() bool { return b.Stats().Submitted == 3 })

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	start := time.Now()
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if time.Since(start) > 500*time.Millisecond {
		t.Fatalf("Shutdown waited out the batching window (%v)", time.Since(start))
	}
	wg.Wait()

	if _, err := b.Submit(context.Background(), []float32{1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("want ErrClosed after shutdown, got %v", err)
	}
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("second Shutdown should be a no-op, got %v", err)
	}
}

func TestShutdownReturnsContextErrorIfEngineIsStuck(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		entered <- struct{}{}
		<-release
		return classFromFirstPixel(ctx, images)
	}}
	b := newBatcher(t, client, testConfig())

	go func() { _, _ = submit(b, 1) }()
	<-entered

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := b.Shutdown(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want DeadlineExceeded, got %v", err)
	}
	close(release)
	ctx2, cancel2 := context.WithTimeout(context.Background(), time.Second)
	defer cancel2()
	if err := b.Shutdown(ctx2); err != nil {
		t.Fatalf("Shutdown after engine recovered: %v", err)
	}
}

func TestConcurrentSubmitAndShutdownNeverPanicsOrLosesAJob(t *testing.T) {
	cfg := testConfig()
	cfg.QueueSize = 16
	cfg.MaxInFlight = 4
	client := &fakeClient{}
	b := newBatcher(t, client, cfg)

	var wg sync.WaitGroup
	var ok, closed, full, other atomic.Int64
	for g := 0; g < 50; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				_, err := submit(b, 1)
				switch {
				case err == nil:
					ok.Add(1)
				case errors.Is(err, ErrClosed):
					closed.Add(1)
				case errors.Is(err, ErrQueueFull):
					full.Add(1)
				default:
					other.Add(1)
					t.Errorf("unexpected error: %v", err)
				}
			}
		}()
	}
	time.Sleep(10 * time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	wg.Wait()
	if total := ok.Load() + closed.Load() + full.Load() + other.Load(); total != 50*50 {
		t.Fatalf("accounted for %d of %d requests", total, 50*50)
	}
	if closed.Load() == 0 {
		t.Log("note: no submit observed ErrClosed; shutdown raced ahead of load")
	}
}

// ---- observability -------------------------------------------------------

func TestStatsCountBatchesItemsAndEngineErrors(t *testing.T) {
	var fail atomic.Bool
	client := &fakeClient{fn: func(ctx context.Context, images [][]float32) ([]Prediction, error) {
		if fail.Load() {
			return nil, errors.New("down")
		}
		return classFromFirstPixel(ctx, images)
	}}
	cfg := testConfig()
	cfg.MaxBatch = 2
	b := newBatcher(t, client, cfg)

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = submit(b, 1) }()
	}
	wg.Wait()
	fail.Store(true)
	_, _ = submit(b, 1)

	eventually(t, "stats settle", func() bool { return b.Stats().Batches == 2 })
	s := b.Stats()
	if s.Items != 3 || s.EngineErrors != 1 || s.Submitted != 3 {
		t.Fatalf("unexpected stats: %+v", s)
	}
}
