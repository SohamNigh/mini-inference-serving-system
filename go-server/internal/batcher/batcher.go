// Package batcher implements dynamic request batching in front of an
// inference engine.
//
// Callers block in Submit. A single collector goroutine groups queued requests
// into batches (flushing when MaxBatch is reached or MaxWait has elapsed since
// the first request in the batch arrived) and hands each batch to a bounded
// pool of engine calls.
package batcher

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrQueueFull means the bounded queue is full; callers should shed load (HTTP 429/503).
	ErrQueueFull = errors.New("batcher: queue full")
	// ErrClosed means Shutdown has begun and no new work is accepted.
	ErrClosed = errors.New("batcher: closed")
	// ErrBadResponse means the engine returned a different number of predictions than images.
	ErrBadResponse = errors.New("batcher: engine returned wrong number of predictions")
	// ErrEnginePanic means the engine client panicked; the batch fails but the batcher survives.
	ErrEnginePanic = errors.New("batcher: engine call panicked")
)

type Prediction struct {
	Class      int32
	Confidence float32
}

type EngineClient interface {
	PredictBatch(ctx context.Context, images [][]float32) ([]Prediction, error)
}

// Observer receives timing signals for metrics. It is optional and must be
// cheap and non-blocking. A panicking Observer is contained: it never affects
// requests.
type Observer interface {
	// ObserveQueueWait is called once per job sent to the engine, with the time
	// from enqueue until its batch's engine call began.
	ObserveQueueWait(d time.Duration)
	// ObserveBatch is called once per engine call. err is nil on success.
	ObserveBatch(size int, engineLatency time.Duration, err error)
}

type Config struct {
	MaxBatch      int           // flush when this many requests are collected
	MaxWait       time.Duration // flush this long after the first request of a batch arrives
	QueueSize     int           // bounded queue; a full queue rejects with ErrQueueFull
	EngineTimeout time.Duration // upper bound on a single engine call
	MaxInFlight   int           // concurrent engine calls
	Observer      Observer      // optional metrics hook
}

// Stats is a point-in-time snapshot of counters, intended for metrics export.
type Stats struct {
	Submitted    int64 // accepted into the queue
	Rejected     int64 // refused because the queue was full
	Expired      int64 // dropped because the caller's context ended before dispatch
	Batches      int64 // engine calls made
	Items        int64 // images sent to the engine
	EngineErrors int64 // batches that failed (error, panic, or bad response)
	QueueDepth   int   // requests waiting right now
}

type job struct {
	ctx      context.Context
	enqueued time.Time
	pixels   []float32 // owned by the caller; must not be mutated until Submit returns a result
	resultCh chan result
}

type result struct {
	prediction Prediction
	err        error
}

type Batcher struct {
	client EngineClient
	cfg    Config
	queue  chan *job
	sem    chan struct{}
	wg     sync.WaitGroup // in-flight engine calls
	done   chan struct{}  // closed when the collector has exited and all calls finished

	mu     sync.RWMutex // guards closed; held (read) while sending on queue
	closed bool

	submitted, rejected, expired, batches, items, engineErrors atomic.Int64
}

func New(client EngineClient, cfg Config) (*Batcher, error) {
	switch {
	case client == nil:
		return nil, errors.New("batcher: nil client")
	case cfg.MaxBatch < 1:
		return nil, errors.New("batcher: MaxBatch must be >= 1")
	case cfg.MaxWait <= 0:
		return nil, errors.New("batcher: MaxWait must be > 0")
	case cfg.QueueSize < 1:
		return nil, errors.New("batcher: QueueSize must be >= 1")
	case cfg.EngineTimeout <= 0:
		return nil, errors.New("batcher: EngineTimeout must be > 0")
	case cfg.MaxInFlight < 1:
		return nil, errors.New("batcher: MaxInFlight must be >= 1")
	}
	b := &Batcher{
		client: client,
		cfg:    cfg,
		queue:  make(chan *job, cfg.QueueSize),
		sem:    make(chan struct{}, cfg.MaxInFlight),
		done:   make(chan struct{}),
	}
	go b.run()
	return b, nil
}

// Submit enqueues one request and blocks until its result, the caller's
// context ending, or rejection. It never blocks on a full queue.
func (b *Batcher) Submit(ctx context.Context, pixels []float32) (Prediction, error) {
	if err := ctx.Err(); err != nil {
		return Prediction{}, err
	}
	j := &job{ctx: ctx, pixels: pixels, resultCh: make(chan result, 1), enqueued: time.Now()}

	// Holding the read lock while sending guarantees Shutdown cannot close the
	// queue underneath us (send on closed channel would panic).
	b.mu.RLock()
	if b.closed {
		b.mu.RUnlock()
		return Prediction{}, ErrClosed
	}
	select {
	case b.queue <- j:
		b.submitted.Add(1)
		b.mu.RUnlock()
	default:
		b.mu.RUnlock()
		b.rejected.Add(1)
		return Prediction{}, ErrQueueFull
	}

	select {
	case r := <-j.resultCh:
		return r.prediction, r.err
	case <-ctx.Done():
		return Prediction{}, ctx.Err()
	}
}

// Shutdown stops accepting work, flushes everything already queued without
// waiting out the batching window, and waits for in-flight engine calls. It
// returns ctx.Err() if that takes longer than ctx allows. It is idempotent.
func (b *Batcher) Shutdown(ctx context.Context) error {
	b.mu.Lock()
	if !b.closed {
		b.closed = true
		close(b.queue)
	}
	b.mu.Unlock()

	select {
	case <-b.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *Batcher) Stats() Stats {
	return Stats{
		Submitted:    b.submitted.Load(),
		Rejected:     b.rejected.Load(),
		Expired:      b.expired.Load(),
		Batches:      b.batches.Load(),
		Items:        b.items.Load(),
		EngineErrors: b.engineErrors.Load(),
		QueueDepth:   len(b.queue),
	}
}

func (b *Batcher) run() {
	for {
		batch, closed := b.collect()
		if len(batch) > 0 {
			b.sem <- struct{}{} // blocks the collector when the engine is saturated: backpressure
			b.wg.Add(1)
			go b.dispatch(batch)
		}
		if closed {
			break
		}
	}
	b.wg.Wait()
	close(b.done)
}

// collect gathers one batch. The batching window starts when the first live
// job arrives, so an idle batcher never spins and dead jobs never start a window.
func (b *Batcher) collect() (batch []*job, closed bool) {
	var timer *time.Timer
	var timeout <-chan time.Time // nil (blocks forever) until the first live job
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()

	for len(batch) < b.cfg.MaxBatch {
		select {
		case j, ok := <-b.queue:
			if !ok {
				return batch, true
			}
			if j.ctx.Err() != nil {
				b.expired.Add(1)
				continue
			}
			batch = append(batch, j)
			if len(batch) == 1 {
				timer = time.NewTimer(b.cfg.MaxWait)
				timeout = timer.C
			}
		case <-timeout:
			return batch, false
		}
	}
	return batch, false
}

func (b *Batcher) dispatch(batch []*job) {
	defer b.wg.Done()
	defer func() { <-b.sem }()

	// Callers may have given up while this batch waited for an engine slot.
	live := batch[:0]
	for _, j := range batch {
		if j.ctx.Err() != nil {
			b.expired.Add(1)
			continue
		}
		live = append(live, j)
	}
	if len(live) == 0 {
		return
	}

	images := make([][]float32, len(live))
	for i, j := range live {
		images[i] = j.pixels
	}

	ctx, cancel := b.engineContext(live)
	defer cancel()

	b.batches.Add(1)
	b.items.Add(int64(len(live)))
	callStart := time.Now()
	for _, j := range live {
		b.observe(func(o Observer) { o.ObserveQueueWait(callStart.Sub(j.enqueued)) })
	}
	preds, err := b.call(ctx, images)
	if err == nil && len(preds) != len(live) {
		err = fmt.Errorf("%w: sent %d, got %d", ErrBadResponse, len(live), len(preds))
	}
	b.observe(func(o Observer) { o.ObserveBatch(len(live), time.Since(callStart), err) })
	if err != nil {
		b.engineErrors.Add(1)
		for _, j := range live {
			j.resultCh <- result{err: err}
		}
		return
	}
	for i, j := range live {
		j.resultCh <- result{prediction: preds[i]}
	}
}

// engineContext bounds the engine call by the LATEST caller deadline, capped by
// EngineTimeout. Using the earliest would let one impatient caller abort work
// that every other caller in the batch is still waiting for; using the latest
// stops the call once nobody could still use the answer. Each caller still
// returns at its own deadline because Submit selects on its own context.
func (b *Batcher) engineContext(live []*job) (context.Context, context.CancelFunc) {
	deadline := time.Now().Add(b.cfg.EngineTimeout)
	var latest time.Time
	allHaveDeadline := true
	for _, j := range live {
		d, ok := j.ctx.Deadline()
		if !ok {
			allHaveDeadline = false
			break
		}
		if d.After(latest) {
			latest = d
		}
	}
	if allHaveDeadline && latest.Before(deadline) {
		deadline = latest
	}
	return context.WithDeadline(context.Background(), deadline)
}

func (b *Batcher) call(ctx context.Context, images [][]float32) (preds []Prediction, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: %v", ErrEnginePanic, r)
		}
	}()
	return b.client.PredictBatch(ctx, images)
}

// observe calls fn with the configured Observer, containing any panic so a
// metrics bug cannot take down a dispatch goroutine (and with it the process).
func (b *Batcher) observe(fn func(Observer)) {
	if b.cfg.Observer == nil {
		return
	}
	defer func() { _ = recover() }()
	fn(b.cfg.Observer)
}
