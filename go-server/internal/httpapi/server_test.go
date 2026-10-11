package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"mini-inference-serving-system/go-server/internal/batcher"
	"mini-inference-serving-system/go-server/internal/engineclient"
)

// ---- helpers ---------------------------------------------------------------

type fakeSubmitter struct {
	mu    sync.Mutex
	calls int
	fn    func(ctx context.Context, pixels []float32) (batcher.Prediction, error)
}

func (f *fakeSubmitter) Submit(ctx context.Context, pixels []float32) (batcher.Prediction, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	if f.fn == nil {
		return batcher.Prediction{Class: 7, Confidence: 0.91}, nil
	}
	return f.fn(ctx, pixels)
}

func (f *fakeSubmitter) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

type record struct {
	status int
	dur    time.Duration
}

type recorder struct {
	mu   sync.Mutex
	recs []record
}

func (r *recorder) record(status int, d time.Duration) {
	r.mu.Lock()
	r.recs = append(r.recs, record{status, d})
	r.mu.Unlock()
}

func (r *recorder) all() []record {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]record(nil), r.recs...)
}

func testOptions() Options {
	return Options{PixelCount: 4, MaxBodyBytes: 1 << 10, RequestTimeout: time.Second}
}

func newServer(t *testing.T, s Submitter, opts Options) *Server {
	t.Helper()
	srv, err := New(s, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return srv
}

func post(srv *Server, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return do(srv, req)
}

func do(srv *Server, req *http.Request) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec
}

const validBody = `{"pixels":[0,0.5,1,0.25]}`

func errorBody(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var e struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil || e.Error == "" {
		t.Fatalf("expected JSON error body, got %q (%v)", rec.Body.String(), err)
	}
	return e.Error
}

// ---- happy path -----------------------------------------------------------

func TestPredictReturnsClassAndConfidenceAsJSON(t *testing.T) {
	sub := &fakeSubmitter{}
	rec := post(newServer(t, sub, testOptions()), validBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, body %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("Content-Type %q", ct)
	}
	var out struct {
		Class      int     `json:"class"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil || out.Class != 7 || math.Abs(out.Confidence-0.91) > 1e-6 {
		t.Fatalf("bad body %s (%v)", rec.Body, err)
	}
}

func TestPixelsReachSubmitterUnchanged(t *testing.T) {
	var got []float32
	sub := &fakeSubmitter{fn: func(_ context.Context, px []float32) (batcher.Prediction, error) {
		got = px
		return batcher.Prediction{}, nil
	}}
	post(newServer(t, sub, testOptions()), validBody)
	want := []float32{0, 0.5, 1, 0.25}
	for i := range want {
		if len(got) != 4 || got[i] != want[i] {
			t.Fatalf("submitter got %v, want %v", got, want)
		}
	}
}

// ---- request validation (the submitter must never be reached) --------------

func TestMalformedRequestsAreRejectedWith400AndNeverReachTheBatcher(t *testing.T) {
	cases := map[string]string{
		"empty body":          ``,
		"not json":            `hello`,
		"truncated json":      `{"pixels":[0,0.5`,
		"unknown field":       `{"pixels":[0,0,0,0],"extra":1}`,
		"missing pixels":      `{}`,
		"null pixels":         `{"pixels":null}`,
		"too few pixels":      `{"pixels":[0,0,0]}`,
		"too many pixels":     `{"pixels":[0,0,0,0,0]}`,
		"empty pixels":        `{"pixels":[]}`,
		"string pixel":        `{"pixels":[0,"x",0,0]}`,
		"above range":         `{"pixels":[0,0,0,1.5]}`,
		"below range":         `{"pixels":[0,0,0,-0.1]}`,
		"overflowing float32": `{"pixels":[0,0,0,1e999]}`,
		"trailing garbage":    `{"pixels":[0,0,0,0]} {"pixels":[0,0,0,0]}`,
		"pixels not array":    `{"pixels":"abc"}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			sub := &fakeSubmitter{}
			rec := post(newServer(t, sub, testOptions()), body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400 (body %s)", rec.Code, rec.Body)
			}
			errorBody(t, rec)
			if sub.callCount() != 0 {
				t.Fatalf("invalid request reached the batcher")
			}
		})
	}
}

func TestOversizedBodyIs413AndNeverReachesTheBatcher(t *testing.T) {
	opts := testOptions()
	opts.MaxBodyBytes = 64
	sub := &fakeSubmitter{}
	body := `{"pixels":[` + strings.Repeat("0,", 1000) + `0]}`
	rec := post(newServer(t, sub, opts), body)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d, want 413", rec.Code)
	}
	if sub.callCount() != 0 {
		t.Fatal("oversized request reached the batcher")
	}
}

func TestWrongMethodIs405WithAllowHeader(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/predict", nil)
	rec := do(newServer(t, &fakeSubmitter{}, testOptions()), req)
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("status %d, Allow %q", rec.Code, rec.Header().Get("Allow"))
	}
}

func TestWrongContentTypeIs415(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(validBody))
	req.Header.Set("Content-Type", "text/plain")
	if rec := do(newServer(t, &fakeSubmitter{}, testOptions()), req); rec.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("status %d, want 415", rec.Code)
	}
}

func TestUnknownPathIs404(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/nope", nil)
	if rec := do(newServer(t, &fakeSubmitter{}, testOptions()), req); rec.Code != http.StatusNotFound {
		t.Fatalf("status %d, want 404", rec.Code)
	}
}

// ---- error mapping --------------------------------------------------------

func TestBackendErrorsMapToDistinctStatusCodes(t *testing.T) {
	secret := "internal-detail-do-not-leak"
	cases := []struct {
		name       string
		err        error
		status     int
		retryAfter bool
	}{
		{"queue full sheds load", batcher.ErrQueueFull, http.StatusTooManyRequests, true},
		{"draining", batcher.ErrClosed, http.StatusServiceUnavailable, true},
		{"engine unreachable", fmt.Errorf("%w: %s", engineclient.ErrUnavailable, secret), http.StatusServiceUnavailable, true},
		{"deadline", context.DeadlineExceeded, http.StatusGatewayTimeout, false},
		{"wrapped deadline", fmt.Errorf("engine: %w", context.DeadlineExceeded), http.StatusGatewayTimeout, false},
		{"engine rejected validated input is our bug", fmt.Errorf("%w: %s", engineclient.ErrInvalidRequest, secret), http.StatusInternalServerError, false},
		{"bad engine response", fmt.Errorf("%w: %s", engineclient.ErrBadResponse, secret), http.StatusBadGateway, false},
		{"engine panic", fmt.Errorf("%w: %s", batcher.ErrEnginePanic, secret), http.StatusBadGateway, false},
		{"unknown error", errors.New(secret), http.StatusBadGateway, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sub := &fakeSubmitter{fn: func(context.Context, []float32) (batcher.Prediction, error) {
				return batcher.Prediction{}, tc.err
			}}
			rec := post(newServer(t, sub, testOptions()), validBody)
			if rec.Code != tc.status {
				t.Fatalf("status %d, want %d", rec.Code, tc.status)
			}
			if got := rec.Header().Get("Retry-After") != ""; got != tc.retryAfter {
				t.Fatalf("Retry-After present=%v, want %v", got, tc.retryAfter)
			}
			if msg := errorBody(t, rec); strings.Contains(msg, secret) {
				t.Fatalf("internal error detail leaked to client: %q", msg)
			}
		})
	}
}

func TestNonFiniteConfidenceFromEngineIsA502NotABrokenBody(t *testing.T) {
	sub := &fakeSubmitter{fn: func(context.Context, []float32) (batcher.Prediction, error) {
		return batcher.Prediction{Class: 1, Confidence: float32(math.NaN())}, nil
	}}
	rec := post(newServer(t, sub, testOptions()), validBody)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d, want 502", rec.Code)
	}
	errorBody(t, rec)
}

// ---- deadlines, disconnects, panics --------------------------------------

func TestRequestTimeoutProducesGatewayTimeoutAndCancelsSubmitterContext(t *testing.T) {
	opts := testOptions()
	opts.RequestTimeout = 30 * time.Millisecond
	var hadDeadline bool
	sub := &fakeSubmitter{fn: func(ctx context.Context, _ []float32) (batcher.Prediction, error) {
		_, hadDeadline = ctx.Deadline()
		<-ctx.Done()
		return batcher.Prediction{}, ctx.Err()
	}}
	start := time.Now()
	rec := post(newServer(t, sub, opts), validBody)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504", rec.Code)
	}
	if !hadDeadline || time.Since(start) > time.Second {
		t.Fatalf("deadline not applied (hadDeadline=%v, took %v)", hadDeadline, time.Since(start))
	}
}

func TestClientDisconnectCancelsSubmitterContextAndIsRecorded(t *testing.T) {
	rc := &recorder{}
	opts := testOptions()
	opts.Recorder = rc.record
	started := make(chan struct{})
	sawCancel := make(chan struct{})
	sub := &fakeSubmitter{fn: func(ctx context.Context, _ []float32) (batcher.Prediction, error) {
		close(started)
		<-ctx.Done()
		close(sawCancel)
		return batcher.Prediction{}, ctx.Err()
	}}
	srv := newServer(t, sub, opts)

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(validBody)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	done := make(chan struct{})
	go func() { do(srv, req); close(done) }()
	<-started
	cancel()
	select {
	case <-sawCancel:
	case <-time.After(time.Second):
		t.Fatal("submitter context was not cancelled when the client went away")
	}
	<-done
	recs := rc.all()
	if len(recs) != 1 || recs[0].status != StatusClientClosedRequest {
		t.Fatalf("recorded %+v, want one 499", recs)
	}
}

func TestPanicInHandlerPathReturns500AndServerKeepsWorking(t *testing.T) {
	var first sync.Once
	sub := &fakeSubmitter{fn: func(context.Context, []float32) (batcher.Prediction, error) {
		first.Do(func() { panic("boom") })
		return batcher.Prediction{Class: 1, Confidence: 1}, nil
	}}
	srv := newServer(t, sub, testOptions())
	if rec := post(srv, validBody); rec.Code != http.StatusInternalServerError {
		t.Fatalf("status %d, want 500", rec.Code)
	}
	if rec := post(srv, validBody); rec.Code != http.StatusOK {
		t.Fatalf("server did not recover, status %d", rec.Code)
	}
}

// ---- health / readiness ----------------------------------------------------

func TestHealthAlwaysOKButReadinessFailsOnceDraining(t *testing.T) {
	srv := newServer(t, &fakeSubmitter{}, testOptions())
	get := func(path string) int {
		return do(srv, httptest.NewRequest(http.MethodGet, path, nil)).Code
	}
	if get("/healthz") != 200 || get("/readyz") != 200 {
		t.Fatal("expected healthy and ready at start")
	}
	srv.SetDraining()
	if get("/healthz") != 200 {
		t.Fatal("liveness must stay OK while draining (otherwise the orchestrator kills the pod mid-drain)")
	}
	if get("/readyz") != http.StatusServiceUnavailable {
		t.Fatal("readiness must fail once draining so the load balancer stops routing here")
	}
}

// ---- observability ---------------------------------------------------------

func TestEveryOutcomeIsRecordedWithItsStatus(t *testing.T) {
	rc := &recorder{}
	opts := testOptions()
	opts.Recorder = rc.record
	srv := newServer(t, &fakeSubmitter{}, opts)
	post(srv, validBody)
	post(srv, `bad`)
	do(srv, httptest.NewRequest(http.MethodGet, "/predict", nil))

	recs := rc.all()
	if len(recs) != 3 || recs[0].status != 200 || recs[1].status != 400 || recs[2].status != 405 {
		t.Fatalf("recorded %+v", recs)
	}
	for _, r := range recs {
		if r.dur < 0 {
			t.Fatalf("negative duration %+v", r)
		}
	}
}

// ---- with the real batcher: backpressure reaches the HTTP layer ------------

type blockingEngine struct {
	entered chan struct{}
	release chan struct{}
}

func (e *blockingEngine) PredictBatch(_ context.Context, images [][]float32) ([]batcher.Prediction, error) {
	e.entered <- struct{}{}
	<-e.release
	return make([]batcher.Prediction, len(images)), nil
}

func TestSaturatedBatcherSurfacesAs429OverHTTP(t *testing.T) {
	eng := &blockingEngine{entered: make(chan struct{}, 8), release: make(chan struct{})}
	b, err := batcher.New(eng, batcher.Config{
		MaxBatch: 1, MaxWait: time.Millisecond, QueueSize: 1, EngineTimeout: 5 * time.Second, MaxInFlight: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = b.Shutdown(ctx)
	}()
	srv := newServer(t, b, testOptions())

	var wg sync.WaitGroup
	launch := func() {
		wg.Add(1)
		go func() { defer wg.Done(); post(srv, validBody) }()
	}
	wait := func(what string, cond func() bool) {
		t.Helper()
		for i := 0; i < 2000 && !cond(); i++ {
			time.Sleep(time.Millisecond)
		}
		if !cond() {
			t.Fatalf("timed out: %s", what)
		}
	}
	launch()
	<-eng.entered
	launch()
	wait("second held by collector", func() bool { s := b.Stats(); return s.Submitted == 2 && s.QueueDepth == 0 })
	launch()
	wait("third queued", func() bool { s := b.Stats(); return s.Submitted == 3 && s.QueueDepth == 1 })

	rec := post(srv, validBody)
	if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") == "" {
		t.Fatalf("status %d, Retry-After %q; want 429 with Retry-After", rec.Code, rec.Header().Get("Retry-After"))
	}
	close(eng.release)
	wg.Wait()
}

// ---- configuration -------------------------------------------------------

func TestNewRejectsInvalidOptions(t *testing.T) {
	bad := []func(*Options){
		func(o *Options) { o.PixelCount = 0 },
		func(o *Options) { o.MaxBodyBytes = 0 },
		func(o *Options) { o.RequestTimeout = 0 },
	}
	for i, m := range bad {
		o := testOptions()
		m(&o)
		if _, err := New(&fakeSubmitter{}, o); err == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
	if _, err := New(nil, testOptions()); err == nil {
		t.Error("expected error for nil submitter")
	}
}
