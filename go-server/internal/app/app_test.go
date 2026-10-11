package app

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	"mini-inference-serving-system/go-server/internal/pb"
)

// ---- fake engine over real TCP ---------------------------------------------

type fakeEngine struct {
	pb.UnimplementedInferenceEngineServer
	entered chan struct{} // signalled on every call when non-nil
	release chan struct{} // calls block until closed when non-nil
}

func (f *fakeEngine) PredictBatch(ctx context.Context, req *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
	if f.entered != nil {
		f.entered <- struct{}{}
	}
	if f.release != nil {
		select {
		case <-f.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	resp := &pb.PredictBatchResponse{}
	for _, img := range req.GetImage() {
		resp.Predictions = append(resp.Predictions,
			&pb.Prediction{PredictedClass: int32(img.GetPixels()[0]*10 + 0.5), Confidence: 0.9})
	}
	return resp, nil
}

func startEngine(t *testing.T, eng *fakeEngine) (addr string, stop func()) {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterInferenceEngineServer(srv, eng)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return lis.Addr().String(), srv.Stop
}

// ---- app harness -------------------------------------------------------------

func testConfig(engineAddr string) Config {
	return Config{
		ListenAddr:      "127.0.0.1:0",
		EngineAddr:      engineAddr,
		PixelCount:      4,
		MaxBatch:        8,
		MaxWait:         20 * time.Millisecond,
		QueueSize:       256,
		EngineTimeout:   3 * time.Second,
		MaxInFlight:     2,
		RequestTimeout:  3 * time.Second,
		MaxBodyBytes:    1 << 12,
		ShutdownTimeout: 2 * time.Second,
		DrainDelay:      0,
	}
}

type running struct {
	addr   string
	cancel context.CancelFunc
	fin    chan struct{} // closed when Run has returned
	err    error         // Run's result; valid after fin is closed
}

func start(t *testing.T, cfg Config) *running {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	addrCh := make(chan string, 1)
	r := &running{cancel: cancel, fin: make(chan struct{})}
	go func() {
		r.err = Run(ctx, cfg, func(a string) { addrCh <- a })
		close(r.fin)
	}()
	select {
	case a := <-addrCh:
		r.addr = a
		t.Cleanup(func() {
			cancel()
			select {
			case <-r.fin:
			case <-time.After(5 * time.Second):
			}
		})
		return r
	case <-r.fin:
		cancel()
		t.Fatalf("Run exited before ready: %v", r.err)
	case <-time.After(3 * time.Second):
		cancel()
		t.Fatal("server did not become ready")
	}
	return nil
}

func (r *running) url(path string) string { return "http://" + r.addr + path }

func (r *running) predict(val float32) (int, map[string]any, error) {
	body := fmt.Sprintf(`{"pixels":[%v,0,0,0]}`, val)
	resp, err := http.Post(r.url("/predict"), "application/json", bytes.NewBufferString(body))
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out, nil
}

func get(t *testing.T, url string) (int, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		return 0, err.Error()
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func metricValue(t *testing.T, body, name string) float64 {
	t.Helper()
	re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(name) + ` ([0-9.e+-]+)$`)
	m := re.FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("metric %s not found", name)
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

// ---- config ----------------------------------------------------------------------

func TestConfigValidation(t *testing.T) {
	bad := map[string]func(*Config){
		"no listen addr":       func(c *Config) { c.ListenAddr = "" },
		"no engine addr":       func(c *Config) { c.EngineAddr = "" },
		"zero pixels":          func(c *Config) { c.PixelCount = 0 },
		"zero batch":           func(c *Config) { c.MaxBatch = 0 },
		"zero max wait":        func(c *Config) { c.MaxWait = 0 },
		"zero queue":           func(c *Config) { c.QueueSize = 0 },
		"zero engine timeout":  func(c *Config) { c.EngineTimeout = 0 },
		"zero in flight":       func(c *Config) { c.MaxInFlight = 0 },
		"zero request timeout": func(c *Config) { c.RequestTimeout = 0 },
		"zero body limit":      func(c *Config) { c.MaxBodyBytes = 0 },
		"zero shutdown":        func(c *Config) { c.ShutdownTimeout = 0 },
		"negative drain":       func(c *Config) { c.DrainDelay = -1 },
		// The drain delay is spent out of the shutdown budget; equal or larger leaves no time to drain.
		"drain exceeds shutdown budget": func(c *Config) { c.DrainDelay = c.ShutdownTimeout },
		// A request budget shorter than the batching window can never succeed.
		"timeout below batching window": func(c *Config) { c.RequestTimeout = c.MaxWait },
	}
	for name, mutate := range bad {
		c := testConfig("127.0.0.1:1")
		mutate(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
	if err := testConfig("127.0.0.1:1").Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

// ---- end to end ------------------------------------------------------------------

func TestEndToEndRequestMetricsAndHealth(t *testing.T) {
	addr, _ := startEngine(t, &fakeEngine{})
	r := start(t, testConfig(addr))

	code, body, err := r.predict(0.3)
	if err != nil || code != 200 || body["class"] != float64(3) {
		t.Fatalf("predict: code=%d body=%v err=%v", code, body, err)
	}
	if c, _ := get(t, r.url("/healthz")); c != 200 {
		t.Fatalf("healthz %d", c)
	}
	if c, _ := get(t, r.url("/readyz")); c != 200 {
		t.Fatalf("readyz %d", c)
	}
	c, metrics := get(t, r.url("/metrics"))
	if c != 200 {
		t.Fatalf("metrics %d", c)
	}
	if v := metricValue(t, metrics, `inference_http_requests_total{code="200"}`); v < 1 {
		t.Fatalf("http counter %v", v)
	}
	if v := metricValue(t, metrics, "inference_batcher_items_total"); v != 1 {
		t.Fatalf("items %v, want 1", v)
	}
}

func TestConcurrentRequestsAreActuallyBatchedAndRoutedCorrectly(t *testing.T) {
	addr, _ := startEngine(t, &fakeEngine{})
	cfg := testConfig(addr)
	cfg.MaxWait = 50 * time.Millisecond
	r := start(t, cfg)

	const n = 64
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			want := i % 10
			code, body, err := r.predict(float32(want) / 10)
			if err != nil || code != 200 || body["class"] != float64(want) {
				t.Errorf("req %d: code=%d body=%v err=%v", i, code, body, err)
			}
		}()
	}
	wg.Wait()

	_, metrics := get(t, r.url("/metrics"))
	items := metricValue(t, metrics, "inference_batcher_items_total")
	batches := metricValue(t, metrics, "inference_batcher_batches_total")
	if items != n || batches >= n/2 {
		t.Fatalf("items=%v batches=%v: expected %d items in far fewer than %d batches", items, batches, n, n/2)
	}
}

func TestEngineOutageSurfacesAs503Quickly(t *testing.T) {
	addr, stop := startEngine(t, &fakeEngine{})
	r := start(t, testConfig(addr))
	if code, _, _ := r.predict(0.1); code != 200 {
		t.Fatalf("warmup failed: %d", code)
	}
	stop()

	begin := time.Now()
	code, _, err := r.predict(0.1)
	if err != nil || code != http.StatusServiceUnavailable {
		t.Fatalf("code=%d err=%v, want 503", code, err)
	}
	if time.Since(begin) > time.Second {
		t.Fatalf("outage took %v to report", time.Since(begin))
	}
	_, metrics := get(t, r.url("/metrics"))
	if v := metricValue(t, metrics, "inference_batcher_engine_errors_total"); v < 1 {
		t.Fatalf("engine errors not counted: %v", v)
	}
}

// ---- graceful shutdown -------------------------------------------------------------

func TestGracefulShutdownDrainsInFlightRequestsAndFlipsReadiness(t *testing.T) {
	eng := &fakeEngine{entered: make(chan struct{}, 8), release: make(chan struct{})}
	addr, _ := startEngine(t, eng)
	cfg := testConfig(addr)
	cfg.DrainDelay = 400 * time.Millisecond
	r := start(t, cfg)

	type res struct {
		code int
		body map[string]any
		err  error
	}
	inflight := make(chan res, 1)
	go func() {
		c, b, e := r.predict(0.7)
		inflight <- res{c, b, e}
	}()
	<-eng.entered // the request is now inside the engine

	r.cancel() // SIGTERM equivalent

	// During the drain delay the listener is still open: readiness must report 503 ...
	deadline := time.Now().Add(300 * time.Millisecond)
	flipped := false
	for time.Now().Before(deadline) {
		if c, _ := get(t, r.url("/readyz")); c == http.StatusServiceUnavailable {
			flipped = true
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !flipped {
		t.Fatal("/readyz did not report 503 during the drain window")
	}
	// ... while liveness stays green so the orchestrator doesn't kill us mid-drain.
	if c, _ := get(t, r.url("/healthz")); c != 200 {
		t.Fatalf("healthz during drain = %d", c)
	}

	select {
	case <-r.fin:
		t.Fatalf("Run returned (%v) while a request was still in flight", r.err)
	case <-time.After(600 * time.Millisecond):
	}

	close(eng.release) // engine finishes
	got := <-inflight
	if got.err != nil || got.code != 200 || got.body["class"] != float64(7) {
		t.Fatalf("in-flight request was not completed during shutdown: %+v", got)
	}
	select {
	case <-r.fin:
		if r.err != nil {
			t.Fatalf("clean shutdown returned error: %v", r.err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Run did not return after in-flight work finished")
	}
	if _, err := http.Get(r.url("/healthz")); err == nil {
		t.Fatal("server still accepting connections after shutdown")
	}
}

func TestShutdownTimeoutIsEnforcedWhenEngineIsStuck(t *testing.T) {
	eng := &fakeEngine{entered: make(chan struct{}, 8), release: make(chan struct{})}
	addr, _ := startEngine(t, eng)
	cfg := testConfig(addr)
	cfg.ShutdownTimeout = 150 * time.Millisecond
	cfg.EngineTimeout = 30 * time.Second
	cfg.RequestTimeout = 30 * time.Second
	r := start(t, cfg)
	defer close(eng.release)

	go func() { _, _, _ = r.predict(0.1) }()
	<-eng.entered

	begin := time.Now()
	r.cancel()
	select {
	case <-r.fin:
		if r.err == nil {
			t.Fatal("expected an error reporting that shutdown timed out")
		}
		if time.Since(begin) > 2*time.Second {
			t.Fatalf("shutdown took %v despite a 150ms limit", time.Since(begin))
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run hung past ShutdownTimeout")
	}
}

func TestRunFailsFastIfListenAddressIsTaken(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	addr, _ := startEngine(t, &fakeEngine{})
	cfg := testConfig(addr)
	cfg.ListenAddr = l.Addr().String()

	done := make(chan error, 1)
	go func() { done <- Run(context.Background(), cfg, nil) }()
	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "listen") {
			t.Fatalf("want a listen error, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not fail on a taken port")
	}
}
