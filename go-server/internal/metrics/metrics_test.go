package metrics

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"mini-inference-serving-system/go-server/internal/batcher"
)

// ---- helpers ---------------------------------------------------------------

func lookup(t *testing.T, m *Metrics, name string) *dto.MetricFamily {
	t.Helper()
	fams, err := m.registry.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range fams {
		if f.GetName() == name {
			return f
		}
	}
	return nil
}

func family(t *testing.T, m *Metrics, name string) *dto.MetricFamily {
	t.Helper()
	f := lookup(t, m, name)
	if f == nil {
		t.Fatalf("metric %q not exposed", name)
	}
	return f
}

func counterWithLabel(t *testing.T, m *Metrics, name, label, value string) float64 {
	t.Helper()
	f := lookup(t, m, name)
	if f == nil {
		return 0 // a labeled series that was never touched is simply absent
	}
	for _, metric := range f.GetMetric() {
		for _, l := range metric.GetLabel() {
			if l.GetName() == label && l.GetValue() == value {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func scalar(t *testing.T, m *Metrics, name string) float64 {
	t.Helper()
	f := family(t, m, name)
	if len(f.GetMetric()) != 1 {
		t.Fatalf("%s: want exactly one series, got %d", name, len(f.GetMetric()))
	}
	met := f.GetMetric()[0]
	switch f.GetType() {
	case dto.MetricType_COUNTER:
		return met.GetCounter().GetValue()
	case dto.MetricType_GAUGE:
		return met.GetGauge().GetValue()
	}
	t.Fatalf("%s: unexpected type %v", name, f.GetType())
	return 0
}

func histogram(t *testing.T, m *Metrics, name string) *dto.Histogram {
	t.Helper()
	f := family(t, m, name)
	if len(f.GetMetric()) != 1 {
		t.Fatalf("%s: want one series", name)
	}
	return f.GetMetric()[0].GetHistogram()
}

// ---- HTTP ------------------------------------------------------------------

func TestRecordHTTPCountsByStatusCodeAndObservesLatency(t *testing.T) {
	m := New()
	m.RecordHTTP(200, 5*time.Millisecond)
	m.RecordHTTP(200, 7*time.Millisecond)
	m.RecordHTTP(429, time.Millisecond)

	if got := counterWithLabel(t, m, "inference_http_requests_total", "code", "200"); got != 2 {
		t.Fatalf("200 count = %v, want 2", got)
	}
	if got := counterWithLabel(t, m, "inference_http_requests_total", "code", "429"); got != 1 {
		t.Fatalf("429 count = %v, want 1", got)
	}
	h := histogram(t, m, "inference_http_request_duration_seconds")
	if h.GetSampleCount() != 3 {
		t.Fatalf("latency samples = %d, want 3", h.GetSampleCount())
	}
}

// ---- batcher observer ------------------------------------------------------

func TestObserveBatchRecordsSizeAndEngineLatency(t *testing.T) {
	m := New()
	m.ObserveBatch(4, 10*time.Millisecond, nil)
	m.ObserveBatch(8, 30*time.Millisecond, errors.New("boom"))

	size := histogram(t, m, "inference_batch_size")
	if size.GetSampleCount() != 2 || size.GetSampleSum() != 12 {
		t.Fatalf("batch size histogram count=%d sum=%v", size.GetSampleCount(), size.GetSampleSum())
	}
	lat := histogram(t, m, "inference_engine_latency_seconds")
	if lat.GetSampleCount() != 2 {
		t.Fatalf("engine latency samples = %d", lat.GetSampleCount())
	}
}

func TestObserveQueueWaitRecordsSeconds(t *testing.T) {
	m := New()
	m.ObserveQueueWait(250 * time.Millisecond)
	h := histogram(t, m, "inference_queue_wait_seconds")
	if h.GetSampleCount() != 1 || h.GetSampleSum() < 0.249 || h.GetSampleSum() > 0.251 {
		t.Fatalf("queue wait count=%d sum=%v", h.GetSampleCount(), h.GetSampleSum())
	}
}

func TestBatchSizeBucketsResolveSmallBatchesIndividually(t *testing.T) {
	// The engine limit is 8, so every size 1..8 needs its own bucket or the
	// histogram can't tell "batching works" from "everything is size 1-2".
	m := New()
	m.ObserveBatch(1, time.Millisecond, nil)
	var bounds []float64
	for _, b := range histogram(t, m, "inference_batch_size").GetBucket() {
		bounds = append(bounds, b.GetUpperBound())
	}
	for want := 1.0; want <= 8; want++ {
		found := false
		for _, b := range bounds {
			if b == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("no bucket boundary at %v (have %v)", want, bounds)
		}
	}
}

// ---- batcher stats ---------------------------------------------------------

type okClient struct{}

func (okClient) PredictBatch(_ context.Context, images [][]float32) ([]batcher.Prediction, error) {
	return make([]batcher.Prediction, len(images)), nil
}

func TestRegisterBatcherExposesLiveStatsAtScrapeTime(t *testing.T) {
	m := New()
	b, err := batcher.New(okClient{}, batcher.Config{
		MaxBatch: 2, MaxWait: time.Millisecond, QueueSize: 8, EngineTimeout: time.Second, MaxInFlight: 1, Observer: m,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Shutdown(context.Background()) }()
	if err := m.RegisterBatcher(b); err != nil {
		t.Fatalf("RegisterBatcher: %v", err)
	}

	if got := scalar(t, m, "inference_batcher_submitted_total"); got != 0 {
		t.Fatalf("submitted before traffic = %v", got)
	}
	for i := 0; i < 3; i++ {
		if _, err := b.Submit(context.Background(), []float32{1}); err != nil {
			t.Fatal(err)
		}
	}
	if got := scalar(t, m, "inference_batcher_submitted_total"); got != 3 {
		t.Fatalf("submitted = %v, want 3 (counters must be read at scrape time, not at registration)", got)
	}
	if got := scalar(t, m, "inference_batcher_items_total"); got != 3 {
		t.Fatalf("items = %v, want 3", got)
	}
	// gauge exists and is readable
	_ = scalar(t, m, "inference_batcher_queue_depth")
	for _, name := range []string{
		"inference_batcher_rejected_total", "inference_batcher_expired_total",
		"inference_batcher_batches_total", "inference_batcher_engine_errors_total",
	} {
		_ = scalar(t, m, name)
	}
}

func TestRegisterBatcherTwiceReturnsErrorInsteadOfPanicking(t *testing.T) {
	m := New()
	b, _ := batcher.New(okClient{}, batcher.Config{
		MaxBatch: 1, MaxWait: time.Millisecond, QueueSize: 1, EngineTimeout: time.Second, MaxInFlight: 1,
	})
	defer func() { _ = b.Shutdown(context.Background()) }()
	if err := m.RegisterBatcher(b); err != nil {
		t.Fatal(err)
	}
	if err := m.RegisterBatcher(b); err == nil {
		t.Fatal("expected an error on duplicate registration")
	}
}

// ---- exposition ------------------------------------------------------------

func TestHandlerServesPrometheusTextFormat(t *testing.T) {
	m := New()
	m.RecordHTTP(200, time.Millisecond)
	m.ObserveBatch(3, time.Millisecond, nil)

	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{
		`inference_http_requests_total{code="200"} 1`,
		"inference_batch_size_bucket",
		"# TYPE inference_engine_latency_seconds histogram",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("exposition missing %q", want)
		}
	}
}

func TestSeparateInstancesDoNotShareGlobalRegistry(t *testing.T) {
	a, b := New(), New() // would panic on duplicate registration if a global registry were used
	a.RecordHTTP(200, time.Millisecond)
	if got := counterWithLabel(t, b, "inference_http_requests_total", "code", "200"); got != 0 {
		t.Fatalf("instances share state: %v", got)
	}
}
