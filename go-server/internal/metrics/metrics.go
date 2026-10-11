// Package metrics exposes Prometheus metrics for the serving path. It uses a
// private registry (never the global one) so tests and multiple instances
// cannot collide.
package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"mini-inference-serving-system/go-server/internal/batcher"
)

type Metrics struct {
	registry *prometheus.Registry

	httpRequests *prometheus.CounterVec
	httpLatency  prometheus.Histogram
	batchSize    prometheus.Histogram
	engineLat    prometheus.Histogram
	queueWait    prometheus.Histogram
}

// Latency buckets from 0.5ms to ~5s: wide enough to show both the batching
// window and a hung engine.
var latencyBuckets = []float64{.0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5}

func New() *Metrics {
	m := &Metrics{
		registry: prometheus.NewRegistry(),
		httpRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "inference_http_requests_total",
			Help: "HTTP requests by response status code.",
		}, []string{"code"}),
		httpLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "inference_http_request_duration_seconds", Help: "End-to-end HTTP request latency.",
			Buckets: latencyBuckets,
		}),
		// One bucket per size up to the engine's limit of 8, so the histogram can
		// tell real batching from "everything is a batch of one".
		batchSize: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "inference_batch_size", Help: "Images per engine call.",
			Buckets: []float64{1, 2, 3, 4, 5, 6, 7, 8, 16, 32},
		}),
		engineLat: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "inference_engine_latency_seconds", Help: "Latency of one engine call.",
			Buckets: latencyBuckets,
		}),
		queueWait: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "inference_queue_wait_seconds", Help: "Time a request waited from enqueue until its engine call began.",
			Buckets: latencyBuckets,
		}),
	}
	m.registry.MustRegister(m.httpRequests, m.httpLatency, m.batchSize, m.engineLat, m.queueWait)
	return m
}

// RecordHTTP matches httpapi.Options.Recorder.
func (m *Metrics) RecordHTTP(status int, d time.Duration) {
	m.httpRequests.WithLabelValues(strconv.Itoa(status)).Inc()
	m.httpLatency.Observe(d.Seconds())
}

// ObserveBatch and ObserveQueueWait implement batcher.Observer.
func (m *Metrics) ObserveBatch(size int, engineLatency time.Duration, _ error) {
	m.batchSize.Observe(float64(size))
	m.engineLat.Observe(engineLatency.Seconds())
}

func (m *Metrics) ObserveQueueWait(d time.Duration) { m.queueWait.Observe(d.Seconds()) }

// RegisterBatcher exposes the batcher's counters, read at scrape time so they
// are always current. Returns an error (rather than panicking) if called twice.
func (m *Metrics) RegisterBatcher(b *batcher.Batcher) error {
	counter := func(name, help string, read func(batcher.Stats) int64) prometheus.Collector {
		return prometheus.NewCounterFunc(prometheus.CounterOpts{Name: name, Help: help},
			func() float64 { return float64(read(b.Stats())) })
	}
	collectors := []prometheus.Collector{
		counter("inference_batcher_submitted_total", "Requests accepted into the queue.", func(s batcher.Stats) int64 { return s.Submitted }),
		counter("inference_batcher_rejected_total", "Requests rejected because the queue was full.", func(s batcher.Stats) int64 { return s.Rejected }),
		counter("inference_batcher_expired_total", "Requests dropped because the caller gave up before dispatch.", func(s batcher.Stats) int64 { return s.Expired }),
		counter("inference_batcher_batches_total", "Engine calls made.", func(s batcher.Stats) int64 { return s.Batches }),
		counter("inference_batcher_items_total", "Images sent to the engine.", func(s batcher.Stats) int64 { return s.Items }),
		counter("inference_batcher_engine_errors_total", "Batches that failed.", func(s batcher.Stats) int64 { return s.EngineErrors }),
		prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name: "inference_batcher_queue_depth", Help: "Requests waiting in the queue right now.",
		}, func() float64 { return float64(b.Stats().QueueDepth) }),
	}
	for i, c := range collectors {
		if err := m.registry.Register(c); err != nil {
			for _, done := range collectors[:i] {
				m.registry.Unregister(done) // don't leave a half-registered set behind
			}
			return err
		}
	}
	return nil
}

func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}
