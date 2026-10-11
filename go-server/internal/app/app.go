// Package app wires the batcher, gRPC client, HTTP layer and metrics into a
// runnable server with an ordered graceful shutdown.
package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	"mini-inference-serving-system/go-server/internal/batcher"
	"mini-inference-serving-system/go-server/internal/engineclient"
	"mini-inference-serving-system/go-server/internal/httpapi"
	"mini-inference-serving-system/go-server/internal/metrics"
)

type Config struct {
	ListenAddr string // HTTP listen address, e.g. ":8080" (":0" picks a free port)
	EngineAddr string // C++ engine gRPC address
	PixelCount int    // required input length

	MaxBatch      int
	MaxWait       time.Duration
	QueueSize     int
	EngineTimeout time.Duration
	MaxInFlight   int

	RequestTimeout time.Duration
	MaxBodyBytes   int64

	ShutdownTimeout time.Duration // total budget for the whole shutdown sequence
	DrainDelay      time.Duration // time /readyz reports 503 before the listener closes

	Logger *slog.Logger // optional
}

func (c Config) Validate() error {
	switch {
	case c.ListenAddr == "":
		return errors.New("ListenAddr is required")
	case c.EngineAddr == "":
		return errors.New("EngineAddr is required")
	case c.PixelCount < 1:
		return errors.New("PixelCount must be >= 1")
	case c.MaxBatch < 1:
		return errors.New("MaxBatch must be >= 1")
	case c.MaxWait <= 0:
		return errors.New("MaxWait must be > 0")
	case c.QueueSize < 1:
		return errors.New("QueueSize must be >= 1")
	case c.EngineTimeout <= 0:
		return errors.New("EngineTimeout must be > 0")
	case c.MaxInFlight < 1:
		return errors.New("MaxInFlight must be >= 1")
	case c.RequestTimeout <= 0:
		return errors.New("RequestTimeout must be > 0")
	case c.MaxBodyBytes < 1:
		return errors.New("MaxBodyBytes must be >= 1")
	case c.ShutdownTimeout <= 0:
		return errors.New("ShutdownTimeout must be > 0")
	case c.DrainDelay < 0:
		return errors.New("DrainDelay must be >= 0")
	case c.RequestTimeout <= c.MaxWait:
		// Every request waits up to MaxWait for its batch; a smaller budget makes timeouts the norm.
		return errors.New("RequestTimeout must exceed MaxWait")
	case c.DrainDelay >= c.ShutdownTimeout:
		return errors.New("DrainDelay must be less than ShutdownTimeout")
	}
	return nil
}

// Run serves until ctx is cancelled, then shuts down in dependency order:
//
//  1. mark not-ready (load balancers stop routing here), wait DrainDelay
//  2. stop the HTTP server, letting in-flight requests finish
//  3. shut down the batcher (nothing can enqueue anymore, so its queue drains)
//  4. close the engine connection
//
// The whole sequence shares ShutdownTimeout. ready, if non-nil, is called with
// the bound listen address once the server is accepting connections.
func Run(ctx context.Context, cfg Config, ready func(addr string)) error {
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}
	log := cfg.Logger
	if log == nil {
		log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}

	lis, err := net.Listen("tcp", cfg.ListenAddr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}

	client, err := engineclient.Dial(cfg.EngineAddr)
	if err != nil {
		_ = lis.Close()
		return fmt.Errorf("engine client: %w", err)
	}
	m := metrics.New()
	b, err := batcher.New(client, batcher.Config{
		MaxBatch: cfg.MaxBatch, MaxWait: cfg.MaxWait, QueueSize: cfg.QueueSize,
		EngineTimeout: cfg.EngineTimeout, MaxInFlight: cfg.MaxInFlight, Observer: m,
	})
	if err != nil {
		_ = lis.Close()
		_ = client.Close()
		return fmt.Errorf("batcher: %w", err)
	}
	if err := m.RegisterBatcher(b); err != nil {
		_ = lis.Close()
		_ = client.Close()
		return fmt.Errorf("metrics: %w", err)
	}
	api, err := httpapi.New(b, httpapi.Options{
		PixelCount: cfg.PixelCount, MaxBodyBytes: cfg.MaxBodyBytes, RequestTimeout: cfg.RequestTimeout,
		Logger: log, Recorder: m.RecordHTTP,
	})
	if err != nil {
		_ = lis.Close()
		_ = client.Close()
		return fmt.Errorf("http api: %w", err)
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", m.Handler())
	mux.Handle("/", api.Handler())
	srv := &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second, // slowloris protection
		IdleTimeout:       60 * time.Second,
	}

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(lis) }()
	log.Info("listening", "addr", lis.Addr().String(), "engine", cfg.EngineAddr)
	if ready != nil {
		ready(lis.Addr().String())
	}

	select {
	case err := <-serveErr:
		// The listener died on its own; still release everything we own.
		shutdownBackend(b, client, cfg.ShutdownTimeout)
		return fmt.Errorf("serve: %w", err)
	case <-ctx.Done():
	}

	// A fresh context: ctx is already cancelled, and shutdown needs its own budget.
	shutCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	log.Info("shutting down", "drain_delay", cfg.DrainDelay)

	var errs []error
	api.SetDraining()
	select {
	case <-time.After(cfg.DrainDelay):
	case <-shutCtx.Done():
	}
	if err := srv.Shutdown(shutCtx); err != nil {
		errs = append(errs, fmt.Errorf("http shutdown: %w", err))
		_ = srv.Close() // give up on stragglers; their connections are cut
	}
	if err := b.Shutdown(shutCtx); err != nil {
		errs = append(errs, fmt.Errorf("batcher shutdown: %w", err))
	}
	if err := client.Close(); err != nil {
		errs = append(errs, fmt.Errorf("engine client close: %w", err))
	}
	if err := <-serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
		errs = append(errs, fmt.Errorf("serve: %w", err))
	}
	log.Info("stopped", "clean", len(errs) == 0)
	return errors.Join(errs...)
}

func shutdownBackend(b *batcher.Batcher, c *engineclient.Client, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	_ = b.Shutdown(ctx)
	_ = c.Close()
}
