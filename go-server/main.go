package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"mini-inference-serving-system/go-server/internal/app"
)

func main() {
	var cfg app.Config
	flag.StringVar(&cfg.ListenAddr, "listen", ":8080", "HTTP listen address")
	flag.StringVar(&cfg.EngineAddr, "engine", "127.0.0.1:50051", "C++ engine gRPC address")
	flag.IntVar(&cfg.PixelCount, "pixels", 784, "required input length")
	flag.IntVar(&cfg.MaxBatch, "max-batch", 8, "max requests per engine call (the engine's limit is 8)")
	flag.DurationVar(&cfg.MaxWait, "max-wait", 5*time.Millisecond, "max time a batch waits to fill")
	flag.IntVar(&cfg.QueueSize, "queue-size", 256, "bounded queue capacity; full queue returns 429")
	flag.DurationVar(&cfg.EngineTimeout, "engine-timeout", 2*time.Second, "cap on a single engine call")
	flag.IntVar(&cfg.MaxInFlight, "max-in-flight", 2, "concurrent engine calls")
	flag.DurationVar(&cfg.RequestTimeout, "request-timeout", time.Second, "end-to-end budget per request")
	flag.Int64Var(&cfg.MaxBodyBytes, "max-body-bytes", 64<<10, "max request body size")
	flag.DurationVar(&cfg.ShutdownTimeout, "shutdown-timeout", 15*time.Second, "total graceful shutdown budget")
	flag.DurationVar(&cfg.DrainDelay, "drain-delay", 2*time.Second, "time /readyz reports 503 before the listener closes")
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	cfg.Logger = log

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := app.Run(ctx, cfg, nil); err != nil {
		log.Error("server exited with error", "err", err)
		os.Exit(1)
	}
}
