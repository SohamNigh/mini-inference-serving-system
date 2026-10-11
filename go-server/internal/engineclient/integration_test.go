//go:build integration

package engineclient

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

// Run against a live C++ engine:
//
//	ENGINE_ADDR=127.0.0.1:50051 go test -tags=integration ./internal/engineclient/
func realClient(t *testing.T) *Client {
	t.Helper()
	addr := os.Getenv("ENGINE_ADDR")
	if addr == "" {
		t.Skip("ENGINE_ADDR not set")
	}
	c, err := Dial(addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func image784() []float32 {
	px := make([]float32, 784)
	for i := range px {
		px[i] = float32(i%7) / 7
	}
	return px
}

func TestRealEngineBatchAndSplit(t *testing.T) {
	c := realClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 20 images exceeds the engine's limit of 8, so the client must split.
	batch := make([][]float32, 20)
	for i := range batch {
		batch[i] = image784()
	}
	preds, err := c.PredictBatch(ctx, batch)
	if err != nil {
		t.Fatalf("PredictBatch: %v", err)
	}
	if len(preds) != 20 {
		t.Fatalf("got %d predictions, want 20", len(preds))
	}
	for i, p := range preds {
		if p.Class < 0 || p.Class > 9 || p.Confidence <= 0 || p.Confidence > 1 {
			t.Fatalf("prediction %d out of range: %+v", i, p)
		}
		if p != preds[0] {
			t.Fatalf("identical inputs gave different outputs: %+v vs %+v", p, preds[0])
		}
	}
}

func TestRealEngineRejectsWrongDimension(t *testing.T) {
	c := realClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := c.PredictBatch(ctx, [][]float32{{1, 2, 3}}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("want ErrInvalidRequest, got %v", err)
	}
}
