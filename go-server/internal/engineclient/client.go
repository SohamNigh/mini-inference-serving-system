// Package engineclient is the Go side of the gRPC link to the C++ inference
// engine. It satisfies batcher.EngineClient.
package engineclient

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"mini-inference-serving-system/go-server/internal/batcher"
	"mini-inference-serving-system/go-server/internal/pb"
)

// DefaultMaxImagesPerCall matches the C++ engine's hard limit of 8 images per request.
const DefaultMaxImagesPerCall = 8

var (
	// ErrInvalidRequest means the engine rejected the input (wrong shape or batch size).
	// It signals a bug or bad input on our side, so retrying will not help.
	ErrInvalidRequest = errors.New("engineclient: engine rejected request")
	// ErrUnavailable means the engine could not be reached. Calls fail fast rather
	// than queueing behind a reconnect, so the batcher can surface the outage immediately.
	ErrUnavailable = errors.New("engineclient: engine unavailable")
	// ErrBadResponse means the engine's reply did not match the request.
	ErrBadResponse = errors.New("engineclient: malformed engine response")
)

type Client struct {
	conn       *grpc.ClientConn
	rpc        pb.InferenceEngineClient
	maxPerCall int
}

// Dial creates a client for the engine at target (e.g. "127.0.0.1:50051").
// Connecting is lazy; use Close to release the connection.
func Dial(target string) (*Client, error) {
	conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return NewFromConn(conn, DefaultMaxImagesPerCall), nil
}

// NewFromConn wraps an existing connection. Batches larger than maxPerCall are
// split into sequential calls.
func NewFromConn(conn *grpc.ClientConn, maxPerCall int) *Client {
	if maxPerCall < 1 {
		maxPerCall = DefaultMaxImagesPerCall
	}
	return &Client{conn: conn, rpc: pb.NewInferenceEngineClient(conn), maxPerCall: maxPerCall}
}

func (c *Client) Close() error { return c.conn.Close() }

// PredictBatch returns one prediction per image, in order. Batches over the
// engine limit are split; if any chunk fails the whole call fails and no
// partial results are returned.
func (c *Client) PredictBatch(ctx context.Context, images [][]float32) ([]batcher.Prediction, error) {
	if len(images) == 0 {
		return nil, fmt.Errorf("%w: empty batch", ErrInvalidRequest)
	}
	out := make([]batcher.Prediction, 0, len(images))
	for start := 0; start < len(images); start += c.maxPerCall {
		end := min(start+c.maxPerCall, len(images))
		preds, err := c.callOnce(ctx, images[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, preds...)
	}
	return out, nil
}

func (c *Client) callOnce(ctx context.Context, images [][]float32) ([]batcher.Prediction, error) {
	req := &pb.PredictBatchRequest{Image: make([]*pb.Image, len(images))}
	for i, px := range images {
		req.Image[i] = &pb.Image{Pixels: px}
	}

	resp, err := c.rpc.PredictBatch(ctx, req)
	if err != nil {
		return nil, mapError(ctx, err)
	}
	if len(resp.GetPredictions()) != len(images) {
		return nil, fmt.Errorf("%w: sent %d images, got %d predictions",
			ErrBadResponse, len(images), len(resp.GetPredictions()))
	}
	out := make([]batcher.Prediction, len(images))
	for i, p := range resp.GetPredictions() {
		out[i] = batcher.Prediction{Class: p.GetPredictedClass(), Confidence: p.GetConfidence()}
	}
	return out, nil
}

// mapError turns gRPC status errors into errors callers can test with errors.Is.
// The caller's own context wins: if it ended, report that, not a transport detail.
func mapError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	switch status.Code(err) {
	case codes.InvalidArgument:
		return fmt.Errorf("%w: %s", ErrInvalidRequest, status.Convert(err).Message())
	case codes.Unavailable:
		return fmt.Errorf("%w: %s", ErrUnavailable, status.Convert(err).Message())
	case codes.DeadlineExceeded:
		return fmt.Errorf("engine: %w", context.DeadlineExceeded)
	case codes.Canceled:
		return fmt.Errorf("engine: %w", context.Canceled)
	default:
		return fmt.Errorf("engine: %w", err)
	}
}
