package engineclient

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"mini-inference-serving-system/go-server/internal/batcher"
	"mini-inference-serving-system/go-server/internal/pb"
)

// ---- in-process fake engine ---------------------------------------------

type fakeEngine struct {
	pb.UnimplementedInferenceEngineServer
	mu        sync.Mutex
	callSizes []int
	deadlines []bool
	handler   func(ctx context.Context, req *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error)
}

func (f *fakeEngine) PredictBatch(ctx context.Context, req *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
	_, hasDeadline := ctx.Deadline()
	f.mu.Lock()
	f.callSizes = append(f.callSizes, len(req.GetImage()))
	f.deadlines = append(f.deadlines, hasDeadline)
	f.mu.Unlock()
	if f.handler != nil {
		return f.handler(ctx, req)
	}
	return classFromFirstPixel(req), nil
}

func (f *fakeEngine) sizes() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]int(nil), f.callSizes...)
}

func classFromFirstPixel(req *pb.PredictBatchRequest) *pb.PredictBatchResponse {
	resp := &pb.PredictBatchResponse{}
	for _, img := range req.GetImage() {
		resp.Predictions = append(resp.Predictions,
			&pb.Prediction{PredictedClass: int32(img.GetPixels()[0]), Confidence: 0.5})
	}
	return resp
}

// start returns a client connected to the fake engine over an in-memory listener.
func start(t *testing.T, eng *fakeEngine, maxPerCall int) (*Client, *grpc.Server) {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	pb.RegisterInferenceEngineServer(srv, eng)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	c := NewFromConn(conn, maxPerCall)
	t.Cleanup(func() { _ = c.Close() })
	return c, srv
}

func imgs(vals ...float32) [][]float32 {
	out := make([][]float32, len(vals))
	for i, v := range vals {
		out[i] = []float32{v}
	}
	return out
}

func ctx2s(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// ---- tests ---------------------------------------------------------------

func TestPredictBatchReturnsPredictionsInRequestOrder(t *testing.T) {
	c, _ := start(t, &fakeEngine{}, 8)
	got, err := c.PredictBatch(ctx2s(t), imgs(3, 1, 2))
	if err != nil {
		t.Fatalf("PredictBatch: %v", err)
	}
	if len(got) != 3 || got[0].Class != 3 || got[1].Class != 1 || got[2].Class != 2 || got[0].Confidence != 0.5 {
		t.Fatalf("wrong predictions/order: %+v", got)
	}
}

func TestOversizedBatchIsSplitToRespectEngineLimitAndOrderPreserved(t *testing.T) {
	eng := &fakeEngine{}
	c, _ := start(t, eng, 3)
	got, err := c.PredictBatch(ctx2s(t), imgs(0, 1, 2, 3, 4, 5, 6))
	if err != nil {
		t.Fatalf("PredictBatch: %v", err)
	}
	for i, p := range got {
		if int(p.Class) != i {
			t.Fatalf("position %d got class %d (order lost across chunks)", i, p.Class)
		}
	}
	if s := eng.sizes(); len(s) != 3 || s[0] != 3 || s[1] != 3 || s[2] != 1 {
		t.Fatalf("expected chunks [3 3 1], got %v", s)
	}
}

func TestChunkFailureFailsWholeCallAndStopsSendingMoreChunks(t *testing.T) {
	calls := 0
	eng := &fakeEngine{handler: func(_ context.Context, req *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		calls++
		if calls == 2 {
			return nil, status.Error(codes.Internal, "kernel exploded")
		}
		return classFromFirstPixel(req), nil
	}}
	c, _ := start(t, eng, 2)
	got, err := c.PredictBatch(ctx2s(t), imgs(0, 1, 2, 3, 4, 5))
	if err == nil || got != nil {
		t.Fatalf("want error and no partial results, got %v, %v", got, err)
	}
	if n := len(eng.sizes()); n != 2 {
		t.Fatalf("sent %d chunks; the third should not be sent after the second failed", n)
	}
}

func TestEmptyBatchIsRejectedWithoutAnRPC(t *testing.T) {
	eng := &fakeEngine{}
	c, _ := start(t, eng, 8)
	if _, err := c.PredictBatch(ctx2s(t), nil); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("want ErrInvalidRequest, got %v", err)
	}
	if n := len(eng.sizes()); n != 0 {
		t.Fatalf("made %d RPCs for an empty batch", n)
	}
}

func TestInvalidArgumentMapsToErrInvalidRequest(t *testing.T) {
	eng := &fakeEngine{handler: func(context.Context, *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		return nil, status.Error(codes.InvalidArgument, "wrong input dimension")
	}}
	c, _ := start(t, eng, 8)
	_, err := c.PredictBatch(ctx2s(t), imgs(1))
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("want ErrInvalidRequest, got %v", err)
	}
}

func TestEngineDownFailsFastAsErrUnavailable(t *testing.T) {
	c, srv := start(t, &fakeEngine{}, 8)
	srv.Stop()

	start := time.Now()
	_, err := c.PredictBatch(ctx2s(t), imgs(1))
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("want ErrUnavailable, got %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("took %v to report a dead engine; calls must fail fast, not wait for reconnect", time.Since(start))
	}
}

func TestCallerDeadlineSurfacesAsContextErrorAndIsPropagatedToEngine(t *testing.T) {
	eng := &fakeEngine{handler: func(ctx context.Context, _ *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(5 * time.Second):
			return nil, errors.New("deadline was not propagated")
		}
	}}
	c, _ := start(t, eng, 8)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	begin := time.Now()
	_, err := c.PredictBatch(ctx, imgs(1))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want context.DeadlineExceeded, got %v", err)
	}
	if time.Since(begin) > time.Second {
		t.Fatalf("returned after %v", time.Since(begin))
	}
	eng.mu.Lock()
	defer eng.mu.Unlock()
	if len(eng.deadlines) != 1 || !eng.deadlines[0] {
		t.Fatalf("engine did not receive a deadline: %v", eng.deadlines)
	}
}

func TestCancellationSurfacesAsContextCanceled(t *testing.T) {
	started := make(chan struct{})
	eng := &fakeEngine{handler: func(ctx context.Context, _ *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	c, _ := start(t, eng, 8)

	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-started; cancel() }()
	if _, err := c.PredictBatch(ctx, imgs(1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestWrongPredictionCountFromEngineIsAnError(t *testing.T) {
	eng := &fakeEngine{handler: func(context.Context, *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		return &pb.PredictBatchResponse{Predictions: []*pb.Prediction{{PredictedClass: 1}}}, nil
	}}
	c, _ := start(t, eng, 8)
	if _, err := c.PredictBatch(ctx2s(t), imgs(1, 2, 3)); !errors.Is(err, ErrBadResponse) {
		t.Fatalf("want ErrBadResponse, got %v", err)
	}
}

// ---- batcher + client end to end (in process) -----------------------------

func TestBatcherOverGRPCRoutesEveryResultToItsCaller(t *testing.T) {
	eng := &fakeEngine{handler: func(_ context.Context, req *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		time.Sleep(2 * time.Millisecond) // give batches time to overlap
		return classFromFirstPixel(req), nil
	}}
	c, _ := start(t, eng, 8)
	b, err := batcher.New(c, batcher.Config{
		MaxBatch: 8, MaxWait: 5 * time.Millisecond, QueueSize: 512,
		EngineTimeout: 5 * time.Second, MaxInFlight: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Shutdown(ctx2s(t)) }()

	const n = 200
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			p, err := b.Submit(ctx2s(t), []float32{float32(i)})
			if err != nil || int(p.Class) != i {
				t.Errorf("request %d: got %+v, %v", i, p, err)
			}
		}()
	}
	wg.Wait()

	total := 0
	for _, s := range eng.sizes() {
		if s > 8 {
			t.Errorf("engine received batch of %d (> its limit of 8)", s)
		}
		total += s
	}
	if total != n {
		t.Fatalf("engine saw %d images, want %d", total, n)
	}
}

func TestEngineInvalidArgumentReachesBatcherCallerAsErrInvalidRequest(t *testing.T) {
	eng := &fakeEngine{handler: func(context.Context, *pb.PredictBatchRequest) (*pb.PredictBatchResponse, error) {
		return nil, status.Error(codes.InvalidArgument, "wrong input dimension")
	}}
	c, _ := start(t, eng, 8)
	b, _ := batcher.New(c, batcher.Config{
		MaxBatch: 8, MaxWait: time.Millisecond, QueueSize: 8, EngineTimeout: time.Second, MaxInFlight: 1,
	})
	defer func() { _ = b.Shutdown(ctx2s(t)) }()

	if _, err := b.Submit(ctx2s(t), []float32{1}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("want ErrInvalidRequest through the batcher, got %v", err)
	}
}
