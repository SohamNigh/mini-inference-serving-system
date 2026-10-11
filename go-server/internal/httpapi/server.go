// Package httpapi is the client-facing HTTP layer: it validates input, applies
// a per-request deadline, and translates batcher/engine failures into
// meaningful status codes.
package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"net/http"
	"sync/atomic"
	"time"

	"mini-inference-serving-system/go-server/internal/batcher"
	"mini-inference-serving-system/go-server/internal/engineclient"
)

// StatusClientClosedRequest is the (nginx-originated) code for "client went away
// before we answered". It is recorded for metrics; the client never sees it.
const StatusClientClosedRequest = 499

type Submitter interface {
	Submit(ctx context.Context, pixels []float32) (batcher.Prediction, error)
}

type Options struct {
	PixelCount     int                                     // required input length (784 for MNIST)
	MaxBodyBytes   int64                                   // request bodies larger than this get 413
	RequestTimeout time.Duration                           // end-to-end budget per request, including queue wait
	Logger         *slog.Logger                            // optional
	Recorder       func(status int, latency time.Duration) // optional; called once per request
}

type Server struct {
	sub      Submitter
	opts     Options
	log      *slog.Logger
	draining atomic.Bool
}

func New(sub Submitter, opts Options) (*Server, error) {
	switch {
	case sub == nil:
		return nil, errors.New("httpapi: nil submitter")
	case opts.PixelCount < 1:
		return nil, errors.New("httpapi: PixelCount must be >= 1")
	case opts.MaxBodyBytes < 1:
		return nil, errors.New("httpapi: MaxBodyBytes must be >= 1")
	case opts.RequestTimeout <= 0:
		return nil, errors.New("httpapi: RequestTimeout must be > 0")
	}
	log := opts.Logger
	if log == nil {
		log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &Server{sub: sub, opts: opts, log: log}, nil
}

// SetDraining makes /readyz fail so load balancers stop sending traffic.
// /healthz (liveness) is unaffected, and in-flight requests keep being served.
func (s *Server) SetDraining() { s.draining.Store(true) }

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		if s.draining.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	return s.middleware(mux)
}

// statusWriter remembers the status code so the recorder can report it.
type statusWriter struct {
	http.ResponseWriter
	status  int
	written bool
}

func (w *statusWriter) WriteHeader(code int) {
	if !w.written {
		w.status, w.written = code, true
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if !w.written {
		w.status, w.written = http.StatusOK, true
	}
	return w.ResponseWriter.Write(b)
}

func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		start := time.Now()
		w := &statusWriter{ResponseWriter: rw}
		defer func() {
			if rec := recover(); rec != nil {
				s.log.Error("handler panic", "panic", fmt.Sprint(rec), "path", r.URL.Path)
				if !w.written {
					writeError(w, http.StatusInternalServerError, "internal error")
				}
			}
			if s.opts.Recorder != nil {
				s.opts.Recorder(w.status, time.Since(start))
			}
		}()
		next.ServeHTTP(w, r)
	})
}

type predictRequest struct {
	Pixels []float32 `json:"pixels"`
}

type predictResponse struct {
	Class      int32   `json:"class"`
	Confidence float32 `json:"confidence"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	if mt, _, err := mime.ParseMediaType(r.Header.Get("Content-Type")); err != nil || mt != "application/json" {
		writeError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return
	}

	pixels, status, msg := s.decode(w, r)
	if status != 0 {
		writeError(w, status, msg)
		return
	}

	// The budget covers queue wait and engine time. The request context also
	// carries client disconnects, so abandoned requests stop consuming the batcher.
	ctx, cancel := context.WithTimeout(r.Context(), s.opts.RequestTimeout)
	defer cancel()

	pred, err := s.sub.Submit(ctx, pixels)
	if err != nil {
		s.writeSubmitError(w, r, err)
		return
	}
	if math.IsNaN(float64(pred.Confidence)) || math.IsInf(float64(pred.Confidence), 0) {
		s.log.Error("engine returned non-finite confidence", "class", pred.Class)
		writeError(w, http.StatusBadGateway, "bad response from inference engine")
		return
	}
	writeJSON(w, http.StatusOK, predictResponse{Class: pred.Class, Confidence: pred.Confidence})
}

// decode parses and validates the body. A non-zero status means reject with msg.
func (s *Server) decode(w http.ResponseWriter, r *http.Request) ([]float32, int, string) {
	body := http.MaxBytesReader(w, r.Body, s.opts.MaxBodyBytes)
	dec := json.NewDecoder(body)
	dec.DisallowUnknownFields()

	var req predictRequest
	if err := dec.Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return nil, http.StatusRequestEntityTooLarge, "request body too large"
		}
		return nil, http.StatusBadRequest, "invalid JSON body"
	}
	// Anything after the first JSON value is a malformed request, not extra data to ignore.
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return nil, http.StatusRequestEntityTooLarge, "request body too large"
		}
		return nil, http.StatusBadRequest, "unexpected data after JSON body"
	}
	if len(req.Pixels) != s.opts.PixelCount {
		return nil, http.StatusBadRequest, fmt.Sprintf("pixels must contain exactly %d values", s.opts.PixelCount)
	}
	for _, p := range req.Pixels {
		if !(p >= 0 && p <= 1) { // also rejects NaN
			return nil, http.StatusBadRequest, "pixel values must be within [0, 1]"
		}
	}
	return req.Pixels, 0, ""
}

func (s *Server) writeSubmitError(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case r.Context().Err() != nil && !errors.Is(r.Context().Err(), context.DeadlineExceeded):
		// Client disconnected; nobody is listening, but keep the code for metrics.
		w.WriteHeader(StatusClientClosedRequest)
	case errors.Is(err, batcher.ErrQueueFull):
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "server is overloaded, retry shortly")
	case errors.Is(err, batcher.ErrClosed), errors.Is(err, engineclient.ErrUnavailable):
		s.log.Warn("unavailable", "err", err)
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, "service temporarily unavailable")
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, http.StatusGatewayTimeout, "request timed out")
	case errors.Is(err, engineclient.ErrInvalidRequest):
		// We validated the input, so an engine rejection means the two sides disagree on the contract.
		s.log.Error("engine rejected validated request", "err", err)
		writeError(w, http.StatusInternalServerError, "internal error")
	default:
		s.log.Error("inference failed", "err", err)
		writeError(w, http.StatusBadGateway, "inference backend error")
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}
