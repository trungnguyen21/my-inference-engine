// Command server exposes the C++ inference engine over HTTP.
//
// Phase 1: one endpoint, one request at a time, no streaming.
// See API.md for the contract you are implementing, and README.md for the Go
// concepts each TODO needs.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/trungnguyen21/my-own-inference-engine/server/engine"
)

// ---------------------------------------------------------------------------
// Wire types. Struct tags control the JSON field names; without them Go would
// marshal `MaxTokens` as "MaxTokens" instead of "max_tokens".
// ---------------------------------------------------------------------------

type generateRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
	Stream    bool   `json:"stream"`
}

type generateResponse struct {
	Text            string `json:"text"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	Truncated       bool   `json:"truncated"`
	MS              int64  `json:"ms"`
}

type errorResponse struct {
	Error string `json:"error"`
}

// Streaming wire types. Each is marshalled into one SSE `data:` field.
type streamToken struct {
	Token string `json:"token"`
}

type streamStats struct {
	PromptTokens    int   `json:"prompt_tokens"`
	GeneratedTokens int   `json:"generated_tokens"`
	Truncated       bool  `json:"truncated"`
	MS              int64 `json:"ms"`
}

// ---------------------------------------------------------------------------
// generator is the slice of *engine.Session this server actually needs.
//
// Depending on an interface rather than the concrete type is what lets the
// tests in main_test.go run instantly with a fake, instead of loading a
// 469 MB model. *engine.Session satisfies this automatically — Go interfaces
// are implicit, so Session never had to declare that it implements anything.
// ---------------------------------------------------------------------------

type generator interface {
	// ctx carries the HTTP request's lifetime. The pool uses it to stop
	// waiting for a worker when the client has already hung up.
	Generate(ctx context.Context, prompt string, maxTokens int, temperature float32,
		onToken engine.TokenFunc) (engine.Stats, error)
}

// server holds everything the handlers need. Bundling dependencies in a struct
// and hanging handlers off it (rather than using globals) is the conventional
// Go shape for this.
type server struct {
	// mu guards the fields that only become known once the model has loaded.
	// The process starts listening immediately so Kubernetes can tell "still
	// starting" (readiness 503) from "broken" (connection refused), which it
	// cannot do if the model loads before the listener exists.
	mu sync.RWMutex

	// gen is a sessionPool in production, and nil until loading finishes. Each
	// in-flight request gets its own session, and therefore its own KV cache —
	// which is what makes concurrent requests safe. Sharing one session is
	// what produced garbage output before Phase 3.
	gen          generator
	workers      int // concurrent requests the pool admits
	contextLimit int // the model's own context ceiling
	maxThreads   int // OpenMP threads per request; see engine.MaxThreads

	// Static configuration, set before serving.
	maxTokensLimit int // largest max_tokens a request may ask for
	promptCapacity int // largest prompt in bytes; from engine.PromptCapacity()
	sessionContext int // the session's context ceiling, for pre-flight checks
}

// ready publishes the loaded engine. Requests are refused until it is called.
func (s *server) ready(gen generator, workers, contextLimit, maxThreads int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gen, s.workers, s.contextLimit, s.maxThreads = gen, workers, contextLimit, maxThreads
}

// generator returns the loaded engine, or nil while the model is still loading.
func (s *server) generator() generator {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.gen
}

// snapshot reads the load-time facts for reporting.
func (s *server) snapshot() (workers, contextLimit, maxThreads int, loaded bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.workers, s.contextLimit, s.maxThreads, s.gen != nil
}

// writeJSON is shared by every handler. Note the ordering: headers must be set
// before WriteHeader, and WriteHeader before the body. Writing the body first
// implicitly sends a 200 and your status code is silently lost.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}

// writeGenerateError turns an error from the generator into a response.
//
// The three cases are distinct on purpose: a client can retry a 503 later, can
// only report a 500, and needs nothing at all once it has disconnected.
func (s *server) writeGenerateError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrBusy):
		// At capacity, not broken. Retry-After gives the client a hint
		// instead of leaving it to guess.
		workers, _, _, _ := s.snapshot()
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable,
			fmt.Sprintf("all %d workers busy; retry shortly", workers))
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		// The caller gave up while queued; nobody is left to read a response.
		log.Printf("request abandoned before a worker was free: %v", err)
	default:
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

// ---------------------------------------------------------------------------
// WORKED EXAMPLE — read this one, then write handleGenerate below.
// ---------------------------------------------------------------------------

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	workers, contextLimit, threads, loaded := s.snapshot()
	status := "ok"
	code := http.StatusOK
	if !loaded {
		status, code = "loading", http.StatusServiceUnavailable
	}
	writeJSON(w, code, map[string]any{
		"status":              status,
		"workers":             workers,
		"model_context_limit": contextLimit,
		"session_context":     s.sessionContext,
		"omp_threads":         threads,
	})
}

// handleLive answers "is the process alive". It must not depend on the model:
// a liveness probe that fails during the ~4 s load would make Kubernetes kill
// the container before it ever finished starting.
func (s *server) handleLive(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "alive"})
}

// handleReady answers "can this pod serve traffic". False until the model has
// loaded, which is what keeps requests off a pod that is still starting.
func (s *server) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	if _, _, _, loaded := s.snapshot(); !loaded {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "loading"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready"})
}

// handleGenerate implements POST /generate as specified in API.md.
func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "bad body request")
		return
	}

	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt cannot be empty")
		return
	}

	if len(prompt) >= s.promptCapacity {
		writeError(w, http.StatusBadRequest, "prompt too long")
		return
	}

	if req.MaxTokens < 1 || req.MaxTokens > s.maxTokensLimit {
		writeError(w, http.StatusBadRequest, "invalid max token limit")
		return
	}

	gen := s.generator()
	if gen == nil {
		w.Header().Set("Retry-After", "5")
		writeError(w, http.StatusServiceUnavailable, "model is still loading")
		return
	}

	if req.Stream {
		s.streamGenerate(w, r, gen, prompt, req.MaxTokens)
		return
	}

	var sb strings.Builder
	start := time.Now()
	stats, err := gen.Generate(r.Context(), prompt, req.MaxTokens, 0,
		func(piece string) bool {
			sb.WriteString(piece)
			return true
		})
	elapsed := time.Since(start)
	if err != nil {
		s.writeGenerateError(w, err)
		return
	}

	response := generateResponse{
		Text:            sb.String(),
		PromptTokens:    stats.PromptTokens,
		GeneratedTokens: stats.GeneratedTokens,
		Truncated:       stats.StoppedEarly,
		MS:              elapsed.Milliseconds(),
	}

	writeJSON(w, http.StatusOK, response)

}

// streamGenerate implements the `"stream": true` path from API.md. The prompt
// arrives already validated and trimmed, so only the streaming mechanics
// remain.
func (s *server) streamGenerate(w http.ResponseWriter, r *http.Request,
	gen generator, prompt string, maxTokens int) {

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "this connection cannot stream")
		return
	}

	clientGone := r.Context().Done()

	// Nothing may be written until the first token arrives. Once a status line
	// is sent it cannot be changed, so writing the SSE header up front would
	// turn an early engine failure into a 200 carrying an error body.
	wroteHeader := false
	disconnected := false

	ensureHeader := func() {
		if wroteHeader {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		wroteHeader = true
	}

	// One SSE event: an optional `event:` name line, then `data:`, then the
	// blank line that terminates it. Without that blank line there is no event.
	writeEvent := func(name string, payload []byte) error {
		if name != "" {
			if _, err := fmt.Fprintf(w, "event: %s\n", name); err != nil {
				return err
			}
		}
		_, err := fmt.Fprintf(w, "data: %s\n\n", payload)
		return err
	}

	// The callback runs synchronously on this goroutine — Generate blocks in C
	// and calls back into Go on the same thread — so these closures need no
	// locking.
	// The pool holds a session for this whole call, so `start` measures
	// generation and excludes any time spent waiting for a free worker.
	start := time.Now()
	stats, genErr := gen.Generate(r.Context(), prompt, maxTokens, 0, func(piece string) bool {
		select {
		case <-clientGone:
			disconnected = true
			return false
		default:
		}

		// Marshal rather than format: a piece may contain quotes, backslashes,
		// or a newline, and a raw newline would split one event into two.
		payload, err := json.Marshal(streamToken{Token: piece})
		if err != nil {
			log.Printf("stream: marshal token: %v", err)
			return false
		}

		ensureHeader()
		if err := writeEvent("", payload); err != nil {
			// The connection is gone; further writes would fail too.
			log.Printf("stream: write token: %v", err)
			disconnected = true
			return false
		}
		flusher.Flush()
		return true
	})

	if genErr != nil {
		if !wroteHeader {
			// Still a normal HTTP error: nothing has been committed yet.
			s.writeGenerateError(w, genErr)
			return
		}
		// The status is already sent, so report in-band and send no [DONE] —
		// its absence is what tells the client the stream was truncated.
		if payload, err := json.Marshal(errorResponse{Error: genErr.Error()}); err != nil {
			log.Printf("stream: marshal error event: %v", err)
		} else if err := writeEvent("error", payload); err != nil {
			log.Printf("stream: write error event: %v", err)
		} else {
			flusher.Flush()
		}
		return
	}

	if disconnected {
		return // nobody is left to read a terminator
	}

	// A request that produced no text still owes the client a valid stream;
	// generation stopping immediately on EOS is a success, not an error.
	ensureHeader()

	payload, err := json.Marshal(streamStats{
		PromptTokens:    stats.PromptTokens,
		GeneratedTokens: stats.GeneratedTokens,
		Truncated:       stats.StoppedEarly,
		MS:              time.Since(start).Milliseconds(),
	})
	if err != nil {
		log.Printf("stream: marshal stats: %v", err)
		return
	}
	if err := writeEvent("stats", payload); err != nil {
		log.Printf("stream: write stats: %v", err)
		return
	}
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		log.Printf("stream: write terminator: %v", err)
		return
	}
	flusher.Flush()
}

// ---------------------------------------------------------------------------
// Wiring. This part is done; read it to see how the pieces connect.
// ---------------------------------------------------------------------------

func main() {
	var (
		modelPath = flag.String("model", envStr("MODEL", "../models/qwen-2.5.gguf"),
			"path to the GGUF model (env: MODEL)")
		addr           = flag.String("addr", envStr("ADDR", ":8080"), "listen address (env: ADDR)")
		sessionContext = flag.Int("context", envInt("CONTEXT", 1024),
			"session context ceiling in tokens, ~27.6 KB each (env: CONTEXT)")
		maxTokensLimit = flag.Int("max-tokens-limit", 512, "largest max_tokens a request may request")
		ompThreadsFlag = flag.Int("threads", envInt("THREADS", 0),
			"OpenMP threads per request (0 = derive from cores/workers) (env: THREADS)")
		workers = flag.Int("workers", envInt("WORKERS", 1),
			"concurrent requests to admit; one session each (env: WORKERS)")
		queueWait = flag.Duration("queue-wait", 30*time.Second,
			"how long a request may wait for a free worker before 503")
	)
	flag.Parse()

	srv := &server{
		maxTokensLimit: *maxTokensLimit,
		promptCapacity: engine.PromptCapacity(),
		sessionContext: *sessionContext,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/health/live", srv.handleLive)
	mux.HandleFunc("/health/ready", srv.handleReady)
	mux.HandleFunc("/generate", srv.handleGenerate)

	httpServer := &http.Server{
		Addr:    *addr,
		Handler: mux,
		// Generation is slow, so the write timeout must be generous or long
		// streaming responses get cut off mid-flight.
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Minute,
	}

	// Listen before loading. The model takes several seconds, and a pod that
	// refuses connections during that window is indistinguishable from a
	// broken one; a pod that answers /health/ready with 503 is not.
	listening := make(chan error, 1)
	go func() {
		log.Printf("listening on %s (readiness 503 until the model is loaded)", *addr)
		err := httpServer.ListenAndServe()
		if !errors.Is(err, http.ErrServerClosed) {
			listening <- err
			return
		}
		listening <- nil
	}()

	// Shut down on SIGINT/SIGTERM. Kubernetes sends SIGTERM and waits, so
	// draining here is what lets in-flight generations finish.
	shutdownCtx, stopSignals := signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	log.Printf("loading %s ...", *modelPath)
	loadStart := time.Now()
	model, err := engine.Open(*modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer model.Close()
	log.Printf("loaded in %s. model context limit = %d tokens",
		time.Since(loadStart).Round(time.Millisecond), model.ContextLimit())

	// Threads multiply by workers, so leaving this to the environment is a
	// trap: 4 workers at "all cores" put 32 threads on 8 cores and turned a
	// 7 s workload into a 120 s one. Derive a safe default instead.
	ompThreads := engine.MaxThreads()
	switch {
	case *ompThreadsFlag > 0:
		ompThreads = engine.SetMaxThreads(*ompThreadsFlag)
		log.Printf("OpenMP: %d threads per request (-threads=%d)", ompThreads, *ompThreadsFlag)
	case *workers > 1:
		derived := runtime.NumCPU() / *workers
		if derived < 1 {
			derived = 1
		}
		ompThreads = engine.SetMaxThreads(derived)
		log.Printf("OpenMP: %d threads per request (derived from %d cores / %d workers; "+
			"pass -threads to override)", ompThreads, runtime.NumCPU(), *workers)
	default:
		if env, ok := os.LookupEnv("OMP_NUM_THREADS"); ok {
			log.Printf("OpenMP: %d threads per request (OMP_NUM_THREADS=%s)", ompThreads, env)
		} else {
			log.Printf("OpenMP: %d threads per request (OMP_NUM_THREADS unset, using all cores; "+
				"pass -threads to pin it)", ompThreads)
		}
	}
	if total := ompThreads * *workers; total > runtime.NumCPU() {
		log.Printf("warning: %d workers x %d threads = %d threads on %d cores; "+
			"oversubscription degrades throughput superlinearly",
			*workers, ompThreads, total, runtime.NumCPU())
	}

	// Each worker holds its own KV cache, so memory scales with both numbers.
	// Report the estimate before allocating: failing here with a clear message
	// beats being OOM-killed halfway through a demo.
	log.Printf("allocating %d worker(s) x %d context tokens (~%d MiB of KV cache)",
		*workers, *sessionContext, estimateKVMiB(*workers, *sessionContext))

	pool, err := newSessionPool(model, *workers, *sessionContext, *queueWait)
	if err != nil {
		log.Fatalf("create session pool: %v", err)
	}

	srv.ready(pool, pool.Workers(), model.ContextLimit(), ompThreads)
	log.Printf("ready: %d worker(s), queue wait %s", pool.Workers(), *queueWait)

	// Wait for a signal, or for the listener to fail.
	select {
	case <-shutdownCtx.Done():
		log.Printf("shutdown signal received; draining")
	case err := <-listening:
		if err != nil {
			log.Printf("serve: %v", err)
		}
	}

	// Stop accepting, let in-flight requests finish, then release the engine.
	// Sessions must outlive every request that holds one, so the pool closes
	// only after the HTTP server has drained.
	drainCtx, cancelDrain := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancelDrain()
	if err := httpServer.Shutdown(drainCtx); err != nil {
		log.Printf("drain did not finish cleanly: %v", err)
	}
	pool.Close()
	log.Printf("stopped")
}

// envStr reads a string environment variable, falling back to def.
func envStr(name, def string) string {
	if raw, ok := os.LookupEnv(name); ok && raw != "" {
		return raw
	}
	return def
}

// envInt reads an integer environment variable, falling back to def when it is
// unset or unparseable. Flags still win: the flag default is this value.
func envInt(name string, def int) int {
	raw, ok := os.LookupEnv(name)
	if !ok {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		log.Printf("ignoring %s=%q: %v", name, raw, err)
		return def
	}
	return n
}

// estimateKVMiB approximates the KV cache plus activations a pool will hold.
//
// Measured at ~27.6 KB per context token per worker for this model: the KV
// cache dominates at 2 caches x 24 layers x 128 values x 4 bytes per token.
func estimateKVMiB(workers, contextTokens int) int {
	const bytesPerContextToken = 28220
	return workers * contextTokens * bytesPerContextToken / (1 << 20)
}
