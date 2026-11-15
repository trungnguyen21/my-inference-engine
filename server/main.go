// Command server exposes the C++ inference engine over HTTP.
//
// Phase 1: one endpoint, one request at a time, no streaming.
// See API.md for the contract you are implementing, and README.md for the Go
// concepts each TODO needs.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
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
	Generate(prompt string, maxTokens int, temperature float32,
		onToken engine.TokenFunc) (engine.Stats, error)
}

// server holds everything the handlers need. Bundling dependencies in a struct
// and hanging handlers off it (rather than using globals) is the conventional
// Go shape for this.
type server struct {
	gen generator

	// A single engine.Session is not safe for concurrent use, but net/http
	// runs every request in its own goroutine. Without this, simultaneous
	// requests corrupt each other's KV cache and can return a 200 carrying
	// garbage text.
	//
	// Holding it for the whole generation serializes requests, which is what
	// the one-session design means. Phase 3 replaces this with a pool of
	// sessions so requests genuinely overlap.
	mu sync.Mutex

	maxTokensLimit int // largest max_tokens a request may ask for
	promptCapacity int // largest prompt in bytes; from engine.PromptCapacity()
	contextLimit   int // reported by /health
	sessionContext int // the session's context ceiling, for pre-flight checks
	maxThreads     int // OpenMP threads per request; see engine.MaxThreads
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

// ---------------------------------------------------------------------------
// WORKED EXAMPLE — read this one, then write handleGenerate below.
// ---------------------------------------------------------------------------

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":              "ok",
		"model_context_limit": s.contextLimit,
		"session_context":     s.sessionContext,
		"omp_threads":         s.maxThreads,
	})
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

	if req.Stream {
		s.streamGenerate(w, r, prompt, req.MaxTokens)
		return
	}

	var sb strings.Builder
	s.mu.Lock()
	// Started after the lock: `ms` reports generation time, not time spent
	// queued behind another request.
	start := time.Now()
	stats, err := s.gen.Generate(prompt, req.MaxTokens, 0,
		func(piece string) bool {
			sb.WriteString(piece)
			return true
		})
	s.mu.Unlock()
	elapsed := time.Since(start)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
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
	prompt string, maxTokens int) {

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
	// Held for the whole stream: one session cannot serve two requests.
	s.mu.Lock()
	// Started after the lock, so `ms` excludes time spent queued.
	start := time.Now()
	stats, genErr := s.gen.Generate(prompt, maxTokens, 0, func(piece string) bool {
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
	s.mu.Unlock()

	if genErr != nil {
		if !wroteHeader {
			// Still a normal HTTP error: nothing has been committed yet.
			writeError(w, http.StatusInternalServerError, genErr.Error())
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
		modelPath      = flag.String("model", "../models/qwen-2.5.gguf", "path to the GGUF model")
		addr           = flag.String("addr", ":8080", "listen address")
		sessionContext = flag.Int("context", 1024, "session context ceiling in tokens (~27.6 KB each)")
		maxTokensLimit = flag.Int("max-tokens-limit", 512, "largest max_tokens a request may request")
		ompThreadsFlag = flag.Int("threads", 0, "OpenMP threads per request (0 = use OMP_NUM_THREADS / all cores)")
	)
	flag.Parse()

	// Loading takes ~4.2 s, so do it before listening. Kubernetes will later
	// need a readiness probe that stays false during this window.
	log.Printf("loading %s ...", *modelPath)
	model, err := engine.Open(*modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer model.Close()
	log.Printf("loaded. model context limit = %d tokens", model.ContextLimit())

	// OpenMP lives inside the C++ kernels, so it is invisible from Go unless
	// reported. Unset means "all cores", which becomes oversubscription once
	// Phase 3 runs several requests at once: threads = workers * this value.
	ompThreads := engine.MaxThreads()
	switch {
	case *ompThreadsFlag > 0:
		ompThreads = engine.SetMaxThreads(*ompThreadsFlag)
		log.Printf("OpenMP: %d threads per request (-threads=%d)", ompThreads, *ompThreadsFlag)
	default:
		if env, ok := os.LookupEnv("OMP_NUM_THREADS"); ok {
			log.Printf("OpenMP: %d threads per request (OMP_NUM_THREADS=%s)", ompThreads, env)
		} else {
			log.Printf("OpenMP: %d threads per request (OMP_NUM_THREADS unset, using all cores; "+
				"pass -threads to pin it)", ompThreads)
		}
	}

	// Phase 1 is one session, so one request at a time. Phase 3 turns this
	// into a pool of N sessions behind a channel.
	session, err := model.NewSession(*sessionContext)
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer session.Close()

	srv := &server{
		gen:            session,
		maxTokensLimit: *maxTokensLimit,
		promptCapacity: engine.PromptCapacity(),
		contextLimit:   model.ContextLimit(),
		sessionContext: *sessionContext,
		maxThreads:     ompThreads,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/generate", srv.handleGenerate)

	httpServer := &http.Server{
		Addr:    *addr,
		Handler: mux,
		// Generation is slow, so the write timeout must be generous or long
		// responses get cut off mid-flight.
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      10 * time.Minute,
	}

	log.Printf("listening on %s", *addr)
	if err := httpServer.ListenAndServe(); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
