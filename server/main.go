// Command server exposes the C++ inference engine over HTTP.
//
// Phase 1: one endpoint, one request at a time, no streaming.
// See API.md for the contract you are implementing, and README.md for the Go
// concepts each TODO needs.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"strings"
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
	gen            generator
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
//
// Run `go test ./...` as you go — main_test.go encodes every rule below, so a
// green suite means you are done. Work in this order:
//
//  1. Reject anything that is not POST with 405.
//     http.MethodPost is the constant; see handleHealth for the shape.
//
//  2. Decode the body into a generateRequest.
//     json.NewDecoder(r.Body).Decode(&req) returns an error for malformed
//     JSON — that is a 400, not a 500. The client sent something bad.
//
//  3. Validate, each failure a 400 with a message naming the problem:
//     - Prompt empty or only whitespace       (strings.TrimSpace)
//     - len(Prompt) > s.promptCapacity        (bytes, not characters:
//     len() on a string counts bytes)
//     - MaxTokens < 1 or > s.maxTokensLimit
//
//  4. Collect the output. Generate hands you pieces, not the whole string:
//
//     var sb strings.Builder
//     stats, err := s.gen.Generate(req.Prompt, req.MaxTokens, 0,
//     func(piece string) bool {
//     sb.WriteString(piece)
//     return true // false would stop generation
//     })
//
//     Use strings.Builder, not `out += piece`. String concatenation in a loop
//     reallocates every time; Builder appends to one growing buffer.
//
//     Time it with time.Now() before and time.Since(start) after.
//
//  5. If err != nil, respond 500 with err.Error().
//     (Pre-flight context overflow could be a 400 — see API.md. Optional.)
//
//  6. On success, respond 200 with a generateResponse.
//     Truncated comes from stats.StoppedEarly.
//     MS is time.Since(start).Milliseconds().
//
// Delete the line below once you start.
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

	var sb strings.Builder
	start := time.Now()
	stats, err := s.gen.Generate(prompt, req.MaxTokens, 0,
		func(piece string) bool {
			sb.WriteString(piece)
			return true
		})
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
