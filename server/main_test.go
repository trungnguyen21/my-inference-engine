package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/trungnguyen21/my-own-inference-engine/server/engine"
)

// fakeGenerator stands in for a real session so these tests run in
// milliseconds instead of loading a 469 MB model. It satisfies the generator
// interface implicitly — no "implements" declaration needed.
type fakeGenerator struct {
	pieces []string // emitted one per callback call
	err    error    // returned instead of generating anything
	stats  engine.Stats

	// errAfterPieces fails only once every piece has been handed over, which
	// is how a mid-stream failure is simulated.
	errAfterPieces bool
	// beforePiece runs just before piece i is handed over, letting a test
	// cancel the request context partway through.
	beforePiece func(i int)

	consumed  int // pieces actually handed to the callback
	gotMax    int // records what the handler passed through
	gotTemp   float32
	gotPrompt string
}

func (f *fakeGenerator) Generate(prompt string, maxTokens int, temperature float32,
	onToken engine.TokenFunc) (engine.Stats, error) {
	f.gotPrompt, f.gotMax, f.gotTemp = prompt, maxTokens, temperature
	if f.err != nil {
		return engine.Stats{}, f.err
	}
	for i, p := range f.pieces {
		if f.beforePiece != nil {
			f.beforePiece(i)
		}
		f.consumed++
		if !onToken(p) {
			break
		}
	}
	if f.errAfterPieces {
		return engine.Stats{}, errEngineBoom
	}
	return f.stats, nil
}

func newTestServer(gen generator) *server {
	return &server{
		gen:            gen,
		maxTokensLimit: 512,
		promptCapacity: 4096,
		contextLimit:   32768,
		sessionContext: 1024,
		maxThreads:     4,
	}
}

// do sends a request body to /generate and returns the recorder.
func do(t *testing.T, srv *server, method, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, "/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.handleGenerate(rec, req)
	return rec
}

func decodeError(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var er errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &er); err != nil {
		t.Fatalf("error body is not JSON: %q", rec.Body.String())
	}
	if er.Error == "" {
		t.Errorf("error body should explain the problem, got %q", rec.Body.String())
	}
	return er.Error
}

// --- the happy path -------------------------------------------------------

func TestGenerateSuccess(t *testing.T) {
	fake := &fakeGenerator{
		pieces: []string{"Par", "is", "."},
		stats: engine.Stats{
			PromptTokens:    36,
			GeneratedTokens: 8,
			StoppedEarly:    false,
		},
	}
	srv := newTestServer(fake)
	rec := do(t, srv, http.MethodPost, `{"prompt":"What is the capital of France?","max_tokens":64}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. body: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var got generateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("body is not JSON: %q", rec.Body.String())
	}
	if got.Text != "Paris." {
		t.Errorf("Text = %q, want %q (did you accumulate every piece?)", got.Text, "Paris.")
	}
	if got.PromptTokens != 36 {
		t.Errorf("PromptTokens = %d, want 36 (copy it from stats)", got.PromptTokens)
	}
	if got.GeneratedTokens != 8 {
		t.Errorf("GeneratedTokens = %d, want 8", got.GeneratedTokens)
	}
	if got.Truncated {
		t.Error("Truncated should be false when StoppedEarly is false")
	}

	// The handler must forward the request's values, and must always ask for
	// greedy decoding: the engine rejects any other temperature.
	if fake.gotPrompt != "What is the capital of France?" {
		t.Errorf("passed prompt %q", fake.gotPrompt)
	}
	if fake.gotMax != 64 {
		t.Errorf("passed max_tokens %d, want 64", fake.gotMax)
	}
	if fake.gotTemp != 0 {
		t.Errorf("passed temperature %v, want 0", fake.gotTemp)
	}
}

func TestGenerateReportsTruncation(t *testing.T) {
	fake := &fakeGenerator{
		pieces: []string{"Par"},
		stats:  engine.Stats{PromptTokens: 5, GeneratedTokens: 1, StoppedEarly: true},
	}
	rec := do(t, newTestServer(fake), http.MethodPost, `{"prompt":"hi","max_tokens":16}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var got generateResponse
	json.Unmarshal(rec.Body.Bytes(), &got)
	if !got.Truncated {
		t.Error("Truncated should mirror stats.StoppedEarly")
	}
}

// --- validation: every one of these is a 400 ------------------------------

func TestGenerateRejectsBadRequests(t *testing.T) {
	longPrompt := strings.Repeat("a", 5000) // over promptCapacity of 4096

	for _, tc := range []struct {
		name string
		body string
	}{
		{"malformed JSON", `{"prompt":`},
		{"not an object", `"hello"`},
		{"missing prompt", `{"max_tokens":16}`},
		{"empty prompt", `{"prompt":"","max_tokens":16}`},
		{"whitespace-only prompt", `{"prompt":"   \n\t ","max_tokens":16}`},
		{"prompt too long", `{"prompt":"` + longPrompt + `","max_tokens":16}`},
		{"missing max_tokens", `{"prompt":"hi"}`},
		{"zero max_tokens", `{"prompt":"hi","max_tokens":0}`},
		{"negative max_tokens", `{"prompt":"hi","max_tokens":-5}`},
		{"max_tokens over limit", `{"prompt":"hi","max_tokens":99999}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fake := &fakeGenerator{pieces: []string{"x"}}
			rec := do(t, newTestServer(fake), http.MethodPost, tc.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400. body: %s", rec.Code, rec.Body.String())
			}
			decodeError(t, rec)
			if fake.gotPrompt != "" {
				t.Error("the engine should not be called for an invalid request")
			}
		})
	}
}

func TestGenerateRejectsNonPost(t *testing.T) {
	for _, method := range []string{http.MethodGet, http.MethodPut, http.MethodDelete} {
		rec := do(t, newTestServer(&fakeGenerator{}), method, "")
		if rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("%s: status = %d, want 405", method, rec.Code)
		}
	}
}

// --- engine failure is a 500, not a 400 ----------------------------------

func TestGenerateEngineFailureIs500(t *testing.T) {
	fake := &fakeGenerator{err: errEngineBoom}
	rec := do(t, newTestServer(fake), http.MethodPost, `{"prompt":"hi","max_tokens":16}`)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500. body: %s", rec.Code, rec.Body.String())
	}
	if msg := decodeError(t, rec); !strings.Contains(msg, "boom") {
		t.Errorf("error should surface the engine's message, got %q", msg)
	}
}

// --- health, which is already written, as a reference --------------------

func TestHealth(t *testing.T) {
	srv := newTestServer(&fakeGenerator{})
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	srv.handleHealth(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["status"] != "ok" {
		t.Errorf("status field = %v", got["status"])
	}
}

// --- regressions: the two bugs found reviewing the first draft -------------

// The engine rejects a prompt at >= PromptCapacity (src/engine_api.cpp:182),
// so the handler must reject at the same boundary. Rejecting only at > would
// let exactly-capacity through and turn the engine's refusal into a 500.
func TestGeneratePromptLengthBoundary(t *testing.T) {
	const capacity = 4096 // matches newTestServer's promptCapacity

	for _, tc := range []struct {
		name       string
		length     int
		wantStatus int
		wantEngine bool
	}{
		{"one under capacity is accepted", capacity - 1, http.StatusOK, true},
		{"exactly capacity is rejected", capacity, http.StatusBadRequest, false},
		{"over capacity is rejected", capacity + 1, http.StatusBadRequest, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fake := &fakeGenerator{pieces: []string{"x"}}
			body, err := json.Marshal(generateRequest{
				Prompt:    strings.Repeat("a", tc.length),
				MaxTokens: 8,
			})
			if err != nil {
				t.Fatal(err)
			}
			rec := do(t, newTestServer(fake), http.MethodPost, string(body))
			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tc.wantStatus)
			}
			if gotEngine := fake.gotPrompt != ""; gotEngine != tc.wantEngine {
				t.Errorf("engine called = %v, want %v", gotEngine, tc.wantEngine)
			}
		})
	}
}

// Validation runs on the trimmed prompt, so the trimmed prompt is what must be
// sent. Validating one string and submitting another means the engine sees text
// that was never checked.
func TestGenerateSendsTheValidatedPrompt(t *testing.T) {
	fake := &fakeGenerator{pieces: []string{"ok"}}
	rec := do(t, newTestServer(fake), http.MethodPost,
		`{"prompt":"  \n hello \t ","max_tokens":8}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. body: %s", rec.Code, rec.Body.String())
	}
	if fake.gotPrompt != "hello" {
		t.Errorf("engine received %q, want %q", fake.gotPrompt, "hello")
	}
}
