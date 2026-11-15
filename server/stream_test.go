package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/trungnguyen21/my-own-inference-engine/server/engine"
)

// ---------------------------------------------------------------------------
// Test plumbing
// ---------------------------------------------------------------------------

// flushRecorder counts Flush calls. httptest.ResponseRecorder already
// implements http.Flusher, so embedding it and overriding Flush lets us assert
// that the handler flushes per token rather than once at the end.
type flushRecorder struct {
	*httptest.ResponseRecorder
	flushes int
}

func newFlushRecorder() *flushRecorder {
	return &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
}

func (f *flushRecorder) Flush() {
	f.flushes++
	f.ResponseRecorder.Flush()
}

// sseEvent is one parsed SSE message.
type sseEvent struct {
	Name string // the "event:" field; empty for a default message
	Data string // the "data:" field
}

// parseSSE splits a stream into events. Events are separated by a blank line,
// which is what the two trailing newlines in "data: ...\n\n" produce.
func parseSSE(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimRight(block, "\n")
		if strings.TrimSpace(block) == "" {
			continue
		}
		var ev sseEvent
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "data:"):
				ev.Data = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			case strings.HasPrefix(line, "event:"):
				ev.Name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			default:
				t.Errorf("unrecognised SSE line %q (fields are 'data:' and 'event:')", line)
			}
		}
		events = append(events, ev)
	}
	return events
}

// tokensFrom returns the token strings from default-named events.
func tokensFrom(t *testing.T, events []sseEvent) []string {
	t.Helper()
	var out []string
	for _, ev := range events {
		if ev.Name != "" || ev.Data == "[DONE]" {
			continue
		}
		var tok streamToken
		if err := json.Unmarshal([]byte(ev.Data), &tok); err != nil {
			t.Errorf("token event data is not JSON: %q", ev.Data)
			continue
		}
		out = append(out, tok.Token)
	}
	return out
}

func doStream(t *testing.T, srv *server, body string, ctx context.Context) *flushRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if ctx != nil {
		req = req.WithContext(ctx)
	}
	rec := newFlushRecorder()
	srv.handleGenerate(rec, req)
	return rec
}

// ---------------------------------------------------------------------------
// The happy path
// ---------------------------------------------------------------------------

func TestStreamEmitsTokenEventsThenStatsThenDone(t *testing.T) {
	fake := &fakeGenerator{
		pieces: []string{"Par", "is", "."},
		stats: engine.Stats{
			PromptTokens: 36, GeneratedTokens: 8, StoppedEarly: false,
		},
	}
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":16,"stream":true}`, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. body: %q", rec.Code, rec.Body.String())
	}
	events := parseSSE(t, rec.Body.String())
	if len(events) < 5 {
		t.Fatalf("got %d events, want 3 tokens + stats + [DONE]. body: %q",
			len(events), rec.Body.String())
	}

	if got := tokensFrom(t, events); strings.Join(got, "") != "Paris." {
		t.Errorf("tokens = %q, want each piece as its own event forming %q", got, "Paris.")
	}

	// The stats event must come before the terminator.
	statsIdx, doneIdx := -1, -1
	for i, ev := range events {
		if ev.Name == "stats" {
			statsIdx = i
		}
		if ev.Data == "[DONE]" {
			doneIdx = i
		}
	}
	if statsIdx < 0 {
		t.Fatal(`no "event: stats" found`)
	}
	if doneIdx < 0 {
		t.Fatal(`no "data: [DONE]" found`)
	}
	if doneIdx != len(events)-1 {
		t.Errorf("[DONE] is at index %d of %d; it must be last", doneIdx, len(events))
	}
	if statsIdx > doneIdx {
		t.Error("stats event must come before [DONE]")
	}

	var st streamStats
	if err := json.Unmarshal([]byte(events[statsIdx].Data), &st); err != nil {
		t.Fatalf("stats data is not JSON: %q", events[statsIdx].Data)
	}
	if st.PromptTokens != 36 || st.GeneratedTokens != 8 {
		t.Errorf("stats = %+v, want PromptTokens 36 and GeneratedTokens 8", st)
	}
	if st.Truncated {
		t.Error("Truncated should mirror stats.StoppedEarly")
	}
}

func TestStreamSetsSSEHeaders(t *testing.T) {
	fake := &fakeGenerator{pieces: []string{"x"}}
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":4,"stream":true}`, nil)

	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if cc := rec.Header().Get("Cache-Control"); !strings.Contains(cc, "no-cache") {
		t.Errorf("Cache-Control = %q, want no-cache", cc)
	}
}

// Without a flush per event the response is buffered and nothing actually
// streams — no error, it just silently stops being a stream.
func TestStreamFlushesPerToken(t *testing.T) {
	fake := &fakeGenerator{pieces: []string{"a", "b", "c", "d"}}
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":16,"stream":true}`, nil)

	if rec.flushes < len(fake.pieces) {
		t.Errorf("flushed %d times for %d tokens; flush after every event",
			rec.flushes, len(fake.pieces))
	}
}

// Token text must be JSON-encoded, or a quote or newline corrupts the stream.
func TestStreamEscapesAwkwardTokens(t *testing.T) {
	fake := &fakeGenerator{pieces: []string{`he said "hi"`, "line\nbreak", `back\slash`}}
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":16,"stream":true}`, nil)

	got := tokensFrom(t, parseSSE(t, rec.Body.String()))
	want := fake.pieces
	if len(got) != len(want) {
		t.Fatalf("got %d tokens, want %d — did a newline split an event? body: %q",
			len(got), len(want), rec.Body.String())
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("token %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Errors: the reason headers are delayed
// ---------------------------------------------------------------------------

func TestStreamValidationErrorIsPlainJSON(t *testing.T) {
	fake := &fakeGenerator{pieces: []string{"x"}}
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"","max_tokens":16,"stream":true}`, nil)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); strings.Contains(ct, "event-stream") {
		t.Errorf("a validation failure must not become an SSE stream (got %q)", ct)
	}
	decodeError(t, rec.ResponseRecorder)
}

func TestStreamEngineFailureBeforeFirstTokenIsJSON500(t *testing.T) {
	fake := &fakeGenerator{err: errEngineBoom} // fails before emitting anything
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":16,"stream":true}`, nil)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500 — headers must not be written until the "+
			"first token succeeds. body: %q", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "data:") {
		t.Error("a pre-first-token failure must not be reported as SSE")
	}
	decodeError(t, rec.ResponseRecorder)
}

func TestStreamEngineFailureMidStreamSendsErrorEvent(t *testing.T) {
	// Emits two pieces, then fails: the status line is already sent, so the
	// failure can only be reported in-band.
	fake := &fakeGenerator{pieces: []string{"Par", "is"}, errAfterPieces: true}
	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":16,"stream":true}`, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (already committed before the failure)", rec.Code)
	}
	events := parseSSE(t, rec.Body.String())

	var sawError, sawDone bool
	for _, ev := range events {
		if ev.Name == "error" {
			sawError = true
			var er errorResponse
			if err := json.Unmarshal([]byte(ev.Data), &er); err != nil || er.Error == "" {
				t.Errorf("error event data should be {\"error\":...}, got %q", ev.Data)
			}
		}
		if ev.Data == "[DONE]" {
			sawDone = true
		}
	}
	if !sawError {
		t.Errorf(`want an "event: error" after the tokens. body: %q`, rec.Body.String())
	}
	if sawDone {
		t.Error("[DONE] must not be sent on failure — its absence signals truncation")
	}
}

// ---------------------------------------------------------------------------
// Client disconnect
// ---------------------------------------------------------------------------

func TestStreamStopsWhenClientDisconnects(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	fake := &fakeGenerator{pieces: []string{"a", "b", "c", "d", "e", "f"}}
	// Hang up once two pieces have been handed over.
	fake.beforePiece = func(i int) {
		if i == 2 {
			cancel()
		}
	}
	defer cancel()

	rec := doStream(t, newTestServer(fake),
		`{"prompt":"hi","max_tokens":16,"stream":true}`, ctx)

	if fake.consumed > 4 {
		t.Errorf("generator produced %d of %d pieces; the callback should return "+
			"false once r.Context() is done", fake.consumed, len(fake.pieces))
	}
	if got := tokensFrom(t, parseSSE(t, rec.Body.String())); len(got) > 4 {
		t.Errorf("wrote %d token events after disconnect", len(got))
	}
}

// ---------------------------------------------------------------------------
// Regression: the non-streaming path is unchanged
// ---------------------------------------------------------------------------

func TestNonStreamingUnaffected(t *testing.T) {
	fake := &fakeGenerator{
		pieces: []string{"Par", "is", "."},
		stats:  engine.Stats{PromptTokens: 36, GeneratedTokens: 8},
	}
	rec := do(t, newTestServer(fake), http.MethodPost,
		`{"prompt":"hi","max_tokens":16}`) // stream omitted => false

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "data:") {
		t.Error("stream defaults to false; this must stay a single JSON object")
	}
	var got generateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("not JSON: %q", rec.Body.String())
	}
	if got.Text != "Paris." {
		t.Errorf("Text = %q, want %q", got.Text, "Paris.")
	}
}
