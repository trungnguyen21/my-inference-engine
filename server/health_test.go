package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// get calls a handler with a GET and returns the recorder.
func get(t *testing.T, h http.HandlerFunc, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec
}

func statusField(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("body is not JSON: %q", rec.Body.String())
	}
	s, _ := body["status"].(string)
	return s
}

// Liveness must never depend on the model. A liveness probe that fails during
// the multi-second load would have Kubernetes kill the container before it
// ever finished starting.
func TestLivenessIsIndependentOfTheModel(t *testing.T) {
	loading := &server{} // gen is nil: still loading
	rec := get(t, loading.handleLive, "/health/live")
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 while still loading", rec.Code)
	}
	if got := statusField(t, rec); got != "alive" {
		t.Errorf("status field = %q, want %q", got, "alive")
	}
}

// Readiness is what keeps traffic off a pod that cannot serve yet.
func TestReadinessFlipsOnceLoaded(t *testing.T) {
	srv := &server{}

	rec := get(t, srv.handleReady, "/health/ready")
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 before the model is loaded", rec.Code)
	}
	if got := statusField(t, rec); got != "loading" {
		t.Errorf("status field = %q, want %q", got, "loading")
	}

	srv.ready(&fakeGenerator{}, 4, 32768, 2)

	rec = get(t, srv.handleReady, "/health/ready")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 once loaded", rec.Code)
	}
	if got := statusField(t, rec); got != "ready" {
		t.Errorf("status field = %q, want %q", got, "ready")
	}
}

// A request arriving during startup must be refused, not crash on a nil engine.
func TestGenerateIsRefusedWhileLoading(t *testing.T) {
	srv := &server{promptCapacity: 4096, maxTokensLimit: 512, sessionContext: 1024}
	rec := do(t, srv, http.MethodPost, `{"prompt":"hi","max_tokens":8}`)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 while loading. body: %s", rec.Code, rec.Body.String())
	}
	if ra := rec.Header().Get("Retry-After"); ra == "" {
		t.Error("want Retry-After so a client knows to come back")
	}
	if msg := decodeError(t, rec); !strings.Contains(msg, "loading") {
		t.Errorf("error = %q, should say the model is still loading", msg)
	}
}

// The combined /health endpoint reports the same distinction.
func TestHealthReportsLoadingState(t *testing.T) {
	srv := &server{sessionContext: 1024}

	rec := get(t, srv.handleHealth, "/health")
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 while loading", rec.Code)
	}
	if got := statusField(t, rec); got != "loading" {
		t.Errorf("status field = %q, want %q", got, "loading")
	}

	srv.ready(&fakeGenerator{}, 4, 32768, 2)
	rec = get(t, srv.handleHealth, "/health")
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 once loaded", rec.Code)
	}
	var body map[string]any
	json.Unmarshal(rec.Body.Bytes(), &body)
	if body["workers"] != float64(4) {
		t.Errorf("workers = %v, want 4", body["workers"])
	}
	if body["omp_threads"] != float64(2) {
		t.Errorf("omp_threads = %v, want 2", body["omp_threads"])
	}
}
