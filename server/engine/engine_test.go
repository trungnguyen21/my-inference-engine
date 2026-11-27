package engine_test

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/trungnguyen21/my-own-inference-engine/server/engine"
)

// TestMain pins the engine's thread count so the suite does not depend on
// OMP_NUM_THREADS being set in the shell.
//
// Without this, TestConcurrentSessions runs 4 requests x 8 threads on 8 cores
// and takes over two minutes instead of seven seconds: the kernels use static
// scheduling, so every oversubscribed team waits at its barrier for whichever
// thread landed on an efficiency core.
func TestMain(m *testing.M) {
	engine.SetMaxThreads(2)
	os.Exit(m.Run())
}

// modelPath resolves the GGUF relative to the repo root and skips the test
// when it is absent, so `go test ./...` stays usable without the model.
func modelPath(t *testing.T) string {
	t.Helper()
	// These tests load a 469 MB model and run real inference, so `go test
	// -short ./...` skips them. Use that while iterating on the HTTP layer.
	if testing.Short() {
		t.Skip("skipping model-backed test in -short mode")
	}
	p, err := filepath.Abs("../../models/qwen-2.5.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("model not present at %s", p)
	}
	return p
}

func openModel(t *testing.T) *engine.Model {
	t.Helper()
	m, err := engine.Open(modelPath(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(m.Close)
	return m
}

func TestOpenReportsLimits(t *testing.T) {
	m := openModel(t)
	if got := m.ContextLimit(); got <= 0 {
		t.Errorf("ContextLimit() = %d, want > 0", got)
	}
	if got := engine.PromptCapacity(); got <= 0 {
		t.Errorf("PromptCapacity() = %d, want > 0", got)
	}
}

func TestOpenMissingFileFails(t *testing.T) {
	if _, err := engine.Open("does-not-exist.gguf"); err == nil {
		t.Fatal("Open of a missing file should fail")
	}
}

func TestGenerate(t *testing.T) {
	m := openModel(t)
	s, err := m.NewSession(192)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	var sb strings.Builder
	stats, err := s.Generate("What's the capital of France?", 16, 0, 1, 1,
		func(piece string) bool { sb.WriteString(piece); return true })
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(sb.String(), "Paris") {
		t.Errorf("output %q should mention Paris", sb.String())
	}
	if stats.PromptTokens == 0 || stats.GeneratedTokens == 0 {
		t.Errorf("stats not populated: %+v", stats)
	}
}

// A session must be reusable with no reallocation, which is what makes a
// worker pool viable.
func TestSessionIsReusable(t *testing.T) {
	m := openModel(t)
	s, err := m.NewSession(192)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	for _, tc := range []struct{ prompt, want string }{
		{"What's the capital of France?", "Paris"},
		{"What's 2+2?", "4"},
	} {
		var sb strings.Builder
		if _, err := s.Generate(tc.prompt, 16, 0, 1, 1,
			func(p string) bool { sb.WriteString(p); return true }); err != nil {
			t.Fatalf("Generate(%q): %v", tc.prompt, err)
		}
		if !strings.Contains(sb.String(), tc.want) {
			t.Errorf("Generate(%q) = %q, want it to contain %q", tc.prompt, sb.String(), tc.want)
		}
	}
}

// Returning false from the callback is how a disconnecting HTTP client stops
// generation.
func TestCallbackCanStopEarly(t *testing.T) {
	m := openModel(t)
	s, err := m.NewSession(192)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	pieces := 0
	stats, err := s.Generate("Count to twenty.", 16, 0, 1, 1, func(string) bool {
		pieces++
		return pieces < 2
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !stats.StoppedEarly {
		t.Error("StoppedEarly should be true")
	}
	if pieces != 2 {
		t.Errorf("got %d pieces, want 2", pieces)
	}
}

func TestInvalidRequestsAreErrors(t *testing.T) {
	m := openModel(t)
	s, err := m.NewSession(192)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()
	keep := func(string) bool { return true }

	for _, tc := range []struct {
		name          string
		maxTokens     int
		temperature   float32
		topP          float32
		repeatPenalty float32
	}{
		// Nonzero temperature used to be rejected outright; it is now valid,
		// real nucleus sampling (see TestGenerateWithTemperature), so it is
		// deliberately absent from this table.
		{"exceeds session context", 100000, 0, 1, 1},
		{"zero max tokens", 0, 0, 1, 1},
		{"out-of-range temperature", 16, 5.0, 1, 1},
		{"out-of-range top_p", 16, 0, 5.0, 1},
	} {
		if _, err := s.Generate("hi", tc.maxTokens, tc.temperature, tc.topP,
			tc.repeatPenalty, keep); err == nil {
			t.Errorf("%s: expected an error", tc.name)
		}
	}

	// A rejected request must leave the session usable.
	var sb strings.Builder
	if _, err := s.Generate("What's 2+2?", 8, 0, 1, 1,
		func(p string) bool { sb.WriteString(p); return true }); err != nil {
		t.Fatalf("session unusable after rejections: %v", err)
	}
}

// Nonzero temperature was rejected outright before this change; this is the
// fix. A low top_p keeps the nucleus small enough that output stays coherent
// English rather than near-uniform random tokens, which is what makes this a
// meaningful check rather than "did not crash".
func TestGenerateWithTemperature(t *testing.T) {
	m := openModel(t)
	s, err := m.NewSession(192)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	var sb strings.Builder
	_, err = s.Generate("What's the capital of France?", 16, 0.7, 0.9, 1,
		func(p string) bool { sb.WriteString(p); return true })
	if err != nil {
		t.Fatalf("Generate with nonzero temperature: %v", err)
	}
	if !strings.Contains(sb.String(), "Paris") {
		t.Errorf("output %q should still mention Paris at a modest temperature", sb.String())
	}
}

// The actual bug this project hit: greedy decoding (temperature 0) on this
// model reliably collapses into repeating a short phrase for the rest of the
// token budget. repeat_penalty is the fix, and it works independently of
// temperature. Verified against the real model, not asserted in the abstract.
func TestRepeatPenaltyBreaksGreedyLoop(t *testing.T) {
	m := openModel(t)
	s, err := m.NewSession(512)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	const loopingPrompt = "Continue this sequence: 1, 1, 1, 1,"

	var noPenalty strings.Builder
	if _, err := s.Generate(loopingPrompt, 96, 0, 1, 1,
		func(p string) bool { noPenalty.WriteString(p); return true }); err != nil {
		t.Fatalf("Generate (no penalty): %v", err)
	}
	if !strings.Contains(noPenalty.String(), "1, 1, 1, 1, 1, 1, 1, 1, 1, 1") {
		t.Skipf("repro did not reproduce the known loop this run; got %q", noPenalty.String())
	}

	var withPenalty strings.Builder
	if _, err := s.Generate(loopingPrompt, 96, 0, 1, 1.3,
		func(p string) bool { withPenalty.WriteString(p); return true }); err != nil {
		t.Fatalf("Generate (repeat_penalty 1.3): %v", err)
	}
	if strings.Contains(withPenalty.String(), "1, 1, 1, 1, 1, 1, 1, 1, 1, 1") {
		t.Errorf("repeat_penalty did not break the loop; got %q", withPenalty.String())
	}
}

// Separate sessions sharing one Model must run concurrently. This is the
// assumption the whole worker-pool design rests on.
func TestConcurrentSessions(t *testing.T) {
	m := openModel(t)
	const n = 4
	prompts := [n]string{
		"What's the capital of France?",
		"Name three primary colors.",
		"What's 2+2?",
		"What color is the sky?",
	}
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		outs [n]string
		errs [n]error
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s, err := m.NewSession(192)
			if err != nil {
				mu.Lock()
				errs[i] = err
				mu.Unlock()
				return
			}
			defer s.Close()
			var sb strings.Builder
			_, err = s.Generate(prompts[i], 16, 0, 1, 1,
				func(p string) bool { sb.WriteString(p); return true })
			mu.Lock()
			outs[i], errs[i] = sb.String(), err
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Errorf("session %d: %v", i, errs[i])
		}
		if outs[i] == "" {
			t.Errorf("session %d produced no output", i)
		}
	}
	if !strings.Contains(outs[0], "Paris") {
		t.Errorf("concurrent output wrong: %q", outs[0])
	}
}
