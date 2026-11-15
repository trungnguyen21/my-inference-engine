// Package engine wraps the C++ inference engine behind a Go API.
//
// This file is the only place in the project that touches C. Everything it
// exports is ordinary Go: strings, errors, and a callback. If you find yourself
// writing C types anywhere else, something belongs in here instead.
//
// Build requirements: libengine must exist in ../build. Run from the repo root:
//
//	cmake --build build --target engine
package engine

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../build -lengine -Wl,-rpath,${SRCDIR}/../../build

#include <stdint.h>
#include <stdlib.h>
#include "engine_api.h"

// goEngineOnToken is implemented in Go (see callback.go). It takes the user
// pointer as uintptr_t rather than void*, which is the documented way to carry
// a cgo.Handle across the boundary: a handle is an integer, and converting an
// integer to unsafe.Pointer is exactly the misuse `go vet` warns about.
extern int goEngineOnToken(const char *text, uintptr_t user);

// The engine's callback type takes void*, so this trampoline adapts it.
static int token_trampoline(const char *text, void *user) {
	return goEngineOnToken(text, (uintptr_t)user);
}

// engine_run cannot be called directly from Go, because Go cannot form a C
// function pointer to an exported Go function at the call site. This tiny
// shim hard-wires the callback instead, so Go only passes data.
static int engine_run_bridge(EngineSession *session, const char *prompt,
                             int max_tokens, float temperature, uintptr_t user,
                             EngineStats *stats, char *error, int cap) {
	return engine_run(session, prompt, max_tokens, temperature,
	                  token_trampoline, (void *)user, stats, error, cap);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime/cgo"
	"unsafe"
)

// Model holds the loaded weights and tokenizer. It is read-only once opened,
// so any number of Sessions on any number of goroutines may share one Model.
type Model struct {
	handle *C.EngineHandle
}

// Session holds one request's mutable state: KV cache, activations, logits.
//
// A Session is NOT safe for concurrent use. One goroutine at a time. It is
// reusable across any number of sequential requests, with no reallocation.
type Session struct {
	ptr   *C.EngineSession
	model *Model // kept so the Model cannot be collected while in use
}

// Stats reports what one Generate call did.
type Stats struct {
	PromptTokens    int
	GeneratedTokens int // counts a terminating EOS, which is never emitted
	StoppedEarly    bool
	PrefillSeconds  float64
	GenerateSeconds float64
	// FirstTokenSeconds is -1 when the request produced no text.
	FirstTokenSeconds float64
}

// errorBuffer allocates a C-writable buffer sized by the engine, plus a
// function to read it back as a Go error.
//
// The buffer is C memory, not Go memory: passing a pointer into a Go slice to C
// is legal only while the call runs, and we would rather not depend on that
// subtlety. C.free is deferred by the caller.
func errorBuffer() (buf *C.char, size C.int, free func()) {
	size = C.engine_error_capacity()
	buf = (*C.char)(C.calloc(C.size_t(size), 1))
	return buf, size, func() { C.free(unsafe.Pointer(buf)) }
}

func bufferError(buf *C.char, fallback string) error {
	if msg := C.GoString(buf); msg != "" {
		return errors.New(msg)
	}
	return errors.New(fallback)
}

// Open loads a GGUF model. The returned Model must be Closed.
//
// Open is not safe to call concurrently with itself or with Close; the
// underlying tokenizer keeps a non-atomic refcount. Open every model during
// startup, before serving traffic.
func Open(ggufPath string) (*Model, error) {
	errBuf, errCap, freeErr := errorBuffer()
	defer freeErr()

	// C.CString allocates with malloc and copies. Go's garbage collector knows
	// nothing about it, so we must free it ourselves.
	cPath := C.CString(ggufPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.engine_open(cPath, errBuf, errCap)
	if handle == nil {
		return nil, fmt.Errorf("open model %q: %w", ggufPath,
			bufferError(errBuf, "unknown engine failure"))
	}
	return &Model{handle: handle}, nil
}

// Close releases the model. Every Session created from it must be closed first.
func (m *Model) Close() {
	if m == nil || m.handle == nil {
		return
	}
	C.engine_close(m.handle)
	m.handle = nil
}

// ContextLimit is the model's own maximum context, in tokens.
func (m *Model) ContextLimit() int {
	if m == nil || m.handle == nil {
		return 0
	}
	return int(C.engine_context_limit(m.handle))
}

// PromptCapacity is the longest prompt Generate accepts, in bytes of UTF-8.
// Checking this yourself lets you reject an oversized request without entering
// the engine at all.
func PromptCapacity() int {
	return int(C.engine_prompt_capacity())
}

// MaxThreads is how many threads one Generate call may use inside the engine's
// kernels, as OpenMP resolved it: OMP_NUM_THREADS when set, otherwise the core
// count.
//
// This is per call, not per process. Running N requests concurrently uses up to
// N*MaxThreads() threads, which is independent of GOMAXPROCS — Go's scheduler
// and OpenMP's thread pool do not coordinate.
func MaxThreads() int {
	return int(C.engine_max_threads())
}

// SetMaxThreads requests how many threads each Generate call uses inside the
// engine's kernels, overriding OMP_NUM_THREADS. Pass 0 to restore the
// environment's default. It returns the count that will be used.
//
// Call this during startup, before serving. Because OpenMP's thread count is
// per-thread state, the value is applied inside each Generate on the goroutine's
// OS thread — setting it once here is enough even when requests run on many
// different threads.
//
// On a machine with heterogeneous cores (Apple Silicon has performance and
// efficiency cores), more threads is not reliably faster: the kernels use
// static scheduling, so a team waits at its barrier for the slowest core.
// Measured on an M2, 4 threads beat 8 by roughly 2x for one request.
func SetMaxThreads(threads int) int {
	return int(C.engine_set_max_threads(C.int(threads)))
}

// NewSession allocates reusable state for up to maxContextTokens of prompt
// plus generated tokens. All memory a request needs is acquired here, so
// Generate allocates nothing.
//
// Memory cost is roughly 27.6 KB per context token.
func (m *Model) NewSession(maxContextTokens int) (*Session, error) {
	if m == nil || m.handle == nil {
		return nil, errors.New("model is not open")
	}
	errBuf, errCap, freeErr := errorBuffer()
	defer freeErr()

	ptr := C.engine_session_create(m.handle, C.int(maxContextTokens), errBuf, errCap)
	if ptr == nil {
		return nil, fmt.Errorf("create session: %w",
			bufferError(errBuf, "unknown engine failure"))
	}
	return &Session{ptr: ptr, model: m}, nil
}

// Close releases the session's buffers.
func (s *Session) Close() {
	if s == nil || s.ptr == nil {
		return
	}
	C.engine_session_destroy(s.ptr)
	s.ptr = nil
}

// TokenFunc receives each decoded piece of output as it is produced.
//
// IMPORTANT: a piece is a byte fragment, not a character. One multi-byte
// character (an emoji, any CJK text) can be split across two calls, so
// accumulate bytes and decode incrementally rather than treating each piece as
// a complete string.
//
// Return false to stop generating. Stopping is not an error; Stats.StoppedEarly
// will be true.
type TokenFunc func(piece string) bool

// Generate runs one completion, calling onToken for each piece.
//
// The session is rewound first, so a session may serve unlimited sequential
// requests. temperature must be 0: only greedy decoding is implemented, and a
// nonzero value is rejected rather than silently ignored.
//
// Not safe for concurrent use on one Session.
func (s *Session) Generate(prompt string, maxTokens int, temperature float32,
	onToken TokenFunc) (Stats, error) {

	if s == nil || s.ptr == nil {
		return Stats{}, errors.New("session is closed")
	}
	if onToken == nil {
		return Stats{}, errors.New("onToken must not be nil")
	}

	errBuf, errCap, freeErr := errorBuffer()
	defer freeErr()

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// A Go pointer may not be stored in C memory, so the callback's user data
	// travels as a cgo.Handle: an opaque integer the Go runtime can map back.
	h := cgo.NewHandle(onToken)
	defer h.Delete()

	var cStats C.EngineStats
	ok := C.engine_run_bridge(s.ptr, cPrompt, C.int(maxTokens),
		C.float(temperature), C.uintptr_t(h),
		&cStats, errBuf, errCap)

	stats := Stats{
		PromptTokens:      int(cStats.prompt_tokens),
		GeneratedTokens:   int(cStats.generated_tokens),
		StoppedEarly:      cStats.stopped_early != 0,
		PrefillSeconds:    float64(cStats.prefill_seconds),
		GenerateSeconds:   float64(cStats.generate_seconds),
		FirstTokenSeconds: float64(cStats.first_token_seconds),
	}
	if ok == 0 {
		return stats, bufferError(errBuf, "unknown engine failure")
	}
	return stats, nil
}
