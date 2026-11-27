package main

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/trungnguyen21/my-own-inference-engine/server/engine"
)

// fakeSession blocks inside Generate until released, so a test can hold
// workers busy on purpose and observe the pool's admission behaviour.
type fakeSession struct {
	id       int
	block    chan struct{} // closed or sent to, to let Generate return
	delay    time.Duration // makes overlap observable in timing tests
	panics   bool
	inFlight *int32 // shared counter of concurrent Generate calls
	peak     *int32 // high-water mark of that counter
	closed   bool
}

func (f *fakeSession) Generate(prompt string, maxTokens int,
	temperature, topP, repeatPenalty float32,
	onToken engine.TokenFunc) (engine.Stats, error) {

	if f.inFlight != nil {
		now := atomic.AddInt32(f.inFlight, 1)
		for {
			old := atomic.LoadInt32(f.peak)
			if now <= old || atomic.CompareAndSwapInt32(f.peak, old, now) {
				break
			}
		}
		defer atomic.AddInt32(f.inFlight, -1)
	}
	if f.panics {
		panic("fake session panic")
	}
	if f.delay > 0 {
		time.Sleep(f.delay)
	}
	if f.block != nil {
		<-f.block
	}
	if onToken != nil {
		onToken("ok")
	}
	return engine.Stats{GeneratedTokens: 1}, nil
}

func (f *fakeSession) Close() { f.closed = true }

func newFakePool(n int, timeout time.Duration, configure func(*fakeSession)) (*sessionPool, []*fakeSession) {
	fakes := make([]*fakeSession, n)
	sessions := make([]session, n)
	for i := 0; i < n; i++ {
		fakes[i] = &fakeSession{id: i}
		if configure != nil {
			configure(fakes[i])
		}
		sessions[i] = fakes[i]
	}
	return newSessionPoolFrom(sessions, timeout), fakes
}

// --- capacity -------------------------------------------------------------

// The pool must never run more Generate calls at once than it has sessions:
// one session per in-flight request is what keeps KV caches from colliding.
func TestPoolCapsConcurrency(t *testing.T) {
	const workers, callers = 2, 8
	var inFlight, peak int32

	pool, _ := newFakePool(workers, time.Minute, func(f *fakeSession) {
		f.inFlight, f.peak = &inFlight, &peak
		f.delay = 25 * time.Millisecond // long enough for callers to overlap
	})

	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil); err != nil {
				t.Errorf("Generate: %v", err)
			}
		}()
	}
	wg.Wait()

	if peak > workers {
		t.Errorf("peak concurrency %d exceeded %d workers", peak, workers)
	}
	if peak < 2 {
		t.Errorf("peak concurrency %d — the pool is not overlapping requests at all", peak)
	}
}

// More requests than workers must all succeed: sessions are returned and
// reused, not consumed.
func TestPoolReusesSessions(t *testing.T) {
	pool, _ := newFakePool(2, time.Minute, nil)
	for i := 0; i < 10; i++ {
		if _, err := pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil); err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
	}
	if len(pool.idle) != 2 {
		t.Errorf("%d sessions idle afterwards, want 2 — one was not released", len(pool.idle))
	}
}

// --- admission control ----------------------------------------------------

func TestPoolReturnsErrBusyWhenSaturated(t *testing.T) {
	release := make(chan struct{})
	pool, _ := newFakePool(1, 50*time.Millisecond, func(f *fakeSession) {
		f.block = release
	})

	busy := make(chan struct{})
	go func() {
		close(busy)
		pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil) // holds the only session
	}()
	<-busy
	time.Sleep(20 * time.Millisecond) // let it acquire

	_, err := pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil)
	if !errors.Is(err, ErrBusy) {
		t.Errorf("err = %v, want ErrBusy once the wait expires", err)
	}
	close(release)
}

// A caller that hangs up while queued must get its context error back and
// must not tie up a session that then generates for nobody.
func TestPoolRespectsCancellationWhileQueued(t *testing.T) {
	release := make(chan struct{})
	pool, _ := newFakePool(1, time.Minute, func(f *fakeSession) {
		f.block = release
	})

	busy := make(chan struct{})
	go func() {
		close(busy)
		pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil)
	}()
	<-busy
	time.Sleep(20 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := pool.Generate(ctx, "hi", 4, 0, 1, 1, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Errorf("waited %v; cancellation should return promptly", elapsed)
	}
	close(release)
}

// --- the failure mode that silently shrinks capacity ----------------------

// A session that is never returned reduces capacity permanently. The release
// must therefore survive a panic in Generate.
func TestPoolReleasesSessionOnPanic(t *testing.T) {
	pool, _ := newFakePool(1, 50*time.Millisecond, func(f *fakeSession) {
		f.panics = true
	})

	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected the panic to propagate to the caller")
			}
		}()
		pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil)
	}()

	if len(pool.idle) != 1 {
		t.Fatal("session was not returned after the panic; capacity is now permanently 0")
	}
	// Prove the pool still works.
	pool.all[0].(*fakeSession).panics = false
	if _, err := pool.Generate(context.Background(), "hi", 4, 0, 1, 1, nil); err != nil {
		t.Errorf("pool unusable after a panic: %v", err)
	}
}

// --- construction ---------------------------------------------------------

func TestNewSessionPoolRejectsZeroWorkers(t *testing.T) {
	if _, err := newSessionPool(nil, 0, 512, time.Second); err == nil {
		t.Error("workers = 0 should be rejected")
	}
}

func TestPoolCloseClosesEverySession(t *testing.T) {
	pool, fakes := newFakePool(3, time.Minute, nil)
	pool.Close()
	for i, f := range fakes {
		if !f.closed {
			t.Errorf("session %d was not closed", i)
		}
	}
}

// --- the HTTP mapping -----------------------------------------------------

func TestHandlerMapsErrBusyTo503(t *testing.T) {
	srv := newTestServer(&fakeGenerator{err: ErrBusy})
	srv.workers = 4
	rec := do(t, srv, http.MethodPost, `{"prompt":"hi","max_tokens":8}`)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 — a saturated server is not a broken one. body: %s",
			rec.Code, rec.Body.String())
	}
	if ra := rec.Header().Get("Retry-After"); ra == "" {
		t.Error("want a Retry-After header so the client knows to retry")
	}
	decodeError(t, rec)
}

func TestHandlerStaysSilentWhenClientAbandonedRequest(t *testing.T) {
	srv := newTestServer(&fakeGenerator{err: context.Canceled})
	rec := do(t, srv, http.MethodPost, `{"prompt":"hi","max_tokens":8}`)

	// Nothing was written, so the recorder keeps its default 200: there is no
	// client left to receive a status.
	if body := rec.Body.String(); body != "" {
		t.Errorf("wrote %q to an abandoned connection", body)
	}
}
