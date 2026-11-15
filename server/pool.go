package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/trungnguyen21/my-own-inference-engine/server/engine"
)

// session is the slice of *engine.Session the pool needs. Narrowing it here
// keeps the pool's own tests free of the 469 MB model: the acquire/release
// logic is pure channel work and deserves to be tested as such.
type session interface {
	Generate(prompt string, maxTokens int, temperature float32,
		onToken engine.TokenFunc) (engine.Stats, error)
	Close()
}

// ErrBusy means every session was in use for longer than the caller was
// willing to wait. Handlers map it to 503 rather than 500: the request was
// fine, the server was simply at capacity.
var ErrBusy = errors.New("all workers busy")

// sessionPool hands out exclusive use of a fixed set of engine sessions.
//
// A buffered channel is the whole mechanism. It is simultaneously:
//
//   - the pool      — it holds the idle sessions
//   - the queue     — a receive blocks while they are all out
//   - the semaphore — its capacity caps concurrency at len(sessions)
//
// That is why there is no mutex here. A session can only be held by whoever
// took it out of the channel, so exclusive use is structural rather than
// something callers have to remember.
//
// Each session owns its own KV cache, which is what makes genuine concurrency
// safe: requests never share mutable state. The weights behind them are
// read-only and shared by every session.
type sessionPool struct {
	idle    chan session
	all     []session     // retained for Close
	timeout time.Duration // how long a request may wait for a session
}

// newSessionPoolFrom wraps sessions that already exist. newSessionPool builds
// them from a model; tests supply fakes.
func newSessionPoolFrom(sessions []session, timeout time.Duration) *sessionPool {
	p := &sessionPool{
		idle:    make(chan session, len(sessions)),
		all:     sessions,
		timeout: timeout,
	}
	for _, s := range sessions {
		p.idle <- s
	}
	return p
}

// newSessionPool allocates workers sessions, each sized to contextTokens.
//
// Memory is roughly 27.6 KB per context token per worker, so the total is
// workers * contextTokens * 27.6 KB. It is acquired here, up front, because a
// pool that can allocate mid-request is a pool that can fail mid-request.
func newSessionPool(model *engine.Model, workers, contextTokens int,
	timeout time.Duration) (*sessionPool, error) {

	if workers < 1 {
		return nil, fmt.Errorf("workers must be at least 1, got %d", workers)
	}
	built := make([]session, 0, workers)
	for i := 0; i < workers; i++ {
		s, err := model.NewSession(contextTokens)
		if err != nil {
			for _, done := range built { // release the ones already built
				done.Close()
			}
			return nil, fmt.Errorf("create session %d of %d: %w", i+1, workers, err)
		}
		built = append(built, s)
	}
	return newSessionPoolFrom(built, timeout), nil
}

func (p *sessionPool) Workers() int { return cap(p.idle) }

// Close releases every session. Call it only once no request is in flight.
func (p *sessionPool) Close() {
	for _, s := range p.all {
		s.Close()
	}
	p.all = nil
}

// acquire takes a session, waiting at most p.timeout.
//
// The three-way select is the entire admission policy:
//
//	a session is free  -> proceed
//	the client hung up -> return ctx.Err(), consuming no session (important:
//	                      handing a session to a dead connection would hold it
//	                      for a full generation with nobody reading)
//	the wait expired   -> ErrBusy, which the handler turns into 503
func (p *sessionPool) acquire(ctx context.Context) (session, error) {
	// Fast path: take one without arming a timer.
	select {
	case s := <-p.idle:
		return s, nil
	default:
	}

	timer := time.NewTimer(p.timeout)
	defer timer.Stop()

	select {
	case s := <-p.idle:
		return s, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
		return nil, ErrBusy
	}
}

// release returns a session to the pool. It must run even when the caller
// panics: a session that is never returned shrinks capacity permanently and
// silently, which is the worst failure mode this type has.
func (p *sessionPool) release(s session) {
	if s == nil {
		return
	}
	select {
	case p.idle <- s:
	default:
		// Unreachable: the buffer has room for every session that exists, so
		// a full channel would mean the same session was released twice.
		log.Printf("pool: dropped a session release; capacity may be reduced")
	}
}

// Generate satisfies the generator interface, so swapping a single session for
// a pool is invisible to the HTTP handlers.
//
// The session is held for the whole call, which for a streaming request means
// the whole stream. That is the intended behaviour — one session cannot
// interleave two conversations — but it does mean a long stream occupies a
// worker for its full duration.
func (p *sessionPool) Generate(ctx context.Context, prompt string, maxTokens int,
	temperature float32, onToken engine.TokenFunc) (engine.Stats, error) {

	s, err := p.acquire(ctx)
	if err != nil {
		return engine.Stats{}, err
	}
	// Deferred so the session comes back even if Generate panics.
	defer p.release(s)

	return s.Generate(prompt, maxTokens, temperature, onToken)
}
