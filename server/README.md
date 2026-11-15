# Go HTTP server

Phases 1-3 of `docs/serving-implementation-plan.md`. One endpoint, optional
streaming via `"stream": true`, and a pool of N sessions so requests overlap.

## What is already done vs. what you write

| | Status |
|---|---|
| `engine/` — cgo bindings to the C++ engine | **done & tested** |
| `main.go` — wiring, types, `/health` | **done**, read as a worked example |
| `main.go` — `handleGenerate`, `streamGenerate` | **done** |
| `API.md` — the contract | the spec you implement against |
| `main_test.go` — the tests | your checklist; green = done |

The cgo layer is given to you deliberately. A mistake there is memory
corruption, not a compiler error — it is cgo trivia rather than Go skill. Read
`engine/engine.go` to see how it works; you should not need to change it.

## Build and run

```sh
# from the repository root — the Go build needs libengine to exist first
cmake --build build --target engine

cd server
go test -short ./...     # fast: no model loaded. Your edit-test loop.
go test ./...            # slow (~25 s): real inference through cgo
go run . -model ../models/qwen-2.5.gguf
```

Then, in another terminal:

```sh
curl -s localhost:8080/health | jq
curl -s localhost:8080/generate -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of France?","max_tokens":32}' | jq
```

Useful flags: `-addr`, `-context` (session size in tokens, ~27.6 KB each),
`-max-tokens-limit`.

### OpenMP threads

The engine's kernels are OpenMP-parallel, and that happens inside C++ where Go
cannot see it. The server reports the resolved value at startup and on
`/health`:

```
OpenMP: 8 threads per request (OMP_NUM_THREADS unset, using all cores)
```

Measured on this machine, single-request generation peaks around **4 threads**
and gets *slower* at 8 — the matvec kernels are memory-bandwidth-bound, so
extra threads only add contention:

| `OMP_NUM_THREADS` | one 16-token request |
|---|---|
| 1 | 5.62 s |
| 2 | 3.66 s |
| 4 | 2.11 s |
| 8 | 2.83 s |

Why 8 is slower: this is an M2 with **4 performance and 4 efficiency cores**,
and the kernels use `schedule(static)` — work is split evenly up front, so the
team waits at its barrier for whichever thread landed on a slow core.

`OMP_NUM_THREADS` and `GOMAXPROCS` are independent: Go schedules goroutines,
OpenMP schedules kernel threads, and they do not coordinate. The total is
`concurrent requests x threads each`, so 4 workers at the default 8 would put
32 threads on 8 cores.

### Pinning it

Prefer the flag over the environment variable:

```sh
go run . -threads 4
```

`engine.SetMaxThreads(n)` is the underlying call. It must be used rather than
setting `OMP_NUM_THREADS` from Go, because OpenMP's thread count is *per-thread*
state: the value is applied inside each `Generate`, on the goroutine's OS
thread, so it reaches every worker.

`engine/engine_test.go` has a `TestMain` that pins it to 2. Without that,
`go test ./...` ran 4 concurrent requests x 8 threads and took **over two
minutes** instead of ~17 s.

### Concurrency does not make this faster

Measured, 4 requests x 16 tokens:

| Configuration | Wall time |
|---|---|
| Sequential, 4 threads | 5.4-5.9 s |
| 4 concurrent, 1 thread each | 6.0 s |

Four at once takes the same total time as four in a row. Concurrency changes
*when* each client gets its answer (all at ~6 s, rather than the first at 1.4 s
and the last at 5.9 s), not how much work the machine does. Every token streams
the full ~400 MB of weights from RAM, so four requests need four times the
memory bandwidth — and bandwidth was already the bottleneck.

The fix for throughput is batching (one weight read serving N requests), which
is an engine change, not a server change.

## The Go you need for `handleGenerate`

Only these, roughly in the order you will use them.

**Handler signature.** Every HTTP handler is `func(w http.ResponseWriter, r *http.Request)`.
You write to `w`; you read the request from `r`. There is no return value, so
"responding" means writing, and **writing twice is a bug** — always `return`
after you have written a response.

**Decoding JSON.**

```go
var req generateRequest
if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
    // malformed JSON is the client's fault: 400
}
```

`&req` passes a pointer so Decode can fill the struct in place. Fields absent
from the JSON are left at their zero value — `""` for strings, `0` for ints —
which is exactly why "missing `max_tokens`" and "`max_tokens: 0`" are the same
case to you.

**Errors are values, not exceptions.** This is the biggest difference from most
languages, and it will feel repetitive. It is idiomatic:

```go
stats, err := s.gen.Generate(...)
if err != nil {
    // handle it here, now
}
```

There is no `try`/`catch`. The `if err != nil` block right after every call *is*
the error handling. It mirrors the `int` return plus error buffer your C++
engine already uses.

**Building the output string.**

```go
var sb strings.Builder
// ... sb.WriteString(piece) for each piece ...
text := sb.String()
```

Avoid `text += piece` in a loop: Go strings are immutable, so that reallocates
and copies on every token. `strings.Builder` appends into one growing buffer.

**Closures capture variables.** The callback you hand to `Generate` is a
function literal that can see `sb` from the enclosing scope:

```go
func(piece string) bool {
    sb.WriteString(piece)
    return true   // false stops generation
}
```

**Timing.**

```go
start := time.Now()
// ... work ...
elapsed := time.Since(start).Milliseconds()
```

**Methods on a struct.** `func (s *server) handleGenerate(...)` makes
`handleGenerate` a method on `*server`, so `s.gen` and `s.maxTokensLimit` are
available. The `s` is the receiver — Go's equivalent of `this`, except you name
it yourself and it is conventionally one letter.

## Two things worth understanding, not just using

**Why `generator` is an interface.** `main.go` declares:

```go
type generator interface {
    Generate(prompt string, maxTokens int, temperature float32,
             onToken engine.TokenFunc) (engine.Stats, error)
}
```

`*engine.Session` satisfies it **without declaring that it does** — Go
interfaces are structural, so any type with a matching method set qualifies.
That is why `main_test.go` can inject a `fakeGenerator` and run every test in
milliseconds instead of loading a 469 MB model. Depending on the narrow
interface you need, rather than the concrete type, is one of the most
load-bearing habits in Go.

**Why the response-writing order matters.**

```go
w.Header().Set("Content-Type", "application/json")  // headers first
w.WriteHeader(status)                               // then status
json.NewEncoder(w).Encode(payload)                  // then body
```

Writing the body first makes Go send an implicit `200`, and your intended
status code is silently discarded. `writeJSON` in `main.go` already does this
correctly — use it.

## Hints, if you get stuck

<details>
<summary>Shape of the handler</summary>

```go
func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost { ...405...; return }

    var req generateRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil { ...400...; return }

    // validate: empty/whitespace prompt, prompt too long, max_tokens range

    start := time.Now()
    var sb strings.Builder
    stats, err := s.gen.Generate(req.Prompt, req.MaxTokens, 0, func(piece string) bool {
        sb.WriteString(piece)
        return true
    })
    if err != nil { ...500...; return }

    writeJSON(w, http.StatusOK, generateResponse{ ... })
}
```
</details>

<details>
<summary>Why does the engine take temperature at all, if it must be 0?</summary>

Because real sampling is a planned feature and the parameter marks where it
goes. Until then the engine *rejects* a nonzero value rather than ignoring it,
so a caller can never silently get greedy decoding when they asked for
sampling. Pass `0` and do not expose the field in your API — see `API.md`.
</details>

## Known rough edges you will meet

- **`generated_tokens` can exceed the visible tokens by one.** A terminating
  EOS is counted but never emitted, so `text` may hold 7 tokens' worth of
  characters while `generated_tokens` says 8. Report it as-is; it matches the
  engine.
- **Prompt length is in bytes, not characters.** `len(someString)` in Go counts
  bytes, which is what `engine.PromptCapacity()` bounds. That is correct here —
  do not reach for `utf8.RuneCountInString`.
- **Cancellation is not wired up yet.** Returning `false` from the callback
  stops generation, and `r.Context().Done()` tells you the client hung up, but
  connecting the two is Phase 2 work along with streaming.

## Streaming

```sh
curl -N -s localhost:8080/generate -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of France?","max_tokens":24,"stream":true}'
```

`-N` disables curl's own buffering; without it you cannot tell whether the
server is streaming. See `API.md` for the wire format.

Two constraints in `streamGenerate` are worth preserving if you refactor it:

1. **The SSE header is not written until the first token arrives.** Once a
   status line is sent it cannot be changed, so writing it up front would turn
   every early failure into a `200` carrying an error body. `wroteHeader`
   tracks this, and it is why the error branch has two cases.
2. **`data: [DONE]` is only sent on success.** Its absence is how a client
   detects truncation, so a mid-stream failure emits `event: error` and no
   terminator.

Client disconnect is handled by returning `false` from the token callback once
`r.Context().Done()` closes. The engine honours it between tokens, so an
abandoned request frees its session within one token instead of running to
completion.

## Workers

```sh
go run . -workers 4                 # or WORKERS=4
go run . -workers 4 -threads 2      # override the derived thread count
```

Each worker owns one `engine.Session`, and therefore one KV cache — which is
what makes concurrency safe. Sharing a single session is what produced
`",,,,,,......"` at HTTP 200 before this phase.

`pool.go` is a buffered channel of sessions, which is at once the pool (it
holds idle sessions), the queue (a receive blocks while they are all out) and
the semaphore (its capacity caps concurrency). That is why there is no mutex:
a session can only be held by whoever took it out of the channel.

Three behaviours to know:

- **Threads are derived from `cores/workers`** unless `-threads` says
  otherwise, because `workers x threads` oversubscription is a severe cliff
  rather than a gentle slope. A bad explicit combination logs a warning.
- **A saturated server returns 503 with `Retry-After`**, after waiting up to
  `-queue-wait` (default 30s). At capacity is not the same as broken.
- **A client that disconnects while queued consumes no worker** — the acquire
  selects on the request context too.

Remember the measured caveat above: concurrency redistributes latency, it does
not add throughput on this hardware. What it buys is time-to-first-token, no
head-of-line blocking, and real backpressure.

## Next

Phase 4: container. Multi-stage Dockerfile, `/health/live` + `/health/ready`
split so Kubernetes can tell "process up" from "model loaded", and a decision
on how the model reaches the image.
