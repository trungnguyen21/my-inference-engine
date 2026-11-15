# Serving the engine over HTTP — implementation plan

Goal: expose the C++ inference engine as an HTTP API, then deploy it on
Kubernetes. Built as a portfolio piece demonstrating deployment, serving, and
distributed-systems practice.

Two demos define "done":

- **Demo A — concurrency.** Four requests streaming at the same time, visibly
  in parallel, each labelled with which pod served it.
- **Demo C — resilience.** Delete a pod mid-request; Kubernetes replaces it and
  traffic continues on the survivors.

Throughput/latency benchmarking is explicitly **out of scope for now**.

---

## Facts this plan is built on

Measured or read from the code, not assumed:

| Fact | Value | Source |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct, Q4_K_M | GGUF metadata |
| Layers / dim / KV heads | 24 / 896 / 2 (head_dim 64) | GGUF metadata |
| Model file | 469 MB, `mmap`'d read-only | `src/gguf_reader.cpp:695` |
| Cold start (load → ready) | **4.2 s** | measured |
| Memory per in-flight request | **~27.6 KB per context token** | KV cache = 2 x 24 x 128 x 4B |
| Machine | 8 cores, 8 GB RAM, arm64 | `sysctl` |

Three properties make concurrent serving possible without touching the kernels:

1. **Weights are read-only and shared.** Every forward function takes
   `const Model *`. Many concurrent requests share one 469 MB copy.
2. **OpenMP is only `parallel for` inside leaf kernels** (`src/quantized_matvec.cpp:153`,
   `src/tensor.cpp:62`) — no shared mutable state, no nested regions. Concurrent
   calls on separate `InferenceState`s are safe.
3. **Resetting a request's state is free.** `token_count` (`src/qwen2_internal.h:24`)
   is the only position marker, and attention length derives from it. Setting it
   to 0 reuses the buffers — no re-allocation between requests.

---

## Phase 0 — Make the engine callable as a library  ✅ DONE

*C++ work only. No Go yet.*

Today everything lives in one executable and generation is welded to the
terminal (`app_run`, `src/app.cpp:142`) — it prints tokens directly and exits
the process on error. Neither works behind a server.

**Tasks**

1. Add `include/engine_api.h` + `src/engine_api.cpp` — a small `extern "C"`
   wrapper exposing *opaque handles* (`EngineModel*`, `EngineWorker*`) rather
   than structs. This means the Go side never has to mirror any struct layout,
   which is the most fragile part of calling C++ from another language.
2. Functions: open/close a model, create/destroy a worker, and `engine_generate`.
3. `engine_generate` is `app_run` with three changes: reset `token_count` instead
   of allocating; hand each token to a **callback** instead of printing it; return
   an error string instead of printing and exiting.
4. Split CMake so sources are shared between the existing `inference` executable
   and a new `engine` shared library.
5. Repoint the existing CLI at the new function.

**Done when:** `./build/inference` behaves exactly as before, and `libengine.dylib` exists.

**Outcome.** All 22 existing tests pass and the CLI produces identical output.
Delivered:

- `include/engine_api.h` — the C ABI: opaque `EngineHandle` (shared, read-only
  weights) and `EngineSession` (one request's mutable state).
- `src/generate.cpp` — the one generation loop, shared by the CLI and the ABI,
  so the two front ends cannot drift apart.
- `inference_state_reset` — rewinds a session for reuse with no allocation.
- `libengine.dylib` via `add_library(engine SHARED ...)`.
- `tests/engine_api_smoke_test.c` — compiled as **C** on purpose, which is what
  proves the header is consumable from cgo. Not a ctest (needs the model file):
  `cmake --build build --target engine_api_smoke_test && ./build/engine_api_smoke_test`

**Verified beyond the plan:** four sessions generating *concurrently* on one
shared handle produce correct output (62 s CPU in 11.6 s wall — 5.4x
parallelism). That was the central unproven assumption behind Phase 3, and it
holds. Session reuse, early cancellation, and rejection-then-reuse are all
covered by the smoke test.

> The CLI still working is the regression test. If output changed, the refactor broke something.

**Watch out for:** server-side validation. The CLI currently exits the process when
the prompt is too long (`src/app.cpp:206`). As a library it must return an error
the server can turn into a 400.

---

## Phase 1 — Simplest possible endpoint

*One request at a time. No streaming. No concurrency.*

```
POST /generate
{ "prompt": "What is the capital of France?", "max_tokens": 64 }

-> { "text": "Paris.", "tokens": 3, "ms": 820 }
```

**Tasks**

1. Create `server/` — a Go module with `net/http` from the standard library.
2. Write the cgo bindings: a Go file that declares the C functions and wraps
   them in normal-looking Go functions. This is the fiddliest file in the
   project; it is written once and rarely touched.
3. Load the model once at startup; keep one worker.
4. One endpoint: validate input, call the engine, return JSON.
5. Reject bad input with 400 — prompt too long, `max_tokens` out of range.

**Done when:** `curl` against a locally running server returns generated text.

**Watch out for:**

- **`temperature` must not be in the API yet.** The sampler only implements
  greedy decoding; a nonzero temperature errors out. Hardcode 0.
- **Memory across the boundary is manual.** Strings passed to C are allocated
  with `C.CString` and must be freed with `defer C.free(...)`. Forgetting this
  leaks on every request.
- **Build setup.** The Go code needs to find the headers and `libengine`, via
  `#cgo CFLAGS`/`LDFLAGS` directives. Expect to fight this once, then never again.

## Phase 2 — Streaming  ✅ DONE

Same endpoint, `"stream": true` switches it to Server-Sent Events — tokens
appear one at a time, like ChatGPT.

**Tasks**

1. Pass a Go function into C++ as the token callback. cgo requires this to be
   marked `//export` and to live in a *different file* from the one holding the
   `import "C"` preamble — a quirk that causes confusing errors otherwise.
2. Use `cgo.Handle` to carry per-request state through the callback. (Raw Go
   pointers may not be passed into C; `cgo.Handle` is the supported way.)
3. Each token flows from the callback into a channel, and from there to the
   HTTP response.
4. **Flush after every token** using `http.Flusher`. Without this the server
   buffers and the whole answer arrives at once — which would silently defeat
   Demo A.
5. Handle client disconnect via `r.Context().Done()`: the callback returns a
   "stop" signal so generation ends instead of burning CPU on a dead connection.

**Done when:** `curl -N` shows words appearing progressively.

**Outcome.** Verified against the real engine: tokens arrive 60-140 ms apart
with correct SSE framing, and a `stats` event plus `data: [DONE]` close the
stream. Needed **no C++ changes** — the callback already supported stopping.

Both review decisions held up in practice:

- Headers are withheld until the first token, so a pre-first-token failure is
  still a real `400`/`500` with a JSON body rather than a `200` carrying an
  error. Confirmed: a whitespace-only prompt with `"stream": true` returns
  `400 Content-Type: application/json`.
- `data: [DONE]` is sent only on success, so its absence marks a truncated
  stream. A mid-stream failure emits `event: error` and no terminator.

Cancellation works end to end: hanging up on a 200-token request freed the
session immediately — the next request answered in 1.2 s instead of waiting
~30 s for the abandoned one to finish.

**Watch out for:** tokens are byte fragments, not characters. A single emoji or
CJK character can be split across two tokens, so bytes must be buffered and
decoded incrementally — otherwise output shows replacement characters.

## Phase 3 — Concurrency  ✅ DONE

Handle several requests at once in one process.

**Tasks**

0. **Replace the `sync.Mutex` added in Phase 2.** (Done.) A single session is not safe
   for concurrent use, but `net/http` runs every request on its own goroutine —
   four simultaneous requests on one shared session returned three errors and
   one **HTTP 200 carrying garbage text** (`",,,,,,......"`). A mutex now
   serializes them correctly; the pool is what makes them actually overlap.
1. Create N workers, each owning its own `InferenceState` (memory cost below).
2. Each worker is a goroutine reading from a channel. `net/http` already runs
   every request in its own goroutine, so the web server stays responsive while
   generation blocks — no special handling needed. (Go notices a goroutine is
   blocked in C and schedules others onto new OS threads automatically.)
3. A **buffered channel** in front acts as the queue, absorbing bursts; return
   503 only when it is full.
4. `WORKERS` is an environment variable, so the same build runs 1 or 4.

**Done when:** four simultaneous requests stream in parallel on one machine.

**Outcome.** `server/pool.go` holds N sessions in a buffered channel, which is
simultaneously the pool, the queue and the semaphore. No engine changes were
needed: `engine_session_create` already produced independent KV caches, and
Phase 0 had already verified 4-way concurrency.

Verified with four concurrent streaming requests at `-workers 4`: tokens from
all four interleave within the same ~100 ms window, and each reports ~5 s of
generation, finishing within 1.9 s of one another. Backpressure also works —
at `-workers 1 -queue-wait 500ms`, three simultaneous requests gave one `200`
and two `503`s carrying `Retry-After: 1`.

Three design points worth keeping:

- **The pool implements the `generator` interface**, so `handleGenerate` and
  `streamGenerate` were untouched and all 37 existing handler tests kept
  passing — the regression check that behaviour did not change.
- **Threads are derived, not inherited.** `-workers 4` with no `-threads`
  computes `cores/workers`, logging `2 threads per request (derived from
  8 cores / 4 workers)`. Leaving this to the environment was the 7 s -> 120 s
  cliff; a bad explicit combination now logs a warning.
- **Release is deferred**, so a panic cannot permanently shrink capacity. That
  is the pool's worst failure mode and it has a dedicated test.

### Measured: concurrency buys latency fairness, not throughput

4 requests x 16 tokens, Qwen2.5-0.5B Q4_K_M, M2 (4 performance + 4 efficiency cores):

| Configuration | Total threads | Wall time |
|---|---|---|
| Sequential, 4 threads | 4 | **5.4-5.9 s** |
| Sequential, 8 threads | 8 | 12.2-12.7 s |
| 4 concurrent, 1 thread each | 4 | 6.0 s |
| 4 concurrent, 2 threads each | 8 | 6.6 s |

Running four requests at once takes the **same total time** as running them one
after another. What changes is the distribution: concurrently every client
finishes around 6 s, sequentially the first finishes at ~1.4 s and the last at
~5.9 s. That is a real property worth demonstrating — it is just not a
throughput win, and the demo should not claim one.

Two causes, both worth understanding:

1. **Memory bandwidth is the ceiling.** Every token streams the whole ~400 MB of
   weight tensors from RAM. Four concurrent requests need four times the memory
   traffic for the same tokens, and bandwidth is already saturated at one
   request. Threads cannot fix a bandwidth problem.
2. **`schedule(static)` plus heterogeneous cores.** Work is split evenly up
   front, so a team waits at its barrier for whichever thread landed on an
   efficiency core — those run roughly 3x slower but got equal work. This is why
   8 threads is **2.2x slower** than 4. Trying `schedule(dynamic)` in
   `src/quantized_matvec.cpp:153,195` and `src/tensor.cpp:62,83` is an open
   experiment.

The real throughput fix is **batching**: run N requests through one weight read
instead of N, which turns the bandwidth cost from N x into 1 x. That is a
substantial engine project, not a serving-layer change, and it is the honest
answer to "how would you make this faster".

### Thread count is set through the API, not the environment

`engine_set_max_threads` / `engine.SetMaxThreads` override `OMP_NUM_THREADS`.
This exists because OpenMP's thread count is **per-thread state**, so setting it
once from a startup thread would not reach workers; the value is applied inside
`engine_run` on the thread that enters the parallel regions.

Why it matters: with 4 workers and the default (all cores), 4 x 8 = 32 threads
land on 8 cores and a 7 s workload becomes a 120 s one. The Go test suite hit
exactly this.

**Memory cost at various context limits** (per worker):

| Max context | Per worker | x4 workers |
|---|---|---|
| 1024 | ~29 MB | ~116 MB |
| 2048 | ~56 MB | ~225 MB |
| 4096 | ~111 MB | ~443 MB |
| 32768 (model max) | ~882 MB | ~3.5 GB — will not fit |

> **Open decision:** the max-context ceiling. See *Deferred decisions*.

---

## Phase 4 — Container

**Tasks**

1. Multi-stage Dockerfile: CMake + Go build stage, then a slim runtime stage.
   Note `CGO_ENABLED=1`, so the result is **not** a static binary — the runtime
   image needs libc, `libengine`, `libllama`, and libomp. Use `debian-slim` or
   `distroless/cc`, not `scratch`.
2. Decide how the model reaches the container (see *Deferred decisions*).
3. Health endpoints: `/health/live` and `/health/ready`. **Ready must stay false
   until the model is loaded** — that's a 4.2 s window, and serving during it
   produces errors.

**Done when:** `docker run` gives the same behaviour as Phase 3.

---

## Phase 5 — Kubernetes (both demos)

**Tasks**

1. `Deployment` (`replicas=4`, `WORKERS=1`), `Service`, resource requests/limits.
2. Readiness probe tuned to the 4.2 s cold start (~`initialDelaySeconds: 5`).
3. Expose the pod name via the Downward API and return it on every response —
   this is what makes Demo C legible.
4. **Demo A:** a small client script firing 4 concurrent streaming requests,
   displayed side by side.
5. **Demo C:** `kubectl delete pod` mid-request; show the failure, the
   replacement pod starting, and traffic continuing.

**Done when:** both demos are reproducible and recordable.

**Watch out for — this will bite:** Kubernetes' default load balancing picks a
pod **at random**, not round-robin. Four simultaneous requests land on four
distinct pods only ~9% of the time; usually one pod gets two (serialised) and
another sits idle. Mitigations, in increasing order of effort:

- Run Demo A as `WORKERS=4, replicas=1` instead, where dispatch is internal and
  therefore perfect;
- or have busy pods return 503 and let the client retry onto an idle pod;
- or build the gateway (Phase 6).

Worth *showing* this problem before fixing it — it is a real distributed-systems
lesson and makes a better story than hiding it.

---

## Phase 6 — Later

- **Gateway**: one front door with a queue that tracks idle workers, giving
  proper least-busy routing. This is Phase 3's queue moved out into its own pod.
- **OpenAI-compatible endpoint** (`/v1/chat/completions`) so off-the-shelf chat
  UIs and SDKs work against this engine. Needs multi-turn prompt formatting —
  `engine_format_prompt` currently handles a single user message only.
- **Benchmarking** — aggregate tokens/sec at 1/2/4 workers.
- **HPA / autoscaling.** Note CPU-based scaling is a poor signal here (CPU is
  pinned at 100% by design during generation); queue depth is the real metric,
  which needs KEDA or prometheus-adapter.
- **Real sampling** (temperature / top-p), which then unlocks the API field.
- **Native tokenizer** — `src/tokenizer.cpp:62` loads the whole model a second
  time through llama.cpp just to get the vocabulary, which is a large part of
  the 4.2 s cold start.

---

## Deferred decisions

Still open; none block Phase 0 or 1.

1. **Max context ceiling** — 1024 / 2048 / 4096. Controls worker memory (table above).
2. **Static vs dynamic linking of the engine** — statically linking
   `libengine.a` into the Go binary would give a single self-contained
   executable, but linking llama.cpp and OpenMP statically is fiddly. Dynamic
   linking is the safe default.
3. **Model delivery** — baked into the image (self-contained, portable to real
   multi-node clusters, ~600 MB image) vs mounted from the host (~30 MB image,
   much faster iteration, laptop-only shortcut).

---

## What Kubernetes does and doesn't buy here

Stated plainly, because a portfolio reviewer will ask.

On a single 8-core machine, Kubernetes provides **no throughput benefit**. Four
pods split the same 8 cores as one pod with four workers, and cost ~180 MB extra
in duplicated runtimes plus a load balancer that misroutes requests.

What it does buy: self-healing, rolling deploys, declarative reproducible config,
and — the real one — **the ability to add machines later without changing the
manifests**. Running it on one node is a rehearsal. The same `Deployment` that
crowds four pods onto a laptop spreads them across four nodes the moment nodes
exist.

Concurrency and Kubernetes are independent — and, measured on this machine,
**concurrency does not improve throughput either.** See below.
