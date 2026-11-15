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

## Phase 1 — Simplest possible endpoint  ✅ DONE

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

**Outcome.** `POST /generate` returns `{"text":...,"prompt_tokens":...,
"generated_tokens":...,"truncated":...,"ms":...}`, with 400 on every invalid
input and 500 only for genuine engine failures.

One bug found late and worth remembering: validation ran on the trimmed prompt
while the untrimmed one was sent to the engine, and the length check used `>`
where the engine rejects at `>=`. Both turned a 400 into a 500. There are now
boundary regression tests at exactly `PromptCapacity()`.

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

## Phase 4 — Container  ✅ DONE (image build pending local verification)

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

**Delivered.** `Dockerfile` (multi-stage), `.dockerignore`, the
liveness/readiness split, and graceful shutdown. See `server/RUNNING.md`.

Three deferred decisions are now resolved:

1. **Max context ceiling — 1024 by default**, as the `-context` / `CONTEXT`
   default. ~29 MB per worker, so four workers cost ~116 MB. Raising it is a
   flag, and the startup log prints the estimate before allocating.
2. **Linking — dynamic.** `CGO_ENABLED=1` makes a static binary impossible
   anyway, so the runtime image is `debian-bookworm-slim` plus libengine and
   the five llama.cpp/ggml shared objects, installed to `/usr/local/lib` with
   `ldconfig`. Not `scratch`.
3. **Model delivery — mounted, not baked in.** Keeps the image at ~120 MB so
   rebuilds take seconds, and guarantees replicas on one node share the same
   inode and therefore the same page cache. Baking it in is correct for a real
   multi-node cluster and is a two-line change; the tradeoff table is in
   `server/RUNNING.md`.

**Two structural changes this phase required:**

- **The server now listens before loading the model.** Previously the listener
  did not exist until loading finished, so a probe during that window got a
  refused connection — indistinguishable from a crashed pod. The engine is
  published behind a `RWMutex` once ready, and requests arriving earlier get
  `503 model is still loading`.
- **Graceful shutdown on SIGTERM**, draining in-flight requests before closing
  sessions. This also fixed a pre-existing bug: `main` used `defer` for
  cleanup, but `ListenAndServe` only returns on error, so those defers never
  actually ran.

**Liveness must not depend on the model.** A liveness probe that fails during
the load would have Kubernetes kill the container before it finished starting,
producing a crash loop that looks like a broken image. `/health/live` therefore
reports only that the process is alive.

---

## Phase 5 — Identity and the demo kit  (local, no cluster)

Everything here works against `go run .`, Compose, or any cluster, so none of
it is wasted if the EC2 step slips. It is also what every later phase depends
on, which is why it comes first.

**Tasks**

1. **Pod and node identity on every response.** Return `X-Served-By` and add
   both to the `stats` event, from the Downward API:

   ```yaml
   env:
     - name: POD_NAME
       valueFrom: { fieldRef: { fieldPath: metadata.name } }
     - name: NODE_NAME
       valueFrom: { fieldRef: { fieldPath: spec.nodeName } }
   ```

   Without the **node** name, four streams look identical whether they are on
   one machine or two — which is the entire point of going multi-node.

2. **`demo/loadgen`** — a closed-loop driver: keep K requests permanently in
   flight, each finishing and immediately starting another. Print a live tally
   per pod and node:

   ```
   elapsed  rps   ok   err  │ node-1              │ node-2
     12.3s  1.8   22    0   │ pod-abc  ███████ 7  │ pod-def  ████████ 8
                            │ pod-ghi  ████    4  │ pod-jkl  ███     3
   ```

   This is what makes disruption legible: delete a pod and its bar stops
   growing while the others absorb the load.

   Two design points that matter more than they look:
   - **Short requests** (`max_tokens: 8`). Each takes 2-5 s, so long outputs
     would show three completions in a whole demo. Expect ~2 req/s with four
     workers; this demo is about *where* requests go, not throughput.
   - **Retry on 503.** Restarts produce `model is still loading` and
     `all N workers busy`. Treating those as fatal makes resilience look like
     an outage. Count them separately so nothing is hidden.

3. **`demo/panes.html`** — four SSE streams side by side with `pod / node`
   badges. Long requests (`max_tokens: 64`) here: this is the shot that proves
   streaming, and it is a different shot from the traffic dashboard.

4. **`demo/bench.sh`** — aggregate tokens/sec, to be run with one node
   cordoned and then with both.

**Done when:** the dashboard and panes work against a local server.

---

## Phase 6 — Docker Compose  (one host)

A deliberate stepping stone, not a detour. ~20 lines, no cost, and it isolates
"is the image right" from "is the cluster right".

**Tasks**

1. `compose.yaml` with `deploy.replicas: 4`, the model mounted read-only, and
   a healthcheck.
2. Point the demo kit at it. Confirm four-way concurrency and the identity
   labels.

**Done when:** `docker compose up --scale engine=4` serves the dashboard.

**What Compose cannot do** — worth seeing first-hand, because it is the
argument for the next phase:

- **No readiness gating.** Compose's DNS hands out every replica's IP
  regardless of healthcheck state, so a container still inside its ~4 s model
  load already receives traffic. The `/health/live` vs `/health/ready` split
  from Phase 4 is inert here.
- **No rolling updates.** No `maxSurge`/`maxUnavailable`, no waiting for a new
  container to be ready before killing the old one, no rollback.
- **No reconciliation loop.** Compose applies the file once; `restart: always`
  covers crashes but not the general case.
- **One machine, by definition.**

A rolling restart under Compose dropping requests, then the same under
Kubernetes not dropping them, is a better demo than either alone.

---

## Phase 7 — k3d multi-node  (local, still free)

Real manifests, nodes as containers. Debugging YAML here costs nothing;
debugging it against rented machines costs money and patience.

**Tasks**

1. `k3d cluster create --agents 2`, then load the image with `k3d image import`.
2. `k8s/`: `Deployment` (`replicas=4`, `WORKERS=1`), `Service`,
   resource requests/limits, and probes tuned to the measured 4.2 s cold start
   (`readinessProbe` ~`initialDelaySeconds: 5, periodSeconds: 2`;
   `livenessProbe` on `/health/live`, which must **not** depend on the model).
3. The Downward API env block from Phase 5.
4. `topologySpreadConstraints` so replicas actually land on different nodes
   rather than piling onto one.
5. **Model delivery flips here.** Mounting from the host only works on one
   machine; a pod must be able to start on any node. Bake it into the image
   (`COPY models/ /models/`, drop `models/` from `.dockerignore`) or pull from
   object storage via an initContainer.

**Done when:** `kubectl get pods -o wide` shows pods on both agents and the
dashboard shows two node names.

**Expose it with a NodePort, not an Ingress.** k3s ships Traefik, and proxies
buffer streaming responses — if tokens arrive in one burst, that is why. Fewer
layers is also more honest about what is being demonstrated.

**This will bite:** Kubernetes' default load balancing picks a backend **at
random**, not round-robin. Four simultaneous requests reach four distinct pods
only ~9 % of the time; usually one pod gets two (serialised) and another idles.
Mitigations, in increasing effort: run the streaming demo as
`WORKERS=4, replicas=1` where dispatch is internal and therefore perfect; rely
on the 503-and-retry the loadgen already does; or build the gateway (Phase 9).

Worth *showing* before fixing — it is a real distributed-systems lesson.

---

## Phase 8 — Two EC2 nodes  (the real demo)

**Why bother, in one sentence:** memory bandwidth is **per machine**, so adding
machines is the only change in this entire project that genuinely increases
aggregate tokens/sec.

Everything measured so far says concurrency on one box does not add throughput
because requests contend for the same RAM bandwidth. Two machines have two
memory subsystems. Expect roughly 2x — a prediction from a diagnosis, which
reads very differently from "I added servers and it got faster".

**Topology**

```
   MacBook                      AWS VPC
   ┌─────────┐          ┌──────────────┬──────────────┐
   │ kubectl ├─ HTTPS ──► node-1       │ node-2       │
   └─────────┘  (SG      │ k3s server  │ k3s agent    │
              locked to  │ + engine pod│ + engine pod │
              your IP)   └──────────────┴──────────────┘
```

**Not** the MacBook as control plane. Two separate things get conflated there:
running the control plane, and running `kubectl`. Only the second is wanted,
and driving a remote cluster from a laptop is the normal arrangement. A
home-NAT control plane means exposing an API server through a dynamic-IP router
to kubelets that must reach it, the control-plane components are Linux-only so
they would sit in a VM behind that NAT, and etcd flaps when the lid closes.

**Tasks**

1. Two **arm64 Graviton** instances (`c7g.xlarge` or similar). arm64 matters:
   the engine's NEON kernels are used rather than the scalar fallback. Avoid
   `t`-class burstable — CPU credits throttle sustained inference, which is
   exactly this workload.
2. k3s server on node-1, agent on node-2. Security group opens the API port to
   your IP only.
3. Same manifests as Phase 7, unchanged. That is the whole point of rehearsing.
4. **Set a billing alarm and stop the instances afterwards.** ~$0.30/hr for the
   pair is a few dollars for an afternoon and ~$210/month if forgotten.

**Done when:** `bench.sh` with one node cordoned, then both, shows the ~2x.

---

## The demo itself

**The deliverable is a recording, not a live cluster.** Nobody will SSH in, and
paying to keep instances up on the chance they might is waste. A recording also
means a failed live demo cannot embarrass you.

Run `demo/loadgen` in one terminal pane and `kubectl` in another, side by side,
so the disruption and the response land in the same frame.

Every step below is a command **you type**. Nothing fires on its own — and that
is the point: you cause a disturbance, Kubernetes reacts without you.

| You type | Kubernetes does, unprompted |
|---|---|
| `kubectl get nodes -o wide` | — (two real machines) |
| `kubectl get pods -o wide` | — (spread across both) |
| *(open panes.html)* | — (badges show two node names) |
| `kubectl cordon` / bench / `uncordon` / bench | — (the ~2x) |
| `kubectl exec <pod> -- kill 1` | kubelet restarts the container |
| `kubectl delete pod` | schedules a replacement, waits for readiness before routing to it |
| `kubectl drain node-2` | evicts and reschedules onto node-1 |
| `kubectl rollout restart` | replaces pods one at a time, each gated on readiness |

`drain` and the throughput measurement are the two steps **only multi-node
gives you**. `drain` is also precisely what Compose structurally cannot do.

**Script it** (`demo/run-demo.sh`) rather than typing live: deterministic,
repeatable after a fix, no dead air during model loads, and readable as a
deliverable in its own right.

**Do not edit out the slow parts.** A replacement pod sits `0/1 Running` for
several seconds while the model loads. That is the system working, and it is
the clearest justification for the liveness/readiness split.

**Mention, but do not film, node failure.** Terminating an instance does
recover automatically — but it takes ~40 s to mark the node `NotReady` plus a
~300 s toleration before pods are evicted. Five to six minutes of nothing.
Knowing why it is slow, and that it is tunable, is worth a paragraph in the
writeup; `drain` is what you actually demo.

---

## Phase 9 — Later

- **Gateway**: one front door with a queue that tracks idle workers, giving
  proper least-busy routing instead of kube-proxy's random choice. This is
  Phase 3's queue moved into its own pod.
- **OpenAI-compatible endpoint** (`/v1/chat/completions`) so off-the-shelf chat
  UIs and SDKs work against this engine. Needs multi-turn prompt formatting —
  `engine_format_prompt` currently handles a single user message only.
- **HPA / autoscaling.** CPU is a poor signal here (pinned at 100 % by design
  during generation); queue depth is the real metric, which needs KEDA or
  prometheus-adapter. On a two-node cluster the useful ceiling is small.
- **Real sampling** (temperature / top-p), which then unlocks the API field.
- **Native tokenizer** — `src/tokenizer.cpp:62` loads the whole model a second
  time through llama.cpp just for the vocabulary. Removing it drops five of the
  six runtime shared libraries and most of the 4.2 s cold start, which also
  makes pod replacement visibly faster.
- **Batching** — the only remaining change that would raise throughput on a
  single machine, and the one thing on this list that is squarely an engine
  project rather than a serving one.

---

## Decisions, resolved

1. **Max context ceiling — 1024**, as the `-context` / `CONTEXT` default.
   ~29 MB per worker; four workers ~116 MB. Startup logs the estimate before
   allocating.
2. **Linking — dynamic.** `CGO_ENABLED=1` makes a static binary impossible, so
   the runtime image is `debian-bookworm-slim` plus libengine and the five
   llama.cpp/ggml shared objects in `/usr/local/lib`. Not `scratch`.
3. **Model delivery — mounted for one host, baked in for a cluster.** Mounting
   keeps the image ~120 MB and rebuilds in seconds, and guarantees replicas on
   one node share an inode. It stops working the moment pods must be
   schedulable on any node, so Phase 7 switches to baking it in.
4. **Control plane — on the cluster, not the MacBook.** `kubectl` runs on the
   laptop; that is all that was ever wanted there.
5. **Ingress — NodePort for the demo.** Proxies buffer SSE, and fewer layers is
   more honest about what is being shown.

---

## What Kubernetes does and doesn't buy here

> Written before the multi-node plan existed. Still accurate **for one
> machine**, which is why Phases 6-8 escalate deliberately rather than jumping
> to a cluster. The one thing missing below: on *two* machines Kubernetes does
> buy throughput, because memory bandwidth is per machine. See Phase 8.

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
