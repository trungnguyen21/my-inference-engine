# Running the server

Three ways, in increasing distance from your laptop.

## 1. Directly

```sh
cmake --build build --target engine      # from the repo root
cd server
go run . -model ../models/qwen-2.5.gguf -workers 4
```

## 2. In a container

The model is **mounted, not baked in**:

```sh
docker build -t inference-engine:dev .          # from the repo root
docker run --rm -p 8080:8080 \
  -v "$PWD/models:/models:ro" \
  -e WORKERS=4 -e CONTEXT=1024 \
  inference-engine:dev
```

Then:

```sh
curl -s localhost:8080/health/ready
curl -s localhost:8080/generate -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of France?","max_tokens":32}'
```

Configuration is by environment variable so a container needs no arguments:
`MODEL`, `ADDR`, `WORKERS`, `CONTEXT`, `THREADS`. Flags still override them.

### Why the model is not in the image

| | Mounted (chosen) | Baked in |
|---|---|---|
| Image size | ~120 MB | ~600 MB |
| Rebuild/reload after a code change | seconds | minutes |
| Page cache shared by replicas on a node | guaranteed (same inode) | yes, via the shared image layer |
| Schedulable on a node that has no model | no | yes |
| cgroup memory accounting | clean | the first pod to fault the pages is charged for all of them |

Mounting wins while iterating, and guarantees the sharing that keeps four
replicas from needing 4 x 469 MB. Baking it in is the right choice for a real
multi-node cluster, where a pod must be able to start on any node. Switching is
two lines: `COPY models/ /models/` and dropping `models/` from `.dockerignore`.

### Why not a `scratch` image

`CGO_ENABLED=1` is unavoidable — the whole point is calling the C++ engine —
so the binary is dynamically linked and needs libc, libstdc++, libgomp, and the
engine's own libraries. `debian-bookworm-slim` plus the six shared objects is
the honest minimum. Static linking llama.cpp and OpenMP would allow a smaller
base, but it is fiddly and was deliberately not attempted.

## 3. Health endpoints

| Endpoint | Means | Fails when |
|---|---|---|
| `/health/live` | the process is running | never, while the process lives |
| `/health/ready` | this instance can serve | model still loading |
| `/health` | both, plus worker/thread/context facts | model still loading |

The split matters: the server **starts listening before the model is loaded**,
so a probe during the several-second load gets a `503` rather than a refused
connection. Kubernetes can then tell "still starting" from "broken".

Critically, liveness must not depend on the model. A liveness probe that failed
during loading would have Kubernetes kill the container before it ever finished
starting — a crash loop that looks like a broken image.

Requests arriving during that window get `503` with `Retry-After: 5` and the
message `model is still loading`, rather than crashing on a nil engine.

## 4. Shutdown

`SIGINT`/`SIGTERM` stops accepting new connections, lets in-flight generations
finish (up to 60 s), then releases the sessions and the model. Kubernetes sends
`SIGTERM` and waits, so this is what prevents a rolling update from cutting
streams mid-sentence.

Ordering matters: the HTTP server drains **before** `pool.Close()`, because a
session must outlive every request holding it.
