# Demo kit

Three tools, each answering a different question. They work against a local
server, Compose, or a cluster — nothing here is Kubernetes-specific.

| Tool | Question | Request shape |
|---|---|---|
| `loadgen` | *where* do requests go, and what happens when a pod dies? | short, non-streaming |
| `/demo` page | does it really stream, concurrently? | long, streaming |
| `bench.sh` | how many tokens/sec in aggregate? | medium, measured |

## loadgen — the traffic dashboard

```sh
cd demo
go run ./loadgen -url http://localhost:8080 -users 6
```

```
loadgen → http://localhost:8080   6 concurrent
────────────────────────────────────────────────────────────────────────
elapsed 1m12s     req/s 1.84    tok/s 14.7     ok 132   busy 3     err 0

  node-1  (71 requests)
     engine-7d4f-abcde      ████████████████████████████   38  p50 2.80s
     engine-7d4f-fghij      ████████████████████████       33  p50 2.91s

  node-2  (61 requests)
     engine-7d4f-klmno      ██████████████████████         31  p50 2.88s
     engine-7d4f-pqrst      █████████████████████          30  p50 2.95s
```

It keeps `-users` requests permanently in flight and redraws in place. That
continuous traffic is the point: **deleting a pod on an idle cluster shows
nothing.** With this running you watch one bar stop growing, the others absorb
the load, then a new pod appear at zero and start climbing.

Three behaviours worth knowing:

- **Short requests by default** (`-max-tokens 8`). Each takes 2-5 s, so a large
  value would show three completions in a whole demo. This tool answers "where
  did requests go", not "how fast is generation".
- **503 is counted apart from errors, and retried.** A restarting pod returns
  `model is still loading`; a saturated one returns `all N workers busy`.
  Treating those as failures makes resilience look like an outage; hiding them
  would be dishonest. So they are counted, shown as `busy`, and retried.
- **A pod that stops answering is marked `·` rather than dropped**, so a
  deletion stays visible instead of silently vanishing from the table.

Identity comes from the `X-Served-By` header, which the server sets on **every**
response including 4xx/5xx — so a pod that *refuses* a request still appears.

## The streaming page

```sh
open http://localhost:8080/demo
```

Served by the server itself, which avoids CORS entirely: the page and the API
share an origin, so its POST needs no cross-origin headers. It is embedded in
the binary, so it works unchanged in a container and behind a NodePort with no
volume mount.

Each pane opens its own streaming request and shows which pod/node answered.
`EventSource` is GET-only, so the page parses SSE by hand off a `fetch` body
stream — the same splitting-on-blank-lines logic as `parseSSE` in
`server/stream_test.go`.

A pane turns red and says `stream ended without [DONE]` if the terminator never
arrives. That is the truncation signal working: the server only sends `[DONE]`
on success.

## bench.sh

```sh
./bench.sh -url http://localhost:8080 -users 6 -duration 60s -label "1 node"
```

Drives `loadgen` rather than reimplementing it, so the request shape and retry
accounting match the traffic demo exactly.

It **warms up first** (`-warmup 10s`, `0` to skip). The first request on a
fresh process pays for OpenMP team creation and for faulting the weights into
the page cache; charging that to the measurement is how you get numbers that
swing by 3x between runs.

The comparison that matters:

```sh
kubectl cordon node-2
./bench.sh -url http://NODE:30080 -label "1 node"
kubectl uncordon node-2
./bench.sh -url http://NODE:30080 -label "2 nodes"
```

Expect roughly 2x. Memory bandwidth is **per machine**, so adding machines is
the only change in this project that raises aggregate tokens/sec — adding
workers on one box does not, which is measured in
`docs/serving-implementation-plan.md`.

## Measuring on a laptop

Don't, for anything you intend to quote. Identical requests measured 1.8 s and
7.5 s on the same machine at load average 8. Thermal state, swap, and whatever
else is running dominate. Use an idle EC2 node for numbers you report.
