# HTTP API — Phases 1 & 2

One endpoint, one request at a time. `"stream": true` switches the response from
a single JSON object to Server-Sent Events.

**Concurrency:** the server holds a single inference session, so simultaneous
requests are **serialized** — each waits for the one before it. Nothing is
rejected and nothing is corrupted, but a request can wait a long time. `ms`
measures generation only and excludes that wait. Phase 3 replaces the single
session with a pool so requests genuinely overlap.

## `POST /generate`

### Request

```json
{
  "prompt": "What is the capital of France?",
  "max_tokens": 64
}
```

| Field        | Type   | Required | Rules |
|--------------|--------|----------|-------|
| `prompt`     | string | yes      | non-empty; at most `engine.PromptCapacity()` bytes |
| `max_tokens` | int    | yes      | `1 <= max_tokens <= maxTokensLimit` (server flag) |
| `stream`     | bool   | no       | `false` (default) returns one JSON object; `true` streams SSE |

There is deliberately **no `temperature`**. Only greedy decoding is implemented,
so accepting the field would advertise behaviour the engine does not have.

### Success — `200`

```json
{
  "text": "Paris.",
  "prompt_tokens": 36,
  "generated_tokens": 8,
  "truncated": false,
  "ms": 820
}
```

| Field              | Meaning |
|--------------------|---------|
| `text`             | the full generated text |
| `prompt_tokens`    | tokens after the instruction template is applied |
| `generated_tokens` | sampled tokens; **counts a terminating EOS that is not in `text`**, so this can exceed the visible token count by one |
| `truncated`        | true when generation stopped early (client disconnected) |
| `ms`               | generation wall time in milliseconds, excluding model load |

### Errors

All errors use the same body:

```json
{ "error": "max_tokens must be between 1 and 512" }
```

| Status | When |
|--------|------|
| `400`  | malformed JSON, missing/empty `prompt`, `prompt` too long, `max_tokens` out of range, prompt + `max_tokens` exceeds the session's context |
| `405`  | any method other than POST |
| `500`  | the engine failed |
| `503`  | the server is busy (**Phase 3**; a single-session server may skip this) |

Distinguishing 400 from 500 matters: a caller can fix a 400 by changing the
request, and can only retry a 500. The engine returns both kinds as a plain Go
`error`, so Phase 1 may reasonably treat context-overflow as 400 by checking it
*before* calling the engine, and treat anything the engine itself returns as
500.

## Streaming — `"stream": true`

### Response

`200`, `Content-Type: text/event-stream`, `Cache-Control: no-cache`.

```
data: {"token":"Par"}

data: {"token":"is"}

data: {"token":"."}

event: stats
data: {"prompt_tokens":36,"generated_tokens":8,"truncated":false,"ms":820}

data: [DONE]

```

Rules, each one pinned by a test:

1. **One event per token piece**, as `data: {"token":"..."}`. Every event ends
   with a **blank line** — that is `\n\n`, and SSE has no framing without it.
2. **Flush after every event.** Without `http.Flusher` the response is buffered
   and the whole answer lands at once. Nothing errors; streaming just silently
   does not happen.
3. **A `stats` event before the terminator**, carrying the same numbers the
   non-streaming response returns. This is what the concurrency demo reads.
4. **`data: [DONE]` last.** Its absence is how a client detects a truncated
   stream, so it must only be sent on success.

### Errors while streaming

**Before the first token** — nothing has been written yet, so respond with a
normal JSON error and the right status, exactly like the non-streaming path:

```
400 {"error":"prompt cannot be empty"}
500 {"error":"tensor operation failed"}
```

This is why the SSE headers must **not** be written until the first token
arrives. Write them up front and every early failure becomes a `200` carrying
an error, which no client can handle properly.

**After the first token** — the status line is already sent and cannot be
changed. Report it in-band and stop, with no `[DONE]`:

```
data: {"token":"Par"}

event: error
data: {"error":"tensor operation failed"}

```

### Client disconnect

When the caller hangs up, `r.Context().Done()` closes. Return `false` from the
token callback so generation stops instead of burning CPU producing output
nobody will read. The engine honours this between tokens, so the worst-case
delay is one token.

A cancelled request needs no response — the connection is gone.

### Manual check

```sh
curl -N -s localhost:8080/generate -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of France?","max_tokens":32,"stream":true}'
```

`-N` disables curl's own buffering. Without it you cannot tell whether the
server is streaming.

## `GET /health`

Returns `200` once the model is loaded:

```json
{
  "status": "ok",
  "model_context_limit": 32768,
  "session_context": 1024,
  "omp_threads": 4
}
```

`omp_threads` is how many threads **one** request may use inside the C++
kernels. It is reported because OpenMP is invisible from Go otherwise, and it
becomes load-bearing in Phase 3: total threads = workers x `omp_threads`.

Phase 4 splits this into `/health/live` and `/health/ready`, because Kubernetes
needs to know the difference between "process is up" and "model has finished
its 4.2 s load".

## Manual check

```sh
curl -s localhost:8080/health | jq
curl -s localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of France?","max_tokens":32}' | jq
```
