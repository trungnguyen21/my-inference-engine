# HTTP API — Phase 1

One endpoint, one request at a time, no streaming. Streaming arrives in Phase 2
behind a `"stream": true` flag, so leave room for it but do not build it yet.

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
