#ifndef INFERENCE_ENGINE_API_H
#define INFERENCE_ENGINE_API_H

/*
 * Stable C ABI for embedding the engine in a host process (for example a Go
 * HTTP server using cgo). Everything here is plain C: opaque handles, flat
 * structs of fixed-width scalars, and NUL-terminated UTF-8 strings. A host
 * language never needs to mirror an internal struct layout.
 *
 * Threading
 *   An EngineHandle is read-only once opened and may be shared by any number
 *   of sessions on any number of threads. An EngineSession owns mutable
 *   per-request state and must be used by one thread at a time.
 *
 *   engine_open and engine_close are NOT thread-safe with respect to each
 *   other; serialize them. Open every handle during startup.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Loaded weights plus tokenizer. Read-only; shared across sessions. */
typedef struct EngineHandle EngineHandle;

/* One request's worth of mutable state (KV cache, activations, logits). */
typedef struct EngineSession EngineSession;

typedef struct EngineStats {
    int prompt_tokens;       /* tokens after applying the instruction template */
    int generated_tokens;    /* includes a terminating EOS, which is never emitted */
    int stopped_early;       /* nonzero when the callback asked to stop */
    double prefill_seconds;  /* model load excluded; prompt processing only */
    double generate_seconds; /* first sampled token through last */
    double first_token_seconds; /* prompt start to first emitted text; -1 if none */
} EngineStats;

/*
 * Receive one decoded token piece. text is NUL-terminated UTF-8, valid only
 * for the duration of the call, and may be a partial multi-byte character:
 * token pieces are byte fragments, so a host must buffer and decode
 * incrementally rather than treating each call as a complete string.
 *
 * Return 1 to continue generating, 0 to stop. Stopping is reported as success.
 */
typedef int (*EngineTokenFn)(const char *text, void *user);

/*
 * Open a GGUF model. Returns NULL on failure, writing a message into error.
 * error may be NULL. The returned handle must be released with engine_close.
 */
EngineHandle *engine_open(const char *gguf_path, char *error, int error_capacity);
void engine_close(EngineHandle *handle);

/* The model's own context limit, in tokens. Returns 0 for a null handle. */
int engine_context_limit(const EngineHandle *handle);

/*
 * Allocate reusable per-request state holding up to max_context_tokens of
 * prompt plus generated tokens. All memory a request needs is acquired here,
 * so engine_run performs no allocation. max_context_tokens must be positive
 * and no greater than engine_context_limit.
 *
 * The session borrows the handle and must not outlive it.
 */
EngineSession *engine_session_create(EngineHandle *handle, int max_context_tokens,
                                     char *error, int error_capacity);
void engine_session_destroy(EngineSession *session);

/*
 * Generate a completion for one user message, reusing the session's buffers.
 *
 * The session is rewound before the prompt is applied, so a session may serve
 * unlimited sequential requests without reallocation. Each decoded piece is
 * handed to on_token as it is produced.
 *
 * Return 1 on success with stats filled in, or 0 with a message in error.
 * Rejects with an error, rather than crashing, when the prompt plus max_tokens
 * exceeds the session's context; a server should map that to a 400.
 *
 * temperature must currently be 0 (greedy decoding); nonzero sampling is not
 * implemented and is reported as an error rather than silently ignored.
 *
 * There is deliberately no top_p parameter: greedy decoding ignores nucleus
 * sampling entirely, so accepting one would advertise behaviour the engine does
 * not have. Add it here only once real sampling exists.
 */
int engine_run(EngineSession *session, const char *prompt, int max_tokens,
               float temperature,
               EngineTokenFn on_token, void *user,
               EngineStats *stats, char *error, int error_capacity);

/* Size a host-side error buffer. */
int engine_error_capacity(void);

/*
 * Longest prompt engine_run accepts, in bytes of UTF-8 excluding the
 * terminator. A host can check this itself to reject an oversized request
 * before entering the engine.
 */
int engine_prompt_capacity(void);

/*
 * Threads one generation call may use inside the engine's kernels, as OpenMP
 * resolved it: OMP_NUM_THREADS when set, otherwise the core count. Returns 1
 * when built without OpenMP.
 *
 * This is per call, not per process. A host running N concurrent requests uses
 * up to N * engine_max_threads() threads, so a server should report this to
 * make oversubscription visible.
 */
int engine_max_threads(void);

/*
 * Request how many threads each generation uses, overriding OMP_NUM_THREADS.
 * Pass a positive count, or 0 to return to the environment's default.
 *
 * Call this during startup, before serving. The value is applied on whichever
 * thread later calls engine_run, which is what makes it work when a host runs
 * requests on a pool of threads: OpenMP's thread count is per-thread state, so
 * setting it once from a startup thread would not otherwise affect workers.
 *
 * Returns the count that will be used, or 0 when built without OpenMP.
 */
int engine_set_max_threads(int threads);

#ifdef __cplusplus
}
#endif

#endif
