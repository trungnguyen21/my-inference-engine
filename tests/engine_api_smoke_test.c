/*
 * Exercises include/engine_api.h the way an out-of-process host does.
 *
 * Deliberately compiled as C, not C++: that is what proves the header is
 * consumable from a C toolchain, which is what cgo (and any other FFI) uses.
 *
 * Needs models/qwen-2.5.gguf, so this is not registered as a ctest. Run it
 * from the repository root:
 *     cmake --build build --target engine_api_smoke_test
 *     ./build/engine_api_smoke_test
 *
 * The thread count is pinned below rather than left to OMP_NUM_THREADS. The
 * concurrency section runs THREADS requests at once, so inheriting "all cores"
 * would put THREADS * ncpu threads on ncpu cores and take minutes instead of
 * seconds.
 */
#include "engine_api.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/qwen-2.5.gguf"
#define SESSION_CONTEXT 192
#define THREADS 4
#define ENGINE_THREADS 2 /* per request; see the note above */

static int failures;

static void check(int condition, const char *what)
{
    printf("  %-44s %s\n", what, condition ? "ok" : "FAILED");
    if (!condition) ++failures;
}

/* ---------- part 1: one session, reused ---------- */

typedef struct {
    char text[2048];
    int pieces;
    int stop_after; /* -1 to never stop early */
} Sink;

static int on_token(const char *text, void *user)
{
    Sink *sink = (Sink *)user;
    ++sink->pieces;
    strncat(sink->text, text, sizeof(sink->text) - strlen(sink->text) - 1);
    return (sink->stop_after >= 0 && sink->pieces >= sink->stop_after) ? 0 : 1;
}

static void sink_reset(Sink *sink, int stop_after)
{
    memset(sink, 0, sizeof(*sink));
    sink->stop_after = stop_after;
}

static void test_sequential(EngineHandle *handle)
{
    char error[512];
    EngineStats stats;
    Sink sink;
    EngineSession *session;

    printf("sequential reuse\n");
    session = engine_session_create(handle, SESSION_CONTEXT, error, sizeof(error));
    check(session != NULL, "session created");
    if (session == NULL) return;

    sink_reset(&sink, -1);
    check(engine_run(session, "What's the capital of France?", 16, 0.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "first request succeeds");
    check(strstr(sink.text, "Paris") != NULL, "first request mentions Paris");
    check(stats.prompt_tokens > 0 && stats.generated_tokens > 0, "stats populated");

    /* The same session again: proves inference_state_reset rewinds correctly
     * and that no per-request allocation is needed. */
    sink_reset(&sink, -1);
    check(engine_run(session, "What's 2+2?", 16, 0.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "reused session succeeds");
    check(strstr(sink.text, "4") != NULL, "reused session answers correctly");

    /* Early cancellation, as a disconnecting HTTP client would cause. */
    sink_reset(&sink, 2);
    check(engine_run(session, "Count to twenty.", 16, 0.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "cancelled request reports success");
    check(stats.stopped_early == 1, "cancellation is reported");
    check(sink.pieces == 2, "cancellation stops at the requested token");

    /* Invalid requests must be errors, not crashes. */
    sink_reset(&sink, -1);
    check(engine_run(session, "hi", 16, 0.8f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "nonzero temperature rejected");
    check(engine_run(session, "hi", 100000, 0.0f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "oversized max_tokens rejected");
    check(engine_run(session, "hi", 0, 0.0f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "zero max_tokens rejected");

    /* A rejected request must leave the session usable. */
    sink_reset(&sink, -1);
    check(engine_run(session, "What's 2+2?", 8, 0.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "session still usable after rejections");

    engine_session_destroy(session);
    engine_session_destroy(NULL); /* must tolerate null */
}

/* ---------- part 2: concurrent sessions sharing one handle ---------- */

static const char *const concurrent_prompts[THREADS] = {
    "What's the capital of France?",
    "Name three primary colors.",
    "What's 2+2?",
    "What color is the sky?"
};

typedef struct {
    int index;
    EngineHandle *handle;
    Sink sink;
    int ok;
} Job;

static void *concurrent_worker(void *argument)
{
    Job *job = (Job *)argument;
    char error[512];
    EngineStats stats;
    int round;
    EngineSession *session =
        engine_session_create(job->handle, SESSION_CONTEXT, error, sizeof(error));

    if (session == NULL) return NULL;
    /* Two rounds each, so reset is also exercised while other threads run. */
    for (round = 0; round < 2; ++round) {
        sink_reset(&job->sink, -1);
        if (!engine_run(session, concurrent_prompts[job->index], 16, 0.0f,
                        on_token, &job->sink, &stats, error, sizeof(error))) {
            engine_session_destroy(session);
            return NULL;
        }
    }
    job->ok = job->sink.text[0] != '\0';
    engine_session_destroy(session);
    return NULL;
}

static void test_concurrent(EngineHandle *handle)
{
    pthread_t threads[THREADS];
    Job jobs[THREADS];
    int i;

    printf("concurrent sessions on one shared handle\n");
    memset(jobs, 0, sizeof(jobs));
    for (i = 0; i < THREADS; ++i) {
        jobs[i].index = i;
        jobs[i].handle = handle;
    }
    for (i = 0; i < THREADS; ++i) {
        pthread_create(&threads[i], NULL, concurrent_worker, &jobs[i]);
    }
    for (i = 0; i < THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
    for (i = 0; i < THREADS; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "thread %d produced output", i);
        check(jobs[i].ok, label);
    }
    check(strstr(jobs[0].sink.text, "Paris") != NULL,
          "concurrent output is still correct");
}

int main(void)
{
    char error[512];
    EngineHandle *handle;
    int default_threads;

    printf("engine_api smoke test (model: %s)\n", MODEL_PATH);
    /* Read the default before changing it: C does not define the order in
     * which printf's arguments are evaluated. */
    default_threads = engine_max_threads();
    printf("pinning %d thread(s) per request (default was %d)\n\n",
           engine_set_max_threads(ENGINE_THREADS), default_threads);
    handle = engine_open(MODEL_PATH, error, sizeof(error));
    if (handle == NULL) {
        printf("could not open model: %s\n", error);
        printf("run this from the repository root, with the model present.\n");
        return 1;
    }
    check(engine_context_limit(handle) > 0, "context limit reported");
    check(engine_error_capacity() > 0, "error capacity reported");
    check(engine_session_create(handle, 1 << 30, error, sizeof(error)) == NULL,
          "oversized session rejected");
    printf("\n");

    test_sequential(handle);
    printf("\n");
    test_concurrent(handle);

    engine_close(handle);
    engine_close(NULL); /* must tolerate null */

    printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL",
           failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
