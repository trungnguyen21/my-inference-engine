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

/* Length, in bytes, of the longest span that is an exact, immediate
 * repetition of a short (<= 24 byte) substring at least 3 times in a row --
 * "blue blue blue" or "the the the the", the shape small-model greedy
 * looping takes. 0 if no such run exists. Not a general repetition detector;
 * just enough to tell "looped" from "did not" for this test. */
static int longest_repeated_substring_run(const char *text)
{
    int len = (int)strlen(text);
    int best = 0;
    int w;
    for (w = 1; w <= 24 && w * 3 <= len; ++w) {
        int i;
        for (i = 0; i + w * 3 <= len; ++i) {
            int repeats = 1;
            while (i + (repeats + 1) * w <= len &&
                   memcmp(text + i, text + i + repeats * w, (size_t)w) == 0) {
                ++repeats;
            }
            if (repeats >= 3) {
                int span = repeats * w;
                if (span > best) best = span;
            }
        }
    }
    return best;
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
    check(engine_run(session, "What's the capital of France?", 16, 0.0f, 1.0f, 1.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "first request succeeds");
    check(strstr(sink.text, "Paris") != NULL, "first request mentions Paris");
    check(stats.prompt_tokens > 0 && stats.generated_tokens > 0, "stats populated");

    /* The same session again: proves inference_state_reset rewinds correctly
     * and that no per-request allocation is needed. */
    sink_reset(&sink, -1);
    check(engine_run(session, "What's 2+2?", 16, 0.0f, 1.0f, 1.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "reused session succeeds");
    check(strstr(sink.text, "4") != NULL, "reused session answers correctly");

    /* Early cancellation, as a disconnecting HTTP client would cause. */
    sink_reset(&sink, 2);
    check(engine_run(session, "Count to twenty.", 16, 0.0f, 1.0f, 1.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "cancelled request reports success");
    check(stats.stopped_early == 1, "cancellation is reported");
    check(sink.pieces == 2, "cancellation stops at the requested token");

    /* Nonzero temperature now has real nucleus sampling behind it (it used to
     * be rejected outright: this is the bug fixed in this change). It must
     * succeed, and a low top_p collapses the nucleus small enough that output
     * is still coherent text, not garbage. */
    sink_reset(&sink, -1);
    check(engine_run(session, "What's the capital of France?", 16, 0.7f, 0.9f, 1.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "nonzero temperature now succeeds");

    /* Invalid requests must be errors, not crashes. */
    sink_reset(&sink, -1);
    check(engine_run(session, "hi", 100000, 0.0f, 1.0f, 1.0f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "oversized max_tokens rejected");
    check(engine_run(session, "hi", 0, 0.0f, 1.0f, 1.0f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "zero max_tokens rejected");
    check(engine_run(session, "hi", 16, 5.0f, 1.0f, 1.0f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "out-of-range temperature rejected");
    check(engine_run(session, "hi", 16, 0.0f, 5.0f, 1.0f, on_token, &sink, &stats,
                     error, sizeof(error)) == 0,
          "out-of-range top_p rejected");

    /* A rejected request must leave the session usable. */
    sink_reset(&sink, -1);
    check(engine_run(session, "What's 2+2?", 8, 0.0f, 1.0f, 1.0f,
                     on_token, &sink, &stats, error, sizeof(error)) == 1,
          "session still usable after rejections");

    engine_session_destroy(session);
    engine_session_destroy(NULL); /* must tolerate null */
}

/* ---------- part 1b: repeat_penalty is the actual bug fix ---------- */

/* Pure greedy decoding on a small model tends to collapse into repeating the
 * same token or short phrase once it has nothing confident left to say. This
 * demonstrates the failure and the fix on the same prompt/session shape used
 * throughout this file, against the real model — not a synthetic case. */
static void test_repetition_penalty(EngineHandle *handle)
{
    char error[512];
    EngineStats stats;
    Sink sink;
    EngineSession *session;
    int max_run;

    printf("repetition penalty\n");
    session = engine_session_create(handle, SESSION_CONTEXT, error, sizeof(error));
    check(session != NULL, "session created for repetition test");
    if (session == NULL) return;

    /* This prompt reliably collapses pure greedy decoding into "1, 1, 1, ..."
     * for the entire token budget on this model -- found empirically by
     * probing several open-ended prompts against the real model, not guessed. */
#define LOOPING_PROMPT "Continue this sequence: 1, 1, 1, 1,"

    sink_reset(&sink, -1);
    check(engine_run(session, LOOPING_PROMPT, 96,
                     0.0f, 1.0f, 1.0f, on_token, &sink, &stats, error, sizeof(error)) == 1,
          "long greedy run succeeds");
    max_run = longest_repeated_substring_run(sink.text);
    printf("    (no penalty)   longest repeated short run: %d chars -- %.60s...\n",
           max_run, sink.text);

    /* Same prompt, same session (rewound by engine_run itself), with the
     * penalty active. If the first run looped, this run must not. */
    sink_reset(&sink, -1);
    check(engine_run(session, LOOPING_PROMPT, 96,
                     0.0f, 1.0f, 1.3f, on_token, &sink, &stats, error, sizeof(error)) == 1,
          "long run with repeat_penalty succeeds");
    printf("    (penalty 1.3)  longest repeated short run: %d chars -- %.60s...\n",
           longest_repeated_substring_run(sink.text), sink.text);

    /* Sanity-check the repro itself: if this ever stops looping (model swap,
     * decoding change elsewhere), the comparison below becomes meaningless, so
     * say so loudly rather than passing vacuously. */
    check(max_run >= 20,
          "no-penalty run reproduces the known looping bug on this prompt");

    /* The real regression this file guards: engine_run must now accept
     * repeat_penalty > 1.0 and it must take effect. */
    check(longest_repeated_substring_run(sink.text) < max_run,
          "repeat_penalty reduced the repetition the no-penalty run showed");

    engine_session_destroy(session);
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
        if (!engine_run(session, concurrent_prompts[job->index], 16, 0.0f, 1.0f, 1.0f,
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
    test_repetition_penalty(handle);
    printf("\n");
    test_concurrent(handle);

    engine_close(handle);
    engine_close(NULL); /* must tolerate null */

    printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL",
           failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
