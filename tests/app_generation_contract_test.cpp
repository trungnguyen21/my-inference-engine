/* The real app loop with a scripted backend. This verifies control flow;
 * the printed fixture pieces are not evidence of real model inference. */
#include "app.h"
#include "engine.h"
#include "metrics.h"
#include "terminal.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>

enum Failure {
    NONE, LOAD, FORMAT, ENCODE, EMPTY_PROMPT, BAD_PROMPT_COUNT, SHORT_CONTEXT,
    INIT, PREFILL, DECODE_FORWARD, SAMPLE, BAD_SAMPLE_LOW, BAD_SAMPLE_HIGH,
    DECODE, UNTERMINATED_PIECE, NULL_LOGITS, WRONG_LOGIT_COUNT
};

static Failure failure;
static std::vector<int> forwards, skipped_logits, samples, decoded;
static std::vector<std::string> events;
static std::string printed, reported_error;
static int loaded, unloaded, initialized, destroyed, finished, sampled, captured;
static int reported_tokens, requested_capacity;
static int profiles_printed;
static double reported_elapsed, reported_first_token;
static int unavailable_capture;
static bool empty_first_piece;
static OperationProfile reported_profile;
static float logits[7];

static void reset(Failure selected, const std::vector<int> &script)
{
    failure = selected;
    forwards.clear(); skipped_logits.clear(); decoded.clear(); events.clear();
    samples = script;
    printed.clear(); reported_error.clear();
    loaded = unloaded = initialized = destroyed = finished = sampled = captured = 0;
    reported_tokens = requested_capacity = 0;
    profiles_printed = 0;
    reported_elapsed = reported_first_token = -1.0;
    unavailable_capture = 0;
    empty_first_piece = false;
    memset(&reported_profile, 0, sizeof(reported_profile));
    memset(logits, 0, sizeof(logits));
}

static int fail(char *error, int capacity)
{
    snprintf(error, (size_t)capacity, "scripted failure");
    return 0;
}

int engine_load_model(const AppConfig *, Model *model, char *error, int capacity)
{
    ++loaded; events.push_back("load");
    model->implementation = logits; /* Also simulate partial failure ownership. */
    model->config.vocab_size = 7;
    model->config.seq_len = failure == SHORT_CONTEXT ? 2 : 100;
    return failure == LOAD ? fail(error, capacity) : 1;
}

void engine_unload_model(Model *model)
{
    ++unloaded; events.push_back("unload");
    assert(model->implementation == logits);
    memset(model, 0, sizeof(*model));
}

int engine_format_prompt(const Model *, const char *user, char *output,
                         int capacity, char *error, int error_capacity)
{
    events.push_back("format");
    assert(strcmp(user, "Hi") == 0);
    if (failure == FORMAT) return fail(error, error_capacity);
    assert(capacity >= 16);
    snprintf(output, (size_t)capacity, "[formatted]Hi");
    return 1;
}

int tokenizer_encode(const Model *, const char *text, int *ids,
                     int capacity, int *count, char *error, int error_capacity)
{
    events.push_back("encode");
    assert(strcmp(text, "[formatted]Hi") == 0 && capacity >= 2);
    if (failure == ENCODE) return fail(error, error_capacity);
    ids[0] = 1; ids[1] = 2; *count = 2;
    if (failure == EMPTY_PROMPT) *count = 0;
    if (failure == BAD_PROMPT_COUNT) *count = capacity + 1;
    return 1;
}

int inference_state_init(const Model *, InferenceState *state, int context,
                         char *error, int capacity)
{
    ++initialized; events.push_back("init"); requested_capacity = context;
    state->implementation = logits;
    state->logits = logits;
    state->logits_count = 7;
    return failure == INIT ? fail(error, capacity) : 1;
}

void inference_state_destroy(InferenceState *state)
{
    ++destroyed; events.push_back("destroy");
    assert(state->implementation == logits);
    memset(state, 0, sizeof(*state));
}

int engine_forward_token(const Model *, InferenceState *state, int id,
                         char *error, int capacity)
{
    forwards.push_back(id); events.push_back("forward");
    if ((failure == PREFILL && forwards.size() == 2) ||
        (failure == DECODE_FORWARD && forwards.size() == 3)) return fail(error, capacity);
    /* A recognizable marker proves the sampler receives the latest forward. */
    logits[0] = (float)id;
    state->logits_count = 7;
    if (failure == NULL_LOGITS) state->logits = nullptr;
    if (failure == WRONG_LOGIT_COUNT) state->logits_count = 6;
    if (state->profile.enabled) {
        ++state->profile.forward_calls;
        state->profile.matvec_seconds += 0.25;
        state->profile.forward_seconds += 0.5;
    }
    return 1;
}

int engine_forward_token_no_logits(const Model *model, InferenceState *state, int id,
                                   char *error, int capacity)
{
    skipped_logits.push_back(id);
    if (!engine_forward_token(model, state, id, error, capacity)) return 0;
    state->logits_count = 0;
    return 1;
}

int sampler_sample(const float *scores, int count, float temperature, float top_p,
                   int *id, char *error, int capacity)
{
    events.push_back("sample");
    assert(scores == logits && count == 7 && temperature == 0 && top_p == 1);
    assert(forwards.size() == (size_t)sampled + 2);
    assert(scores[0] == (sampled == 0 ? 2.0f : (float)samples[sampled - 1]));
    if (failure == SAMPLE) return fail(error, capacity);
    if (failure == BAD_SAMPLE_LOW) { *id = -1; return 1; }
    if (failure == BAD_SAMPLE_HIGH) { *id = 7; return 1; }
    assert((size_t)sampled < samples.size());
    *id = samples[sampled++];
    return 1;
}

int tokenizer_is_eos(const Model *, int id) { return id == 6; }

int tokenizer_decode_one(const Model *, int id, char *text, int capacity,
                         char *error, int error_capacity)
{
    assert(id != 6); decoded.push_back(id); events.push_back("decode");
    if (failure == DECODE) return fail(error, error_capacity);
    if (failure == UNTERMINATED_PIECE) { memset(text, 'x', (size_t)capacity); return 1; }
    assert(id == 3 || id == 4);
    if (empty_first_piece && decoded.size() == 1) { text[0] = '\0'; return 1; }
    snprintf(text, (size_t)capacity, "%s", id == 3 ? "Par" : "is");
    return 1;
}

void terminal_print_startup(const AppConfig *) {}
void terminal_print_prompt(const char *user) { assert(strcmp(user, "Hi") == 0); }
void terminal_print_generated_text(const char *text) { printed += text; events.push_back("print"); }
void terminal_print_error(const char *message) { reported_error = message; }
void terminal_finish_generation(int count, double elapsed, double first_token,
                                const UsageSnapshot *, const UsageSnapshot *)
{
    ++finished; reported_tokens = count;
    reported_elapsed = elapsed; reported_first_token = first_token;
}
void terminal_print_operation_profile(const OperationProfile *profile)
{
    assert(finished == 1 && destroyed == 0 && profile->enabled);
    ++profiles_printed;
    reported_profile = *profile;
}
void metrics_capture(UsageSnapshot *snapshot)
{
    ++captured; memset(snapshot, 0, sizeof(*snapshot));
    snapshot->wall_seconds = 10.0 * captured;
    snapshot->available = captured != unavailable_capture;
    events.push_back("clock");
}
double metrics_elapsed_seconds(const UsageSnapshot *start, const UsageSnapshot *end)
{
    assert(start->available && end->available && end->wall_seconds >= start->wall_seconds);
    return end->wall_seconds - start->wall_seconds;
}

static AppConfig config_for(int max_tokens)
{
    AppConfig config;
    app_config_init(&config);
    strcpy(config.model_path, "fixture.gguf"); strcpy(config.prompt, "Hi");
    config.max_tokens = max_tokens; config.temperature = 0; config.top_p = 1;
    return config;
}

static void expect_success(int limit, const std::vector<int> &script,
                           const std::vector<int> &expected_forwards,
                           const char *text, int count, int profile = 0)
{
    reset(NONE, script);
    AppConfig config = config_for(limit);
    config.profile = profile;
    assert(app_run(&config) == 0);
    assert(forwards == expected_forwards && printed == text);
    assert(skipped_logits == std::vector<int>(1, 1)); /* Only first prompt token. */
    assert(loaded == 1 && unloaded == 1 && initialized == 1 && destroyed == 1);
    assert(finished == 1 && reported_tokens == count && sampled == count);
    const bool has_text = text[0] != '\0';
    assert(captured == (has_text ? 4 : 3));
    assert(reported_first_token == (has_text ? 20.0 : -1.0));
    assert(reported_elapsed == (has_text ? 20.0 : 10.0));
    assert(requested_capacity == 2 + limit && reported_error.empty());
    assert(events[0] == "load" && events[1] == "clock" &&
           events[2] == "format" && events[3] == "encode");
    auto prefill_end = events.begin() + 7; /* load, clock, format, encode, init, two forwards */
    assert(*prefill_end == "clock");
    if (has_text) {
        auto first_print = std::find(events.begin(), events.end(), "print");
        assert(first_print != events.end() && first_print + 1 != events.end());
        assert(*(first_print + 1) == "clock");
    }
    assert(events[events.size() - 2] == "destroy" && events.back() == "unload");
    assert(profiles_printed == profile);
    if (profile) {
        assert(reported_profile.forward_calls == forwards.size());
        assert(reported_profile.sampling_calls == (unsigned long long)count);
        assert(reported_profile.matvec_seconds == 0.25 * forwards.size());
        assert(reported_profile.sampling_seconds >= 0.0);
    }
}

static void test_cli()
{
    AppConfig config;
    app_config_init(&config);
    assert(config.profile == 0);
    char program[] = "inference", model[] = "--model", path[] = "x.gguf";
    char prompt[] = "--prompt", text[] = "Hi", temperature[] = "--temperature";
    char zero[] = "0", top_p[] = "--top-p", one[] = "1", nan[] = "nan";
    char max_tokens[] = "--max-tokens", negative[] = "-1", help[] = "--help";
    char *valid[] = {program, model, path, prompt, text, temperature, zero, top_p, one};
    assert(app_parse_args(9, valid, &config) == 0 && config.temperature == 0 && config.top_p == 1);
    char profile[] = "--profile";
    char *profile_first[] = {program, profile, model, path, prompt, text};
    assert(app_parse_args(6, profile_first, &config) == 0 && config.profile == 1);
    app_config_init(&config);
    char *profile_last[] = {program, model, path, prompt, text, profile};
    assert(app_parse_args(6, profile_last, &config) == 0 && config.profile == 1);
    char unexpected[] = "true";
    char *bad_profile[] = {program, model, path, prompt, text, profile, unexpected};
    assert(app_parse_args(7, bad_profile, &config) == -1);
    char *bad_float[] = {program, model, path, prompt, text, temperature, nan};
    assert(app_parse_args(7, bad_float, &config) == -1);
    char *bad_count[] = {program, model, path, prompt, text, max_tokens, negative};
    assert(app_parse_args(7, bad_count, &config) == -1);
    char *missing[] = {program, model};
    assert(app_parse_args(2, missing, &config) == -1);
    char *help_args[] = {program, help};
    assert(app_parse_args(2, help_args, &config) == 1);
}

static void test_first_token_edge_cases()
{
    AppConfig config = config_for(2);
    reset(NONE, {3, 4});
    empty_first_piece = true;
    assert(app_run(&config) == 0 && printed == "is");
    assert(reported_first_token == 20.0 && captured == 4);
    /* Skip the empty piece; capture immediately after the second piece. */
    auto first_print = std::find(events.begin(), events.end(), "print");
    assert(first_print != events.end() && first_print + 1 != events.end());
    assert(*(first_print + 1) == "forward");

    reset(NONE, {3, 4});
    unavailable_capture = 1; /* Prompt start clock failed. */
    assert(app_run(&config) == 0 && printed == "Paris");
    assert(reported_first_token == -1.0 && captured == 4);
    reset(NONE, {3, 4});
    unavailable_capture = 3; /* First output clock failed; never substitute a later token. */
    assert(app_run(&config) == 0 && printed == "Paris");
    assert(reported_first_token == -1.0 && captured == 4);

    reset(NONE, {3});
    empty_first_piece = true;
    config.max_tokens = 1;
    assert(app_run(&config) == 0 && printed.empty());
    assert(reported_first_token == -1.0 && captured == 3);
}

int main()
{
    expect_success(2, {3, 4}, {1, 2, 3}, "Paris", 2);
    assert((decoded == std::vector<int>{3, 4}));
    expect_success(1, {3}, {1, 2}, "Par", 1);
    expect_success(4, {6}, {1, 2}, "", 1); /* EOS counts, but is never printed. */
    assert(decoded.empty());
    expect_success(4, {3, 6}, {1, 2, 3}, "Par", 2);
    assert((decoded == std::vector<int>{3}));

    expect_success(2, {3, 4}, {1, 2, 3}, "Paris", 2, 1);
    expect_success(1, {3}, {1, 2}, "Par", 1, 1);
    expect_success(4, {6}, {1, 2}, "", 1, 1);
    expect_success(4, {3, 6}, {1, 2, 3}, "Par", 2, 1);

    for (int profile = 0; profile <= 1; ++profile) {
        for (int selected = LOAD; selected <= WRONG_LOGIT_COUNT; ++selected) {
            reset((Failure)selected, {3, 4});
            AppConfig config = config_for(2);
            config.profile = profile;
            assert(app_run(&config) == 1);
            assert(loaded == 1 && unloaded == 1 && finished == 0 && !reported_error.empty());
            const bool attempted_init = selected >= INIT;
            assert(initialized == (int)attempted_init && destroyed == (int)attempted_init);
            assert(events.back() == "unload");
            assert(printed == (selected == DECODE_FORWARD ? "Par" : ""));
            assert(profiles_printed == 0);
        }
    }
    test_cli();
    test_first_token_edge_cases();
    puts("App generation sequencing, EOS, limits, errors, cleanup, and CLI passed.");
    return 0;
}
