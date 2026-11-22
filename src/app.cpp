#include "app.h"

#include "engine.h"
#include "generate.h"
#include "metrics.h"
#include "terminal.h"
#include "sampler.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_TOKEN_CAPACITY 32768

static int copy_argument(char *destination, int capacity, const char *source)
{
    size_t length = strlen(source);
    if (length >= (size_t)capacity) {
        return 0;
    }
    memcpy(destination, source, length + 1);
    return 1;
}

static int parse_positive_int(const char *text, int *value)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

static int parse_unit_float(const char *text, float *value)
{
    char *end = NULL;
    errno = 0;
    float parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) ||
        parsed < 0.0f || parsed > 1.0f) {
        return 0;
    }
    *value = parsed;
    return 1;
}

static int parse_seed(const char *text, unsigned long *value)
{
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *value = parsed;
    return 1;
}

static int parse_repeat_penalty(const char *text, float *value)
{
    char *end = NULL;
    errno = 0;
    float parsed = strtof(text, &end);
    /* 1.0 disables the penalty; values much above 2.0 degrade quality. */
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) ||
        parsed < 1.0f || parsed > 2.0f) {
        return 0;
    }
    *value = parsed;
    return 1;
}

void app_config_init(AppConfig *config)
{
    memset(config, 0, sizeof(*config));
    config->max_tokens = 128;
    config->temperature = 0.8f;
    config->top_p = 0.95f;
    config->repeat_penalty = 1.1f;
}

void app_print_usage(const char *program_name)
{
    fprintf(stderr,
            "Usage: %s --model PATH --prompt TEXT [options]\n"
            "Options:\n"
            "  --max-tokens N   Maximum generated tokens (default: 128)\n"
            "  --temperature X  Sampling temperature in [0, 1] (default: 0.8)\n"
            "  --top-p X        Nucleus sampling threshold in [0, 1] (default: 0.95)\n"
            "  --repeat-penalty X  Penalty for already-seen tokens in [1, 2] (default: 1.1, 1 disables)\n"
            "  --seed N         RNG seed for temperature > 0 (default: random each run)\n"
            "  --profile        Collect and display operation timings (prefill + decode)\n"
            "  --help           Show this help\n",
            program_name);
}

int app_parse_args(int argc, char **argv, AppConfig *config)
{
    int has_model = 0;
    int has_prompt = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        const char *option = argv[i];
        const char *value;
        if (strcmp(option, "--help") == 0 || strcmp(option, "-h") == 0) {
            app_print_usage(argv[0]);
            return 1;
        }
        if (strcmp(option, "--profile") == 0) {
            config->profile = 1;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for %s\n", option);
            app_print_usage(argv[0]);
            return -1;
        }
        value = argv[++i];

        if (strcmp(option, "--model") == 0) {
            if (!copy_argument(config->model_path, MODEL_PATH_CAPACITY, value)) {
                fprintf(stderr, "Model path is too long (maximum %d characters).\n",
                        MODEL_PATH_CAPACITY - 1);
                return -1;
            }
            has_model = 1;
        } else if (strcmp(option, "--prompt") == 0) {
            if (!copy_argument(config->prompt, PROMPT_CAPACITY, value)) {
                fprintf(stderr, "Prompt is too long (maximum %d characters).\n",
                        PROMPT_CAPACITY - 1);
                return -1;
            }
            has_prompt = 1;
        } else if (strcmp(option, "--max-tokens") == 0) {
            if (!parse_positive_int(value, &config->max_tokens)) {
                fprintf(stderr, "Invalid --max-tokens value: %s\n", value);
                return -1;
            }
        } else if (strcmp(option, "--temperature") == 0) {
            if (!parse_unit_float(value, &config->temperature)) {
                fprintf(stderr, "Invalid --temperature value (expected 0 to 1): %s\n", value);
                return -1;
            }
        } else if (strcmp(option, "--top-p") == 0) {
            if (!parse_unit_float(value, &config->top_p)) {
                fprintf(stderr, "Invalid --top-p value (expected 0 to 1): %s\n", value);
                return -1;
            }
        } else if (strcmp(option, "--repeat-penalty") == 0) {
            if (!parse_repeat_penalty(value, &config->repeat_penalty)) {
                fprintf(stderr, "Invalid --repeat-penalty value (expected 1 to 2): %s\n", value);
                return -1;
            }
        } else if (strcmp(option, "--seed") == 0) {
            if (!parse_seed(value, &config->seed)) {
                fprintf(stderr, "Invalid --seed value: %s\n", value);
                return -1;
            }
            config->has_seed = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", option);
            app_print_usage(argv[0]);
            return -1;
        }
    }

    if (!has_model || !has_prompt) {
        fprintf(stderr, "Both --model and --prompt are required.\n");
        app_print_usage(argv[0]);
        return -1;
    }
    return 0;
}

static int app_emit_token(const char *text, void *user)
{
    (void)user;
    terminal_print_generated_text(text);
    return 1; /* The CLI never stops early. */
}

int app_run(const AppConfig *config)
{
    Model model;
    InferenceState state;
    GenerateResult result;
    char error[ENGINE_ERROR_CAPACITY];
    int *prompt_token_ids;
    int token_capacity = PROMPT_TOKEN_CAPACITY;
    int prompt_token_count = 0;
    int requested_context;
    int model_loaded = 0;
    int state_initialized = 0;
    int exit_code = 1;
    UsageSnapshot prompt_start_usage;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&result, 0, sizeof(result));
    error[0] = '\0';
    prompt_token_ids = (int *)malloc((size_t)token_capacity * sizeof(*prompt_token_ids));
    if (prompt_token_ids == NULL) {
        terminal_print_error("Could not allocate token context buffer.");
        return 1;
    }
    terminal_print_startup(config);
    terminal_print_prompt(config->prompt);

    /* Load GGUF metadata and retain its tensor directory for engine work. */
    model_loaded = 1; /* Cleanup must also handle a partially failed load. */
    if (!engine_load_model(config->model_path, &model, error, sizeof(error))) {
        terminal_print_error(error);
        goto cleanup;
    }

    /* First-token latency includes prompt processing, starting with a ready model. */
    metrics_capture(&prompt_start_usage);
    if (!generate_prepare_prompt(&model, config->prompt, prompt_token_ids,
                                 token_capacity, &prompt_token_count,
                                 error, sizeof(error))) {
        terminal_print_error(error);
        goto cleanup;
    }
    if (model.config.vocab_size <= 0 || model.config.seq_len <= 0) {
        terminal_print_error("Model metadata has an invalid vocabulary or context size.");
        goto cleanup;
    }
    requested_context = prompt_token_count + config->max_tokens;
    if (requested_context > model.config.seq_len) {
        snprintf(error, sizeof(error),
                 "Prompt plus requested output (%d tokens) exceeds model context (%d tokens).",
                 requested_context, model.config.seq_len);
        terminal_print_error(error);
        goto cleanup;
    }
    state_initialized = 1; /* Destroy must also handle a partially failed init. */
    if (!inference_state_init(&model, &state, requested_context,
                              error, sizeof(error))) {
        terminal_print_error(error);
        goto cleanup;
    }

    state.profile.enabled = config->profile;

    if (config->has_seed) {
        sampler_seed_rng(config->seed);
    }

    if (!generate_run(&model, &state, prompt_token_ids, prompt_token_count,
                      config->max_tokens, config->temperature, config->top_p,
                      config->repeat_penalty,
                      &prompt_start_usage, app_emit_token, NULL,
                      &result, error, sizeof(error))) {
        terminal_print_error(error);
        goto cleanup;
    }
    free(prompt_token_ids);
    prompt_token_ids = NULL;

    terminal_finish_generation(result.generated_tokens,
                               metrics_elapsed_seconds(&result.start_usage,
                                                       &result.end_usage),
                               result.time_to_first_token_seconds,
                               &result.start_usage, &result.end_usage);
    if (config->profile) terminal_print_operation_profile(&state.profile);
    exit_code = 0;

cleanup:
    if (state_initialized) {
        inference_state_destroy(&state);
    }
    if (model_loaded) {
        engine_unload_model(&model);
    }
    free(prompt_token_ids);
    return exit_code;
}
