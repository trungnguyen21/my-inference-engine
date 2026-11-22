#ifndef INFERENCE_APP_H
#define INFERENCE_APP_H

#include "engine.h"

/* The CLI's argument buffers track the engine's input limits. */
#define MODEL_PATH_CAPACITY ENGINE_MODEL_PATH_CAPACITY
#define PROMPT_CAPACITY ENGINE_PROMPT_CAPACITY

typedef struct AppConfig {
    char model_path[MODEL_PATH_CAPACITY];
    char prompt[PROMPT_CAPACITY];
    int max_tokens;
    float temperature;
    float top_p;
    float repeat_penalty; /* 1.0 disables; >1.0 discourages already-seen tokens */
    unsigned long seed; /* RNG seed for temperature > 0; ignored otherwise */
    int has_seed; /* 0 means --seed was not passed: use a random seed */
    int profile; /* collect and print operation timings; disabled by default */
} AppConfig;

void app_config_init(AppConfig *config);
int app_parse_args(int argc, char **argv, AppConfig *config);
int app_run(const AppConfig *config);
void app_print_usage(const char *program_name);

#endif
