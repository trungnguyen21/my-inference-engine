#ifndef INFERENCE_APP_H
#define INFERENCE_APP_H

#define MODEL_PATH_CAPACITY 1024
#define PROMPT_CAPACITY 4096

typedef struct AppConfig {
    char model_path[MODEL_PATH_CAPACITY];
    char prompt[PROMPT_CAPACITY];
    int max_tokens;
    float temperature;
    float top_p;
} AppConfig;

void app_config_init(AppConfig *config);
int app_parse_args(int argc, char **argv, AppConfig *config);
int app_run(const AppConfig *config);
void app_print_usage(const char *program_name);

#endif
