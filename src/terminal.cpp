#include "terminal.h"

#include "metrics.h"

#include <stdio.h>

void terminal_print_startup(const AppConfig *config)
{
    printf("\033[1;36mInference skeleton\033[0m\n");
    printf("Model: %s\n", config->model_path);
    printf("Generation: max_tokens=%d temperature=%.3f top_p=%.3f\n\n",
           config->max_tokens, config->temperature, config->top_p);
}

void terminal_print_prompt(const char *prompt)
{
    printf("\033[1;33mPrompt:\033[0m %s\n\n\033[1;32mOutput:\033[0m ", prompt);
    fflush(stdout);
}

void terminal_print_generated_text(const char *text)
{
    fputs(text, stdout);
    fflush(stdout);
}

void terminal_finish_generation(int generated_tokens, double elapsed_seconds,
                                const UsageSnapshot *start,
                                const UsageSnapshot *end)
{
    double cpu_percent = metrics_cpu_percent(start, end);
    printf("\n\n\033[1;36mGeneration complete\033[0m\n");
    printf("Generated tokens: %d\n", generated_tokens);
    printf("Elapsed time: %.3f s\n", elapsed_seconds);
    printf("Throughput: %.2f tokens/s\n",
           metrics_tokens_per_second(generated_tokens, elapsed_seconds));
    if (cpu_percent >= 0.0) {
        printf("Process CPU: %.1f%% (can exceed 100%% on multiple cores)\n", cpu_percent);
    } else {
        printf("Process CPU: unavailable\n");
    }
    if (end != NULL && end->resident_bytes > 0) {
        printf("Resident RAM: %.1f MiB\n",
               (double)end->resident_bytes / (1024.0 * 1024.0));
    } else {
        printf("Resident RAM: unavailable\n");
    }
}

void terminal_print_error(const char *message)
{
    fprintf(stderr, "\n\033[1;31mError:\033[0m %s\n", message);
}
