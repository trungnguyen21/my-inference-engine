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
                                double time_to_first_token_seconds,
                                const UsageSnapshot *start,
                                const UsageSnapshot *end)
{
    double cpu_percent = metrics_cpu_percent(start, end);
    printf("\n\n\033[1;36mGeneration complete\033[0m\n");
    printf("Generated tokens: %d\n", generated_tokens);
    if (time_to_first_token_seconds >= 0.0) {
        printf("Time to first token: %.3f s (excludes model loading)\n",
               time_to_first_token_seconds);
    } else {
        printf("Time to first token: unavailable\n");
    }
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

void terminal_print_operation_profile(const OperationProfile *profile)
{
    if (profile == NULL || !profile->enabled) return;

    double stages = profile->matvec_seconds + profile->attention_seconds +
                    profile->rmsnorm_seconds + profile->rope_seconds +
                    profile->output_seconds;
    double other = profile->forward_seconds > stages ?
                   profile->forward_seconds - stages : 0.0;
    const double seconds[] = {
        profile->matvec_seconds, profile->attention_seconds,
        profile->rmsnorm_seconds, profile->rope_seconds,
        profile->output_seconds + profile->sampling_seconds, other
    };
    const char *labels[] = {
        "matvec (projections + FFN)", "attention (scores/softmax/V)",
        "RMSNorm", "RoPE", "output matrix + sampling", "other forward work"
    };
    double total = 0.0;
    for (int i = 0; i < 6; ++i) total += seconds[i];

    printf("\nOperation profile (prefill + decode)\n");
    printf("Forward calls: %llu | Sampling calls: %llu\n",
           profile->forward_calls, profile->sampling_calls);
    printf("%-30s %12s %9s\n", "Operation", "Time (ms)", "Share");
    for (int i = 0; i < 6; ++i) {
        printf("%-30s %12.3f %8.2f%%\n", labels[i], seconds[i] * 1000.0,
               total > 0.0 ? seconds[i] / total * 100.0 : 0.0);
    }
    printf("%-30s %12.3f\n", "Total profiled time", total * 1000.0);

    const char *formats[PROFILE_WEIGHT_FORMAT_COUNT] = {
        "F32", "Q5_0", "Q8_0", "Q4_K", "Q6_K"
    };
    double weight_total = 0.0;
    for (int i = 0; i < PROFILE_WEIGHT_FORMAT_COUNT; ++i) {
        weight_total += profile->weight_layer_seconds[i] + profile->weight_output_seconds[i];
    }
    printf("\nWeight kernels by format (prefill + decode)\n");
    printf("%-8s %11s %11s %9s %9s %8s\n",
           "Format", "Layer ms", "Output ms", "Total ms", "Calls", "Share");
    for (int i = 0; i < PROFILE_WEIGHT_FORMAT_COUNT; ++i) {
        double layer = profile->weight_layer_seconds[i];
        double output = profile->weight_output_seconds[i];
        printf("%-8s %11.3f %11.3f %9.3f %9llu %7.2f%%\n",
               formats[i], layer * 1000.0, output * 1000.0,
               (layer + output) * 1000.0,
               profile->weight_layer_calls[i] + profile->weight_output_calls[i],
               weight_total > 0.0 ? (layer + output) / weight_total * 100.0 : 0.0);
    }
    printf("Kernel-only times are subsets of the stage rows above (not additive).\n");
    printf("Excludes model loading, tokenization, and terminal output.\n");
}
