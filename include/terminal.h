#ifndef INFERENCE_TERMINAL_H
#define INFERENCE_TERMINAL_H

#include "app.h"
#include "metrics.h"

void terminal_print_startup(const AppConfig *config);
void terminal_print_prompt(const char *prompt);
void terminal_print_generated_text(const char *text);
void terminal_finish_generation(int generated_tokens, double elapsed_seconds,
                                const UsageSnapshot *start,
                                const UsageSnapshot *end);
void terminal_print_error(const char *message);
void terminal_print_operation_profile(const OperationProfile *profile);

#endif
