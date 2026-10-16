#ifndef INFERENCE_METRICS_H
#define INFERENCE_METRICS_H

typedef struct UsageSnapshot {
    double wall_seconds;
    double process_cpu_seconds;
    unsigned long long resident_bytes;
    int available;
} UsageSnapshot;

void metrics_capture(UsageSnapshot *snapshot);
double metrics_elapsed_seconds(const UsageSnapshot *start,
                               const UsageSnapshot *end);
double metrics_tokens_per_second(int generated_tokens, double elapsed_seconds);
double metrics_cpu_percent(const UsageSnapshot *start,
                           const UsageSnapshot *end);

#endif
