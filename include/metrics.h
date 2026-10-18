#ifndef INFERENCE_METRICS_H
#define INFERENCE_METRICS_H

/* Per-request wall-clock totals. Forward stages exclude sampling, and the
 * output projection is separate from transformer-layer matrix products. */
typedef struct OperationProfile {
    int enabled;
    double matvec_seconds;
    double attention_seconds;
    double rmsnorm_seconds;
    double rope_seconds;
    double output_seconds;
    double sampling_seconds;
    double forward_seconds;
    unsigned long long forward_calls;
    unsigned long long sampling_calls;
} OperationProfile;

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
