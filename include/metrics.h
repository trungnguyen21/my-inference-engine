#ifndef INFERENCE_METRICS_H
#define INFERENCE_METRICS_H

/* Per-request wall-clock totals. Forward stages exclude sampling, and the
 * output projection is separate from transformer-layer matrix products. */
enum { PROFILE_F32, PROFILE_Q5_0, PROFILE_Q8_0, PROFILE_Q4_K, PROFILE_Q6_K,
       PROFILE_WEIGHT_FORMAT_COUNT };
typedef struct OperationProfile {
    int enabled;
    double matvec_seconds;
    double attention_seconds;
    double rmsnorm_seconds;
    double rope_seconds;
    double output_seconds;
    double sampling_seconds;
    /* Kernel-only time by weight format; output projections are kept separate.
     * These are subsets of matvec_seconds and output_seconds, not extra stages. */
    double weight_layer_seconds[PROFILE_WEIGHT_FORMAT_COUNT];
    double weight_output_seconds[PROFILE_WEIGHT_FORMAT_COUNT];
    unsigned long long weight_layer_calls[PROFILE_WEIGHT_FORMAT_COUNT];
    unsigned long long weight_output_calls[PROFILE_WEIGHT_FORMAT_COUNT];
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
