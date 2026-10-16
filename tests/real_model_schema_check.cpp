#include <math.h>
#include <stdio.h>
#include <string.h>

#include "app.h"
#include "engine.h"

int main(int argc, char **argv)
{
    AppConfig config;
    Model model;
    char error[ENGINE_ERROR_CAPACITY];

    if (argc != 2) {
        fprintf(stderr, "Usage: %s path/to/model.gguf\n", argv[0]);
        return 2;
    }

    memset(&config, 0, sizeof(config));
    memset(&model, 0, sizeof(model));
    snprintf(config.model_path, sizeof(config.model_path), "%s", argv[1]);
    error[0] = '\0';

    if (!engine_load_model(&config, &model, error, (int)sizeof(error))) {
        fprintf(stderr, "Could not load and validate model: %s\n", error);
        return 1;
    }
    if (!isfinite(model.config.rms_norm_epsilon) ||
        model.config.rms_norm_epsilon <= 0.0f ||
        !isfinite(model.config.rope_frequency_base) ||
        model.config.rope_frequency_base <= 0.0f) {
        fprintf(stderr, "Model loader did not retain valid RMSNorm/RoPE metadata.\n");
        engine_unload_model(&model);
        return 1;
    }

    /* Guard the adapter flag needed by the instruction formatter. Ordinary
     * BPE text pieces here would make a correctly formatted prompt ineffective. */
    const char *markers[] = {"<|im_start|>", "<|im_end|>"};
    int marker_ids[2] = {-1, -1};
    for (int i = 0; i < 2; ++i) {
        int ids[16], count = 0;
        if (!tokenizer_encode(&model, markers[i], ids, 16, &count, error, sizeof(error)) ||
            count != 1 || ids[0] < 0 || ids[0] >= model.config.vocab_size) {
            fprintf(stderr, "Instruction marker did not encode as one special token: %s\n", markers[i]);
            engine_unload_model(&model);
            return 1;
        }
        marker_ids[i] = ids[0];
    }
    if (marker_ids[0] == marker_ids[1] || !tokenizer_is_eos(&model, marker_ids[1])) {
        fprintf(stderr, "Instruction marker IDs or end-of-generation detection are invalid.\n");
        engine_unload_model(&model);
        return 1;
    }

    printf("Validated Qwen schema: dim=%d, layers=%d, heads=%d/%d, "
           "rms_epsilon=%g, rope_base=%g\n",
           model.config.dim, model.config.n_layers, model.config.n_heads,
           model.config.n_kv_heads, model.config.rms_norm_epsilon,
           model.config.rope_frequency_base);
    puts("Validated instruction-marker tokenization and assistant end-token detection.");
    engine_unload_model(&model);
    return 0;
}
