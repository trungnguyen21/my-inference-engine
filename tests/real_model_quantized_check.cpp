#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf_reader.h"
#include "quantized_matvec.h"

static const GgufTensorInfo *find_first_matrix_of_type(
    const GgufFile *file,
    uint32_t ggml_type
)
{
    for (uint64_t i = 0; i < file->tensor_count; ++i) {
        const GgufTensorInfo *tensor = &file->tensors[i];

        if (tensor->type == ggml_type && tensor->dimension_count == 2 &&
            tensor->dimensions[0] > 0 && tensor->dimensions[0] <= INT_MAX &&
            tensor->dimensions[1] > 0) {
            return tensor;
        }
    }
    return nullptr;
}

static int check_one_type(const GgufFile *file, uint32_t ggml_type)
{
    const GgufTensorInfo *tensor = find_first_matrix_of_type(file, ggml_type);
    GgufQuantizedTensorView full_view;
    GgufQuantizedTensorView first_row_view;
    float *input;
    float output = 0.0f;
    int input_count;
    int success = 0;

    if (tensor == nullptr) {
        fprintf(stderr, "No two-dimensional tensor found for GGML type %u.\n", ggml_type);
        return 0;
    }
    if (!gguf_get_quantized_tensor_view(file, tensor->name, &full_view)) {
        fprintf(stderr, "Could not create packed view for %s.\n", tensor->name);
        return 0;
    }

    /* Check only the first row: it keeps this integration check quick. */
    first_row_view = full_view;
    first_row_view.dimension_count = 2;
    first_row_view.dimensions[1] = 1;
    first_row_view.element_count = first_row_view.dimensions[0];
    first_row_view.block_count =
        first_row_view.element_count / first_row_view.block_elements;

    input_count = (int)first_row_view.dimensions[0];
    input = (float *)malloc((size_t)input_count * sizeof(*input));
    if (input == nullptr) {
        fprintf(stderr, "Could not allocate activation fixture.\n");
        return 0;
    }
    for (int i = 0; i < input_count; ++i) {
        input[i] = ((float)((i % 7) - 3)) * 0.125f;
    }

    if (quantized_tensor_matvec_f32(
        &first_row_view,
        input,
        input_count,
        &output,
        1
    ) && isfinite(output)) {
        printf("%s (type %u, row width %d): %f\n",
               tensor->name, ggml_type, input_count, output);
        success = 1;
    } else {
        fprintf(stderr, "Packed matrix-vector check failed for %s.\n", tensor->name);
    }

    free(input);
    return success;
}

int main(int argc, char **argv)
{
    const uint32_t types[] = {
        ENGINE_GGML_TYPE_Q5_0,
        ENGINE_GGML_TYPE_Q8_0,
        ENGINE_GGML_TYPE_Q4_K,
        ENGINE_GGML_TYPE_Q6_K
    };
    GgufFile file;
    char error[512];
    int success = 1;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s path/to/model.gguf\n", argv[0]);
        return 2;
    }
    if (!gguf_open(argv[1], &file, error, (int)sizeof(error))) {
        fprintf(stderr, "Could not open GGUF: %s\n", error);
        return 1;
    }

    for (int i = 0; i < (int)(sizeof(types) / sizeof(types[0])); ++i) {
        if (!check_one_type(&file, types[i])) {
            success = 0;
        }
    }

    gguf_close(&file);
    if (!success) {
        return 1;
    }
    printf("All real-model packed tensor checks passed.\n");
    return 0;
}
