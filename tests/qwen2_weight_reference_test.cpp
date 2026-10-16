#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "qwen2_quantized_test_utils.h"

static const float SENTINEL = -12345.0f;

static void expect_row_failure(
    const GgufQuantizedTensorView *view, int row, int dim, int null_output = 0
)
{
    float output[512];
    for (int i = 0; i < 512; ++i) output[i] = SENTINEL;
    char error[512] = {};
    assert(qwen2_decode_quantized_embedding_row(
        view, row, dim, null_output ? nullptr : output, error, sizeof(error)) == 0);
    assert(error[0] != '\0');
    for (int i = 0; i < 512; ++i) assert(output[i] == SENTINEL);
}

static void test_invalid_embedding_views(void)
{
    PackedWeightFixture fixture;
    fixture_init_packed(&fixture, ENGINE_GGML_TYPE_Q8_0);
    GgufQuantizedTensorView valid;
    assert(gguf_get_quantized_tensor_view(&fixture.file, fixture.tensor.name, &valid) == 1);
    expect_row_failure(nullptr, 0, fixture.width);
    expect_row_failure(&valid, 0, fixture.width, 1);
    expect_row_failure(&valid, -1, fixture.width);
    expect_row_failure(&valid, 3, fixture.width);
    expect_row_failure(&valid, 0, 0);
    expect_row_failure(&valid, 0, fixture.width - 1);
    for (int invalid_case = 0; invalid_case < 9; ++invalid_case) {
        GgufQuantizedTensorView view = valid;
        if (invalid_case == 0) view.dimension_count = 1;
        if (invalid_case == 1) view.ggml_type = 99;
        if (invalid_case == 2) view.block_elements += 1;
        if (invalid_case == 3) view.block_bytes += 1;
        if (invalid_case == 4) view.block_count -= 1;
        if (invalid_case == 5) view.element_count -= 1;
        if (invalid_case == 6) view.data = nullptr;
        if (invalid_case == 7) view.dimensions[1] = UINT64_MAX;
        if (invalid_case == 8) view.dimensions[1] = 0;
        expect_row_failure(&view, 0, fixture.width);
    }
    /* The whole tensor fits blocks, but individual rows do not. */
    GgufQuantizedTensorView split_rows = valid;
    split_rows.dimensions[0] = 48;
    split_rows.dimensions[1] = 4;
    expect_row_failure(&split_rows, 0, 48);
    assert(qwen2_decode_quantized_embedding_row(nullptr, 0, 64, nullptr, nullptr, 0) == 0);
}

static void test_embedding_type(uint32_t type)
{
    PackedWeightFixture fixture;
    fixture_init_packed(&fixture, type);
    GgufQuantizedTensorView view;
    assert(gguf_get_quantized_tensor_view(&fixture.file, fixture.tensor.name, &view) == 1);
    unsigned char original[sizeof(fixture.storage)];
    memcpy(original, fixture.storage, sizeof(original));
    for (int row = 0; row < 3; ++row) {
        float guarded[514];
        for (int i = 0; i < 514; ++i) guarded[i] = SENTINEL;
        char error[512] = {};
        int result = qwen2_decode_quantized_embedding_row(
            &view, row, fixture.width, guarded + 1, error, sizeof(error));
        if (!result) fprintf(stderr, "Embedding type %u, row %d: %s\n", type, row, error);
        assert(result == 1);
        for (int i = 0; i < fixture.width; ++i) {
            int block = row * 2 + i / fixture_block_elements(type);
            assert(guarded[i + 1] == fixture.block_values[block]);
        }
        assert(guarded[0] == SENTINEL && guarded[fixture.width + 1] == SENTINEL);
        assert(memcmp(original, fixture.storage, sizeof(original)) == 0);
    }
    float output[512];
    assert(qwen2_decode_quantized_embedding_row(&view, 1, fixture.width, output, nullptr, 0) == 1);
}

static void expect_matvec_failure(
    const GgufFile *file, const char *name, const float *input,
    int columns, int rows, int null_output = 0
)
{
    float output[] = {SENTINEL, SENTINEL, SENTINEL};
    char error[512] = {};
    assert(qwen2_weight_matvec(file, name, input, columns, null_output ? nullptr : output,
                               rows, error, sizeof(error)) == 0);
    assert(error[0] != '\0');
    for (int i = 0; i < 3; ++i) assert(output[i] == SENTINEL);
}

static void test_invalid_matvec_inputs(void)
{
    PackedWeightFixture fixture;
    fixture_init_packed(&fixture, ENGINE_GGML_TYPE_Q8_0);
    const char *name = fixture.tensor.name;
    float input[64] = {};
    expect_matvec_failure(nullptr, name, input, 64, 3);
    expect_matvec_failure(&fixture.file, nullptr, input, 64, 3);
    expect_matvec_failure(&fixture.file, name, nullptr, 64, 3);
    expect_matvec_failure(&fixture.file, name, input, 64, 3, 1);
    expect_matvec_failure(&fixture.file, name, input, 0, 3);
    expect_matvec_failure(&fixture.file, name, input, 64, 0);
    expect_matvec_failure(&fixture.file, "missing.weight", input, 64, 3);
    expect_matvec_failure(&fixture.file, name, input, 63, 3);
    expect_matvec_failure(&fixture.file, name, input, 64, 2);
    fixture.tensor.dimension_count = 1;
    expect_matvec_failure(&fixture.file, name, input, 64, 3);
    fixture.tensor.dimension_count = 2;
    fixture.tensor.type = 99;
    expect_matvec_failure(&fixture.file, name, input, 64, 3);
    fixture.tensor.type = ENGINE_GGML_TYPE_Q8_0;
    fixture.tensor.byte_size -= 1;
    expect_matvec_failure(&fixture.file, name, input, 64, 3);
    fixture.tensor.byte_size += 1;
    fixture.file.file_size -= 1;
    expect_matvec_failure(&fixture.file, name, input, 64, 3);
    fixture.file.file_size += 1;
    fixture.tensor.dimensions[0] = 48;
    fixture.tensor.dimensions[1] = 4;
    expect_matvec_failure(&fixture.file, name, input, 48, 4);
    char error[512] = {};
    assert(qwen2_weight_matvec(&fixture.file, name, input, 48, input, 4, error, sizeof(error)) == 0);
    assert(error[0] != '\0');
}

static void test_f32_matvec(void)
{
    float weights[] = {1, 2, 3, 4, 5, 6};
    float input[] = {1, 0, -1};
    float guarded[] = {SENTINEL, 0, 0, SENTINEL};
    GgufTensorInfo tensor = {};
    tensor.name = (char *)"output.weight";
    tensor.type = ENGINE_GGML_TYPE_F32;
    tensor.dimension_count = 2;
    tensor.dimensions[0] = 3;
    tensor.dimensions[1] = 2;
    tensor.byte_size = sizeof(weights);
    GgufFile file = {};
    file.data = (const unsigned char *)weights;
    file.file_size = sizeof(weights);
    file.tensor_count = 1;
    file.tensors = &tensor;
    assert(qwen2_weight_matvec(&file, tensor.name, input, 3, guarded + 1, 2, nullptr, 0) == 1);
    assert(guarded[1] == -2 && guarded[2] == -2);
    assert(guarded[0] == SENTINEL && guarded[3] == SENTINEL);
}

static void test_matvec_type(uint32_t type)
{
    PackedWeightFixture fixture;
    fixture_init_packed(&fixture, type);
    float input[512];
    int block_elements = fixture_block_elements(type);
    for (int i = 0; i < fixture.width; ++i) input[i] = i < block_elements ? 1.0f : 2.0f;
    unsigned char original[sizeof(fixture.storage)];
    memcpy(original, fixture.storage, sizeof(original));
    const char *names[] = {"blk.2.attn_k.weight", "output.weight"};
    for (int name_index = 0; name_index < 2; ++name_index) {
        fixture.tensor.name = (char *)names[name_index];
        float guarded[] = {SENTINEL, 0, 0, 0, SENTINEL};
        char error[512] = {};
        int result = qwen2_weight_matvec(&fixture.file, fixture.tensor.name, input, fixture.width,
                                         guarded + 1, 3, error, sizeof(error));
        if (!result) fprintf(stderr, "Matvec type %u, %s: %s\n", type, fixture.tensor.name, error);
        assert(result == 1);
        for (int row = 0; row < 3; ++row) {
            float expected = (float)block_elements *
                (fixture.block_values[row * 2] + 2.0f * fixture.block_values[row * 2 + 1]);
            assert(guarded[row + 1] == expected);
        }
        assert(guarded[0] == SENTINEL && guarded[4] == SENTINEL);
        assert(memcmp(original, fixture.storage, sizeof(original)) == 0);
    }
    float output[3];
    assert(qwen2_weight_matvec(&fixture.file, fixture.tensor.name, input, fixture.width,
                               output, 3, nullptr, 0) == 1);
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3 ||
        (strcmp(argv[1], "embedding") != 0 && strcmp(argv[1], "matvec") != 0)) {
        fprintf(stderr, "Usage: %s embedding|matvec [q5_0|q8_0|q4_k|q6_k|invalid|f32]\n", argv[0]);
        return 2;
    }
    int embedding = strcmp(argv[1], "embedding") == 0;
    if (embedding) test_invalid_embedding_views();
    else {
        test_f32_matvec();
        test_invalid_matvec_inputs();
    }
    if (argc == 3 && strcmp(argv[2], "invalid") == 0) return 0;
    if (argc == 3 && strcmp(argv[2], "f32") == 0) return embedding ? 2 : 0;
    const uint32_t types[] = {
        ENGINE_GGML_TYPE_Q5_0, ENGINE_GGML_TYPE_Q8_0,
        ENGINE_GGML_TYPE_Q4_K, ENGINE_GGML_TYPE_Q6_K
    };
    uint32_t selected = argc == 3 ? fixture_parse_type(argv[2]) : 0;
    if (argc == 3 && selected == 0) return 2;
    for (int i = 0; i < 4; ++i) {
        if (selected && types[i] != selected) continue;
        if (embedding) test_embedding_type(types[i]);
        else test_matvec_type(types[i]);
    }
    printf("Qwen2 %s reference checks passed.\n", argv[1]);
    return 0;
}
