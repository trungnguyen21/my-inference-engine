#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "gguf_reader.h"

static const char *FIXTURE_NAME = "fixture.quantized";

static void setup_quantized_fixture(
    GgufFile *file,
    GgufTensorInfo *tensor,
    const unsigned char *data,
    uint32_t ggml_type,
    uint64_t element_count,
    uint64_t byte_size
)
{
    memset(file, 0, sizeof(*file));
    memset(tensor, 0, sizeof(*tensor));

    tensor->name = (char *)FIXTURE_NAME;
    tensor->dimension_count = 1;
    tensor->dimensions[0] = element_count;
    tensor->type = ggml_type;
    tensor->offset = 0;
    tensor->byte_size = byte_size;

    file->data = data;
    file->file_size = byte_size;
    file->tensor_data_offset = 0;
    file->tensor_count = 1;
    file->tensors = tensor;
}

static void set_view_sentinel(GgufQuantizedTensorView *view)
{
    view->data = (const unsigned char *)"stale";
    view->ggml_type = 99u;
    view->dimension_count = 4;
    view->dimensions[0] = 1;
    view->dimensions[1] = 2;
    view->dimensions[2] = 3;
    view->dimensions[3] = 4;
    view->element_count = 24;
    view->block_elements = 32;
    view->block_bytes = 34;
    view->block_count = 1;
}

static void assert_empty_view(const GgufQuantizedTensorView *view)
{
    assert(view->data == nullptr);
    assert(view->ggml_type == 0u);
    assert(view->dimension_count == 0u);
    assert(view->dimensions[0] == 0u);
    assert(view->dimensions[1] == 0u);
    assert(view->dimensions[2] == 0u);
    assert(view->dimensions[3] == 0u);
    assert(view->element_count == 0u);
    assert(view->block_elements == 0u);
    assert(view->block_bytes == 0u);
    assert(view->block_count == 0u);
}

static void test_supported_views(void)
{
    struct Case {
        uint32_t ggml_type;
        uint64_t element_count;
        uint64_t block_bytes;
    } cases[] = {
        {ENGINE_GGML_TYPE_Q5_0, 32u, 22u},
        {ENGINE_GGML_TYPE_Q8_0, 32u, 34u},
        {ENGINE_GGML_TYPE_Q4_K, 256u, 144u},
        {ENGINE_GGML_TYPE_Q6_K, 256u, 210u}
    };
    unsigned char data[210];

    memset(data, 0, sizeof(data));
    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); ++i) {
        GgufFile file;
        GgufTensorInfo tensor;
        GgufQuantizedTensorView view;

        setup_quantized_fixture(
            &file,
            &tensor,
            data,
            cases[i].ggml_type,
            cases[i].element_count,
            cases[i].block_bytes
        );
        memset(&view, 0, sizeof(view));

        assert(gguf_get_quantized_tensor_view(&file, FIXTURE_NAME, &view) == 1);
        assert(view.data == data);
        assert(view.ggml_type == cases[i].ggml_type);
        assert(view.dimension_count == 1u);
        assert(view.dimensions[0] == cases[i].element_count);
        assert(view.element_count == cases[i].element_count);
        assert(view.block_elements == cases[i].element_count);
        assert(view.block_bytes == cases[i].block_bytes);
        assert(view.block_count == 1u);
    }
}

static void test_invalid_requests_clear_view(void)
{
    unsigned char data[34];
    GgufFile file;
    GgufTensorInfo tensor;
    GgufQuantizedTensorView view;

    memset(data, 0, sizeof(data));
    setup_quantized_fixture(
        &file,
        &tensor,
        data,
        ENGINE_GGML_TYPE_Q8_0,
        32u,
        34u
    );

    set_view_sentinel(&view);
    assert(gguf_get_quantized_tensor_view(nullptr, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    set_view_sentinel(&view);
    assert(gguf_get_quantized_tensor_view(&file, nullptr, &view) == 0);
    assert_empty_view(&view);

    assert(gguf_get_quantized_tensor_view(&file, FIXTURE_NAME, nullptr) == 0);

    set_view_sentinel(&view);
    assert(gguf_get_quantized_tensor_view(&file, "missing.tensor", &view) == 0);
    assert_empty_view(&view);

    tensor.type = ENGINE_GGML_TYPE_F32;
    set_view_sentinel(&view);
    assert(gguf_get_quantized_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    tensor.type = ENGINE_GGML_TYPE_Q8_0;
    tensor.dimensions[0] = 31u;
    set_view_sentinel(&view);
    assert(gguf_get_quantized_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    tensor.dimensions[0] = 32u;
    tensor.byte_size = 33u;
    file.file_size = 33u;
    set_view_sentinel(&view);
    assert(gguf_get_quantized_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);
}

int main(void)
{
    printf("Running GGUF quantized-view tests...\n");

    test_supported_views();
    test_invalid_requests_clear_view();

    printf("All GGUF quantized-view tests passed.\n");
    return 0;
}
