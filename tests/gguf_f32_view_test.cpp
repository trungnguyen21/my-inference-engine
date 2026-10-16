#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gguf_reader.h"

static char FIXTURE_NAME[] = "f32.fixture";

static void setup_f32_fixture(
    GgufFile *file,
    GgufTensorInfo *tensor,
    const float *values,
    uint64_t byte_size
)
{
    memset(file, 0, sizeof(*file));
    memset(tensor, 0, sizeof(*tensor));

    tensor->name = FIXTURE_NAME;
    tensor->dimension_count = 2;
    tensor->dimensions[0] = 2;
    tensor->dimensions[1] = 2;
    tensor->type = 0;
    tensor->offset = 0;
    tensor->byte_size = byte_size;

    file->data = (const unsigned char *)values;
    file->file_size = byte_size;
    file->tensor_data_offset = 0;
    file->tensor_count = 1;
    file->tensors = tensor;
}

static void set_view_sentinel(GgufF32TensorView *view, const float *data)
{
    view->data = data;
    view->dimension_count = 4;
    view->dimensions[0] = 1;
    view->dimensions[1] = 2;
    view->dimensions[2] = 3;
    view->dimensions[3] = 4;
    view->element_count = 24;
}

static void assert_empty_view(const GgufF32TensorView *view)
{
    assert(view->data == nullptr);
    assert(view->dimension_count == 0);
    assert(view->dimensions[0] == 0);
    assert(view->dimensions[1] == 0);
    assert(view->dimensions[2] == 0);
    assert(view->dimensions[3] == 0);
    assert(view->element_count == 0);
}

static void test_valid_f32_view(void)
{
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    GgufFile file;
    GgufTensorInfo tensor;
    GgufF32TensorView view;

    setup_f32_fixture(&file, &tensor, values, sizeof(values));
    memset(&view, 0, sizeof(view));

    assert(gguf_get_f32_tensor_view(&file, FIXTURE_NAME, &view) == 1);
    assert(view.data == values);
    assert(view.dimension_count == 2);
    assert(view.dimensions[0] == 2);
    assert(view.dimensions[1] == 2);
    assert(view.element_count == 4);
    assert(view.data[0] == 1.0f);
    assert(view.data[3] == 4.0f);
}

static void test_invalid_requests_clear_view(void)
{
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    GgufFile file;
    GgufTensorInfo tensor;
    GgufF32TensorView view;

    setup_f32_fixture(&file, &tensor, values, sizeof(values));

    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(nullptr, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(&file, nullptr, &view) == 0);
    assert_empty_view(&view);

    assert(gguf_get_f32_tensor_view(&file, FIXTURE_NAME, nullptr) == 0);

    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(&file, "missing.tensor", &view) == 0);
    assert_empty_view(&view);
}

static void test_rejected_tensor_layouts(void)
{
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    GgufFile file;
    GgufTensorInfo tensor;
    GgufF32TensorView view;

    setup_f32_fixture(&file, &tensor, values, sizeof(values));
    tensor.type = 6;
    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    setup_f32_fixture(&file, &tensor, values, sizeof(values));
    tensor.dimensions[1] = 0;
    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    setup_f32_fixture(&file, &tensor, values, sizeof(values));
    tensor.dimensions[0] = UINT64_MAX;
    tensor.dimensions[1] = 2;
    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);

    setup_f32_fixture(&file, &tensor, values, sizeof(values));
    tensor.byte_size = sizeof(values) - 1;
    set_view_sentinel(&view, values);
    assert(gguf_get_f32_tensor_view(&file, FIXTURE_NAME, &view) == 0);
    assert_empty_view(&view);
}

static void test_real_qwen_model(const char *model_path)
{
    GgufFile file;
    GgufF32TensorView view;
    char error[256];

    assert(gguf_open(model_path, &file, error, sizeof(error)) == 1);
    assert(gguf_get_f32_tensor_view(&file, "blk.0.attn_norm.weight", &view) == 1);
    assert(view.dimension_count == 1);
    assert(view.dimensions[0] == 896);
    assert(view.element_count == 896);
    assert(isfinite(view.data[0]));
    assert(isfinite(view.data[view.element_count - 1]));

    gguf_close(&file);
}

int main(int argc, char **argv)
{
    printf("Running GGUF F32 view tests...\n");

    test_valid_f32_view();
    test_invalid_requests_clear_view();
    test_rejected_tensor_layouts();

    if (argc == 2) {
        test_real_qwen_model(argv[1]);
    }

    printf("All GGUF F32 view tests passed.\n");
    return 0;
}
