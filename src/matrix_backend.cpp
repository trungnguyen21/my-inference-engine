#include "matrix_backend.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

int matrix_weight_dense(MatrixWeight *weight, const float *data, int rows, int columns)
{
    if (weight == nullptr) return 0;
    *weight = MatrixWeight();
    if (data == nullptr || rows <= 0 || columns <= 0) return 0;
    weight->kind = MATRIX_WEIGHT_DENSE_F32;
    weight->rows = rows;
    weight->columns = columns;
    weight->dense = data;
    return 1;
}

int matrix_weight_packed(MatrixWeight *weight, const GgufQuantizedTensorView *view)
{
    if (weight == nullptr) return 0;
    *weight = MatrixWeight();
    if (view == nullptr || view->data == nullptr || view->dimension_count != 2 ||
        view->dimensions[0] == 0 || view->dimensions[0] > INT_MAX ||
        view->dimensions[1] == 0 || view->dimensions[1] > INT_MAX) {
        return 0;
    }
    weight->kind = MATRIX_WEIGHT_PACKED;
    weight->columns = (int)view->dimensions[0];
    weight->rows = (int)view->dimensions[1];
    weight->packed = view;
    return 1;
}

const MatrixBackend *matrix_backend_default(void)
{
    return matrix_backend_cpu();
}

const MatrixBackend *matrix_backend_get(MatrixBackendType type)
{
    switch (type) {
        case MATRIX_BACKEND_CPU: return matrix_backend_cpu();
        case MATRIX_BACKEND_GPU: return matrix_backend_gpu();
        default: return nullptr;
    }
}

/* The descriptor must be self-consistent so backends never re-derive shapes. */
static int weight_is_valid(const MatrixWeight *weight)
{
    if (weight == nullptr || weight->rows <= 0 || weight->columns <= 0) return 0;
    switch (weight->kind) {
        case MATRIX_WEIGHT_DENSE_F32:
            return weight->dense != nullptr;
        case MATRIX_WEIGHT_PACKED:
            return weight->packed != nullptr && weight->packed->data != nullptr &&
                   weight->packed->dimension_count == 2 &&
                   weight->packed->dimensions[0] == (uint64_t)weight->columns &&
                   weight->packed->dimensions[1] == (uint64_t)weight->rows;
        default:
            return 0;
    }
}

static int regions_overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    uintptr_t a_start = (uintptr_t)a;
    uintptr_t b_start = (uintptr_t)b;
    return a_start < b_start + b_bytes && b_start < a_start + a_bytes;
}

/* Validate buffers shared by matvec and matmul: sizes must be representable and
 * the output may overlap neither the inputs nor a dense matrix. */
static int buffers_are_usable(
    const MatrixWeight *weight, int batch_count,
    const float *inputs, const float *outputs)
{
    if (inputs == nullptr || outputs == nullptr || batch_count <= 0) return 0;
    const size_t columns = (size_t)weight->columns;
    const size_t rows = (size_t)weight->rows;
    const size_t batch = (size_t)batch_count;
    if (columns > SIZE_MAX / sizeof(float) / batch ||
        rows > SIZE_MAX / sizeof(float) / batch) {
        return 0;
    }
    if (regions_overlap(inputs, batch * columns * sizeof(float),
                        outputs, batch * rows * sizeof(float))) {
        return 0;
    }
    if (weight->kind == MATRIX_WEIGHT_DENSE_F32) {
        if (columns > SIZE_MAX / sizeof(float) / rows) return 0;
        if (regions_overlap(weight->dense, rows * columns * sizeof(float),
                            outputs, batch * rows * sizeof(float))) {
            return 0;
        }
    }
    return 1;
}

int matrix_backend_matvec(
    const MatrixBackend *backend, const MatrixWeight *weight,
    const float *input, float *output)
{
    if (backend == nullptr || !backend->available || backend->matvec == nullptr ||
        !weight_is_valid(weight) || !buffers_are_usable(weight, 1, input, output)) {
        return 0;
    }
    return backend->matvec(weight, input, output);
}

int matrix_backend_matmul(
    const MatrixBackend *backend, const MatrixWeight *weight,
    const float *inputs, int batch_count, float *outputs)
{
    if (backend == nullptr || !backend->available || backend->matmul == nullptr ||
        !weight_is_valid(weight) || !buffers_are_usable(weight, batch_count, inputs, outputs)) {
        return 0;
    }
    return backend->matmul(weight, inputs, batch_count, outputs);
}
