#include "matrix_backend.h"

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

int matrix_backend_matvec_f32(
    const MatrixBackend *backend,
    const float *matrix, int rows, int columns,
    const float *input, float *output)
{
    if (backend == nullptr || !backend->available || backend->matvec_f32 == nullptr)
        return 0;
    return backend->matvec_f32(matrix, rows, columns, input, output);
}

int matrix_backend_matmul_f32(
    const MatrixBackend *backend,
    const float *matrix, int rows, int columns,
    const float *inputs, int batch_count, float *outputs)
{
    if (backend == nullptr || !backend->available || backend->matmul_f32 == nullptr)
        return 0;
    return backend->matmul_f32(matrix, rows, columns, inputs, batch_count, outputs);
}

int matrix_backend_matvec_quantized(
    const MatrixBackend *backend,
    const GgufQuantizedTensorView *matrix,
    const float *input, int input_count,
    float *output, int output_count)
{
    if (backend == nullptr || !backend->available || backend->matvec_quantized == nullptr)
        return 0;
    return backend->matvec_quantized(matrix, input, input_count, output, output_count);
}

int matrix_backend_matmul_quantized(
    const MatrixBackend *backend,
    const GgufQuantizedTensorView *matrix,
    const float *inputs, int input_count, int batch_count,
    float *outputs, int output_count)
{
    if (backend == nullptr || !backend->available || backend->matmul_quantized == nullptr)
        return 0;
    return backend->matmul_quantized(
        matrix, inputs, input_count, batch_count, outputs, output_count);
}
