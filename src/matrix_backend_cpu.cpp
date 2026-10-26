#include "matrix_backend.h"

#include "quantized_matvec.h"
#include "tensor.h"

/* The CPU backend is the one place that bridges MatrixWeight to the two
 * existing kernel signatures (dense: rows/columns; packed: counts from the view). */

static int cpu_matvec(const MatrixWeight *weight, const float *input, float *output)
{
    switch (weight->kind) {
        case MATRIX_WEIGHT_DENSE_F32:
            return matrix_matvec_dense_f32(
                weight->dense, weight->rows, weight->columns, input, output);
        case MATRIX_WEIGHT_PACKED:
            return matrix_matvec_packed_f32(
                weight->packed, input, weight->columns, output, weight->rows);
        default:
            return 0;
    }
}

static int cpu_matmul(
    const MatrixWeight *weight, const float *inputs, int batch_count, float *outputs)
{
    switch (weight->kind) {
        case MATRIX_WEIGHT_DENSE_F32:
            return matrix_matmul_dense_f32(
                weight->dense, weight->rows, weight->columns, inputs, batch_count, outputs);
        case MATRIX_WEIGHT_PACKED:
            return matrix_matmul_packed_f32(
                weight->packed, inputs, weight->columns, batch_count, outputs, weight->rows);
        default:
            return 0;
    }
}

const MatrixBackend *matrix_backend_cpu(void)
{
    static const MatrixBackend backend = {
        MATRIX_BACKEND_CPU,
        "cpu",
        1,
        cpu_matvec,
        cpu_matmul
    };
    return &backend;
}
