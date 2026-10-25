#include "matrix_backend.h"

#include "quantized_matvec.h"
#include "tensor.h"

const MatrixBackend *matrix_backend_cpu(void)
{
    static const MatrixBackend backend = {
        MATRIX_BACKEND_CPU,
        "cpu",
        1,
        tensor_matvec_f32,
        tensor_matmul_f32,
        quantized_tensor_matvec_f32,
        quantized_tensor_matmul_f32
    };
    return &backend;
}
