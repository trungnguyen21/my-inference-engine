#ifndef MATRIX_BACKEND_H
#define MATRIX_BACKEND_H

#include "gguf_reader.h"

/* Backend-neutral matrix execution boundary. Inputs and outputs are currently
 * host-visible pointers. A future GPU backend may add prepared/device-resident
 * weight ownership without changing Qwen's projection schedule. */
typedef enum MatrixBackendType {
    MATRIX_BACKEND_CPU = 0,
    MATRIX_BACKEND_GPU = 1
} MatrixBackendType;

typedef int (*MatrixMatvecF32Fn)(
    const float *matrix, int rows, int columns,
    const float *input, float *output
);

typedef int (*MatrixMatmulF32Fn)(
    const float *matrix, int rows, int columns,
    const float *inputs, int batch_count, float *outputs
);

typedef int (*MatrixMatvecQuantizedFn)(
    const GgufQuantizedTensorView *matrix,
    const float *input, int input_count,
    float *output, int output_count
);

typedef int (*MatrixMatmulQuantizedFn)(
    const GgufQuantizedTensorView *matrix,
    const float *inputs, int input_count, int batch_count,
    float *outputs, int output_count
);

typedef struct MatrixBackend {
    MatrixBackendType type;
    const char *name;
    int available;
    MatrixMatvecF32Fn matvec_f32;
    MatrixMatmulF32Fn matmul_f32;
    MatrixMatvecQuantizedFn matvec_quantized;
    MatrixMatmulQuantizedFn matmul_quantized;
} MatrixBackend;

/* Concrete backend descriptors. The GPU descriptor is intentionally
 * unavailable until a Metal implementation supplies its operations. */
const MatrixBackend *matrix_backend_cpu(void);
const MatrixBackend *matrix_backend_gpu(void);
const MatrixBackend *matrix_backend_default(void);
const MatrixBackend *matrix_backend_get(MatrixBackendType type);

/* Checked calls fail with 0 when the backend or operation is unavailable.
 * Kernel-specific shape/storage validation remains inside each backend. */
int matrix_backend_matvec_f32(
    const MatrixBackend *backend,
    const float *matrix, int rows, int columns,
    const float *input, float *output
);
int matrix_backend_matmul_f32(
    const MatrixBackend *backend,
    const float *matrix, int rows, int columns,
    const float *inputs, int batch_count, float *outputs
);
int matrix_backend_matvec_quantized(
    const MatrixBackend *backend,
    const GgufQuantizedTensorView *matrix,
    const float *input, int input_count,
    float *output, int output_count
);
int matrix_backend_matmul_quantized(
    const MatrixBackend *backend,
    const GgufQuantizedTensorView *matrix,
    const float *inputs, int input_count, int batch_count,
    float *outputs, int output_count
);

#endif
