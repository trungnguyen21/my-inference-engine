#ifndef MATRIX_BACKEND_H
#define MATRIX_BACKEND_H

#include "gguf_reader.h"

/*
 * Backend-neutral matrix execution boundary.
 *
 * A backend implements exactly two operations, matvec and matmul, over a
 * MatrixWeight that hides whether the weights are dense F32 or packed GGUF
 * blocks. Activations and outputs are always F32.
 *
 * Layout (the same for every backend):
 *   - A weight has `rows` output features and `columns` input features, stored
 *     row-major with one output feature per row: output[row] = dot(row, input).
 *   - matvec:  input[columns]               -> output[rows]
 *   - matmul:  inputs[batch_count, columns] -> outputs[batch_count, rows]
 *              (token-major batches)
 *
 * Callers should use the matrix_backend_matvec/matmul wrappers. They validate
 * the weight and buffers once, so a backend can assume: valid backend, valid
 * weight, non-null buffers, batch_count > 0, and no overlap between the output
 * and the input (or a dense weight). Kernel-specific storage checks, such as
 * packed block layout, remain the backend's job. Inputs and outputs are
 * host-visible pointers for now; a device-resident weight handle belongs in
 * MatrixWeight when a GPU backend needs one.
 */

typedef enum MatrixWeightKind {
    MATRIX_WEIGHT_DENSE_F32 = 0, /* row-major F32 matrix */
    MATRIX_WEIGHT_PACKED = 1     /* GGUF block-quantized matrix */
} MatrixWeightKind;

typedef struct MatrixWeight {
    MatrixWeightKind kind;
    int rows;                                /* output features */
    int columns;                             /* input features */
    const float *dense;                      /* MATRIX_WEIGHT_DENSE_F32 only */
    const GgufQuantizedTensorView *packed;   /* MATRIX_WEIGHT_PACKED only; borrowed */
} MatrixWeight;

/* Describe dense F32 weights (rows * columns floats). Return 0 and leave a
 * cleared descriptor for null data or non-positive sizes. */
int matrix_weight_dense(MatrixWeight *weight, const float *data, int rows, int columns);

/* Describe a packed 2-D GGUF view without copying it; the view must outlive the
 * descriptor. rows = dimensions[1] and columns = dimensions[0]. Return 0 and
 * leave a cleared descriptor for a null/non-2-D view or dimensions outside
 * [1, INT_MAX]. */
int matrix_weight_packed(MatrixWeight *weight, const GgufQuantizedTensorView *view);

typedef enum MatrixBackendType {
    MATRIX_BACKEND_CPU = 0,
    MATRIX_BACKEND_GPU = 1
} MatrixBackendType;

typedef int (*MatrixMatvecFn)(
    const MatrixWeight *weight, const float *input, float *output
);

typedef int (*MatrixMatmulFn)(
    const MatrixWeight *weight, const float *inputs, int batch_count, float *outputs
);

typedef struct MatrixBackend {
    MatrixBackendType type;
    const char *name;
    int available;          /* 0 until every operation is implemented and verified */
    MatrixMatvecFn matvec;
    MatrixMatmulFn matmul;
} MatrixBackend;

/* Concrete backend descriptors. The GPU descriptor stays unavailable until a
 * GPU implementation supplies working operations. */
const MatrixBackend *matrix_backend_cpu(void);
const MatrixBackend *matrix_backend_gpu(void);
const MatrixBackend *matrix_backend_default(void);
const MatrixBackend *matrix_backend_get(MatrixBackendType type);

/* Checked entry points. Return 0 for a null/unavailable backend, a missing
 * operation, an invalid weight, null or overlapping buffers, or a failed
 * kernel; return 1 only after every output element has been written. Output
 * contents are unspecified on failure. */
int matrix_backend_matvec(
    const MatrixBackend *backend, const MatrixWeight *weight,
    const float *input, float *output
);
int matrix_backend_matmul(
    const MatrixBackend *backend, const MatrixWeight *weight,
    const float *inputs, int batch_count, float *outputs
);

#endif
