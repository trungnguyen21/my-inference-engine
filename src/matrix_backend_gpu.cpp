#include "matrix_backend.h"

/*
 * GPU backend stubs (not implemented). The API is GPU-vendor neutral: a Metal,
 * CUDA, or other implementation fills in the same two operations.
 *
 * A backend supplies exactly two functions, with the contract documented in
 * matrix_backend.h. The matrix_backend_matvec/matmul wrappers have already
 * checked the weight descriptor, buffer pointers, batch size, and overlap by
 * the time these run, so an implementation only needs to handle:
 *
 *   MATRIX_WEIGHT_DENSE_F32  weight->dense is rows*columns row-major floats.
 *   MATRIX_WEIGHT_PACKED     weight->packed is a GGUF view; rows are contiguous
 *                            blocks of Q5_0, Q8_0, Q4_K, or Q6_K. Block bytes may
 *                            be unaligned in the mapped file, so read them with
 *                            an alignment-safe load. Return 0 for a type you do
 *                            not support yet (start with Q8_0).
 *
 * Return 1 only after every output element has been written; return 0 on any
 * failure. Results may differ from the CPU backend in the last bits because a
 * parallel reduction changes summation order; tests compare with a tolerance.
 *
 * Keep `available` at 0 until both operations are implemented and verified
 * against the CPU backend; the checked wrappers refuse an unavailable backend,
 * so a half-finished GPU path can never be selected.
 */

/* output[rows] = weight[rows, columns] * input[columns] */
static int gpu_matvec(const MatrixWeight *, const float * /*input*/, float * /*output*/)
{
    /* TODO(gpu): one threadgroup/block per row, strided partial sums + reduction. */
    return 0;
}

/* outputs[batch_count, rows] = inputs[batch_count, columns] * weight^T */
static int gpu_matmul(
    const MatrixWeight *, const float * /*inputs*/, int /*batch_count*/, float * /*outputs*/)
{
    /* TODO(gpu): reuse each loaded (and, if packed, decoded) weight row across the batch. */
    return 0;
}

const MatrixBackend *matrix_backend_gpu(void)
{
    static const MatrixBackend backend = {
        MATRIX_BACKEND_GPU,
        "gpu",
        0, /* flip to 1 only when both operations are implemented and tested */
        gpu_matvec,
        gpu_matmul
    };
    return &backend;
}
