#include "matrix_backend.h"

/* Placeholder for a future Metal implementation. Every operation is null and
 * the descriptor is unavailable, so checked calls fail without touching data. */
const MatrixBackend *matrix_backend_gpu(void)
{
    static const MatrixBackend backend = {
        MATRIX_BACKEND_GPU,
        "gpu",
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    return &backend;
}
