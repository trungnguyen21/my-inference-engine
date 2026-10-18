#ifndef INFERENCE_OPERATION_TIMER_H
#define INFERENCE_OPERATION_TIMER_H

#include <chrono>

/* No allocation or clock reads when disabled. Scope exit records elapsed
 * wall time even when an operation returns early with an error. */
class OperationTimer {
public:
    OperationTimer(int enabled, double *total)
        : total_(enabled ? total : nullptr), start_()
    {
        if (total_ != nullptr) start_ = Clock::now();
    }

    ~OperationTimer()
    {
        if (total_ != nullptr) {
            *total_ += std::chrono::duration<double>(Clock::now() - start_).count();
        }
    }

    OperationTimer(const OperationTimer &) = delete;
    OperationTimer &operator=(const OperationTimer &) = delete;

private:
    typedef std::chrono::steady_clock Clock;
    double *total_;
    Clock::time_point start_;
};

#endif
