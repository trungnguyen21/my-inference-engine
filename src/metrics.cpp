#include "metrics.h"

#include <string.h>
#include <stdio.h>
#include <sys/resource.h>
#include <time.h>

#if defined(__linux__)
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

static double process_cpu_seconds(void)
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    return (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1000000.0 +
           (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1000000.0;
}

static unsigned long long resident_memory_bytes(void)
{
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (unsigned long long)info.resident_size;
#elif defined(__linux__)
    FILE *statm = fopen("/proc/self/statm", "r");
    unsigned long total_pages = 0;
    unsigned long resident_pages = 0;
    long page_size = sysconf(_SC_PAGESIZE);
    if (statm == NULL || page_size <= 0 ||
        fscanf(statm, "%lu %lu", &total_pages, &resident_pages) != 2) {
        if (statm != NULL) {
            fclose(statm);
        }
        return 0;
    }
    fclose(statm);
    (void)total_pages;
    return (unsigned long long)resident_pages * (unsigned long long)page_size;
#else
    return 0;
#endif
}

void metrics_capture(UsageSnapshot *snapshot)
{
    struct timespec now;
    memset(snapshot, 0, sizeof(*snapshot));
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return;
    }
    snapshot->wall_seconds = (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
    snapshot->process_cpu_seconds = process_cpu_seconds();
    snapshot->resident_bytes = resident_memory_bytes();
    snapshot->available = 1;
}

double metrics_elapsed_seconds(const UsageSnapshot *start,
                               const UsageSnapshot *end)
{
    if (start == NULL || end == NULL || !start->available || !end->available ||
        end->wall_seconds < start->wall_seconds) {
        return 0.0;
    }
    return end->wall_seconds - start->wall_seconds;
}

double metrics_tokens_per_second(int generated_tokens, double elapsed_seconds)
{
    if (generated_tokens <= 0 || elapsed_seconds <= 0.0) {
        return 0.0;
    }
    return (double)generated_tokens / elapsed_seconds;
}

double metrics_cpu_percent(const UsageSnapshot *start,
                           const UsageSnapshot *end)
{
    double elapsed;
    double cpu_elapsed;
    if (start == NULL || end == NULL || !start->available || !end->available) {
        return -1.0;
    }
    elapsed = metrics_elapsed_seconds(start, end);
    cpu_elapsed = end->process_cpu_seconds - start->process_cpu_seconds;
    if (elapsed <= 0.0 || cpu_elapsed < 0.0) {
        return -1.0;
    }
    return 100.0 * cpu_elapsed / elapsed;
}
