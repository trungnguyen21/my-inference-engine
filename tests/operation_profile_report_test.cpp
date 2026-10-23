#include "terminal.h"

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string>

template <typename PrintFn>
static std::string capture_output(PrintFn print)
{
    fflush(stdout);
    FILE *capture = tmpfile();
    assert(capture != nullptr);
    int saved_stdout = dup(STDOUT_FILENO);
    assert(saved_stdout >= 0);
    assert(dup2(fileno(capture), STDOUT_FILENO) >= 0);
    print();
    fflush(stdout);
    assert(dup2(saved_stdout, STDOUT_FILENO) >= 0);
    close(saved_stdout);
    rewind(capture);
    std::string output;
    char buffer[512];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), capture)) > 0) {
        output.append(buffer, count);
    }
    fclose(capture);
    return output;
}

static std::string capture_report(const OperationProfile *profile)
{
    return capture_output([profile]() { terminal_print_operation_profile(profile); });
}

static void expect_row(const std::string &output, const char *label,
                       const char *milliseconds, const char *percentage)
{
    size_t begin = output.find(label);
    assert(begin != std::string::npos);
    std::string row = output.substr(begin, output.find('\n', begin) - begin);
    assert(row.find(milliseconds) != std::string::npos);
    if (percentage != nullptr) assert(row.find(percentage) != std::string::npos);
}

int main()
{
    OperationProfile profile = {};
    assert(capture_report(nullptr).empty());
    assert(capture_report(&profile).empty());
    profile.enabled = 1;
    std::string empty = capture_report(&profile);
    assert(empty.find("nan") == std::string::npos);
    assert(empty.find("inf") == std::string::npos);
    expect_row(empty, "Total profiled time", "0.000", nullptr);
    expect_row(empty, "matvec (projections + FFN)", "0.000", "0.00%");
    assert(empty.find("Weight kernels by format") != std::string::npos);
    expect_row(empty, "Q5_0", "0.000", "0.00%");

    profile.matvec_seconds = 0.78;
    profile.attention_seconds = 0.09;
    profile.rmsnorm_seconds = 0.02;
    profile.rope_seconds = 0.01;
    profile.output_seconds = 0.07;
    profile.sampling_seconds = 0.03;
    profile.forward_seconds = 0.97;
    profile.forward_calls = 24;
    profile.sampling_calls = 4;
    profile.weight_layer_seconds[PROFILE_Q5_0] = 0.4;
    profile.weight_layer_calls[PROFILE_Q5_0] = 20;
    profile.weight_output_seconds[PROFILE_Q8_0] = 0.1;
    profile.weight_output_calls[PROFILE_Q8_0] = 4;
    std::string output = capture_report(&profile);
    assert(output.find("Forward calls: 24 | Sampling calls: 4") != std::string::npos);
    expect_row(output, "matvec (projections + FFN)", "780.000", "78.00%");
    expect_row(output, "attention (scores/softmax/V)", "90.000", "9.00%");
    expect_row(output, "RMSNorm", "20.000", "2.00%");
    expect_row(output, "RoPE", "10.000", "1.00%");
    expect_row(output, "output matrix + sampling", "100.000", "10.00%");
    expect_row(output, "Total profiled time", "1000.000", nullptr);
    expect_row(output, "Q5_0", "400.000", "80.00%");
    expect_row(output, "Q8_0", "100.000", "20.00%");

    /* Unclassified forward work contributes to the denominator and is visible. */
    profile.forward_seconds += 1.0;
    output = capture_report(&profile);
    expect_row(output, "other forward work", "1000.000", "50.00%");
    expect_row(output, "matvec (projections + FFN)", "780.000", "39.00%");
    expect_row(output, "output matrix + sampling", "100.000", "5.00%");
    expect_row(output, "Total profiled time", "2000.000", nullptr);

    UsageSnapshot start = {}, end = {};
    start.available = end.available = 1;
    end.wall_seconds = 2.0;
    std::string summary = capture_output([&]() {
        terminal_finish_generation(2, 2.0, 12.3456, &start, &end);
    });
    assert(summary.find("Time to first token: 12.346 s (excludes model loading)\n") !=
           std::string::npos);
    assert(summary.find("Elapsed time: 2.000 s\n") != std::string::npos);
    summary = capture_output([&]() {
        terminal_finish_generation(2, 2.0, -1.0, &start, &end);
    });
    assert(summary.find("Time to first token: unavailable\n") != std::string::npos);
    summary = capture_output([&]() {
        terminal_finish_generation(2, 2.0, 0.0, &start, &end);
    });
    assert(summary.find("Time to first token: 0.000 s") != std::string::npos);
    puts("Operation report percentages, totals, disabled and empty cases passed.");
    return 0;
}
