#include "engine.h"
#include "sampler.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <initializer_list>

static void expect_pick(const float *logits, int count, int expected)
{
    float saved[8];
    assert(count <= 8);
    memcpy(saved, logits, (size_t)count * sizeof(float));
    char error[128] = {};
    int id = -1;
    int ok = sampler_greedy_sample(logits, count, &id, error, sizeof(error));
    if (!ok) fprintf(stderr, "Greedy checkpoint: %s\n", error);
    assert(ok && id == expected);
    assert(memcmp(saved, logits, (size_t)count * sizeof(float)) == 0);
}

static void test_wrapper_validation()
{
    const float logits[] = {1.0f, 2.0f};
    const float bad[] = {-0.1f, 1.1f, NAN, INFINITY};
    char error[128];
    int id;
    for (float value : bad) {
        error[0] = '\0'; id = 123;
        assert(!sampler_sample(logits, 2, value, 1.0f, &id, error, sizeof(error)));
        assert(id == -1 && error[0]);
        error[0] = '\0'; id = 123;
        assert(!sampler_sample(logits, 2, 0.0f, value, &id, error, sizeof(error)));
        assert(id == -1 && error[0]);
    }
    assert(!sampler_sample(nullptr, 2, 0, 1, &id, error, sizeof(error)) && id == -1);
    assert(!sampler_sample(logits, 0, 0, 1, &id, error, sizeof(error)) && id == -1);
    assert(!sampler_sample(logits, -1, 0, 1, &id, nullptr, 0) && id == -1);
    assert(!sampler_sample(logits, 2, 0, 1, nullptr, error, sizeof(error)));
    /* Positive-temperature sampling is explicitly deferred. */
    assert(!sampler_sample(logits, 2, 0.8f, 0.95f, &id, error, sizeof(error)));
    assert(id == -1 && strstr(error, "--temperature 0"));
}

int main(int argc, char **argv)
{
    test_wrapper_validation();
    if (argc == 2 && strcmp(argv[1], "validation") == 0) return 0;

    const float ordinary[] = {-3.0f, 0.5f, 7.0f, 2.0f};
    const float negative[] = {-10.0f, -2.0f, -5.0f};
    const float tied[] = {4.0f, 4.0f, -1.0f, 4.0f};
    const float later_tie[] = {-1.0f, 9.0f, 9.0f};
    const float singleton[] = {-FLT_MAX};
    const float extremes[] = {-FLT_MAX, FLT_MAX, 0.0f};
    const float zero[] = {-0.0f, 0.0f};
    expect_pick(ordinary, 4, 2);
    expect_pick(negative, 3, 1);
    expect_pick(tied, 4, 0);
    expect_pick(later_tie, 3, 1);
    expect_pick(singleton, 1, 0);
    expect_pick(extremes, 3, 1);
    expect_pick(zero, 2, 0);

    char error[128];
    int id = 99;
    for (float top_p : {0.0f, 0.5f, 1.0f}) {
        assert(sampler_sample(ordinary, 4, 0, top_p, &id, error, sizeof(error)));
        assert(id == 2);
    }
    assert(sampler_greedy_sample(ordinary, 4, &id, nullptr, 0) && id == 2);
    /* A later invalid entry must fail even if the winner was seen earlier. */
    for (float bad : {NAN, INFINITY, -INFINITY}) {
        for (int at = 0; at < 3; ++at) {
            float logits[] = {3.0f, 2.0f, 1.0f};
            logits[at] = bad;
            error[0] = '\0'; id = 99;
            assert(!sampler_greedy_sample(logits, 3, &id, error, sizeof(error)));
            assert(id == -1 && error[0]);
        }
    }
    assert(!sampler_greedy_sample(nullptr, 3, &id, error, sizeof(error)) && id == -1);
    assert(!sampler_greedy_sample(ordinary, 0, &id, error, sizeof(error)) && id == -1);
    assert(!sampler_greedy_sample(ordinary, -1, &id, nullptr, 0) && id == -1);
    assert(!sampler_greedy_sample(ordinary, 4, nullptr, error, sizeof(error)));
    char guard[] = {'L', '?', 'R'};
    assert(!sampler_greedy_sample(nullptr, 0, &id, guard + 1, 1));
    assert(guard[0] == 'L' && guard[1] == '\0' && guard[2] == 'R');
    puts("Greedy selection and routing contract passed.");
    return 0;
}
