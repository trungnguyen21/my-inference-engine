#include "architecture.h"
#include "engine_internal.h"
#include "qwen2_architecture.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>

static int dispatch_calls;
static int fake_format(const char *user, char *output, int capacity,
                       char *error, int error_capacity)
{
    ++dispatch_calls;
    assert(strcmp(user, "hello") == 0);
    if (capacity < 3) {
        if (error && error_capacity > 0) snprintf(error, error_capacity, "backend capacity error");
        return 0;
    }
    memcpy(output, "ok", 3);
    return 1;
}

static void test_dispatch()
{
    ArchitectureOps ops = *qwen2_architecture_ops();
    ops.format_prompt = fake_format;
    EngineModel implementation = {};
    implementation.architecture = &ops;
    Model model = {};
    model.implementation = &implementation;
    char output[32], error[128] = {};
    assert(engine_format_prompt(&model, "hello", output, sizeof(output), error, sizeof(error)));
    assert(dispatch_calls == 1 && strcmp(output, "ok") == 0);
    assert(!engine_format_prompt(&model, "hello", output, 2, error, sizeof(error)));
    assert(output[0] == '\0' && strcmp(error, "backend capacity error") == 0);
    assert(dispatch_calls == 2);
    assert(!engine_format_prompt(nullptr, "hello", output, sizeof(output), error, sizeof(error)));
    assert(output[0] == '\0' && error[0]);
    Model unloaded = {};
    assert(!engine_format_prompt(&unloaded, "hello", output, sizeof(output), nullptr, 0));
    assert(!engine_format_prompt(&model, nullptr, output, sizeof(output), error, sizeof(error)));
    assert(!engine_format_prompt(&model, "hello", nullptr, 10, error, sizeof(error)));
    output[0] = 'X';
    assert(!engine_format_prompt(&model, "hello", output, 0, error, sizeof(error)));
    assert(output[0] == 'X');
    ops.format_prompt = nullptr;
    assert(!engine_format_prompt(&model, "hello", output, sizeof(output), error, sizeof(error)));
    assert(error[0] && dispatch_calls == 2);
    implementation.architecture = nullptr;
    assert(!engine_format_prompt(&model, "hello", output, sizeof(output), error, sizeof(error)));
}

static void expect_prompt(const char *user)
{
    /* Expected bytes from the local GGUF tokenizer.chat_template, restricted
     * to one user message, default system, no tools, add_generation_prompt. */
    const std::string expected =
        std::string("<|im_start|>system\n") +
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    const std::string saved(user);
    const ArchitectureOps *ops = qwen2_architecture_ops();
    std::vector<char> output(expected.size() + 3, '#');
    char error[128] = {};
    int ok = ops->format_prompt(user, output.data() + 1, (int)expected.size() + 1,
                                error, sizeof(error));
    if (!ok) fprintf(stderr, "Prompt checkpoint: %s\n", error);
    assert(ok);
    assert(strcmp(output.data() + 1, expected.c_str()) == 0);
    assert(output.front() == '#' && output.back() == '#');
    assert(saved == user);

    std::fill(output.begin(), output.end(), '#');
    assert(!ops->format_prompt(user, output.data() + 1, (int)expected.size(), error, sizeof(error)));
    assert(output[1] == '\0' && error[0]);
    assert(output.front() == '#' && output[expected.size() + 1] == '#');
    assert(ops->format_prompt(user, output.data(), (int)output.size(), nullptr, 0));
}

int main(int argc, char **argv)
{
    test_dispatch();
    if (argc == 2 && strcmp(argv[1], "dispatch") == 0) return 0;
    expect_prompt("Hello");
    expect_prompt("");
    expect_prompt(u8"Hello\nViệt Nam 世界\n");
    const std::string longest(PROMPT_CAPACITY - 1, 'a');
    expect_prompt(longest.c_str());

    const auto format = qwen2_architecture_ops()->format_prompt;
    char output[32], error[128] = {};
    assert(!format(nullptr, output, sizeof(output), error, sizeof(error)));
    assert(output[0] == '\0' && error[0]);
    assert(!format("hello", nullptr, 10, error, sizeof(error)));
    output[0] = 'X';
    assert(!format("hello", output, 0, error, sizeof(error)) && output[0] == 'X');
    assert(!format("hello", output, -1, nullptr, 0) && output[0] == 'X');
    char guard[] = {'L', '?', 'R'};
    assert(!format("hello", guard + 1, 1, error, sizeof(error)));
    assert(guard[0] == 'L' && guard[1] == '\0' && guard[2] == 'R');
    puts("Prompt formatting and architecture dispatch contract passed.");
    return 0;
}
