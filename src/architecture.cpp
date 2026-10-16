#include "architecture.h"

#include "qwen2_architecture.h"

#include <string.h>

const ArchitectureOps *architecture_find(
    const char *identifier,
    uint64_t identifier_length
)
{
    const ArchitectureOps *backends[] = {
        qwen2_architecture_ops()
    };

    if (identifier == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < sizeof(backends) / sizeof(backends[0]); ++i) {
        const ArchitectureOps *candidate = backends[i];
        const size_t candidate_length = strlen(candidate->identifier);

        if (identifier_length == (uint64_t)candidate_length &&
            memcmp(identifier, candidate->identifier, candidate_length) == 0) {
            return candidate;
        }
    }
    return nullptr;
}
