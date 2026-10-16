#ifndef QWEN2_ARCHITECTURE_H
#define QWEN2_ARCHITECTURE_H

#include "architecture.h"

/* Return the single callback table for GGUF files whose architecture is qwen2. */
const ArchitectureOps *qwen2_architecture_ops(void);

#endif
