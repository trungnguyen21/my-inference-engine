#ifndef GGUF_READER_H
#define GGUF_READER_H

#include <stdint.h>
#include <stddef.h>

#define ENGINE_GGML_TYPE_F32 0u
#define ENGINE_GGML_TYPE_Q5_0 6u
#define ENGINE_GGML_TYPE_Q8_0 8u
#define ENGINE_GGML_TYPE_Q4_K 12u
#define ENGINE_GGML_TYPE_Q6_K 14u

typedef struct GgufTensorInfo {
    char *name;
    uint32_t dimension_count;
    uint64_t dimensions[4];
    uint32_t type;
    uint64_t offset;
    uint64_t byte_size;
} GgufTensorInfo;

/* Read-only F32 data owned by the mapped GgufFile. Do not retain after close. */
typedef struct GgufF32TensorView {
    const float *data;
    uint32_t dimension_count;
    uint64_t dimensions[4];
    uint64_t element_count;
} GgufF32TensorView;

/* Read-only packed data owned by the mapped GgufFile. Do not retain after close. */
typedef struct GgufQuantizedTensorView {
    const unsigned char *data;      // point to the first packed byte in memory-mapped GGUF file
    uint32_t ggml_type;             // determine the type of these bytes (Q5_0, Q8_0, etc.)
    uint32_t dimension_count;       // how many dimensions the tensor has (2 if it's a matrix)
    uint64_t dimensions[4];         // hold the size of each dimension ([0] = columns and [1] rows if count = 2)
    uint64_t element_count;         // total number of logical weights (for matrix, it's columns*rows)
    uint64_t block_elements;        // how many weights one packed block represent (32 for Q5_0, etc.)
    uint64_t block_bytes;           // how many bytes are stored in 1 packed block
    uint64_t block_count;           // how many packed blocks are in the whole tensor (element_count / block_elements)
} GgufQuantizedTensorView;

typedef struct GgufMetadataEntry {
    char *key;
    uint32_t type;
    uint64_t value_offset;
    uint32_t array_type;
    uint64_t array_count;
} GgufMetadataEntry;

typedef struct GgufFile {
    int file_descriptor;
    const unsigned char *data;
    uint64_t file_size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t tensor_data_offset;
    uint32_t alignment;
    GgufMetadataEntry *metadata;
    GgufTensorInfo *tensors;
} GgufFile;

/* A sequential view of one GGUF array. Values point into GgufFile::data. */
typedef struct GgufArrayCursor {
    const GgufFile *file;
    uint64_t position;
    uint64_t remaining;
    uint32_t element_type;
} GgufArrayCursor;

int gguf_open(const char *path, GgufFile *file, char *error, int error_capacity);
void gguf_close(GgufFile *file);
int gguf_get_uint(const GgufFile *file, const char *key, uint64_t *value);
/*
 * Read scalar floating-point metadata as a double without losing an F64 value.
 *
 * file is an opened GGUF file, key is a NUL-terminated metadata name, and value
 * is caller-provided output storage. Accept GGUF F32 and F64 scalar metadata
 * only. Reject null arguments, missing keys, other metadata types, and NaN or
 * infinite stored values. Return 1 after writing the finite result, otherwise
 * return 0 and leave value unchanged.
 */
int gguf_get_float(const GgufFile *file, const char *key, double *value);
int gguf_get_string(const GgufFile *file, const char *key,
                    const char **value, uint64_t *length);
int gguf_get_array_info(const GgufFile *file, const char *key,
                        uint32_t *element_type, uint64_t *count);
int gguf_get_bool(const GgufFile *file, const char *key, int *value);
int gguf_array_cursor_init(const GgufFile *file, const char *key,
                           GgufArrayCursor *cursor);
int gguf_array_next_string(GgufArrayCursor *cursor, const char **value,
                           uint64_t *length);
int gguf_array_next_uint(GgufArrayCursor *cursor, uint64_t *value);
int gguf_array_next_int(GgufArrayCursor *cursor, int64_t *value);
const GgufTensorInfo *gguf_find_tensor(const GgufFile *file, const char *name);
const void *gguf_tensor_data(const GgufFile *file, const GgufTensorInfo *tensor);

// file: a successfully opened GGUF file that owns the mapped model bytes.
// name: the NUL-terminated name of one tensor in that file.
// view: caller-provided output storage for a read-only F32 tensor view.
//
// First ensure view is not null, then clear every field in *view so a
// failure can never leave stale data from an earlier successful lookup.
// Reject null file or name. Find the named GgufTensorInfo and require
// tensor->type == 0, which is GGML F32.
//
// Validate every tensor dimension and multiply them into element_count.
// Reject zero dimensions and multiplication overflow. Check that
// tensor->byte_size equals element_count * sizeof(float), without allowing
// that multiplication to overflow.
//
// Use gguf_tensor_data to obtain the mapped address. If it is valid, store
// it as view->data, copy the dimension count and dimensions, and store the
// calculated element_count. The pointer is owned by file, read-only, and
// invalid as soon as gguf_close(file) is called. Return 1 on success and 0
// for every invalid case.
int gguf_get_f32_tensor_view(const GgufFile *file, const char *name,
                             GgufF32TensorView *view);

// file: a successfully opened GGUF file that owns the mapped model bytes.
// name: the NUL-terminated name of one tensor in that file.
// view: caller-provided output storage for a read-only packed tensor view.
//
// First ensure view is not null, then clear every field in *view so a failure
// cannot leave stale data from an earlier successful lookup. Reject null file
// or name. Find the named tensor and accept only ENGINE_GGML_TYPE_Q5_0,
// ENGINE_GGML_TYPE_Q8_0, ENGINE_GGML_TYPE_Q4_K, or ENGINE_GGML_TYPE_Q6_K.
//
// Validate all tensor dimensions, multiply them into element_count without
// overflow, and obtain the format's block_elements and block_bytes. Require
// element_count to divide evenly into blocks. Calculate block_count and verify
// tensor->byte_size equals block_count * block_bytes without overflow. Finally,
// use gguf_tensor_data to obtain the mapped address, copy the type and shape,
// and write the block layout fields.
//
// view->data is read-only and belongs to file. It becomes invalid immediately
// after gguf_close(file). Return 1 on success and 0 for all invalid cases.
int gguf_get_quantized_tensor_view(
    const GgufFile *file,
    const char *name,
    GgufQuantizedTensorView *view
);

#endif
