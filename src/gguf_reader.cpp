#include "gguf_reader.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_HEADER_SIZE 24
#define GGUF_MAX_METADATA_ENTRIES 1000000
#define GGUF_MAX_TENSORS 1000000

typedef struct Reader {
    const unsigned char *data;
    uint64_t size;
    uint64_t position;
} Reader;

static void set_error(char *error, int capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

static int has_bytes(const Reader *reader, uint64_t count)
{
    return reader->position <= reader->size &&
           count <= reader->size - reader->position;
}

static int read_u32(Reader *reader, uint32_t *value)
{
    const unsigned char *p;
    if (!has_bytes(reader, 4)) {
        return 0;
    }
    p = reader->data + reader->position;
    *value = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
             ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    reader->position += 4;
    return 1;
}

static int read_u64(Reader *reader, uint64_t *value)
{
    const unsigned char *p;
    int i;
    uint64_t result = 0;
    if (!has_bytes(reader, 8)) {
        return 0;
    }
    p = reader->data + reader->position;
    for (i = 0; i < 8; ++i) {
        result |= (uint64_t)p[i] << (i * 8);
    }
    reader->position += 8;
    *value = result;
    return 1;
}

static char *read_string(Reader *reader)
{
    uint64_t length;
    char *text;
    if (!read_u64(reader, &length) || length > SIZE_MAX - 1 ||
        !has_bytes(reader, length)) {
        return NULL;
    }
    text = (char *)malloc((size_t)length + 1);
    if (text == NULL) {
        return NULL;
    }
    memcpy(text, reader->data + reader->position, (size_t)length);
    text[length] = '\0';
    reader->position += length;
    return text;
}

static int fixed_type_size(uint32_t type, uint32_t *size)
{
    static const uint8_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 8, 8, 8, 8, 8};
    if (type >= sizeof(sizes)) {
        return 0;
    }
    *size = sizes[type];
    return 1;
}

static int skip_value(Reader *reader, uint32_t type, int depth,
                      uint32_t *array_type, uint64_t *array_count)
{
    uint32_t size;
    uint64_t count;
    uint64_t i;
    if (depth > 2) {
        return 0;
    }
    if (type == 8) {
        uint64_t length;
        if (!read_u64(reader, &length) || !has_bytes(reader, length)) {
            return 0;
        }
        reader->position += length;
        return 1;
    }
    if (type == 9) {
        uint32_t element_type;
        if (!read_u32(reader, &element_type) || !read_u64(reader, &count) ||
            count > GGUF_MAX_METADATA_ENTRIES) {
            return 0;
        }
        if (depth == 0 && array_type != NULL && array_count != NULL) {
            *array_type = element_type;
            *array_count = count;
        }
        for (i = 0; i < count; ++i) {
            if (!skip_value(reader, element_type, depth + 1, NULL, NULL)) {
                return 0;
            }
        }
        return 1;
    }
    if (!fixed_type_size(type, &size) || !has_bytes(reader, size)) {
        return 0;
    }
    if (type == 7 && reader->data[reader->position] > 1) {
        return 0;
    }
    reader->position += size;
    return 1;
}

static int tensor_type_layout(uint32_t type, uint64_t *block_elements,
                              uint64_t *block_bytes)
{
    switch (type) {
    case 0: *block_elements = 1;   *block_bytes = 4;   return 1; /* F32 */
    case 6: *block_elements = 32;  *block_bytes = 22;  return 1; /* Q5_0 */
    case 8: *block_elements = 32;  *block_bytes = 34;  return 1; /* Q8_0 */
    case 12:*block_elements = 256; *block_bytes = 144; return 1; /* Q4_K */
    case 14:*block_elements = 256; *block_bytes = 210; return 1; /* Q6_K */
    default: return 0;
    }
}

static int tensor_byte_size(const GgufTensorInfo *tensor, uint64_t *byte_size)
{
    uint64_t block_elements;
    uint64_t block_bytes;
    uint64_t elements = 1;
    uint32_t i;
    if (!tensor_type_layout(tensor->type, &block_elements, &block_bytes)) {
        return 0;
    }
    for (i = 0; i < tensor->dimension_count; ++i) {
        if (tensor->dimensions[i] == 0 ||
            elements > UINT64_MAX / tensor->dimensions[i]) {
            return 0;
        }
        elements *= tensor->dimensions[i];
    }
    if (elements % block_elements != 0 ||
        elements / block_elements > UINT64_MAX / block_bytes) {
        return 0;
    }
    *byte_size = (elements / block_elements) * block_bytes;
    return 1;
}

static int parse_metadata(Reader *reader, GgufFile *file)
{
    uint64_t i;
    file->metadata = (GgufMetadataEntry *)calloc(
        (size_t)file->metadata_count, sizeof(*file->metadata));
    if (file->metadata == NULL && file->metadata_count != 0) {
        return 0;
    }
    for (i = 0; i < file->metadata_count; ++i) {
        GgufMetadataEntry *entry = &file->metadata[i];
        entry->key = read_string(reader);
        if (entry->key == NULL || !read_u32(reader, &entry->type)) {
            return 0;
        }
        entry->value_offset = reader->position;
        if (!skip_value(reader, entry->type, 0,
                        &entry->array_type, &entry->array_count)) {
            return 0;
        }
    }
    return 1;
}

static int parse_tensors(Reader *reader, GgufFile *file)
{
    uint64_t i;
    file->tensors = (GgufTensorInfo *)calloc(
        (size_t)file->tensor_count, sizeof(*file->tensors));
    if (file->tensors == NULL && file->tensor_count != 0) {
        return 0;
    }
    for (i = 0; i < file->tensor_count; ++i) {
        GgufTensorInfo *tensor = &file->tensors[i];
        uint32_t dimension;
        tensor->name = read_string(reader);
        if (tensor->name == NULL || !read_u32(reader, &tensor->dimension_count) ||
            tensor->dimension_count == 0 || tensor->dimension_count > 4) {
            return 0;
        }
        for (dimension = 0; dimension < tensor->dimension_count; ++dimension) {
            if (!read_u64(reader, &tensor->dimensions[dimension])) {
                return 0;
            }
        }
        if (!read_u32(reader, &tensor->type) || !read_u64(reader, &tensor->offset) ||
            !tensor_byte_size(tensor, &tensor->byte_size)) {
            return 0;
        }
        if (tensor->offset % file->alignment != 0) {
            return 0;
        }
    }
    return 1;
}

static int validate_tensor_ranges(const GgufFile *file)
{
    uint64_t i;
    uint64_t available;
    if (file->tensor_data_offset > file->file_size) {
        return 0;
    }
    available = file->file_size - file->tensor_data_offset;
    for (i = 0; i < file->tensor_count; ++i) {
        const GgufTensorInfo *tensor = &file->tensors[i];
        if (tensor->offset > available || tensor->byte_size > available - tensor->offset) {
            return 0;
        }
    }
    return 1;
}

static int metadata_index(const GgufFile *file, const char *key)
{
    uint64_t i;
    if (file == NULL || key == NULL) {
        return -1;
    }
    for (i = 0; i < file->metadata_count; ++i) {
        if (strcmp(file->metadata[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

int gguf_get_uint(const GgufFile *file, const char *key, uint64_t *value)
{
    int index = metadata_index(file, key);
    const GgufMetadataEntry *entry;
    const unsigned char *p;
    uint32_t size;
    uint32_t i;
    uint64_t result = 0;
    if (index < 0 || value == NULL) {
        return 0;
    }
    entry = &file->metadata[index];
    if (!(entry->type == 0 || entry->type == 2 || entry->type == 4 || entry->type == 10)) {
        return 0;
    }
    fixed_type_size(entry->type, &size);
    p = file->data + entry->value_offset;
    for (i = 0; i < size; ++i) {
        result |= (uint64_t)p[i] << (i * 8);
    }
    *value = result;
    return 1;
}

int gguf_get_float(const GgufFile *file, const char *key, double *value)
{
    int index;
    const GgufMetadataEntry *entry;
    Reader reader;
    double parsed_value;

    if (file == NULL || key == NULL || value == NULL || file->data == NULL ||
        file->metadata == NULL) {
        return 0;
    }

    index = metadata_index(file, key);
    if (index < 0) {
        return 0;
    }
    entry = &file->metadata[index];
    if (entry->type != 6 && entry->type != 12) {
        return 0;
    }

    reader.data = file->data;
    reader.size = file->file_size;
    reader.position = entry->value_offset;

    if (entry->type == 6) {
        uint32_t bits;
        float parsed_f32;
        if (!read_u32(&reader, &bits)) {
            return 0;
        }
        memcpy(&parsed_f32, &bits, sizeof(parsed_f32));
        if (!isfinite(parsed_f32)) {
            return 0;
        }
        parsed_value = (double)parsed_f32;
    } else {
        uint64_t bits;
        if (!read_u64(&reader, &bits)) {
            return 0;
        }
        memcpy(&parsed_value, &bits, sizeof(parsed_value));
        if (!isfinite(parsed_value)) {
            return 0;
        }
    }

    *value = parsed_value;
    return 1;
}

int gguf_get_string(const GgufFile *file, const char *key,
                    const char **value, uint64_t *length)
{
    int index = metadata_index(file, key);
    const GgufMetadataEntry *entry;
    Reader reader;
    uint64_t string_length;
    if (index < 0 || value == NULL || length == NULL) {
        return 0;
    }
    entry = &file->metadata[index];
    if (entry->type != 8) {
        return 0;
    }
    reader.data = file->data;
    reader.size = file->file_size;
    reader.position = entry->value_offset;
    if (!read_u64(&reader, &string_length) || !has_bytes(&reader, string_length)) {
        return 0;
    }
    *value = (const char *)(file->data + reader.position);
    *length = string_length;
    return 1;
}

int gguf_get_array_info(const GgufFile *file, const char *key,
                        uint32_t *element_type, uint64_t *count)
{
    int index = metadata_index(file, key);
    if (index < 0 || count == NULL || element_type == NULL ||
        file->metadata[index].type != 9) {
        return 0;
    }
    *element_type = file->metadata[index].array_type;
    *count = file->metadata[index].array_count;
    return 1;
}

int gguf_get_bool(const GgufFile *file, const char *key, int *value)
{
    int index = metadata_index(file, key);
    const GgufMetadataEntry *entry;
    if (index < 0 || value == NULL) {
        return 0;
    }
    entry = &file->metadata[index];
    if (entry->type != 7 || entry->value_offset >= file->file_size) {
        return 0;
    }
    *value = file->data[entry->value_offset] != 0;
    return 1;
}

int gguf_array_cursor_init(const GgufFile *file, const char *key,
                           GgufArrayCursor *cursor)
{
    int index = metadata_index(file, key);
    const GgufMetadataEntry *entry;
    Reader reader;
    uint32_t element_type;
    uint64_t count;
    if (index < 0 || cursor == NULL) {
        return 0;
    }
    entry = &file->metadata[index];
    if (entry->type != 9) {
        return 0;
    }
    reader.data = file->data;
    reader.size = file->file_size;
    reader.position = entry->value_offset;
    if (!read_u32(&reader, &element_type) || !read_u64(&reader, &count) ||
        element_type != entry->array_type || count != entry->array_count) {
        return 0;
    }
    cursor->file = file;
    cursor->position = reader.position;
    cursor->remaining = count;
    cursor->element_type = element_type;
    return 1;
}

int gguf_array_next_string(GgufArrayCursor *cursor, const char **value,
                           uint64_t *length)
{
    Reader reader;
    uint64_t string_length;
    if (cursor == NULL || value == NULL || length == NULL ||
        cursor->file == NULL || cursor->remaining == 0 ||
        cursor->element_type != 8) {
        return 0;
    }
    reader.data = cursor->file->data;
    reader.size = cursor->file->file_size;
    reader.position = cursor->position;
    if (!read_u64(&reader, &string_length) || !has_bytes(&reader, string_length)) {
        return 0;
    }
    *value = (const char *)(reader.data + reader.position);
    *length = string_length;
    cursor->position = reader.position + string_length;
    --cursor->remaining;
    return 1;
}

int gguf_array_next_uint(GgufArrayCursor *cursor, uint64_t *value)
{
    uint32_t size;
    uint32_t i;
    uint64_t result = 0;
    if (cursor == NULL || value == NULL || cursor->file == NULL ||
        cursor->remaining == 0 ||
        !(cursor->element_type == 0 || cursor->element_type == 2 ||
          cursor->element_type == 4 || cursor->element_type == 10) ||
        !fixed_type_size(cursor->element_type, &size) ||
        cursor->position > cursor->file->file_size ||
        size > cursor->file->file_size - cursor->position) {
        return 0;
    }
    for (i = 0; i < size; ++i) {
        result |= (uint64_t)cursor->file->data[cursor->position + i] << (i * 8);
    }
    cursor->position += size;
    --cursor->remaining;
    *value = result;
    return 1;
}

int gguf_array_next_int(GgufArrayCursor *cursor, int64_t *value)
{
    uint32_t size;
    uint32_t i;
    uint64_t bits = 0;
    if (cursor == NULL || value == NULL || cursor->file == NULL ||
        cursor->remaining == 0 ||
        !(cursor->element_type == 1 || cursor->element_type == 3 ||
          cursor->element_type == 5 || cursor->element_type == 11) ||
        !fixed_type_size(cursor->element_type, &size) ||
        cursor->position > cursor->file->file_size ||
        size > cursor->file->file_size - cursor->position) {
        return 0;
    }
    for (i = 0; i < size; ++i) {
        bits |= (uint64_t)cursor->file->data[cursor->position + i] << (i * 8);
    }
    if (size < 8 && (bits & ((uint64_t)1 << (size * 8 - 1))) != 0) {
        bits |= ~(((uint64_t)1 << (size * 8)) - 1);
    }
    cursor->position += size;
    --cursor->remaining;
    *value = (int64_t)bits;
    return 1;
}

const GgufTensorInfo *gguf_find_tensor(const GgufFile *file, const char *name)
{
    uint64_t i;
    if (file == NULL || name == NULL) {
        return NULL;
    }
    for (i = 0; i < file->tensor_count; ++i) {
        if (strcmp(file->tensors[i].name, name) == 0) {
            return &file->tensors[i];
        }
    }
    return NULL;
}

const void *gguf_tensor_data(const GgufFile *file, const GgufTensorInfo *tensor)
{
    if (file == NULL || tensor == NULL || file->data == NULL ||
        tensor->offset > file->file_size - file->tensor_data_offset) {
        return NULL;
    }
    return file->data + file->tensor_data_offset + tensor->offset;
}

int gguf_get_f32_tensor_view(const GgufFile *file, const char *name,
                             GgufF32TensorView *view)
{
    // promise: clear the view first before continuing
    if (view == nullptr) {
        return 0;
    }
    // clear every field in view
    *view = {};

    if ((file == nullptr) || (name == nullptr)) {
        return 0;
    }

    if (file->tensor_count == 0) {
        return 0;
    }
    
    const GgufTensorInfo *tensor = gguf_find_tensor(file, name);
    
    if ((tensor == nullptr) || (tensor->type != 0)) {
        return 0;
    }

    if ((tensor->dimension_count > 4) || (tensor->dimension_count == 0)) {
        return 0;
    }

    uint64_t elem_count = 1;
    for (uint32_t i = 0; i < tensor->dimension_count; i++) {
        if (tensor->dimensions[i] > 0) {
            if (elem_count > UINT64_MAX / tensor->dimensions[i]) {
                return 0;
            }
            elem_count *= tensor->dimensions[i];
        } else {
            return 0;
        }
    }

    if ((elem_count > UINT64_MAX / sizeof(float)) || (tensor->byte_size != elem_count * sizeof(float))) {
        return 0;
    }

    const float *data = (const float *)gguf_tensor_data(file, tensor);
    if (data == nullptr) {
        return 0;
    }

    view->data = data;
    view->element_count = elem_count;
    view->dimension_count = tensor->dimension_count;
    for (uint32_t i = 0; i < tensor->dimension_count; i++) {
        view->dimensions[i] = tensor->dimensions[i];
    }

    return 1;
}

int gguf_get_quantized_tensor_view(
    const GgufFile *file,
    const char *name,
    GgufQuantizedTensorView *view
)
{
    if (view == nullptr) {
        return 0;
    }
    *view = {};
    if (file == nullptr || name == nullptr || file->tensor_count == 0) {
        return 0;
    }

    const GgufTensorInfo *tensor = gguf_find_tensor(file, name);
    if (tensor == nullptr || tensor->dimension_count == 0 || tensor->dimension_count > 4) {
        return 0;
    }

    uint64_t block_elements;
    uint64_t block_bytes;
    switch (tensor->type) {
        case ENGINE_GGML_TYPE_Q5_0: block_elements = 32; block_bytes = 22; break;
        case ENGINE_GGML_TYPE_Q8_0: block_elements = 32; block_bytes = 34; break;
        case ENGINE_GGML_TYPE_Q4_K: block_elements = 256; block_bytes = 144; break;
        case ENGINE_GGML_TYPE_Q6_K: block_elements = 256; block_bytes = 210; break;
        default: return 0;
    }

    uint64_t element_count = 1;
    for (uint32_t i = 0; i < tensor->dimension_count; ++i) {
        const uint64_t dimension = tensor->dimensions[i];
        if (dimension == 0 || element_count > UINT64_MAX / dimension) {
            return 0;
        }
        element_count *= dimension;
    }
    if (element_count % block_elements != 0) {
        return 0;
    }
    const uint64_t block_count = element_count / block_elements;
    if (block_count > UINT64_MAX / block_bytes ||
        tensor->byte_size != block_count * block_bytes ||
        file->tensor_data_offset > file->file_size ||
        tensor->offset > file->file_size - file->tensor_data_offset ||
        tensor->byte_size > file->file_size - file->tensor_data_offset - tensor->offset) {
        return 0;
    }

    const unsigned char *data = (const unsigned char *)gguf_tensor_data(file, tensor);
    if (data == nullptr) {
        return 0;
    }
    view->data = data;
    view->ggml_type = tensor->type;
    view->dimension_count = tensor->dimension_count;
    for (uint32_t i = 0; i < tensor->dimension_count; ++i) {
        view->dimensions[i] = tensor->dimensions[i];
    }
    view->element_count = element_count;
    view->block_elements = block_elements;
    view->block_bytes = block_bytes;
    view->block_count = block_count;
    return 1;
}

void gguf_close(GgufFile *file)
{
    uint64_t i;
    if (file == NULL) {
        return;
    }
    if (file->metadata != NULL) {
        for (i = 0; i < file->metadata_count; ++i) {
            free(file->metadata[i].key);
        }
    }
    if (file->tensors != NULL) {
        for (i = 0; i < file->tensor_count; ++i) {
            free(file->tensors[i].name);
        }
    }
    free(file->metadata);
    free(file->tensors);
    if (file->data != NULL && file->data != MAP_FAILED) {
        munmap((void *)file->data, (size_t)file->file_size);
    }
    if (file->file_descriptor >= 0) {
        close(file->file_descriptor);
    }
    memset(file, 0, sizeof(*file));
    file->file_descriptor = -1;
}

int gguf_open(const char *path, GgufFile *file, char *error, int error_capacity)
{
    struct stat stat_buffer;
    Reader reader;
    uint32_t version;
    uint64_t metadata_count;
    uint64_t tensor_count;
    uint64_t value;
    const char *architecture;
    uint64_t architecture_length;
    uint64_t directory_end;
    uint64_t i;

    memset(file, 0, sizeof(*file));
    file->file_descriptor = -1;
    if (path == NULL || path[0] == '\0') {
        set_error(error, error_capacity, "Model path is empty.");
        return 0;
    }
    file->file_descriptor = open(path, O_RDONLY);
    if (file->file_descriptor < 0 || fstat(file->file_descriptor, &stat_buffer) != 0 ||
        stat_buffer.st_size < GGUF_HEADER_SIZE ||
        (uint64_t)stat_buffer.st_size > SIZE_MAX) {
        set_error(error, error_capacity, "Could not open model file or file is too small.");
        gguf_close(file);
        return 0;
    }
    file->file_size = (uint64_t)stat_buffer.st_size;
    file->data = (const unsigned char *)mmap(NULL, (size_t)file->file_size,
                                             PROT_READ, MAP_PRIVATE,
                                             file->file_descriptor, 0);
    if (file->data == MAP_FAILED) {
        file->data = NULL;
        set_error(error, error_capacity, "Could not memory-map model file.");
        gguf_close(file);
        return 0;
    }

    reader.data = file->data;
    reader.size = file->file_size;
    reader.position = 0;
    if (memcmp(reader.data, "GGUF", 4) != 0) {
        set_error(error, error_capacity, "Model file does not have GGUF magic.");
        gguf_close(file);
        return 0;
    }
    reader.position = 4;
    if (!read_u32(&reader, &version) || !read_u64(&reader, &tensor_count) ||
        !read_u64(&reader, &metadata_count) || version != 3 ||
        tensor_count == 0 || tensor_count > GGUF_MAX_TENSORS ||
        metadata_count > GGUF_MAX_METADATA_ENTRIES) {
        set_error(error, error_capacity, "Unsupported or malformed GGUF header (expected version 3).");
        gguf_close(file);
        return 0;
    }
    file->version = version;
    file->tensor_count = tensor_count;
    file->metadata_count = metadata_count;
    file->alignment = 32;
    if (!parse_metadata(&reader, file)) {
        set_error(error, error_capacity, "Malformed GGUF metadata section.");
        gguf_close(file);
        return 0;
    }
    if (gguf_get_uint(file, "general.alignment", &value)) {
        if (value < 8 || value > 4096 || (value & (value - 1)) != 0) {
            set_error(error, error_capacity, "GGUF alignment metadata is invalid.");
            gguf_close(file);
            return 0;
        }
        file->alignment = (uint32_t)value;
    }
    if (!gguf_get_string(file, "general.architecture", &architecture,
                         &architecture_length) || architecture_length == 0) {
        set_error(error, error_capacity, "GGUF file is missing general.architecture metadata.");
        gguf_close(file);
        return 0;
    }
    (void)architecture;

    /* Reject duplicate tensor names before clients look tensors up by name. */
    if (!parse_tensors(&reader, file)) {
        set_error(error, error_capacity,
                  "Malformed tensor directory, unsupported tensor type, or tensor outside file bounds.");
        gguf_close(file);
        return 0;
    }
    for (i = 0; i < tensor_count; ++i) {
        uint64_t j;
        for (j = 0; j < i; ++j) {
            if (strcmp(file->tensors[i].name, file->tensors[j].name) == 0) {
                set_error(error, error_capacity, "GGUF tensor directory contains duplicate names.");
                gguf_close(file);
                return 0;
            }
        }
    }
    directory_end = reader.position;
    if (directory_end > UINT64_MAX - (file->alignment - 1)) {
        set_error(error, error_capacity, "GGUF tensor directory size overflows.");
        gguf_close(file);
        return 0;
    }
    file->tensor_data_offset =
        (directory_end + file->alignment - 1) & ~((uint64_t)file->alignment - 1);
    if (file->tensor_data_offset > file->file_size) {
        set_error(error, error_capacity, "GGUF tensor data section is outside the file.");
        gguf_close(file);
        return 0;
    }
    if (!validate_tensor_ranges(file)) {
        set_error(error, error_capacity, "GGUF tensor data extends beyond the end of the file.");
        gguf_close(file);
        return 0;
    }
    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }
    return 1;
}
