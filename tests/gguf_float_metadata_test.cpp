#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gguf_reader.h"

static char FIXTURE_KEY[] = "fixture.float";

static void store_u32_le(unsigned char *output, uint32_t value)
{
    for (int i = 0; i < 4; ++i) {
        output[i] = (unsigned char)(value >> (i * 8));
    }
}

static void store_u64_le(unsigned char *output, uint64_t value)
{
    for (int i = 0; i < 8; ++i) {
        output[i] = (unsigned char)(value >> (i * 8));
    }
}

static void setup_fixture(
    GgufFile *file,
    GgufMetadataEntry *entry,
    const unsigned char *data,
    uint32_t type,
    uint64_t file_size
)
{
    memset(file, 0, sizeof(*file));
    memset(entry, 0, sizeof(*entry));
    entry->key = FIXTURE_KEY;
    entry->type = type;
    entry->value_offset = 0;
    file->data = data;
    file->file_size = file_size;
    file->metadata_count = 1;
    file->metadata = entry;
}

static void test_reads_f32_and_widens_to_double(void)
{
    unsigned char data[4];
    GgufFile file;
    GgufMetadataEntry entry;
    double value = -1.0;

    /* IEEE-754 F32 1.25, encoded as little-endian GGUF bytes. */
    store_u32_le(data, 0x3fa00000u);
    setup_fixture(&file, &entry, data, 6u, sizeof(data));

    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 1);
    assert(value == 1.25);
}

static void test_reads_f64_without_losing_precision(void)
{
    unsigned char data[8];
    GgufFile file;
    GgufMetadataEntry entry;
    double value = -1.0;

    /* The next representable F64 above 1.0 proves the getter preserves F64. */
    store_u64_le(data, UINT64_C(0x3ff0000000000001));
    setup_fixture(&file, &entry, data, 12u, sizeof(data));

    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 1);
    assert(value == 1.0000000000000002);
}

static void test_rejects_missing_or_wrong_type_without_changing_output(void)
{
    unsigned char data[8] = {0};
    GgufFile file;
    GgufMetadataEntry entry;
    double value = 123.5;

    setup_fixture(&file, &entry, data, 4u, sizeof(data));
    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 0);
    assert(value == 123.5);

    entry.type = 6u;
    assert(gguf_get_float(&file, "missing.key", &value) == 0);
    assert(value == 123.5);
}

static void test_rejects_non_finite_values_without_changing_output(void)
{
    unsigned char data[8];
    GgufFile file;
    GgufMetadataEntry entry;
    double value = 123.5;

    store_u32_le(data, 0x7f800000u); /* F32 +infinity */
    setup_fixture(&file, &entry, data, 6u, 4u);
    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 0);
    assert(value == 123.5);

    store_u32_le(data, 0x7fc00000u); /* F32 quiet NaN */
    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 0);
    assert(value == 123.5);

    store_u64_le(data, UINT64_C(0x7ff0000000000000)); /* F64 +infinity */
    setup_fixture(&file, &entry, data, 12u, 8u);
    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 0);
    assert(value == 123.5);

    store_u64_le(data, UINT64_C(0x7ff8000000000000)); /* F64 quiet NaN */
    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 0);
    assert(value == 123.5);
}

static void test_rejects_truncated_value_and_null_arguments(void)
{
    unsigned char data[4];
    GgufFile file;
    GgufMetadataEntry entry;
    double value = 123.5;

    store_u32_le(data, 0x3fa00000u);
    setup_fixture(&file, &entry, data, 6u, 3u);
    assert(gguf_get_float(&file, FIXTURE_KEY, &value) == 0);
    assert(value == 123.5);

    setup_fixture(&file, &entry, data, 6u, sizeof(data));
    assert(gguf_get_float(NULL, FIXTURE_KEY, &value) == 0);
    assert(gguf_get_float(&file, NULL, &value) == 0);
    assert(gguf_get_float(&file, FIXTURE_KEY, NULL) == 0);
    assert(value == 123.5);
}

int main(void)
{
    printf("Running GGUF floating-point metadata tests...\n");

    test_reads_f32_and_widens_to_double();
    test_reads_f64_without_losing_precision();
    test_rejects_missing_or_wrong_type_without_changing_output();
    test_rejects_non_finite_values_without_changing_output();
    test_rejects_truncated_value_and_null_arguments();

    printf("All GGUF floating-point metadata tests passed.\n");
    return 0;
}
