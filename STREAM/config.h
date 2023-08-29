#ifndef STREAM_CONFIG_H_
#define STREAM_CONFIG_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t stream_array_size;
    size_t ntimes;
    size_t offset;
    size_t block_size;
} StreamConfig_t;

#define STREAM_ARRAY_SIZE config.stream_array_size
#define NTIMES config.ntimes
#define OFFSET config.offset
#ifdef CONFIG_HIP
#define BLOCK_SIZE config.block_size
#endif

void initStreamConfig(int argc, char **argv);

#ifdef __cplusplus
} // end extern C
#endif
#endif // STREAM_CONFIG_H_
