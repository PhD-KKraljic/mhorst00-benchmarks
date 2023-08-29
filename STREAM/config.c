#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

StreamConfig_t config = {
    .stream_array_size = 1342177280,
    .ntimes = 10,
    .offset = 0,
    .block_size = 512,
};

static size_t readInt(char* optarg) {
    char* end;
    return (size_t) strtoumax(optarg, &end, 10);
}

void initStreamConfig(int argc, char **argv) {

  static struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"stream-array-size", required_argument, 0, 's'},
      {"offset", required_argument, 0, 'o'},
      {"ntimes", required_argument, 0, 'n'},
#ifdef CONFIG_HIP
      {"block-size", required_argument, 0, 'b'},
#endif
      {0, 0, 0, 0}};

  static char *description[] = {
      "Display this page",
      "Set the stream array size",
      "Set the offset",
      "How often to repeat",
#ifdef CONFIG_HIP
      "Blocksize to use on the GPU",
#endif
  };

  int c;
  while (1) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

#ifdef CONFIG_HIP
    c = getopt_long(argc, argv, "hs:o:n:b:", long_options, &option_index);
#else
    c = getopt_long(argc, argv, "hs:o:n:", long_options, &option_index);
#endif

    /* Detect the end of the options. */
    if (-1 == c) break;

    switch (c) {
    case 'h':
      printf("STREAM Benchmark help menu\n");
      printf("Usage:\n");
      for (int i = 0; i < sizeof(description) / sizeof(description[0]); i++) {
        printf("    -%c, --%s:\n", long_options[i].val, long_options[i].name);
        printf("        %s\n", description[i]);
      }
      exit(0);

    case 's':
      config.stream_array_size = readInt(optarg);
      break;

    case 'o':
      config.offset = readInt(optarg);
      break;

    case 'n':
      if (readInt(optarg) > 1) {
        config.ntimes = readInt(optarg);
      }
      break;

#ifdef CONFIG_HIP
    case 'b':
      config.block_size = readInt(optarg);
      break;
#endif

    case '?':
      /* getopt_long already printed an error message. */
      break;

    default:
      abort();
    }
  }
}
