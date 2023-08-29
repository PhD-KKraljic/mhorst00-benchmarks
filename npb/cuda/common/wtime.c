#include "wtime.h"
#include <time.h>
#ifndef DOS
#include <sys/time.h>
#endif


void wtime(double *t)
{
  static time_t sec = -1;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  if (sec < 0) sec = ts.tv_sec;
  *t = (ts.tv_sec - sec) + 1.0e-9*ts.tv_nsec;
}
