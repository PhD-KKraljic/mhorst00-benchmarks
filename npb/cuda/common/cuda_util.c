
#include "type.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define CUCHK(x)\
	do {\
		cudaError_t err = (x);\
		if (err != cudaSuccess) {\
			const char *name = cudaGetErrorName(err);\
			const char *desc = cudaGetErrorString(err);\
			fprintf(stderr, "[%s (%s:%d)]: %s\n", name, __FILE__, __LINE__, desc);\
			exit(1);\
		}\
	}while(0)

size_t RoundWorkSize(size_t work_size, size_t group_size) {
	size_t rem = work_size % group_size;
	return (rem == 0) ? work_size : (work_size + group_size - rem);
}

/****************************************************************************/
/* CUDA Profiling Functions                                                 */
/****************************************************************************/

#define MAX_NUM_LOG         65536
#define MAX_NUM_EVENT       100
#define MAX_EVENT_NAME      35
#define WRITE_EVENT_NAME    "Host to Device Data Transfer"
#define READ_EVENT_NAME     "Device to Host Data Transfer"
#define COPY_EVENT_NAME     "Device to Device Data Transfer"

typedef struct cuda_ProfilerLog_t {
  char name[MAX_EVENT_NAME];
  cudaEvent_t ev_s;
  cudaEvent_t ev_e;
} cuda_ProfilerLog;

typedef struct cuda_ProfilerEvent_t {
  char name[MAX_EVENT_NAME];
  double elapsed;
} cuda_ProfilerEvent;

logical             npb_timeron             = false;
logical             cuda_profiler_timeron   = false;
cuda_ProfilerLog    *cuda_profiler_logs;
unsigned long long  cuda_profiler_log_cnt   = 0;
unsigned long long  cuda_profiler_log_max   = MAX_NUM_LOG;
cuda_ProfilerEvent  *cuda_profiler_events;
int                 cuda_profiler_event_cnt = 0;

static double cuda_GetEventElapsedTime(cuda_ProfilerLog e) {
  double ret;
  float ms;
  CUCHK(cudaEventElapsedTime(&ms, e.ev_s, e.ev_e));

  ret = (double)ms;
  ret /= 1.0E+3;

  return ret;
}

static int cuda_GetEventIndex(const char *name) {
  int i, ret;

  for (i = 0; i < cuda_profiler_event_cnt; i++) {
    if (strcmp(name, cuda_profiler_events[i].name) == 0)
      return i;
  }

  // does NOT find event

  if (i == MAX_NUM_EVENT) {
    fprintf(stderr, " Too many event types to profile\n");
    exit(EXIT_FAILURE);
  }

  strcpy(cuda_profiler_events[cuda_profiler_event_cnt].name, name);
  cuda_profiler_events[cuda_profiler_event_cnt].elapsed = 0.0;

  ret = cuda_profiler_event_cnt++;

  return ret;
}

static void cuda_ProfilerAccumLogs(void) {
  unsigned long long i;
  int ev_idx;
  double elapsed;

  for (i = 0; i < cuda_profiler_log_cnt; i++) {
    elapsed = cuda_GetEventElapsedTime(cuda_profiler_logs[i]);
    ev_idx = cuda_GetEventIndex(cuda_profiler_logs[i].name);
    cuda_profiler_events[ev_idx].elapsed += elapsed;
  }
}

void cuda_ProfilerPrintElapsedTime(const char *name, double elapsed) {
  if (elapsed != 0.0) {
    printf(" %35s : %12.6f sec\n", name, elapsed);
  }
}

static void cuda_ProfilerPrintEventWithIndex(int ev_idx) {
  cuda_ProfilerPrintElapsedTime(cuda_profiler_events[ev_idx].name,
                                cuda_profiler_events[ev_idx].elapsed);
}


// User APIs

void cuda_ProfilerStartEventRecord(const char *ev_name, cudaStream_t stream) {

  // doubling
  if (cuda_profiler_log_cnt == cuda_profiler_log_max) {
    cuda_ProfilerLog *tmp;
    tmp = (cuda_ProfilerLog *)malloc(sizeof(cuda_ProfilerLog)*cuda_profiler_log_max*2);
    unsigned long long i;

    for (i = 0; i < cuda_profiler_log_max; i++) {
      tmp[i].ev_s = cuda_profiler_logs[i].ev_s;
      tmp[i].ev_e = cuda_profiler_logs[i].ev_e;
      strcpy(tmp[i].name, cuda_profiler_logs[i].name);
    }

    free(cuda_profiler_logs);
    cuda_profiler_logs = tmp;
    cuda_profiler_log_max *= 2;
  }

  cudaEvent_t ev_s;
  cudaEventCreate(&ev_s);
  cudaEventRecord(ev_s, stream);

  cuda_profiler_logs[cuda_profiler_log_cnt].ev_s = ev_s;
  strcpy(cuda_profiler_logs[cuda_profiler_log_cnt].name, ev_name);
}

void cuda_ProfilerEndEventRecord(const char *ev_name, cudaStream_t stream) {

  // FIXME
  assert(strcmp(cuda_profiler_logs[cuda_profiler_log_cnt].name, ev_name) == 0);

  cudaEvent_t ev_e;
  cudaEventCreate(&ev_e);
  cudaEventRecord(ev_e, stream);

  cuda_profiler_logs[cuda_profiler_log_cnt].ev_e = ev_e;

  cuda_profiler_log_cnt++;
}

void cuda_ProfilerSetup(void) {
  FILE *fp;

  if ((fp = fopen("timer.flag", "r")) != NULL) {
    fclose(fp);
    npb_timeron = true;
  }

  int i;

  cuda_profiler_logs = (cuda_ProfilerLog *)malloc(sizeof(cuda_ProfilerLog)*cuda_profiler_log_max);
  cuda_profiler_events = (cuda_ProfilerEvent *)malloc(sizeof(cuda_ProfilerEvent)*MAX_NUM_EVENT);

  for (i = 0; i < MAX_NUM_EVENT; i++) {
    cuda_profiler_events[i].elapsed = 0.0;
  }
}

void cuda_ProfilerRelease(void) {
  unsigned long long i;
  for (i = 0; i < cuda_profiler_log_cnt; i++) {
    cudaEventDestroy(cuda_profiler_logs[i].ev_s);
    cudaEventDestroy(cuda_profiler_logs[i].ev_e);
  }

  free(cuda_profiler_logs);
  free(cuda_profiler_events);
}

void cuda_ProfilerStart(void) {
  cuda_profiler_timeron = true;
}

void cuda_ProfilerStop(void) {
  cuda_profiler_timeron = false;
}

void cuda_ProfilerClear(void) {
  unsigned long long i;
  for (i = 0; i < cuda_profiler_log_cnt; i++) {
    cudaEventDestroy(cuda_profiler_logs[i].ev_s);
    cudaEventDestroy(cuda_profiler_logs[i].ev_e);
  }
  cuda_profiler_log_cnt = 0;

  int j;
  for (j = 0; j < cuda_profiler_event_cnt; j++) {
    cuda_profiler_events[j].elapsed = 0.0;
    cuda_profiler_events[j].name[0] = '\0';
  }
  cuda_profiler_event_cnt = 0;
}

void cuda_ProfilerPrintResult(void) {
  int i;
  int wb_idx, rb_idx, copy_idx;

  cuda_ProfilerAccumLogs();

  wb_idx = cuda_GetEventIndex(WRITE_EVENT_NAME);
  rb_idx = cuda_GetEventIndex(READ_EVENT_NAME);
  copy_idx = cuda_GetEventIndex(COPY_EVENT_NAME);

  printf(" =========================================================================\n");
  printf("                       CUDA Event Profiler Results  \n");
  printf(" %35s   %12s \n",
      "Event Name    ", "      Time (sec)");
  for (i = 0; i < cuda_profiler_event_cnt; i++) {
    if (i != wb_idx &&
        i != rb_idx &&
        i != copy_idx)
      cuda_ProfilerPrintEventWithIndex(i);
  }
  cuda_ProfilerPrintEventWithIndex(wb_idx);
  cuda_ProfilerPrintEventWithIndex(rb_idx);
  cuda_ProfilerPrintEventWithIndex(copy_idx);
}

cudaError_t cuda_Profiler_cudaMemcpy(void *dst,
                                     const void *src,
                                     size_t count,
                                     enum cudaMemcpyKind kind) {
#undef cudaMemcpy

  cudaError_t err;

  if (kind == cudaMemcpyDeviceToHost) {
    cuda_ProfilerStartEventRecord(READ_EVENT_NAME, 0);
    err = cudaMemcpy(dst, src, count, kind);
    cuda_ProfilerEndEventRecord(READ_EVENT_NAME, 0);
  }
  else if (kind == cudaMemcpyHostToDevice) {
    cuda_ProfilerStartEventRecord(WRITE_EVENT_NAME, 0);
    err = cudaMemcpy(dst, src, count, kind);
    cuda_ProfilerEndEventRecord(WRITE_EVENT_NAME, 0);
  }
  else if (kind == cudaMemcpyDeviceToDevice) {
    cuda_ProfilerStartEventRecord(COPY_EVENT_NAME, 0);
    err = cudaMemcpy(dst, src, count, kind);
    cuda_ProfilerEndEventRecord(COPY_EVENT_NAME, 0);
  }
  else {
    fprintf(stderr, "Unsupported Memcpy Kind\n");
    exit(EXIT_FAILURE);
  }

  return err;
}

cudaError_t cuda_Profiler_cudaMemcpyAsync(void *dst,
                                          const void *src,
                                          size_t count,
                                          enum cudaMemcpyKind kind,
                                          cudaStream_t stream) {
#undef cudaMemcpyAsync

  if (!npb_timeron || !cuda_profiler_timeron)
    return cudaMemcpyAsync(dst, src, count, kind, stream);

  cudaError_t err;

  if (kind == cudaMemcpyDeviceToHost) {
    cuda_ProfilerStartEventRecord(READ_EVENT_NAME, stream);
    err = cudaMemcpyAsync(dst, src, count, kind, stream);
    cuda_ProfilerEndEventRecord(READ_EVENT_NAME, stream);
  }
  else if (kind == cudaMemcpyHostToDevice) {
    cuda_ProfilerStartEventRecord(WRITE_EVENT_NAME, stream);
    err = cudaMemcpyAsync(dst, src, count, kind, stream);
    cuda_ProfilerEndEventRecord(WRITE_EVENT_NAME, stream);
  }
  else if (kind == cudaMemcpyDeviceToDevice) {
    cuda_ProfilerStartEventRecord(COPY_EVENT_NAME, stream);
    err = cudaMemcpyAsync(dst, src, count, kind, stream);
    cuda_ProfilerEndEventRecord(COPY_EVENT_NAME, stream);
  }
  else {
    fprintf(stderr, "Unsupported Memcpy Kind\n");
    exit(EXIT_FAILURE);
  }

  return err;
}

cudaError_t cuda_Profiler_cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
                                            cudaStream_t stream) {
#undef cudaMemcpy3DAsync

  if (!npb_timeron || !cuda_profiler_timeron)
    return cudaMemcpy3DAsync(p, stream);

  cudaError_t err;
  enum cudaMemcpyKind kind = p->kind;

  if (kind == cudaMemcpyDeviceToHost) {
    cuda_ProfilerStartEventRecord(READ_EVENT_NAME, stream);
    err = cudaMemcpy3DAsync(p, stream);
    cuda_ProfilerEndEventRecord(READ_EVENT_NAME, stream);
  }
  else if (kind == cudaMemcpyHostToDevice) {
    cuda_ProfilerStartEventRecord(WRITE_EVENT_NAME, stream);
    err = cudaMemcpy3DAsync(p, stream);
    cuda_ProfilerEndEventRecord(WRITE_EVENT_NAME, stream);
  }
  else if (kind == cudaMemcpyDeviceToDevice) {
    cuda_ProfilerStartEventRecord(COPY_EVENT_NAME, stream);
    err = cudaMemcpy3DAsync(p, stream);
    cuda_ProfilerEndEventRecord(COPY_EVENT_NAME, stream);
  }
  else {
    fprintf(stderr, "Unsupported Memcpy Kind\n");
    exit(EXIT_FAILURE);
  }

  return err;

}
