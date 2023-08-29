#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include "type.h"

#include <cuda_runtime.h>
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


extern "C" {
  size_t RoundWorkSize(size_t work_size, size_t group_size);
  void cuda_ProfilerStartEventRecord(const char *ev_name, cudaStream_t stream);
  void cuda_ProfilerEndEventRecord(const char *ev_name, cudaStream_t stream);
  void cuda_ProfilerSetup(void);
  void cuda_ProfilerRelease(void);
  void cuda_ProfilerStart(void);
  void cuda_ProfilerStop(void);
  void cuda_ProfilerClear(void);
  void cuda_ProfilerPrintElapsedTime(const char *name, double elapsed);
  void cuda_ProfilerPrintResult(void);
  cudaError_t cuda_Profiler_cudaMemcpy(void *dst,
                                       const void *src,
                                       size_t count,
                                       cudaMemcpyKind kind);
  cudaError_t cuda_Profiler_cudaMemcpyAsync(void *dst,
                                            const void *src,
                                            size_t count,
                                            enum cudaMemcpyKind kind,
                                            cudaStream_t stream);
  cudaError_t cuda_Profiler_cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
                                              cudaStream_t stream);

}

#define cudaMemcpy(...) cuda_Profiler_cudaMemcpy(__VA_ARGS__)
#define cudaMemcpyAsync(...) cuda_Profiler_cudaMemcpyAsync(__VA_ARGS__)
#define cudaMemcpy3DAsync(...) cuda_Profiler_cudaMemcpy3DAsync(__VA_ARGS__)

#endif //CUDA_UTIL_H
