#include <hip/hip_runtime.h>
#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include "config.h"

#define GPU_ERROR(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true) {
    if (code != hipSuccess) {
        std::cerr << "GPUassert: \"" << hipGetErrorString(code) << "\"  in "
                << file << ": " << line << "\n";
        if (abort)
        exit(code);
    }
}

template <typename T>
__global__ void init_kernel(T *A, const double value, const size_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = value;
    }
}

template <typename T>
__global__ void stream_reference_kernel(T *A, const size_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = 2.0E0 * A[i];
    }
}

template <typename T>
__global__ void stream_copy_kernel(const T *__restrict__ A,
                                  T *C,
                                  const int64_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        C[i] = A[i];
    }
}

template <typename T>
__global__ void stream_scale_kernel(T *B,
                                  const T *__restrict__ C,
                                  const T *__restrict__ scalar,
                                  const int64_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        B[i] = scalar[0] * C[i];
    }
}

template <typename T>
__global__ void stream_add_kernel(const T *__restrict__ A,
                                const T *__restrict__ B,
                                T *C,
                                const int64_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void stream_triad_kernel(T *A,
                                 const T *__restrict__ B,
                                 const T *__restrict__ C,
                                 const T *__restrict__ scalar,
                                 const int64_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = B[i] + scalar[0] * C[i];
    }
}

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif
STREAM_TYPE *a, *b, *c, *scalar;
int max_blocks;

extern StreamConfig_t config;

extern "C" {
    void hip_STREAM_Init() {
        // reserve space on GPU
        GPU_ERROR(hipMalloc(&a, STREAM_ARRAY_SIZE * sizeof(STREAM_TYPE) + OFFSET));
        GPU_ERROR(hipMalloc(&b, STREAM_ARRAY_SIZE * sizeof(STREAM_TYPE) + OFFSET));
        GPU_ERROR(hipMalloc(&c, STREAM_ARRAY_SIZE * sizeof(STREAM_TYPE) + OFFSET));
        GPU_ERROR(hipMalloc(&scalar, sizeof(STREAM_TYPE)));
        if (a == NULL || b == NULL || c == NULL) {
            printf("Memory not allocated.\n");
            exit(1);
        }

        // get GPU device properties
        hipDeviceProp_t prop;
        int deviceId = 0;
        GPU_ERROR(hipGetDevice(&deviceId));
        GPU_ERROR(hipGetDeviceProperties(&prop, deviceId));
        int smCount = prop.multiProcessorCount;
        int maxActiveBlocks = 0;
        GPU_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, stream_triad_kernel<STREAM_TYPE>, BLOCK_SIZE, 0));

        max_blocks = maxActiveBlocks * smCount;

        hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, a, 1.0, STREAM_ARRAY_SIZE);
        hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, b, 2.0, STREAM_ARRAY_SIZE);
        hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, c, 0.0, STREAM_ARRAY_SIZE);
        hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, scalar, 3.0, 1);
        GPU_ERROR(hipDeviceSynchronize());
    }

    void hip_STREAM_Reference() {
        hipLaunchKernelGGL((stream_reference_kernel<STREAM_TYPE>),
            dim3(max_blocks), dim3(BLOCK_SIZE), 0, 0,
            a, STREAM_ARRAY_SIZE
        );
        GPU_ERROR(hipGetLastError());
        GPU_ERROR(hipDeviceSynchronize());
    }

    void hip_STREAM_Copy() {
        hipLaunchKernelGGL((stream_copy_kernel<STREAM_TYPE>),
            dim3(max_blocks), dim3(BLOCK_SIZE), 0, 0,
            a, c, STREAM_ARRAY_SIZE
        );
        GPU_ERROR(hipGetLastError());
        GPU_ERROR(hipDeviceSynchronize());
    }

    void hip_STREAM_Scale() {
        hipLaunchKernelGGL((stream_scale_kernel<STREAM_TYPE>),
            dim3(max_blocks), dim3(BLOCK_SIZE), 0, 0,
            b, c, scalar, STREAM_ARRAY_SIZE
        );
        GPU_ERROR(hipGetLastError());
        GPU_ERROR(hipDeviceSynchronize());
    }

    void hip_STREAM_Add() {
        hipLaunchKernelGGL((stream_add_kernel<STREAM_TYPE>),
            dim3(max_blocks), dim3(BLOCK_SIZE), 0, 0,
            a, b, c, STREAM_ARRAY_SIZE
        );
        GPU_ERROR(hipGetLastError());
        GPU_ERROR(hipDeviceSynchronize());
    }

    void hip_STREAM_Triad() {
        hipLaunchKernelGGL((stream_triad_kernel<STREAM_TYPE>),
            dim3(max_blocks), dim3(BLOCK_SIZE), 0, 0,
            a, b, c, scalar, STREAM_ARRAY_SIZE
        );
        GPU_ERROR(hipGetLastError());
        GPU_ERROR(hipDeviceSynchronize());
    }

    void hip_STREAM_Close() {
        GPU_ERROR(hipFree(a));
        GPU_ERROR(hipFree(b));
        GPU_ERROR(hipFree(c));
        GPU_ERROR(hipFree(scalar));
    }
}
