# SNUNPB 2019
Optimized OpenCL & CUDA implementations for NAS Parallel Benchmark Suite (NPB3.3.1)

Features:
- OpenCL & CUDA support
  - *SNU_NPB_2019-OCL/* # opencl implementations
  - *SNU_NPB_2019-CUDA/* # cuda implementations
- Consist of 8 benchmark applications (FT, CG, MG, BT, SP, LU, IS, EP)
- Multiple optimization levels for each application

## Compile options
Makefile options are in `config/make.def`.

### OpenCL
Nvidia OpenCL example flags
```bash
C_LIB  = -lm -L/usr/local/cuda/lib64 -lOpenCL
C_INC = -I../common -I/usr/local/cuda/include
```

AMD ROCm OpenCL example flags
```bash
C_LIB  = -lm -L/opt/rocm/opencl/lib/x86_64 -lOpenCL
C_INC = -I../common -I/opt/rocm/opencl/include
```

### CUDA
Nvidia CUDA example flags
```bash
# modify `-arch` option to support your gpu system
NVCCFLAGS = -arch sm_70 -Xcompiler -mcmodel=medium -Xcompiler -Wall -Xcompiler -fopenmp -O3 -c
```

## Usage examples

### OpenCL
```bash
# for OpenCL version, set environment variable first.
export OPENCL_DEVICE_TYPE=gpu

# compilation specific benchmark (CG) with problem size (C)
make CG CLASS=C

# run with baseline benchmark with OpenCL kernel root directory (e.g. CG)
bin/cg.C.x -o 0 -s ./CG

# run with fully-optimized benchmark with OpenCL kernel root directory (e.g. CG)
bin/cg.C.x -s ./CG
```

If `-s` option is not specified, default value is used (../<benchmark-name>)


### CUDA
```bash
# compilation specific benchmark (CG) with problem size (C)
make CG CLASS=C

# run with optimize group 2 benchmark (e.g. CG)
bin/cg.C.x -o 2

# run with fully-optimized benchmark (e.g. CG)
bin/cg.C.x
```

## Optimization groups
Optimization groups are set by adding `-o <number>` to executable. (exclusive)
- '0' for baseline
  - every benchmark
- '1' for Optimization group 1: to increase GPU utilization by increasing parallelism
  - FT, BT, LU, SP
- '2' for Optimization group 2: to exploit the local (shared, for CUDA) and private memory
  - CG, MG, FT, LU, IS, SP, EP
- '3' for Optimization group 3: to expoloit global memory access coalescing
  - BT, SP
- '4' for Optimization group 4: to reduce kernel launch and synchronize overhead
  - LU

If no option is specified, the benchmark runs as fully-optimized version.

If unavailable option is specified, the benchmark returns immediately.
