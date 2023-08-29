# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

include ./common.mk

.PHONY: all clean rebuild

all:
	mkdir -p bin
ifdef CUDA_INSTALL_PATH
	$(info compiling with CUDA and OpenCL)
	$(MAKE) -C cachebench-cuda
	$(MAKE) -C shmembench-cuda
	$(MAKE) -C constbench-cuda
	cp cachebench-cuda/cachebench cachebench-cuda/cachebench-l2-only cachebench-cuda/cachebench-tex-loads shmembench-cuda/shmembench constbench-cuda/constbench bin/
endif
	$(MAKE) -C cachebench-ocl
	$(MAKE) -C shmembench-ocl
	$(MAKE) -C constbench-ocl
	cp cachebench-ocl/cachebench-ocl shmembench-ocl/shmembench-ocl constbench-ocl/constbench-ocl bin/

clean:
	$(MAKE) -C cachebench-cuda clean
	$(MAKE) -C cachebench-ocl clean
	$(MAKE) -C shmembench-cuda clean
	$(MAKE) -C shmembench-ocl clean
	$(MAKE) -C constbench-cuda clean
	$(MAKE) -C constbench-ocl clean

rebuild:
	$(MAKE) -C cachebench-cuda rebuild
	$(MAKE) -C cachebench-ocl rebuild
	$(MAKE) -C shmembench-cuda rebuild
	$(MAKE) -C shmembench-ocl rebuild
	$(MAKE) -C constbench-cuda rebuild
	$(MAKE) -C constbench-ocl rebuild
