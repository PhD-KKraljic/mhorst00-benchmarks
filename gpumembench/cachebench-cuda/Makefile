# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

LIKWID = /var/tmp/likwid-nvidia
LIKWID_NVMON = TRUE

include ../common.mk
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g -ftz=true ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_60,code=\"compute_60\" 
#NVCODE = -gencode=arch=compute_30,code=\"compute_30,sm_30\"

ifdef LIKWID
$(info building with likwid enabled)
	LIKWID_LIB ?= -L$(LIKWID)/lib
	LIKWID_INC ?= -I$(LIKWID)/include
	#LIKWID_DEFINES ?= -DLIKWID_PERFMON
#        ifdef ROCM_HOME
#$(info building with rocm enabled)
#		LIKWID_DEFINES += -DLIKWID_ROCMON
#        endif
        ifdef CUDA
$(info building with cuda enabled)
		LIKWID_DEFINES += -DLIKWID_NVMON
        endif
	LIKWID_DEFINES += -DLIKWID
	LIKWID_LIB += -lm -llikwid
endif
.PHONY: all

all: cachebench cachebench-l2-only cachebench-tex-loads

cachebench: main.o cache_kernels.o
	${CC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -o $@ $^ ${LFLAGS}

cachebench-l2-only: main.o cache_kernels_l2.o
	${CC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -o $@ $^ ${LFLAGS}

cachebench-tex-loads: main.o cache_kernels_texloads.o
	${CC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -o $@ $^ ${LFLAGS}

main.o: main.cpp cache_kernels.h
	${CC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -c ${FLAGS} $<

cache_kernels.o: cache_kernels.cu
	${NVCC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

cache_kernels_l2.o: cache_kernels.cu
	${NVCC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${NVCODE} ${NVFLAGS} -DUNIX -DL2_ONLY -c $< -o $@

cache_kernels_texloads.o: cache_kernels.cu
	${NVCC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${NVCODE} ${NVFLAGS} -DUNIX -DTEX_LOADS -c $< -o $@

clean:
	\rm -f cachebench cachebench-l2-only cachebench-tex-loads main.o cache_kernels.o cache_kernels_l2.o cache_kernels_texloads.o

rebuild: clean all

