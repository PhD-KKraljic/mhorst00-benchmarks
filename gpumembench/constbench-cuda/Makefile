# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

LIKWID = /var/tmp/likwid-nvidia
LIKWID_NVMON = TRUE
include ../common.mk
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -ffast-math -fassociative-math -msse2 -msse
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -ftz=true

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

constbench: main.o const_kernels.o
	${CC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -o $@ $^ ${LFLAGS}

main.o: main.cpp const_kernels.h
	${CC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -c ${FLAGS} $<

const_kernels.o: const_kernels.cu
	${NVCC} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${NVCODE} ${NVFLAGS} -DUNIX -c $<

clean:
	\rm -f constbench main.o const_kernels.o

rebuild: clean constbench

