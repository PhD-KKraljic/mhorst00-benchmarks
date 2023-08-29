# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

include ../common.mk
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -ffast-math -fassociative-math -msse2 -msse
INCDIR = ../common
FLAGS = ${OPTFLAG} -I${OPENCL_INSTALL_PATH}/include -Wall -g -I${INCDIR}
LFLAGS = -L${OPENCL_LIBRARY_PATH} -lm -lstdc++ -lOpenCL -lrt

LIKWID = /var/tmp/likwid-nvidia
#LIKWID_ROCMON = TRUE
#ROCM_HOME = /opt/rocm
LIKWID_NVMON = TRUE
CUDA_HOME = ${CUDA}

ifdef LIKWID
$(info building with likwid enabled)
	LIKWID_LIB ?= -L$(LIKWID)/lib
	LIKWID_INC ?= -I$(LIKWID)/include
	#LIKWID_DEFINES ?= -DLIKWID_PERFMON
        ifdef ROCM_HOME
$(info building with rocm enabled)
		LIKWID_DEFINES += -DLIKWID_ROCMON
        endif
        ifdef CUDA_HOME
$(info building with cuda enabled)
		LIKWID_DEFINES += -DLIKWID_NVMON
        endif
	DEFINES += -DLIKWID
	LIBS += -lm -llikwid
endif

cachebench-ocl: main.o cache_kernels.o
	${CC} ${DEFINES} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${LIBS} -o $@ $^ ${LFLAGS}

main.o: main.cpp cache_kernels.h ${INCDIR}/loclutil.h
	${CC} ${DEFINES} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${LIBS}  -c ${FLAGS} $<

cache_kernels.o: cache_kernels.cpp kernel.h
	${CC}  ${DEFINES} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${LIBS} ${FLAGS} -c $<

kernel.h: kernel.cl
	echo "const char c_kernel[]={" >kernel.h
	hexdump -ve '1/1 "0x%.2x,"' kernel.cl >>kernel.h
	echo "0x00};" >>kernel.h

clean:
	\rm -f cachebench-ocl main.o cache_kernels.o kernel.h

rebuild: clean cachebench-ocl

