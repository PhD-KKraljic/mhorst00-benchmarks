CC = gcc
CFLAGS = -O2 -fopenmp -g
CFILES = stream.c

FC = gfortran
FFLAGS = -O2 -fopenmp

HIPCC ?= ${ROCM_HOME}/hip/bin/hipcc
HIPFLAGS = -O3 -std=c++17
HIP_PLATFORM ?= hip

LIKWID = /var/tmp/likwid-lua
LIKWID_NVMON = FALSE
LIKWID_ROCMON = TRUE
ROCM_HOME = /opt/rocm
STREAM_CONFIG = TRUE

ifeq ($(STREAM_CONFIG),TRUE)
$(info building with config enabled)
	DEFINES += -DCONFIG
	CFILES += config.c
endif

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

all: stream_f.exe stream_c.exe stream_hip.exe

stream_f.exe: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FC) $(FFLAGS) -c stream.f
	$(FC) $(FFLAGS) stream.o mysecond.o -o stream_f.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) $(DEFINES) $(LIKWID_DEFINES) $(LIKWID_INC) $(LIKWID_LIB) $(CFILES) -DNTIMES=200 -o stream_c.exe $(LIBS)

clean:
	rm -f *.exe *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -ffreestanding -qopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc

stream_hip.exe:
ifeq (,$(wildcard $(HIPCC)))
	$(error hipcc not found)
endif
	$(HIPCC) -DCONFIG_HIP -c stream.hip.cpp
	$(CC)    -DCONFIG_HIP -c config.c
	$(CC) -DHIP $(DEFINES) $(LIKWID_DEFINES) $(LIKWID_INC) -g -c stream.c
	$(HIPCC) $(HIPFLAGS) -DHIP $(LIKWID_LIB) stream.o config.o stream.hip.o -o stream_$(HIP_PLATFORM).exe $(LIBS)
