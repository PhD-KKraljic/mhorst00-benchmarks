include ../../../common/make.config

#Can be changed by `make TYPE=CPU`
TYPE = GPU
LIKWID = /var/tmp/likwid-nvidia
#LIKWID_ROCMON = TRUE
LIKWID_NVMON = TRUE
#ROCM_HOME = /opt/rocm
CUDA_HOME = $(CUDA)

#Library
ifeq ($(TYPE),GPU)
	OPENCL_INC = $(NV_OPENCL_INC)
	OPENCL_LIB = $(NV_OPENCL_LIB)
$(info using $(NV_OPENCL_INC) for OpenCL)
	#OPENCL_INC = /opt/rocm/include
	#OPENCL_LIB = /opt/rocm/lib
else
	OPENCL_INC = $(INTEL_OPENCL_INC)
	OPENCL_LIB = $(INTEL_OPENCL_LIB)
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

INC = -I$(OPENCL_INC) -I../common
LIB = -L$(OPENCL_LIB)

#C compiler
CC = g++
CC_FLAGS = -g -O3 -Wall

EXE = lud

$(EXE): lud.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(DEFINES) $(LIKWID_DEFINES) $(LIKWID_INC) $(LIKWID_LIB) \
		-o $(EXE) \
		lud.cpp ../common/common.c ../../util/timing.c \
		$(INC) $(LIB) $(LIBS) -lOpenCL \
		-I../../util -DTIMING \
		-Wno-unused-result

clean:
	rm -f ../$(EXE)
