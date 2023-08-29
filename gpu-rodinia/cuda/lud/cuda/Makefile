CC = gcc
NVCC = nvcc

ifeq ($(LIKWID_NVMON),true)
	LIKWID_FLAGS = -DLIKWID -DLIKWID_NVMON -L/var/tmp/likwid-nvidia/lib -I/var/tmp/likwid-nvidia/include -llikwid
endif
DEFS += \
		-DGPU_TIMER \
		$(SPACE)

NVCCFLAGS += -I../common \
			 -O3 \
			 -use_fast_math \
			 -arch=sm_60 \
			 -lm $(LIKWID_FLAGS) \
			 $(SPACE)

CFLAGS += -I../common \
					-I$(CUDA)/include \
		  -O3 \
		  -Wall \
		  $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<

%.o:	%.cu 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o $@ -c $<

clang: $(SRC)
	clang++ lud.cu lud_kernel.cu ../common/common.c -o $(EXECUTABLE) \
		-I../common -I../../util \
		-L$(CUDA)/lib64 -lm $(LIKWID_FLAGS) -lcudart_static -ldl -lrt -pthread -DTIMING

$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -lm $(LIKWID_FLAGS) -o $@  $?

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
