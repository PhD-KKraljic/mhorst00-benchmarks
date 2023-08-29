SHELL=/bin/sh
BENCHMARK=mg
BENCHMARKU=MG
ROCM_HOME=/opt/rocm
LIKWID=/var/tmp/likwid-lua
LIKWID_ROCMON=TRUE

include ../config/make.def

include ../sys/make.common

ifdef LIKWID
$(info building with LIKIWD enabled)
	LIKWID_LIB ?= -L$(LIKWID)/lib
	LIKWID_INC ?= -I$(LIKWID)/include
	ifdef ROCM_HOME
$(info building with LIKWID ROCm enabled)
		LIKWID_DEFINES += -DLIKWID_ROCMON
	endif
	ifdef CUDA_HOME
$(info building with LIKWID CUDA enabled)
		LIKWID_DEFINES += -DLIKWID_NVMON
	endif
	DEFINES += -DLIKWID
	LIBS += -lm -llikwid
endif


OBJS = mg.o \
       ${COMMON}/c_print_results.o  \
       ${COMMON}/${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/wtime.o \
	${COMMON}/cl_util.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} ${DEFINES} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${LIBS} -o ${PROGRAM} ${OBJS} ${C_LIB} 

.c.o:
	${CCOMPILE} ${DEFINES} ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} ${LIBS} $<

mg.o:		  mg.c globals.h npbparams.h 

clean:
	- rm -f *.o *~
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
