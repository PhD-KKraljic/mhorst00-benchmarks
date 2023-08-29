SHELL=/bin/sh
BENCHMARK=mg
BENCHMARKU=MG
VEC=
LIKWID=/var/tmp/likwid-nvidia
CUDA_HOME=$(CUDA)

include ../config/make.def

include ../sys/make.common

ifdef LIKWID
$(info building with LIKIWD enabled)
	LIKWID_LIB ?= -L$(LIKWID)/lib
	LIKWID_INC ?= -I$(LIKWID)/include
	ifdef CUDA_HOME
$(info building with LIKWID CUDA enabled)
		LIKWID_DEFINES += -DLIKWID_NVMON
	endif
	LIKWID_DEFINES += -DLIKWID
	LIBS += -lm -llikwid
endif
OBJS = mg.o mg_gpu.o \
       ${COMMON}/c_print_results.o  \
       ${COMMON}/${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/wtime.o ${COMMON}/cuda_util.o

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS}  ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE}  ${LIKWID_DEFINES} ${LIKWID_INC} ${LIKWID_LIB} $< 

mg.o: mg.cu globals.h npbparams.h
	${NVCCCOMPILE} $<

mg_gpu.o: mg_gpu.cu globals.h npbparams.h
	${NVCCCOMPILE} $<

clean:
	- rm -f *.o *~
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
