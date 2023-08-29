SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

include ../sys/make.common

OBJS = cg.o cg_gpu.o \
        ${COMMON}/${RAND}.o \
        ${COMMON}/c_timers.o \
        ${COMMON}/wtime.o\
		${COMMON}/cuda_util.o\
		${COMMON}/c_print_results.o\


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

cg.o:		cg.cu  globals.h npbparams.h
	${NVCCCOMPILE} $<

cg_gpu.o:		cg_gpu.cu  globals.h npbparams.h
	${NVCCCOMPILE} $<

clean:
	- rm -f *.o *~ *.bin
	- rm -f npbparams.h core
