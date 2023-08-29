SHELL=/bin/sh
BENCHMARK=ep
BENCHMARKU=EP

include ../config/make.def

include ../sys/make.common

OBJS = ep.o \
	   ${COMMON}/c_print_results.o \
	   ${COMMON}/cuda_util.o \
	   ${COMMON}/${RAND}.o \
	   ${COMMON}/c_timers.o \
	   ${COMMON}/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

ep.o: ep.cu npbparams.h
	${NVCCCOMPILE} $<

clean:
	- rm -f *.o *~ *.bin
	- rm -f npbparams.h core

