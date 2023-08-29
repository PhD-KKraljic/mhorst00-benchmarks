SHELL=/bin/sh
BENCHMARK=is
BENCHMARKU=IS

include ../config/make.def

include ../sys/make.common

OBJS = is.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o \
       ${COMMON}/c_print_results.o ${COMMON}/cuda_util.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

is.o:             is.cu  npbparams.h
	${NVCCCOMPILE} $<

clean:
	- rm -f *.o *~ mputil*
	- rm -f ../common/*.o
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
