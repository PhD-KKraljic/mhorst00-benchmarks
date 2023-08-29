SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT

include ../config/make.def

include ../sys/make.common

OBJS = ft.o ${COMMON}/${RAND}.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/wtime.o ${COMMON}/cuda_util.o\
	   ft_gpu.o 

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

.cu.o:
	${NVCCCOMPILE} $<

ft.o: ft.cu global.h npbparams.h
	${NVCCCOMPILE} $<

ft_gpu.o: ft_gpu.cu global.h npbparams.h

clean:
	- rm -f *.o *~ mputil* *.bin
	- rm -f ft npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
