SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=LU
VEC=

include ../config/make.def

include ../sys/make.common

OBJS = lu.o read_input.o \
       domain.o setcoeff.o setbv.o exact.o setiv.o \
       erhs.o ssor$(VEC).o ssor_baseline.o \
       kernel_ssor_baseline.o \
       rhs$(VEC).o rhs_baseline.o rhs_gmem.o rhs_parallel.o rhs_fullopt.o \
       kernel_rhs_baseline.o kernel_rhs_gmem.o kernel_rhs_parallel.o kernel_rhs_fullopt.o \
       l2norm.o l2norm_baseline.o l2norm_gmem.o \
       kernel_l2norm.o kernel_l2norm_baseline.o kernel_l2norm_gmem.o \
       jacld_blts.o jacld_blts_baseline.o jacld_blts_gmem.o jacld_blts_sync.o jacld_blts_fullopt.o \
       kernel_jacld_blts_baseline.o kernel_jacld_blts_gmem.o kernel_jacld_blts_sync.o kernel_jacld_blts_fullopt.o \
       jacu_buts.o jacu_buts_baseline.o jacu_buts_gmem.o jacu_buts_sync.o jacu_buts_fullopt.o \
       kernel_jacu_buts_baseline.o kernel_jacu_buts_gmem.o kernel_jacu_buts_sync.o kernel_jacu_buts_fullopt.o \
       error.o  \
       pintgr.o verify.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/wtime.o ${COMMON}/cuda_util.o

# npbparams.h is included by applu.incl
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

${PROGRAM}: config
	@if [ x$(VERSION) = xvec ] ; then	\
		${MAKE} VEC=_vec exec;		\
	elif [ x$(VERSION) = xVEC ] ; then	\
		${MAKE} VEC=_vec exec;		\
	else					\
		${MAKE} exec;			\
	fi

exec: $(OBJS)
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

.cu.o:
	${NVCCCOMPILE} $<

lu.o:		lu.cu applu.incl npbparams.h
blts$(VEC).o:	blts$(VEC).c
blts$(VEC).o:	blts$(VEC).c applu.incl npbparams.h
buts$(VEC).o:	buts$(VEC).c
buts$(VEC).o:	buts$(VEC).c applu.incl npbparams.h
erhs.o:		erhs.cu applu.incl npbparams.h
error.o:	error.cu applu.incl npbparams.h
exact.o:	exact.cu applu.incl npbparams.h
jacld_blts.o:	jacld_blts.cu applu.incl npbparams.h
jacld_blts_baseline.o:	jacld_blts_baseline.cu applu.incl npbparams.h
jacld_blts_gmem.o:	jacld_blts_gmem.cu applu.incl npbparams.h
jacld_blts_sync.o:	jacld_blts_sync.cu applu.incl npbparams.h
jacld_blts_fullopt.o:	jacld_blts_fullopt.cu applu.incl npbparams.h
jacu_buts.o:	jacu_buts.cu applu.incl npbparams.h
jacu_buts_baseline.o:	jacu_buts_baseline.cu applu.incl npbparams.h
jacu_buts_gmem.o:	jacu_buts_gmem.cu applu.incl npbparams.h
jacu_buts_sync.o:	jacu_buts_sync.cu applu.incl npbparams.h
jacu_buts_fullopt.o:	jacu_buts_fullopt.cu applu.incl npbparams.h
l2norm.o:	l2norm.cu applu.incl npbparams.h 
l2norm_baseline.o:	l2norm_baseline.cu applu.incl npbparams.h 
l2norm_gmem.o:	l2norm_gmem.cu applu.incl npbparams.h 
pintgr.o:	pintgr.cu applu.incl npbparams.h
read_input.o:	read_input.cu applu.incl npbparams.h
rhs$(VEC).o:	rhs$(VEC).cu applu.incl npbparams.h
rhs_baseline.o: rhs_baseline.cu applu.incl npbparams.h
rhs_gmem.o: rhs_gmem.cu applu.incl npbparams.h
rhs_parallel.o: rhs_parallel.cu applu.incl npbparams.h
rhs_fullopt.o:	rhs_fullopt.cu applu.incl npbparams.h
setbv.o:	setbv.cu applu.incl npbparams.h
setiv.o:	setiv.cu applu.incl npbparams.h
setcoeff.o:	setcoeff.cu applu.incl npbparams.h
ssor$(VEC).o:	ssor$(VEC).cu applu.incl npbparams.h
ssor_baseline.o:	ssor_baseline.cu applu.incl npbparams.h
domain.o:	domain.cu applu.incl npbparams.h
verify.o:	verify.cu applu.incl npbparams.h

kernel_ssor_baseline.o: kernel_ssor_baseline.cu applu.incl npbparams.h
kernel_l2norm.o : kernel_l2norm.cu applu.incl npbparams.h
kernel_l2norm_baseline.o : kernel_l2norm_baseline.cu applu.incl npbparams.h
kernel_l2norm_gmem.o : kernel_l2norm_gmem.cu applu.incl npbparams.h
kernel_rhs_baseline.o : kernel_rhs_baseline.cu applu.incl npbparams.h
kernel_rhs_gmem.o : kernel_rhs_gmem.cu applu.incl npbparams.h
kernel_rhs_parallel.o : kernel_rhs_parallel.cu applu.incl npbparams.h
kernel_rhs_fullopt.o : kernel_rhs_fullopt.cu applu.incl npbparams.h
kernel_jacld_blts_baseline.o : kernel_jacld_blts_baseline.cu applu.incl npbparams.h kernel_constants.h
kernel_jacld_blts_gmem.o : kernel_jacld_blts_gmem.cu applu.incl npbparams.h kernel_constants.h
kernel_jacld_blts_sync.o : kernel_jacld_blts_sync.cu applu.incl npbparams.h kernel_constants.h
kernel_jacld_blts_fullopt.o : kernel_jacld_blts_fullopt.cu applu.incl npbparams.h
kernel_jacu_buts_baseline.o : kernel_jacu_buts_baseline.cu applu.incl npbparams.h kernel_constants.h
kernel_jacu_buts_gmem.o : kernel_jacu_buts_gmem.cu applu.incl npbparams.h kernel_constants.h
kernel_jacu_buts_sync.o : kernel_jacu_buts_sync.cu applu.incl npbparams.h kernel_constants.h
kernel_jacu_buts_fullopt.o : kernel_jacu_buts_fullopt.cu applu.incl npbparams.h

clean:
	- /bin/rm -f npbparams.h
	- /bin/rm -f *.o *~
	- if [ -d rii_files ]; then rm -r rii_files; fi
