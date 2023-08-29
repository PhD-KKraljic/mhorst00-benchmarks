SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=LU
VEC=

include ../config/make.def

include ../sys/make.common

OBJS = lu.o read_input.o \
       domain.o setcoeff.o setbv.o exact.o setiv.o \
       erhs.o ssor$(VEC).o rhs$(VEC).o l2norm.o \
       jacld_blts.o jacu_buts.o error.o  \
			 rhs_baseline.o rhs_gmem.o \
       rhs_parallel.o rhs_fullopt.o \
			 l2norm_baseline.o l2norm_gmem.o \
			 ssor_baseline.o \
			 jacld_blts_baseline.o jacld_blts_gmem.o \
			 jacld_blts_sync.o jacld_blts_fullopt.o \
			 jacu_buts_baseline.o jacu_buts_gmem.o \
			 jacu_buts_sync.o jacu_buts_fullopt.o \
       pintgr.o verify.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/wtime.o ${COMMON}/cl_util.o

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

lu.o:		lu.c applu.incl npbparams.h
erhs.o:		erhs.c applu.incl npbparams.h
error.o:	error.c applu.incl npbparams.h
exact.o:	exact.c applu.incl npbparams.h
jacld_blts.o:	jacld_blts.c applu.incl npbparams.h
jacu_buts.o:	jacu_buts.c applu.incl npbparams.h
l2norm.o:	l2norm.c applu.incl npbparams.h 
pintgr.o:	pintgr.c applu.incl npbparams.h
read_input.o:	read_input.c applu.incl npbparams.h
rhs$(VEC).o:	rhs$(VEC).c applu.incl npbparams.h
setbv.o:	setbv.c applu.incl npbparams.h
setiv.o:	setiv.c applu.incl npbparams.h
setcoeff.o:	setcoeff.c applu.incl npbparams.h
ssor$(VEC).o:	ssor$(VEC).c applu.incl npbparams.h
domain.o:	domain.c applu.incl npbparams.h
verify.o:	verify.c applu.incl npbparams.h
ssor_fullopt.o: ssor_baseline.c applu.incl npbparams.h
rhs_baseline.o: rhs_baseline.c applu.incl npbparams.h
rhs_gmem.o: rhs_gmem.c applu.incl npbparams.h
rhs_parallel.o: rhs_parallel.c applu.incl npbparams.h
rhs_fullopt.o: rhs_fullopt.c applu.incl npbparams.h
l2norm_baseline.o: l2norm_baseline.c applu.incl npbparams.h
l2norm_gmem.o: l2norm_gmem.c applu.incl npbparams.h 
jacld_blts_baseline.o: jacld_blts_baseline.c applu.incl npbparams.h 
jacld_blts_gmem.o: jacld_blts_gmem.c applu.incl npbparams.h 
jacld_blts_sync.o: jacld_blts_sync.c applu.incl npbparams.h 
jacld_blts_fullopt.o: jacld_blts_fullopt.c applu.incl npbparams.h 
jacu_buts_baseline.o: jacu_buts_baseline.c applu.incl npbparams.h 
jacu_buts_gmem.o: jacu_buts_gmem.c applu.incl npbparams.h 
jacu_buts_sync.o: jacu_buts_sync.c applu.incl npbparams.h
jacu_buts_fullopt.o: jacu_buts_fullopt.c applu.incl npbparams.h 

cl_util.o: cl_util.c

clean:
	- /bin/rm -f npbparams.h
	- /bin/rm -f *.o *~
	- if [ -d rii_files ]; then rm -r rii_files; fi
