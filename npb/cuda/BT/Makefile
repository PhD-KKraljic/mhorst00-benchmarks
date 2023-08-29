SHELL=/bin/sh
BENCHMARK=bt
BENCHMARKU=BT
VEC=

include ../config/make.def

include ../sys/make.common

OBJS = bt.o  initialize.o exact_solution.o exact_rhs.o \
       set_constants.o adi.o \
       rhs.o  rhs_parallel.o rhs_baseline.o \
       x_solve$(VEC).o x_solve_baseline.o x_solve_parallel.o x_solve_memlayout.o x_solve_fullopt$(VEC).o \
       y_solve$(VEC).o y_solve_baseline.o y_solve_parallel.o y_solve_memlayout.o y_solve_fullopt.o \
       z_solve$(VEC).o z_solve_baseline.o z_solve_parallel.o z_solve_memlayout.o z_solve_fullopt.o \
       add.o error.o verify.o \
       kernel_add.o \
       kernel_compute_rhs_baseline.o kernel_compute_rhs_parallel.o \
       kernel_solve_baseline.o kernel_solve_parallel.o kernel_solve_fullopt.o \
       kernel_x_solve_memlayout.o kernel_y_solve_memlayout.o kernel_z_solve_memlayout.o \
	   ${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o ${COMMON}/cuda_util.o



# npbparams.h is included by header.h
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

bt.o:             bt.cu  header.h npbparams.h
initialize.o:     initialize.cu  header.h npbparams.h
exact_solution.o: exact_solution.cu  header.h npbparams.h
exact_rhs.o:      exact_rhs.cu  header.h npbparams.h
set_constants.o:  set_constants.cu  header.h npbparams.h
adi.o:            adi.cu  header.h npbparams.h
rhs.o:            rhs.cu  header.h npbparams.h
rhs_baseline.o:   rhs_baseline.cu  header.h npbparams.h
rhs_prarllel.o:   rhs_parallel.cu  header.h npbparams.h

x_solve$(VEC).o:  x_solve$(VEC).cu  header.h npbparams.h
x_solve_baseline.o:  x_solve_baseline.cu  header.h npbparams.h
x_solve_parallel.o:  x_solve_parallel.cu  header.h npbparams.h
x_solve_memlayout.o:  x_solve_memlayout.cu  header.h npbparams.h
x_solve_fullopt.o:  x_solve_fullopt.cu  header.h npbparams.h
y_solve$(VEC).o:  y_solve$(VEC).cu  header.h npbparams.h
y_solve_baseline.o:  y_solve_baseline.cu  header.h npbparams.h
y_solve_parallel.o:  y_solve_parallel.cu  header.h npbparams.h
y_solve_memlayout.o:  y_solve_memlayout.cu  header.h npbparams.h
y_solve_fullopt.o:  y_solve_fullopt.cu  header.h npbparams.h
z_solve$(VEC).o:  z_solve$(VEC).cu  header.h npbparams.h
z_solve_baseline.o:  z_solve_baseline.cu  header.h npbparams.h
z_solve_parallel.o:  z_solve_parallel.cu  header.h npbparams.h
z_solve_memlayout.o:  z_solve_memlayout.cu  header.h npbparams.h
z_solve_fullopt.o:  z_solve_fullopt.cu  header.h npbparams.h
add.o:            add.cu  header.h npbparams.h
error.o:          error.cu  header.h npbparams.h
verify.o:         verify.cu  header.h npbparams.h
kernel_add.o: kernel_add.cu header.h npbparams.h
kernel_compute_rhs_parallel.o: kernel_compute_rhs_parallel.cu header.h npbparams.h
kernel_compute_rhs_baseline.o: kernel_compute_rhs_baseline.cu header.h npbparams.h
kernel_solve_baseline.o: kernel_solve_baseline.cu header.h npbparams.h
kernel_solve_parallel.o: kernel_solve_parallel.cu header.h npbparams.h
kernel_solve_fullopt.o: kernel_solve_fullopt.cu header.h npbparams.h
kernel_x_solve_memlayout.o: kernel_x_solve_memlayout.cu header.h npbparams.h
kernel_y_solve_memlayout.o: kernel_y_solve_memlayout.cu header.h npbparams.h
kernel_z_solve_memlayout.o: kernel_z_solve_memlayout.cu header.h npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
