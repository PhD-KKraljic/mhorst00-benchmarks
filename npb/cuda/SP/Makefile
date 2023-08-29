SHELL=/bin/sh
BENCHMARK=sp
BENCHMARKU=SP

include ../config/make.def

include ../sys/make.common

OBJS = sp.o initialize.o exact_solution.o exact_rhs.o \
       set_constants.o adi.o rhs.o      \
       x_solve.o ninvr.o y_solve.o pinvr.o    \
       z_solve.o tzetar.o add.o txinvr.o error.o verify.o \
	   kernel_add.o kernel_add_base.o kernel_add_opt.o \
	   kernel_util.o kernel_util_base.o kernel_util_opt.o \
	   kernel_x_solve.o kernel_x_solve_base.o kernel_x_solve_parallel.o \
	   kernel_x_solve_layout.o kernel_x_solve_opt.o \
	   kernel_y_solve.o kernel_y_solve_base.o kernel_y_solve_opt.o \
	   kernel_z_solve.o kernel_z_solve_base.o kernel_z_solve_opt.o \
	   kernel_compute_rhs.o kernel_compute_rhs_base.o kernel_compute_rhs_opt.o \
	   kernel_txinvr.o kernel_txinvr_base.o kernel_txinvr_opt.o \
	   ${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o ${COMMON}/cuda_util.o

# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.cu.o:
	${NVCCCOMPILE} $<

.c.o:
	${CCOMPILE} $<

sp.o:             sp.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
initialize.o:     initialize.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
exact_solution.o: exact_solution.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
exact_rhs.o:      exact_rhs.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
set_constants.o:  set_constants.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
adi.o:            adi.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
rhs.o:            rhs.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
x_solve.o:        x_solve.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
ninvr.o:          ninvr.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
y_solve.o:        y_solve.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
pinvr.o:          pinvr.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
z_solve.o:        z_solve.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
tzetar.o:         tzetar.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
add.o:            add.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
txinvr.o:         txinvr.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
error.o:          error.cu  header.h npbparams.h
	${NVCCCOMPILE} $<
verify.o:         verify.cu  header.h npbparams.h
	${NVCCCOMPILE} $<

kernel_add.o:    kernel_add.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_util.o:    kernel_util.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_x_solve.o:    kernel_x_solve.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_y_solve.o:    kernel_y_solve.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_z_solve.o:    kernel_z_solve.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_compute_rhs.o:    kernel_compute_rhs.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_txinvr.o:    kernel_txinvr.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<

kernel_add_base.o:    kernel_add_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_add_opt.o:    kernel_add_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_util_base.o:    kernel_util_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_util_opt.o:    kernel_util_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_x_solve_base.o:    kernel_x_solve_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_x_solve_parallel.o:    kernel_x_solve_parallel.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_x_solve_layout.o:    kernel_x_solve_layout.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_x_solve_opt.o:    kernel_x_solve_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_y_solve_base.o:    kernel_y_solve_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_y_solve_opt.o:    kernel_y_solve_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_z_solve_base.o:    kernel_z_solve_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_z_solve_opt.o:    kernel_z_solve_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_compute_rhs_base.o:    kernel_compute_rhs_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_compute_rhs_opt.o:    kernel_compute_rhs_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_txinvr_base.o:    kernel_txinvr_base.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<
kernel_txinvr_opt.o:    kernel_txinvr_opt.cu header.h kernel_header.h npbparams.h
	${NVCCCOMPILE} $<

#util.o:           util.cu  header.h npbparams.h
	#${NVCCCOMPILE} $<

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
