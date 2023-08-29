//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB BT code. This CUDA® C  //
//  version is a part of SNU-NPB 2019 developed by the Center for Manycore //
//  Programming at Seoul National University and derived from the serial   //
//  Fortran versions in "NPB3.3.1-SER" developed by NAS.                   //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on original NPB 3.3.1, including the technical report, the //
//  original specifications, source code, results and information on how   //
//  to submit new results, is available at:                                //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Information on SNU-NPB 2019, including the conference paper and source //
//  code, is available at:                                                 //
//                                                                         //
//           http://aces.snu.ac.kr                                         //
//                                                                         //
//  Send comments or suggestions for this CUDA® C version to               //
//  snunpb@aces.snu.ac.kr                                                  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 08826, Korea                                             //
//                                                                         //
//          E-mail: snunpb@aces.snu.ac.kr                                  //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Youngdong Do, Hyung Mo Kim, Pyeongseok Oh, Daeyoung Park,      //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//---------------------------------------------------------------------
//  program mg
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
}

#include "globals.h"
#include <time.h>
#ifdef LIKWID
# include <likwid.h>
#endif

//---------------------------------------------------------------------------
// CUDA part
//---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include "cuda_util.h"

static int device;
static cudaStream_t cmd_queue[2];

static size_t max_work_group_size;
static size_t max_mem_alloc_size;
static size_t global_mem_size;
static size_t shared_mem_size;

// memory objects
struct grid {
  double *d[2];

  double *h;

  int dim_x;
  int dim_y;
  int dim_z;

  int created;
  int reserved;
};

static struct grid m_r[LT_DEFAULT+1];
static struct grid m_u[LT_DEFAULT+1];
static struct grid m_v;

static int NUM_PARTITION[LT_DEFAULT+1];

static double *m_a;
static double *m_c;

#define MAX_PARTITION 32
static cudaEvent_t write_event[MAX_PARTITION][3];

static logical use_shared_mem = true;

static void parse_arg(int argc, char *argv[]);
static void setup_timers(char *t_names[]);
static void setup_cuda(int argc, char *argv[], double a[4], double c[4]);
static void release_cuda();

static void init_grid(struct grid *g, double *h, int k, int x, int y, int z);
static void create_grid(struct grid *g, int n);
static void release_grid(struct grid *g);
static void push_grid(struct grid *g, int p, int n, int d, bool sync);
static void pull_grid(struct grid *g, int p, int n, int d, bool sync);
static void pushv_grid(struct grid *g, int p, int n, int d, bool sync);
//---------------------------------------------------------------------------

static void setup(int *n1, int *n2, int *n3);
static void mg3P(int n1, int n2, int n3);

static void psinv(int n1, int n2, int n3, int k);
static void psinv_nb(int n1, int n2, int n3, int k);
static void psinv_db(int n1, int n2, int n3, int k);

static void resid(int n1, int n2, int n3, int k);
static void resid_nb(int n1, int n2, int n3, int k);
static void resid_db(int n1, int n2, int n3, int k);
static void resid_db_top(int n1, int n2, int n3, int k);

static void rprj3(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k);
static void rprj3_nb(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k);
static void rprj3_sbk(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k);
static void rprj3_sbkj(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k);

static void interp(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k);
static void interp_nb(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k);
static void interp_sbk(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k);
static void interp_sbkj(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k);

static void norm2u3(int n1, int n2, int n3, double *rnm2, double *rnmu,
                    int nx, int ny, int nz);

static void zero3_kernel(double **m_uk, int n1, int n2, int n3);
static void comm3_kernel(double **m_ru, int n1, int n2, int n3);

static void comm3(void *ou, int n1, int n2, int n3);
static void zran3(void *oz, int n1, int n2, int n3, int nx, int ny, int k);
static double power(double a, int n);
static void bubble(double ten[][2], int j1[][2], int j2[][2], int j3[][2],
                   int m, int ind);
static void zero3(void *oz, int n1, int n2, int n3);

static size_t get_lws(int gws);
static size_t get_wg(int gws);

//-------------------------------------------------------------------------c
// These arrays are in common because they are quite large
// and probably shouldn't be allocated on the stack. They
// are always passed as subroutine args.
//-------------------------------------------------------------------------c
/* common /noautom/ */
static double u[NR];
static double v[NR];
static double r[NR];

/* common /grid/ */
static int is1, is2, is3, ie1, ie2, ie3;

#define nextpow(x) pow(2, ceil(log((int)x)/log(2)))

int main(int argc, char *argv[])
{
  //-------------------------------------------------------------------------c
  // k is the current level. It is passed down through subroutine args
  // and is NOT global. it is the current iteration, int offsetion
  //-------------------------------------------------------------------------c
  int k, it;
  double t, tinit, mflops;

  double a[4], c[4];

  double rnm2, rnmu, epsilon;

  int n1, n2, n3, nit;
  double nn, verify_value, err;
  logical verified;

  int i;
  char *t_names[T_last];
  double tmax;

  parse_arg(argc, argv);

  setup_timers(t_names);

  mg_timer_start(T_init);

  //---------------------------------------------------------------------
  // Read in and broadcast input data
  //---------------------------------------------------------------------
  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-CUDA) - MG Benchmark\n\n");

  lt = LT_DEFAULT;
  nit = NIT_DEFAULT;
  nx[lt] = NX_DEFAULT;
  ny[lt] = NY_DEFAULT;
  nz[lt] = NZ_DEFAULT;

  if ( (nx[lt] != ny[lt]) || (nx[lt] != nz[lt]) ) {
    Class = 'U';
  } else if ( nx[lt] == 32 && nit == 4 ) {
    Class = 'S';
  } else if ( nx[lt] == 128 && nit == 4 ) {
    Class = 'W';
  } else if ( nx[lt] == 256 && nit == 4 ) {
    Class = 'A';
  } else if ( nx[lt] == 256 && nit == 20 ) {
    Class = 'B';
  } else if ( nx[lt] == 512 && nit == 20 ) {
    Class = 'C';
  } else if ( nx[lt] == 1024 && nit == 50 ) {
    Class = 'D';
  } else if ( nx[lt] == 2048 && nit == 50 ) {
    Class = 'E';
  } else {
    Class = 'U';
  }

  a[0] = -8.0/3.0;
  a[1] =  0.0;
  a[2] =  1.0/6.0;
  a[3] =  1.0/12.0;

  if (Class == 'A' || Class == 'S' || Class =='W') {
    //---------------------------------------------------------------------
    // Coefficients for the S(a) smoother
    //---------------------------------------------------------------------
    c[0] =  -3.0/8.0;
    c[1] =  +1.0/32.0;
    c[2] =  -1.0/64.0;
    c[3] =   0.0;
  } else {
    //---------------------------------------------------------------------
    // Coefficients for the S(b) smoother
    //---------------------------------------------------------------------
    c[0] =  -3.0/17.0;
    c[1] =  +1.0/33.0;
    c[2] =  -1.0/61.0;
    c[3] =   0.0;
  }
  lb = 1;

  printf(" Size: %4dx%4dx%4d  (class %c)\n", nx[lt], ny[lt], nz[lt], Class);
  printf(" Iterations: %3d\n", nit);
  printf("\n");

  setup(&n1, &n2, &n3);
  zero3(u, n1, n2, n3);
  zran3(v, n1, n2, n3, nx[lt], ny[lt], lt);

  setup_cuda(argc, argv, a, c);

  for (k = lb; k <= lt; k++) {
    init_grid(&m_u[k], u, k, m1[k], m2[k], m3[k]);
    init_grid(&m_r[k], r, k, m1[k], m2[k], m3[k]);
  }
  if (NUM_PARTITION[lt] == 1) {
    init_grid(&m_v, v, lt, m1[lt], m2[lt], m3[lt]);
  }

  CUCHK(cudaMemcpyAsync(m_a, a, 4 * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[0]));

  CUCHK(cudaMemcpyAsync(m_c, c, 4 * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[0]));

  for (k = lb; k <= lt; k++) {
    if (NUM_PARTITION[k] == 1) {
      create_grid(&m_r[k], 1);
      push_grid(&m_r[k], 0, 1, 0, false);

      create_grid(&m_u[k], 1);
      push_grid(&m_u[k], 0, 1, 1, false);
    }
  }
  if (NUM_PARTITION[lt] == 1) {
    create_grid(&m_v, 1);
    push_grid(&m_v, 0, 1, 2, false);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[0]));

  mg_timer_stop(T_init);
  tinit = timer_read(T_init);

  printf(" Initialization time: %15.4f seconds\n\n", tinit);

#ifdef LIKWID_NVMON
    LIKWID_NVMARKER_INIT;
    LIKWID_NVMARKER_REGISTER("nv_npbmg");
    LIKWID_NVMARKER_START("nv_npbmg");
#endif
  mg_timer_start(T_bench);
  cuda_ProfilerStart();

  resid(n1, n2, n3, lt);

  norm2u3(n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

  for (it = 1; it <= nit; it++) {
    printf("  iter %3d\n", it);

    mg3P(n1, n2, n3);
    resid(n1, n2, n3, lt);
  }

  norm2u3(n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

  cuda_ProfilerStop();
  mg_timer_stop(T_bench);

#ifdef LIKWID_NVMON
    LIKWID_NVMARKER_STOP("nv_npbmg");
    LIKWID_NVMARKER_CLOSE;
#endif
  t = timer_read(T_bench);

  verified = false;
  verify_value = 0.0;

  printf("\n Benchmark completed\n");

  epsilon = 1.0e-8;
  if (Class != 'U') {
    if (Class == 'S') {
      verify_value = 0.5307707005734e-04;
    } else if (Class == 'W') {
      verify_value = 0.6467329375339e-05;
    } else if (Class == 'A') {
      verify_value = 0.2433365309069e-05;
    } else if (Class == 'B') {
      verify_value = 0.1800564401355e-05;
    } else if (Class == 'C') {
      verify_value = 0.5706732285740e-06;
    } else if (Class == 'D') {
      verify_value = 0.1583275060440e-09;
    } else if (Class == 'E') {
      verify_value = 0.8157592357404e-10;
    }

    err = fabs(rnm2 - verify_value);
    if (err < epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" L2 Norm is %20.13E\n", rnm2);
      printf(" Error is   %20.13E\n", err);
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" L2 Norm is             %20.13E\n", rnm2);
      printf(" The correct L2 Norm is %20.13E\n", verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
    printf(" L2 Norm is %20.13E\n", rnm2);
  }

  nn = 1.0 * nx[lt] * ny[lt] * nz[lt];

  if (t != 0.0) {
    mflops = 58.0 * nit * nn * 1.0e-6 / t;
  } else {
    mflops = 0.0;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  c_print_results("MG", Class, nx[lt], ny[lt], nz[lt],
                  nit, t,
                  mflops, "          floating point",
                  verified, NPBVERSION, COMPILETIME,
                  CS1, CS2, CS3, CS4, CS5, CS6, CS7,
                  (const char *)prop.name);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    tmax = timer_read(T_bench);
    if (tmax == 0.0) tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = T_bench; i < T_last; i++) {
      t = timer_read(i);
      printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100./tmax);
    }

    cuda_ProfilerPrintResult();
  }

  release_cuda();

  fflush(stdout);

  return 0;
}


static void setup(int *n1, int *n2, int *n3)
{
  int k, j;

  int mi[MAXLEVEL+1][3];
  int ng[MAXLEVEL+1][3];

  ng[lt][0] = nx[lt];
  ng[lt][1] = ny[lt];
  ng[lt][2] = nz[lt];

  for (k = lt-1; k >= 1; k--) {
    nx[k] = ng[k][0] = ng[k+1][0] / 2;
    ny[k] = ng[k][1] = ng[k+1][1] / 2;
    nz[k] = ng[k][2] = ng[k+1][2] / 2;
  }

  for (k = lt; k >= 1; k--) {
    m1[k] = mi[k][0] = 2 + ng[k][0];
    m2[k] = mi[k][1] = 2 + ng[k][1];
    m3[k] = mi[k][2] = 2 + ng[k][2];
  }

  k = lt;

  is1 = 2 + ng[k][0] - ng[lt][0];
  ie1 = 1 + ng[k][0];
  *n1 = 3 + ie1 - is1;

  is2 = 2 + ng[k][1] - ng[lt][1];
  ie2 = 1 + ng[k][1];
  *n2 = 3 + ie2 - is2;

  is3 = 2 + ng[k][2] - ng[lt][2];
  ie3 = 1 + ng[k][2];
  *n3 = 3 + ie3 - is3;

  ir[lt] = 0;

  for (j = lt-1; j >= 1; j--) {
    ir[j] = ir[j+1] + ONE * m1[j+1] * m2[j+1] * m3[j+1];
  }
}


//---------------------------------------------------------------------
// multigrid V-cycle routine
//---------------------------------------------------------------------
static void mg3P(int n1, int n2, int n3)
{
  if (timeron) mg_timer_start(T_mg3P);

  int j, k;

  //---------------------------------------------------------------------
  // down cycle.
  // restrict the residual from the find grid to the coarse
  //---------------------------------------------------------------------
  for (k = lt; k >= lb+1; k--) {
    j = k - 1;
    rprj3(m1[k], m2[k], m3[k], m1[j], m2[j], m3[j], k);
  }

  k = lb;
  //---------------------------------------------------------------------
  // compute an approximate solution on the coarsest grid
  //---------------------------------------------------------------------
  zero3_kernel(&m_u[k].d[0], m1[k], m2[k], m3[k]);

  psinv(m1[k], m2[k], m3[k], k);

  for (k = lb+1; k <= lt-1; k++) {
    j = k - 1;

    //---------------------------------------------------------------------
    // prolongate from level k-1  to k
    //---------------------------------------------------------------------
    if (NUM_PARTITION[k] == 1) {
      zero3_kernel(&m_u[k].d[0], m1[k], m2[k], m3[k]);
    }
    else {
      zero3(u + ir[k], m1[k], m2[k], m3[k]);
    }

    interp(m1[j], m2[j], m3[j], m1[k], m2[k], m3[k], k);

    //---------------------------------------------------------------------
    // compute residual for level k
    //---------------------------------------------------------------------
    resid(m1[k], m2[k], m3[k], k);

    //---------------------------------------------------------------------
    // apply smoother
    //---------------------------------------------------------------------
    psinv(m1[k], m2[k], m3[k], k);
  }

  k = lt;
  j = k - 1;
  interp(m1[j], m2[j], m3[j], n1, n2, n3, k);
  resid(n1, n2, n3, k);
  psinv(n1, n2, n3, k);

  if (timeron) mg_timer_stop(T_mg3P);
}


//---------------------------------------------------------------------
// psinv applies an approximate inverse as smoother:  u = u + Cr
//
// This  implementation costs  15A + 4M per result, where
// A and M denote the costs of Addition and Multiplication.
// Presuming coefficient c(3) is zero (the NPB assumes this,
// but it is thus not a general case), 2A + 1M may be eliminated,
// resulting in 13A + 3M.
// Note that this vectorizes, and is also fine for cache
// based machines.
//---------------------------------------------------------------------
static void psinv(int n1, int n2, int n3, int k)
{
  if (timeron) mg_timer_start(T_psinv);

  int nk = NUM_PARTITION[k];

  if (nk == 1) {
    psinv_nb(n1, n2, n3, k);
  }
  else {
    psinv_db(n1, n2, n3, k);
  }

  if (timeron) mg_timer_stop(T_psinv);

  //---------------------------------------------------------------------
  // exchange boundary points
  //---------------------------------------------------------------------
  if (m_u[k].reserved) {
    comm3_kernel(&m_u[k].d[0], n1, n2, n3);
  }
  else {
    comm3(u + ir[k], n1, n2, n3);
  }
}
/* psinv no buffering */
static void psinv_nb(int n1, int n2, int n3, int k)
{
  size_t psinv_lws[3], psinv_gws[3];

  psinv_lws[0] = get_lws(n1-2);
  psinv_lws[1] = 1;
  psinv_lws[2] = 1;

  psinv_gws[0] = get_wg(n1-2) * psinv_lws[0];
  psinv_gws[1] = n3-2;
  psinv_gws[2] = n3-2;

  dim3 blockSize(psinv_gws[0] / psinv_lws[0],
                 psinv_gws[1] / psinv_lws[1],
                 psinv_gws[2] / psinv_lws[2]);
  dim3 threadSize(psinv_lws[0], psinv_lws[1], psinv_lws[2]);

  int bound = (n1-2) / get_wg(n1-2) + 2;

  if (timeron) mg_timer_start(T_psinv_kern);
  if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("kernel_psinv_opt", cmd_queue[0] );
    kernel_psinv_opt<<< blockSize,
                        threadSize,
                        sizeof(double) * bound * 2,
                        cmd_queue[0] >>>
        (m_r[k].d[0], m_u[k].d[0], m_c, n1, n2, bound);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_psinv_opt", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("kernel_psinv_base", cmd_queue[0] );
    kernel_psinv_base<<< blockSize,
                         threadSize,
                         0,
                         cmd_queue[0] >>>
        (m_r[k].d[0], m_u[k].d[0], m_c, n1, n2, bound);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_psinv_base", cmd_queue[0] );
  }
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    CUCHK(cudaGetLastError());
    mg_timer_stop(T_psinv_kern);
  }
}

/* psinv double buffering optimized */
static void psinv_db(int n1, int n2, int n3, int k)
{
  size_t psinv_lws[3], psinv_gws[3];

  int p;
  int nk = NUM_PARTITION[k];
  int pad = n1 * n2;

  create_grid(&m_r[k], nk);
  if (timeron) mg_timer_start(T_psinv_comm);
  push_grid(&m_r[k], 0, nk, 0, false);
  if (timeron) mg_timer_stop(T_psinv_comm);

  if (timeron) mg_timer_start(T_psinv_comm);
  create_grid(&m_u[k], nk);
  push_grid(&m_u[k], 0, nk, 1, false);
  if (timeron) mg_timer_stop(T_psinv_comm);

  for (p = 0; p < nk; p++) {
    psinv_lws[0] = get_lws(n1-2);
    psinv_lws[1] = 1;
    psinv_lws[2] = 1;

    psinv_gws[0] = get_wg(n1-2) * psinv_lws[0];
    psinv_gws[1] = n2-2;
    psinv_gws[2] = (n3-2) / nk;

    dim3 blockSize(psinv_gws[0] / psinv_lws[0],
                   psinv_gws[1] / psinv_lws[1],
                   psinv_gws[2] / psinv_lws[2]);
    dim3 threadSize(psinv_lws[0], psinv_lws[1], psinv_lws[2]);

    int bound = (n1-2) / get_wg(n1-2) + 2;

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][0], 0));
    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][1], 0));

    if (timeron) mg_timer_start(T_psinv_kern);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("kernel_psinv_opt", cmd_queue[1] );
      kernel_psinv_opt<<< blockSize,
                          threadSize,
                          sizeof(double) * bound * 2,
                          cmd_queue[1] >>>
            (m_r[k].d[p%2], m_u[k].d[p%2], m_c, n1, n2, bound);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_psinv_opt", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("kernel_psinv_base", cmd_queue[1] );
      kernel_psinv_base<<< blockSize,
                           threadSize,
                           0,
                           cmd_queue[1] >>>
            (m_r[k].d[p%2], m_u[k].d[p%2], m_c, n1, n2, bound);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_psinv_base", cmd_queue[1] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_psinv_kern);
    }

    if (p+1 < nk) {
      if (timeron) mg_timer_start(T_psinv_comm);
      push_grid(&m_r[k], p+1, nk, 0, true);
      push_grid(&m_u[k], p+1, nk, 1, true);
      if (timeron) mg_timer_stop(T_psinv_comm);
    }

    if (timeron) mg_timer_start(T_psinv_comm);
    pull_grid(&m_u[k], p, nk, pad, false);
    if (timeron) mg_timer_stop(T_psinv_comm);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));

  release_grid(&m_u[k]);
  release_grid(&m_r[k]);
}


//---------------------------------------------------------------------
// resid computes the residual:  r = v - Au
//
// This  implementation costs  15A + 4M per result, where
// A and M denote the costs of Addition (or Subtraction) and
// Multiplication, respectively.
// Presuming coefficient a(1) is zero (the NPB assumes this,
// but it is thus not a general case), 3A + 1M may be eliminated,
// resulting in 12A + 3M.
// Note that this vectorizes, and is also fine for cache
// based machines.
//---------------------------------------------------------------------
static void resid(int n1, int n2, int n3, int k)
{
  if (timeron) mg_timer_start(T_resid);

  int nk = NUM_PARTITION[k];

  if (nk == 1) {
    resid_nb(n1, n2, n3, k);
  }
  else if (k < lt) {
    resid_db(n1, n2, n3, k);
  }
  else {
    resid_db_top(n1, n2, n3, k);
  }

  if (timeron) mg_timer_stop(T_resid);

  //---------------------------------------------------------------------
  // exchange boundary data
  //---------------------------------------------------------------------
  if (m_r[k].reserved) {
    comm3_kernel(&m_r[k].d[0], n1, n2, n3);
  }
  else {
    comm3(r + ir[k], n1, n2, n3);
  }
}

/* resid no buffering */
static void resid_nb(int n1, int n2, int n3, int k)
{
  size_t resid_lws[3], resid_gws[3];

  resid_lws[0] = get_lws(n1-2);
  resid_lws[1] = 1;
  resid_lws[2] = 1;

  resid_gws[0] = get_wg(n1-2) * resid_lws[0];
  resid_gws[1] = n2-2;
  resid_gws[2] = n3-2;

  dim3 blockSize(resid_gws[0] / resid_lws[0],
                 resid_gws[1] / resid_lws[1],
                 resid_gws[2] / resid_lws[2]);
  dim3 threadSize(resid_lws[0], resid_lws[1], resid_lws[2]);

  int bound = (n1-2) / get_wg(n1-2) + 2;
  double *m_in = (k < lt) ? m_r[k].d[0] : m_v.d[0];

  if (timeron) mg_timer_start(T_resid_kern);
  if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("kernel_resid_opt", cmd_queue[0] );
    kernel_resid_opt<<< blockSize,
                        threadSize,
                        sizeof(double) * bound * 2,
                        cmd_queue[0] >>>
        (m_r[k].d[0], m_u[k].d[0], m_in, m_a, n1, n2, bound);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_resid_opt", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("kernel_resid_base", cmd_queue[0] );
    kernel_resid_base<<< blockSize,
                         threadSize,
                         0,
                         cmd_queue[0] >>>
        (m_r[k].d[0], m_u[k].d[0], m_in, m_a, n1, n2, bound);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_resid_base", cmd_queue[0] );
  }
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    CUCHK(cudaGetLastError());
    mg_timer_stop(T_resid_kern);
  }
}

/* resid double buffering optimized */
static void resid_db(int n1, int n2, int n3, int k)
{
  size_t resid_lws[3], resid_gws[3];

  int p;
  int nk = NUM_PARTITION[k];
  int pad = n1 * n2;

  create_grid(&m_r[k], nk);
  if (timeron) mg_timer_start(T_resid_comm);
  push_grid(&m_r[k], 0, nk, 0, false);
  if (timeron) mg_timer_stop(T_resid_comm);

  create_grid(&m_u[k], nk);
  if (timeron) mg_timer_start(T_resid_comm);
  push_grid(&m_u[k], 0, nk, 1, false);
  if (timeron) mg_timer_stop(T_resid_comm);

  for (p = 0; p < nk; p++) {
    resid_lws[0] = get_lws(n1-2);
    resid_lws[1] = 1;
    resid_lws[2] = 1;

    resid_gws[0] = get_wg(n1-2) * resid_lws[0];
    resid_gws[1] = n2-2;
    resid_gws[2] = (n3-2) / nk;

    dim3 blockSize(resid_gws[0] / resid_lws[0],
                   resid_gws[1] / resid_lws[1],
                   resid_gws[2] / resid_lws[2]);
    dim3 threadSize(resid_lws[0], resid_lws[1], resid_lws[2]);

    int bound = (n1-2) / get_wg(n1-2) + 2;

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][0], 0));
    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][1], 0));

    if (timeron) mg_timer_start(T_resid_kern);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("kernel_resid_opt", cmd_queue[1] );
      kernel_resid_opt<<< blockSize,
                          threadSize,
                          sizeof(double) * bound * 2,
                          cmd_queue[1] >>>
            (m_r[k].d[p%2], m_u[k].d[p%2], m_r[k].d[p%2], m_a, n1, n2, bound);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_resid_opt", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("kernel_resid_base", cmd_queue[1] );
      kernel_resid_base<<< blockSize,
                           threadSize,
                           0,
                           cmd_queue[1] >>>
            (m_r[k].d[p%2], m_u[k].d[p%2], m_r[k].d[p%2], m_a, n1, n2, bound);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_resid_base", cmd_queue[1] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_resid_kern);
    }

    if (p+1 < nk) {
      if (timeron) mg_timer_start(T_resid_comm);
      push_grid(&m_r[k], p+1, nk, 0, true);
      push_grid(&m_u[k], p+1, nk, 1, true);
      if (timeron) mg_timer_stop(T_resid_comm);
    }

    if (timeron) mg_timer_start(T_resid_comm);
    pull_grid(&m_r[k], p, nk, pad, false);
    if (timeron) mg_timer_stop(T_resid_comm);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  CUCHK(cudaGetLastError());

  release_grid(&m_r[k]);
  release_grid(&m_u[k]);
}

/* resid double buffering top grid optimized */
static void resid_db_top(int n1, int n2, int n3, int k)
{
  size_t resid_lws[3], resid_gws[3];

  int p;
  int nk = NUM_PARTITION[k];
  int pad = n1 * n2;

  create_grid(&m_r[k], nk);
  if (timeron) mg_timer_start(T_resid_comm);
  pushv_grid(&m_r[k], 0, nk, 0, false);
  if (timeron) mg_timer_stop(T_resid_comm);

  create_grid(&m_u[k], nk);
  if (timeron) mg_timer_start(T_resid_comm);
  push_grid(&m_u[k], 0, nk, 1, false);
  if (timeron) mg_timer_stop(T_resid_comm);

  for (p = 0; p < nk; p++) {
    resid_lws[0] = get_lws(n1-2);
    resid_lws[1] = 1;
    resid_lws[2] = 1;

    resid_gws[0] = get_wg(n1-2) * resid_lws[0];
    resid_gws[1] = n2-2;
    resid_gws[2] = (n3-2) / nk;

    dim3 blockSize(resid_gws[0] / resid_lws[0],
                   resid_gws[1] / resid_lws[1],
                   resid_gws[2] / resid_lws[2]);
    dim3 threadSize(resid_lws[0], resid_lws[1], resid_lws[2]);

    int bound = (n1-2) / get_wg(n1-2) + 2;

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][0], 0));
    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][1], 0));

    if (timeron) mg_timer_start(T_resid_kern);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("kernel_resid_opt", cmd_queue[1] );
      kernel_resid_opt<<< blockSize,
                          threadSize,
                          sizeof(double) * bound * 2,
                          cmd_queue[1] >>>
            (m_r[k].d[p%2], m_u[k].d[p%2], m_r[k].d[p%2], m_a, n1, n2, bound);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_resid_opt", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("kernel_resid_base", cmd_queue[1] );
      kernel_resid_base<<< blockSize,
                           threadSize,
                           0,
                           cmd_queue[1] >>>
            (m_r[k].d[p%2], m_u[k].d[p%2], m_r[k].d[p%2], m_a, n1, n2, bound);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_resid_base", cmd_queue[1] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_resid_kern);
    }

    if (p+1 < nk) {
      if (timeron) mg_timer_start(T_resid_comm);
      pushv_grid(&m_r[k], p+1, nk, 0, true);
      push_grid(&m_u[k], p+1, nk, 1, true);
      if (timeron) mg_timer_stop(T_resid_comm);
    }

    if (timeron) mg_timer_start(T_resid_comm);
    pull_grid(&m_r[k], p, nk, pad, false);
    if (timeron) mg_timer_stop(T_resid_comm);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  CUCHK(cudaGetLastError());

  release_grid(&m_r[k]);
  release_grid(&m_u[k]);
}


//---------------------------------------------------------------------
// rprj3 projects onto the next coarser grid,
// using a trilinear Finite Element projection:  s = r' = P r
//
// This  implementation costs  20A + 4M per result, where
// A and M denote the costs of Addition and Multiplication.
// Note that this vectorizes, and is also fine for cache
// based machines.
//---------------------------------------------------------------------
static void rprj3(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k)
{
  if (timeron) mg_timer_start(T_rprj3);

  int j = k-1;
  int nk = NUM_PARTITION[k];
  int nj = NUM_PARTITION[j];

  if (nk == 1 && nj == 1) {
    rprj3_nb(m1k, m2k, m3k, m1j, m2j, m3j, k);
  }
  else if (nj == 1) {
    rprj3_sbk(m1k, m2k, m3k, m1j, m2j, m3j, k);
  }
  else {
    rprj3_sbkj(m1k, m2k, m3k, m1j, m2j, m3j, k);
  }

  if (timeron) mg_timer_stop(T_rprj3);

  if (m_r[j].reserved) {
    comm3_kernel(&m_r[j].d[0], m1j, m2j, m3j);
  }
  else {
    comm3(r + ir[j], m1j, m2j, m3j);
  }
}

/* rprj3 no buffering */
static void rprj3_nb(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k)
{
  size_t rprj3_lws[2], rprj3_gws[2];

  int j = k-1;
  size_t ofs = 0;

  rprj3_lws[0] = (m1j-2 > max_work_group_size) ? max_work_group_size : m1j-2;
  rprj3_lws[1] = 1;

  rprj3_gws[0] = (m1j-2) * (m2j-2);
  rprj3_gws[1] = (m3j-2);

  dim3 blockSize(rprj3_gws[0] / rprj3_lws[0], rprj3_gws[1] / rprj3_lws[1]);
  dim3 threadSize(rprj3_lws[0], rprj3_lws[1]);

  CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[0][0], 0));

  if (timeron) mg_timer_start(T_rprj3_kern);
  cuda_ProfilerStartEventRecord("kernel_rprj3", cmd_queue[1] );
  kernel_rprj3<<< blockSize, threadSize, 0, cmd_queue[1] >>>
              (m_r[k].d[0], m_r[j].d[0], m1k, m2k, m1j, m2j, ofs);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("kernel_rprj3", cmd_queue[1] );
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
    CUCHK(cudaGetLastError());
    mg_timer_stop(T_rprj3_kern);
  }
}

/* rprj3 single buffering with grid k */
static void rprj3_sbk(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k)
{
  size_t rprj3_lws[2], rprj3_gws[2];

  int p, j = k-1;
  int nk = NUM_PARTITION[k];
  size_t ofsj = (size_t) m1j * m2j * (m3j-2) / nk;

  create_grid(&m_r[k], nk);

  for (p = 0; p < nk; p++) {
    size_t ofs = ofsj * p;

    if (timeron) mg_timer_start(T_rprj3_comm);
    push_grid(&m_r[k], p, nk, 0, false);
    if (timeron) mg_timer_stop(T_rprj3_comm);

    rprj3_lws[0] = (m1j-2 > max_work_group_size) ? max_work_group_size : m1j-2;
    rprj3_lws[1] = 1;

    rprj3_gws[0] = (m1j-2) * (m2j-2);
    rprj3_gws[1] = (m3j-2) / nk;

    dim3 blockSize(rprj3_gws[0] / rprj3_lws[0], rprj3_gws[1] / rprj3_lws[1]);
    dim3 threadSize(rprj3_lws[0], rprj3_lws[1]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][0], 0));

    if (timeron) mg_timer_start(T_rprj3_kern);
    cuda_ProfilerStartEventRecord("kernel_rprj3", cmd_queue[1] );
    kernel_rprj3<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                (m_r[k].d[p%2], m_r[j].d[0], m1k, m2k, m1j, m2j, ofs);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_rprj3", cmd_queue[1] );
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_rprj3_kern);
    }
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  CUCHK(cudaGetLastError());

  release_grid(&m_r[k]);
}

/* rprj3 single buffering with grid k,j */
static void rprj3_sbkj(int m1k, int m2k, int m3k, int m1j, int m2j, int m3j, int k)
{
  size_t rprj3_lws[2], rprj3_gws[2];

  int p, j = k-1;
  int nk = NUM_PARTITION[k];
  size_t ofs = 0;
  size_t padj = m1j * m2j;

  create_grid(&m_r[k], nk);
  create_grid(&m_r[j], nk);

  for (p = 0; p < nk; p++) {
    if (timeron) mg_timer_start(T_rprj3_comm);
    push_grid(&m_r[k], p, nk, 0, false);
    push_grid(&m_r[j], p, nk, 0, false);
    if (timeron) mg_timer_stop(T_rprj3_comm);

    rprj3_lws[0] = (m1j-2 > max_work_group_size) ? max_work_group_size : m1j-2;
    rprj3_lws[1] = 1;

    rprj3_gws[0] = (m1j-2) * (m2j-2);
    rprj3_gws[1] = (m3j-2) / nk;

    dim3 blockSize(rprj3_gws[0] / rprj3_lws[0], rprj3_gws[1] / rprj3_lws[1]);
    dim3 threadSize(rprj3_lws[0], rprj3_lws[1]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][0], 0));

    if (timeron) mg_timer_start(T_rprj3_kern);
    cuda_ProfilerStartEventRecord("kernel_rprj3", cmd_queue[1] );
    kernel_rprj3<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                (m_r[k].d[p%2], m_r[j].d[p%2], m1k, m2k, m1j, m2j, ofs);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_rprj3", cmd_queue[1] );
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_rprj3_kern);
    }

    if (timeron) mg_timer_start(T_rprj3_comm);
    pull_grid(&m_r[j], p, nk, padj, true);
    if (timeron) mg_timer_stop(T_rprj3_comm);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  CUCHK(cudaGetLastError());

  release_grid(&m_r[k]);
  release_grid(&m_r[j]);
}


//---------------------------------------------------------------------
// interp adds the trilinear interpolation of the correction
// from the coarser grid to the current approximation:  u = u + Qu'
//
// Observe that this  implementation costs  16A + 4M, where
// A and M denote the costs of Addition and Multiplication.
// Note that this vectorizes, and is also fine for cache
// based machines.  Vector machines may get slightly better
// performance however, with 8 separate "do i1" loops, rather than 4.
//---------------------------------------------------------------------
static void interp(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k)
{
  if (timeron) mg_timer_start(T_interp);

  int j = k-1;
  int nk = NUM_PARTITION[k];
  int nj = NUM_PARTITION[j];

  if (nk == 1 && nj == 1) {
    interp_nb(mm1, mm2, mm3, n1, n2, n3, k);
  }
  else if (nj == 1) {
    interp_sbk(mm1, mm2, mm3, n1, n2, n3, k);
  }
  else {
    interp_sbkj(mm1, mm2, mm3, n1, n2, n3, k);
  }

  if (timeron) mg_timer_stop(T_interp);
}

/* interp no buffering */
static void interp_nb(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k)
{
  size_t interp_lws[3], interp_gws[3];

  int j = k-1;
  size_t ofs = 0;

  interp_lws[0] = (mm1-1 > max_work_group_size) ? max_work_group_size : mm1-1;
  interp_lws[1] = 1;
  interp_lws[2] = 1;

  interp_gws[0] = RoundWorkSize((size_t) (mm1-1), interp_lws[0]);
  interp_gws[1] = mm2-1;
  interp_gws[2] = mm3-1;

  dim3 blockSize(interp_gws[0] / interp_lws[0],
                 interp_gws[1] / interp_lws[1],
                 interp_gws[2] / interp_lws[2]);
  dim3 threadSize(interp_lws[0], interp_lws[1], interp_lws[2]);

  CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[0][1], 0));

  if (timeron) mg_timer_start(T_interp_kern);
  cuda_ProfilerStartEventRecord("kernel_interp", cmd_queue[1] );
  kernel_interp<<< blockSize, threadSize, 0, cmd_queue[1] >>>
               (m_u[j].d[0], m_u[k].d[0], mm1, mm2, n1, n2, ofs);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("kernel_interp", cmd_queue[1] );
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
    CUCHK(cudaGetLastError());
    mg_timer_stop(T_interp_kern);
  }
}

/* interp single buffering with grid k */
static void interp_sbk(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k)
{
  size_t interp_lws[3], interp_gws[3];

  int p, j = k-1;
  int nk = NUM_PARTITION[k];
  size_t ofsj = mm1 * mm2 * (mm3-2) / nk;

  create_grid(&m_u[k], nk);

  for (p = 0; p < nk; p++) {
    size_t ofs = ofsj * p;

    if (timeron) mg_timer_start(T_interp_comm);
    push_grid(&m_u[k], p, nk, 1, false);
    if (timeron) mg_timer_stop(T_interp_comm);

    interp_lws[0] = (mm1-1 > max_work_group_size) ? max_work_group_size : mm1-1;
    interp_lws[1] = 1;
    interp_lws[2] = 1;

    interp_gws[0] = RoundWorkSize((size_t) (mm1-1), interp_lws[0]);
    interp_gws[1] = mm2-1;
    interp_gws[2] = (mm3-2) / nk;

    if (p == nk-1) interp_gws[2] += 1;

    dim3 blockSize(interp_gws[0] / interp_lws[0],
                   interp_gws[1] / interp_lws[1],
                   interp_gws[2] / interp_lws[2]);
    dim3 threadSize(interp_lws[0], interp_lws[1], interp_lws[2]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][1], 0));

    if (timeron) mg_timer_start(T_interp_kern);
    cuda_ProfilerStartEventRecord("kernel_interp", cmd_queue[1] );
    kernel_interp<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                 (m_u[j].d[0], m_u[k].d[p%2], mm1, mm2, n1, n2, ofs);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_interp", cmd_queue[1] );
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_interp_kern);
    }

    if (timeron) mg_timer_start(T_interp_comm);
    pull_grid(&m_u[k], p, nk, 0, true);
    if (timeron) mg_timer_stop(T_interp_comm);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  CUCHK(cudaGetLastError());

  release_grid(&m_u[k]);
}

/* interp single buffering with grid k,j */
static void interp_sbkj(int mm1, int mm2, int mm3, int n1, int n2, int n3, int k)
{
  size_t interp_lws[3], interp_gws[3];

  int p, j = k-1;
  int nk = NUM_PARTITION[k];
  size_t ofs = 0;

  create_grid(&m_u[k], nk);
  create_grid(&m_u[j], nk);

  for (p = 0; p < nk; p++) {
    if (timeron) mg_timer_start(T_interp_comm);
    push_grid(&m_u[k], p, nk, 1, false);
    push_grid(&m_u[j], p, nk, 1, false);
    if (timeron) mg_timer_stop(T_interp_comm);

    interp_lws[0] = (mm1-1 > max_work_group_size) ? max_work_group_size : mm1-1;
    interp_lws[1] = 1;
    interp_lws[2] = 1;

    interp_gws[0] = RoundWorkSize((size_t) (mm1-1), interp_lws[0]);
    interp_gws[1] = mm2-1;
    interp_gws[2] = (mm3-2) / nk;

    if (p == nk-1) interp_gws[2] += 1;

    dim3 blockSize(interp_gws[0] / interp_lws[0],
                   interp_gws[1] / interp_lws[1],
                   interp_gws[2] / interp_lws[2]);
    dim3 threadSize(interp_lws[0], interp_lws[1], interp_lws[2]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[p][1], 0));

    if (timeron) mg_timer_start(T_interp_kern);
    cuda_ProfilerStartEventRecord("kernel_interp", cmd_queue[1] );
    kernel_interp<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                 (m_u[j].d[p%2], m_u[k].d[p%2], mm1, mm2, n1, n2, ofs);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("kernel_interp", cmd_queue[1] );
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_interp_kern);
    }

    if (timeron) mg_timer_start(T_interp_comm);
    pull_grid(&m_u[k], p, nk, 0, true);
    if (timeron) mg_timer_stop(T_interp_comm);
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  CUCHK(cudaGetLastError());

  release_grid(&m_u[k]);
  release_grid(&m_u[j]);
}


//---------------------------------------------------------------------
// norm2u3 evaluates approximations to the L2 norm and the
// uniform (or L-infinity or Chebyshev) norm, under the
// assumption that the boundaries are periodic or zero.  Add the
// boundaries in with half weight (quarter weight on the edges
// and eighth weight at the corners) for inhomogeneous boundaries.
//---------------------------------------------------------------------
static void norm2u3(int n1, int n2, int n3,
                    double *rnm2, double *rnmu,
                    int nx, int ny, int nz)
{
  double *m_sum, *m_max, *m_norm2u3[2];

  double s, dn, g_rnmu;

  size_t norm2u3_lws[2], norm2u3_gws[2];
  size_t temp_size;

  if (timeron) mg_timer_start(T_norm2u3);

  dn = 1.0 * nx * ny * nz;
  s = 0.0;
  g_rnmu = 0.0;

  int p, j;
  int nk = NUM_PARTITION[lt];

  create_grid(&m_r[lt], nk);

  norm2u3_lws[0] = NORM2U3_LWS;
  norm2u3_lws[1] = 1;

  norm2u3_gws[0] = (n2-2) * norm2u3_lws[0];
  norm2u3_gws[1] = (n3-2) / nk;

  temp_size = (norm2u3_gws[0] * norm2u3_gws[1])
            / (norm2u3_lws[0] * norm2u3_lws[1]);

  CUCHK(cudaMalloc(&m_sum, temp_size * sizeof(double)));
  CUCHK(cudaMalloc(&m_max, temp_size * sizeof(double)));

  CUCHK(cudaMalloc(&m_norm2u3[0], norm2u3_gws[0] * norm2u3_gws[1] * sizeof(double)));
  CUCHK(cudaMalloc(&m_norm2u3[1], norm2u3_gws[0] * norm2u3_gws[1] * sizeof(double)));

  for (p = 0; p < nk; p++) {
    if (nk > 1) {
      if (timeron) mg_timer_start(T_norm2u3_comm);
      push_grid(&m_r[lt], p, nk, 0, false);
      if (timeron) mg_timer_stop(T_norm2u3_comm);
    }

    double *g_sum = (double *) malloc(temp_size * sizeof(double));
    double *g_max = (double *) malloc(temp_size * sizeof(double));

    dim3 blockSize(norm2u3_gws[0] / norm2u3_lws[0],
                   norm2u3_gws[1] / norm2u3_lws[1]);
    dim3 threadSize(norm2u3_lws[0], norm2u3_lws[1]);

    if (timeron) mg_timer_start(T_norm2u3_kern);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("kernel_norm2u3_opt", cmd_queue[0] );
      kernel_norm2u3_opt<<< blockSize,
                            threadSize,
                            sizeof(double) * norm2u3_lws[0] * 2,
                            cmd_queue[0] >>>
            (m_r[lt].d[p%2], n1, n2, m_sum, m_max);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_norm2u3_opt", cmd_queue[0] );
    }
    else {
      cuda_ProfilerStartEventRecord("kernel_norm2u3_base", cmd_queue[0] );
      kernel_norm2u3_base<<< blockSize,
                             threadSize,
                             0,
                             cmd_queue[0] >>>
            (m_r[lt].d[p%2], n1, n2, m_sum, m_max, m_norm2u3[0], m_norm2u3[1]);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("kernel_norm2u3_base", cmd_queue[0] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
      CUCHK(cudaGetLastError());
      mg_timer_stop(T_norm2u3_kern);
    }

    if (timeron) mg_timer_start(T_norm2u3_comm);
    CUCHK(cudaMemcpyAsync(g_sum, m_sum, temp_size * sizeof(double),
                          cudaMemcpyDeviceToHost, cmd_queue[0]));

    CUCHK(cudaMemcpyAsync(g_max, m_max, temp_size * sizeof(double),
                          cudaMemcpyDeviceToHost, cmd_queue[0]));

    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    CUCHK(cudaGetLastError());
    if (timeron) mg_timer_stop(T_norm2u3_comm);

    for (j = 0; j < temp_size; j++) {
      s = s + g_sum[j];
      if (g_rnmu < g_max[j]) g_rnmu = g_max[j];
    }

    free(g_sum);
    free(g_max);
  }

  *rnmu = g_rnmu;
  *rnm2 = sqrt(s / dn);

  release_grid(&m_r[lt]);

  cudaFree(m_sum);
  cudaFree(m_max);
  cudaFree(m_norm2u3[0]);
  cudaFree(m_norm2u3[1]);

  if (timeron) mg_timer_stop(T_norm2u3);
}


//---------------------------------------------------------------------
// comm3 organizes the communication on all borders
//---------------------------------------------------------------------
static void comm3(void *ou, int n1, int n2, int n3)
{
  if (timeron) mg_timer_start(T_comm3);

  double (*u)[n2][n1] = (double (*)[n2][n1]) ou;

  int i1, i2, i3;

  if (timeron) mg_timer_start(T_comm3);

  if (timeron) mg_timer_start(T_comm3_host);
  #pragma omp parallel for
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i2 = 1; i2 < n2-1; i2++) {
      u[i3][i2][   0] = u[i3][i2][n1-2];
      u[i3][i2][n1-1] = u[i3][i2][   1];
    }
  }

  #pragma omp parallel for
  for (i3 = 1; i3 < n3-1; i3++) {
    for (i1 = 0; i1 < n1; i1++) {
      u[i3][   0][i1] = u[i3][n2-2][i1];
      u[i3][n2-1][i1] = u[i3][   1][i1];
    }
  }

  #pragma omp parallel for
  for (i2 = 0; i2 < n2; i2++) {
    for (i1 = 0; i1 < n1; i1++) {
      u[   0][i2][i1] = u[n3-2][i2][i1];
      u[n3-1][i2][i1] = u[   1][i2][i1];
    }
  }
  if (timeron) mg_timer_stop(T_comm3_host);

  if (timeron) mg_timer_stop(T_comm3);
}

//---------------------------------------------------------------------
// comm3_kernel organizes the communication on all borders
//---------------------------------------------------------------------
static void comm3_kernel(double **m_ru, int n1, int n2, int n3)
{
  size_t comm3_lws[2], comm3_gws[2];

  if (timeron) mg_timer_start(T_comm3);

  comm3_lws[1] = 1;
  comm3_lws[0] = 32;

  comm3_gws[1] = n3 - 2;
  comm3_gws[0] = RoundWorkSize((size_t) (n2-2), comm3_lws[0]);

  dim3 blockSize(comm3_gws[0] / comm3_lws[0], comm3_gws[1] / comm3_lws[1]);
  dim3 threadSize(comm3_lws[0], comm3_lws[1]);

  if (timeron) mg_timer_start(T_comm3_kern);
  cuda_ProfilerStartEventRecord("kernel_comm3_1", cmd_queue[0] );
  kernel_comm3_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (*m_ru, n1, n2, n3);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("kernel_comm3_1", cmd_queue[0] );
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    CUCHK(cudaGetLastError());
    mg_timer_stop(T_comm3_kern);
  }

  comm3_lws[1] = 1;
  comm3_lws[0] = 32;

  comm3_gws[1] = n3 - 2;
  comm3_gws[0] = RoundWorkSize((size_t) n1, comm3_lws[0]);

  threadSize.x = comm3_lws[0];
  threadSize.y = comm3_lws[1];
  threadSize.z = 1;

  blockSize.x = comm3_gws[0] / comm3_lws[0];
  blockSize.y = comm3_gws[1] / comm3_lws[1];
  blockSize.z = 1;
  
  if (timeron) mg_timer_start(T_comm3_kern);
  cuda_ProfilerStartEventRecord("kernel_comm3_2", cmd_queue[0] );
  kernel_comm3_2<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (*m_ru, n1, n2, n3);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("kernel_comm3_2", cmd_queue[0] );
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    CUCHK(cudaGetLastError());
    mg_timer_stop(T_comm3_kern);
  }

  comm3_lws[1] = 1;
  comm3_lws[0] = 64;

  comm3_gws[1] = n2;
  comm3_gws[0] = RoundWorkSize((size_t) n1, comm3_lws[0]);

  threadSize.x = comm3_lws[0];
  threadSize.y = comm3_lws[1];
  threadSize.z = 1;

  blockSize.x = comm3_gws[0] / comm3_lws[0];
  blockSize.y = comm3_gws[1] / comm3_lws[1];
  blockSize.z = 1;
  
  if (timeron) mg_timer_start(T_comm3_kern);
  cuda_ProfilerStartEventRecord("kernel_comm3_3", cmd_queue[0] );
  kernel_comm3_3<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (*m_ru, n1, n2, n3);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("kernel_comm3_3", cmd_queue[0] );
  CUCHK(cudaStreamSynchronize(cmd_queue[0]));
  CUCHK(cudaGetLastError());
  if (timeron) mg_timer_stop(T_comm3_kern);

  if (timeron) mg_timer_stop(T_comm3);
}


//---------------------------------------------------------------------
// zran3  loads +1 at ten randomly chosen points,
// loads -1 at a different ten random points,
// and zero elsewhere.
//---------------------------------------------------------------------
static void zran3(void *oz, int n1, int n2, int n3, int nx, int ny, int k)
{
  double (*z)[n2][n1] = (double (*)[n2][n1]) oz;

  int i0, m0, m1;

  int i1, i2, i3, d1, e1, e2, e3;
  double xx, x0, x1, a1, a2, ai;

  const int mm = 10;
  const double a = pow(5.0, 13.0);
  const double x = 314159265.0;
  double ten[mm][2], best;
  int i, j1[mm][2], j2[mm][2], j3[mm][2];
  int jg[4][mm][2];

  double rdummy;

  a1 = power(a, nx);
  a2 = power(a, nx*ny);

  zero3(z, n1, n2, n3);

  i = is1-2+nx*(is2-2+ny*(is3-2));

  ai = power(a, i);
  d1 = ie1 - is1 + 1;
  e1 = ie1 - is1 + 2;
  e2 = ie2 - is2 + 2;
  e3 = ie3 - is3 + 2;
  x0 = x;
  rdummy = randlc(&x0, ai);

  for (i3 = 1; i3 < e3; i3++) {
    x1 = x0;
    for (i2 = 1; i2 < e2; i2++) {
      xx = x1;
      vranlc(d1, &xx, a, &(z[i3][i2][1]));
      rdummy = randlc(&x1,a1);
    }
    rdummy = randlc(&x0, a2);
  }

  //---------------------------------------------------------------------
  // comm3(z,n1,n2,n3);
  // showall(z,n1,n2,n3);
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // each processor looks for twenty candidates
  //---------------------------------------------------------------------
  for (i = 0; i < mm; i++) {
    ten[i][1] = 0.0;
    j1[i][1] = 0;
    j2[i][1] = 0;
    j3[i][1] = 0;
    ten[i][0] = 1.0;
    j1[i][0] = 0;
    j2[i][0] = 0;
    j3[i][0] = 0;
  }

  for (i3 = 1; i3 < n3-1; i3++) {
    for (i2 = 1; i2 < n2-1; i2++) {
      for (i1 = 1; i1 < n1-1; i1++) {
        if (z[i3][i2][i1] > ten[0][1]) {
          ten[0][1] = z[i3][i2][i1];
          j1[0][1] = i1;
          j2[0][1] = i2;
          j3[0][1] = i3;
          bubble(ten, j1, j2, j3, mm, 1);
        }
        if (z[i3][i2][i1] < ten[0][0]) {
          ten[0][0] = z[i3][i2][i1];
          j1[0][0] = i1;
          j2[0][0] = i2;
          j3[0][0] = i3;
          bubble(ten, j1, j2, j3, mm, 0);
        }
      }
    }
  }


  //---------------------------------------------------------------------
  // Now which of these are globally best?
  //---------------------------------------------------------------------
  i1 = mm - 1;
  i0 = mm - 1;
  for (i = mm - 1; i >= 0; i--) {
    best = 0.0;
    if (best < ten[i1][1]) {
      jg[0][i][1] = 0;
      jg[1][i][1] = is1 - 2 + j1[i1][1];
      jg[2][i][1] = is2 - 2 + j2[i1][1];
      jg[3][i][1] = is3 - 2 + j3[i1][1];
      i1 = i1-1;
    } else {
      jg[0][i][1] = 0;
      jg[1][i][1] = 0;
      jg[2][i][1] = 0;
      jg[3][i][1] = 0;
    }

    best = 1.0;
    if (best > ten[i0][0]) {
      jg[0][i][0] = 0;
      jg[1][i][0] = is1 - 2 + j1[i0][0];
      jg[2][i][0] = is2 - 2 + j2[i0][0];
      jg[3][i][0] = is3 - 2 + j3[i0][0];
      i0 = i0-1;
    } else {
      jg[0][i][0] = 0;
      jg[1][i][0] = 0;
      jg[2][i][0] = 0;
      jg[3][i][0] = 0;
    }
  }

  m1 = 0;
  m0 = 0;

  for (i3 = 0; i3 < n3; i3++) {
    for (i2 = 0; i2 < n2; i2++) {
      for (i1 = 0; i1 < n1; i1++) {
        z[i3][i2][i1] = 0.0;
      }
    }
  }
  for (i = mm-1; i >= m0; i--) {
    z[jg[3][i][0]][jg[2][i][0]][jg[1][i][0]] = -1.0;
  }
  for (i = mm-1; i >= m1; i--) {
    z[jg[3][i][1]][jg[2][i][1]][jg[1][i][1]] = +1.0;
  }

  comm3(z, n1, n2, n3);
}


//---------------------------------------------------------------------
// power  raises an integer, disguised as a double
// precision real, to an integer power
//---------------------------------------------------------------------
static double power(double a, int n)
{
  double aj;
  int nj;
  double rdummy;
  double power;

  power = 1.0;
  nj = n;
  aj = a;

  while (nj != 0) {
    if ((nj % 2) == 1) rdummy = randlc(&power, aj);
    rdummy = randlc(&aj, aj);
    nj = nj/2;
  }

  return power;
}


//---------------------------------------------------------------------
// bubble        does a bubble sort in direction dir
//---------------------------------------------------------------------
static void bubble(double ten[][2], int j1[][2], int j2[][2], int j3[][2],
                   int m, int ind)
{
  double temp;
  int i, j_temp;

  if (ind == 1) {
    for (i = 0; i < m-1; i++) {
      if (ten[i][ind] > ten[i+1][ind]) {
        temp = ten[i+1][ind];
        ten[i+1][ind] = ten[i][ind];
        ten[i][ind] = temp;

        j_temp = j1[i+1][ind];
        j1[i+1][ind] = j1[i][ind];
        j1[i][ind] = j_temp;

        j_temp = j2[i+1][ind];
        j2[i+1][ind] = j2[i][ind];
        j2[i][ind] = j_temp;

        j_temp = j3[i+1][ind];
        j3[i+1][ind] = j3[i][ind];
        j3[i][ind] = j_temp;
      } else {
        return;
      }
    }
  } else {
    for (i = 0; i < m-1; i++) {
      if (ten[i][ind] < ten[i+1][ind]) {

        temp = ten[i+1][ind];
        ten[i+1][ind] = ten[i][ind];
        ten[i][ind] = temp;

        j_temp = j1[i+1][ind];
        j1[i+1][ind] = j1[i][ind];
        j1[i][ind] = j_temp;

        j_temp = j2[i+1][ind];
        j2[i+1][ind] = j2[i][ind];
        j2[i][ind] = j_temp;

        j_temp = j3[i+1][ind];
        j3[i+1][ind] = j3[i][ind];
        j3[i][ind] = j_temp;
      } else {
        return;
      }
    }
  }
}


static void zero3(void *oz, int n1, int n2, int n3)
{
  memset(oz, 0, sizeof(double) * n1 * n2 * n3);
}


static void zero3_kernel(double **m_uk, int n1, int n2, int n3)
{
  if (timeron) mg_timer_start(T_zero3);

  size_t zero3_lws, zero3_gws;

  size_t problem_size = (size_t) n1 * n2 * n3;
  zero3_lws = problem_size > max_work_group_size
            ? max_work_group_size
            : problem_size;
  zero3_gws = RoundWorkSize(problem_size, zero3_lws);

  cuda_ProfilerStartEventRecord("kernel_zero3", cmd_queue[0] );
  kernel_zero3<<< zero3_gws / zero3_lws, zero3_lws, 0, cmd_queue[0] >>> 
              (*m_uk, n1, n2, n3);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("kernel_zero3", cmd_queue[0] );

  CUCHK(cudaStreamSynchronize(cmd_queue[0]));

  if (timeron) mg_timer_stop(T_zero3);
}


static size_t get_lws(int gws) {
  size_t lws = max_work_group_size;
  size_t lws_min = 32;

  while (lws > lws_min && gws <= lws / 2) {
    lws /= 2;
  }

  return lws;
}


static size_t get_wg(int gws)
{
  size_t lws = get_lws(gws);
  return (gws + lws - 1) / lws;
}


static void parse_arg(int argc, char *argv[])
{
  int c, l;
  char opt_level[100];

  while ((c = getopt(argc, argv, "o:")) != -1) {
    switch (c) {
      case 'o':
        memcpy(opt_level, optarg, strlen(optarg) + 1);
        l = atoi(opt_level);

        if (l == 0) {
          use_shared_mem = false;
        }
        else if (l == 1) {
          exit(0);
        }
        else if (l == 2) {
          use_shared_mem = true;
        }
        else if (l == 3) {
          exit(0);
        }
        else if (l == 4) {
          exit(0);
        }
        break;

      case '?':
        if (optopt == 'o') {
           printf("option -o requires OPTLEVEL\n");
        }
        break;
    }
  }
}

static void setup_timers(char *t_names[])
{
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;

    t_names[T_init] = "init";
    t_names[T_bench] = "bench";
    t_names[T_mg3P] = "mg3P";
    t_names[T_psinv] = "psinv";
    t_names[T_psinv_kern] = "psinv_kern";
    t_names[T_psinv_comm] = "psinv_comm";
    t_names[T_resid] = "resid";
    t_names[T_resid_kern] = "resid_kern";
    t_names[T_resid_comm] = "resid_comm";
    t_names[T_rprj3] = "rprj3";
    t_names[T_rprj3_kern] = "rprj3_kern";
    t_names[T_rprj3_comm] = "rprj3_comm";
    t_names[T_interp] = "interp";
    t_names[T_interp_kern] = "interp_kern";
    t_names[T_interp_comm] = "interp_comm";
    t_names[T_norm2u3] = "norm2u3";
    t_names[T_norm2u3_kern] = "norm2u3_kern";
    t_names[T_norm2u3_comm] = "norm2u3_comm";
    t_names[T_comm3] = "comm3";
    t_names[T_comm3_kern] = "comm3_kern";
    t_names[T_comm3_host] = "comm3_host";
    t_names[T_zero3] = "zero3";

    fclose(fp);
  }
  else {
    timeron = false;
  }

  int i;
  for (i = T_init; i < T_last; i++) {
    timer_clear(i);
  }
}

//---------------------------------------------------------------------
// Set up the CUDA environment.
//---------------------------------------------------------------------
static void setup_cuda(int argc, char *argv[], double a[4], double c[4])
{
  cuda_ProfilerSetup();

  int j, k;

  //-----------------------------------------------------------------------
  // 1. Find the default device type and get a device for the device type
  //-----------------------------------------------------------------------
  int devCount;

  CUCHK(cudaGetDeviceCount(&devCount));
  CUCHK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUCHK(cudaGetDeviceProperties(&prop, device));

  max_work_group_size = prop.maxThreadsPerBlock;
  max_mem_alloc_size = prop.totalGlobalMem / 4;
  global_mem_size = prop.totalGlobalMem;
  shared_mem_size = prop.sharedMemPerBlock;

  size_t required[LT_DEFAULT+1];

  for (k = lb; k <= lt; k++) {
    required[k] = (size_t) m1[k] * m2[k] * m3[k] * sizeof(double) * 2;
  }

  NUM_PARTITION[lb] = 1;

  for (k = lb+1; k <= lt; k++) {
    NUM_PARTITION[k] = NUM_PARTITION[k-1];

    while (global_mem_size / 2 < required[k] / NUM_PARTITION[k]) {
      NUM_PARTITION[k] *= 4;
    }

    NUM_PARTITION[k] = (NUM_PARTITION[k] < MAX_PARTITION)
                     ? NUM_PARTITION[k]
                     : MAX_PARTITION;
  }

  //for (k = lb; k <= lt; k++) {
  //  fprintf(stderr, " NUM_PARTITION[%d] = %d\n", k, (int) NUM_PARTITION[k]);
  //}
  //for (k = lb; k <= lt; k++) {
  //  fprintf(stderr, " LSIZE[%d] = %d x %d\n",
  //          k, (int) get_lws(1 << k), (int) get_wg(1 << k));
  //}
  //fprintf(stderr, " Global memory size: %d MB\n", (int) (global_mem_size >> 20));
  //fprintf(stderr, " Shared memory size: %d KB\n", (int) (shared_mem_size >> 10));

  //-----------------------------------------------------------------------
  // 3. Create a command queue
  //-----------------------------------------------------------------------
  CUCHK(cudaStreamCreate(&cmd_queue[0]));
  CUCHK(cudaStreamCreate(&cmd_queue[1]));

  //-----------------------------------------------------------------------
  // 6. Creating buffers
  //-----------------------------------------------------------------------
  CUCHK(cudaMalloc(&m_a, 4 * sizeof(double)));
  CUCHK(cudaMalloc(&m_c, 4 * sizeof(double)));

  //-----------------------------------------------------------------------
  // 7. Creating Events
  //-----------------------------------------------------------------------

  for (int i = 0; i < MAX_PARTITION; ++i) {
    for (int j = 0; j < 3; ++j) {
      CUCHK(cudaEventCreate(&write_event[i][j]));
    }
  }
}

static void release_cuda()
{
  cudaFree(m_a);
  cudaFree(m_c);

  cudaStreamDestroy(cmd_queue[0]);
  cudaStreamDestroy(cmd_queue[1]);

  for (int i = 0; i < MAX_PARTITION; ++i) {
    for (int j = 0; j < 3; ++j) {
      CUCHK(cudaEventDestroy(write_event[i][j]));
    }
  }

  cuda_ProfilerRelease();
}

/* grid operations */
static void init_grid(struct grid *g, double *h, int k, int x, int y, int z)
{
  g->h = h + ir[k];

  g->dim_x = x;
  g->dim_y = y;
  g->dim_z = z;

  g->created = 0;
  g->reserved = NUM_PARTITION[k] == 1;
}

static void create_grid(struct grid *g, int nk)
{
  size_t len = (size_t) g->dim_x * g->dim_y * ((g->dim_z - 2) / nk + 2);

  if (!g->created) {
    CUCHK(cudaMalloc(&g->d[0], len * sizeof(double)));

    if (!g->reserved) {
      CUCHK(cudaMalloc(&g->d[1], len * sizeof(double)));
    }

    g->created = 1;
  }
}

static void release_grid(struct grid *g)
{
  if (!g->reserved) {
    cudaFree(g->d[0]);
    cudaFree(g->d[1]);
    g->created = 0;
  }
}

static void push_grid(struct grid *g, int p, int n, int evt, bool sync)
{
  size_t len = (size_t) g->dim_x * g->dim_y * ((g->dim_z-2) / n + 2);
  size_t ofs = (size_t) g->dim_x * g->dim_y * (g->dim_z-2) / n * p;

  if (g->created) {
    CUCHK(cudaMemcpyAsync(g->d[p%2], g->h + ofs,
                          len * sizeof(double),
                          cudaMemcpyHostToDevice, cmd_queue[0]));
    CUCHK(cudaEventRecord(write_event[p][evt], cmd_queue[0]));

    if (sync) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    }
  }
}

static void pull_grid(struct grid *g, int p, int n, int pad, bool sync)
{
  size_t len = (size_t) g->dim_x * g->dim_y * ((g->dim_z-2) / n + 2) - 2*pad;
  size_t ofs = (size_t) g->dim_x * g->dim_y * (g->dim_z-2) / n * p + pad;

  if (g->created) {
    CUCHK(cudaMemcpyAsync(g->h + ofs, 
                          g->d[p%2] + pad,
                          len * sizeof(double),
                          cudaMemcpyDeviceToHost, cmd_queue[1]));

    if (sync) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
    }
  }
}

static void pushv_grid(struct grid *g, int p, int n, int evt, bool sync)
{
  size_t len = (size_t) g->dim_x * g->dim_y * ((g->dim_z-2) / n + 2);
  size_t ofs = (size_t) g->dim_x * g->dim_y * (g->dim_z-2) / n * p;

  if (g->created) {
    CUCHK(cudaMemcpyAsync(g->d[p%2], v + ofs,
                          len * sizeof(double),
                          cudaMemcpyHostToDevice, cmd_queue[0]));
    CUCHK(cudaEventRecord(write_event[p][evt], cmd_queue[0]));

    if (sync) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    }
  }
}
