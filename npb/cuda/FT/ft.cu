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
// FT benchmark
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
extern "C" {
#include "print_results.h"
#include "timers.h"
}

//---------------------------------------------------------------------------
// CUDA part
//---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include "cuda_util.h"

//---------------------------------------------------------------------
// u0, u1, u2 are the main arrays in the problem.
// Depending on the decomposition, these arrays will have different
// dimensions. To accomodate all possibilities, we allocate them as
// one-dimensional arrays and pass them to subroutines for different
// views
//  - u0 contains the initial (transformed) initial condition
//  - u1 and u2 are working arrays
//  - twiddle contains exponents for the time evolution operator.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// Large arrays are in common so that they are allocated on the
// heap rather than the stack. This common block is not
// referenced directly anywhere else. Padding is to avoid accidental
// cache problems, since all array sizes are powers of two.
//---------------------------------------------------------------------
/* common /bigarrays/ */
dcomplex u0[NTOTALP];
dcomplex u1[NTOTALP];
double twiddle[NTOTALP];

//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
#define MIN(x, y) (((size_t)x) < ((size_t)y) ? (x) : (y))

#define BUFFERING 2
static int device;
static const char *device_name;
static cudaStream_t cmd_queue[BUFFERING];

static size_t work_item_sizes[3];
static size_t max_work_group_size;
static size_t max_mem_alloc_size;
static size_t global_mem_size;
static size_t shared_mem_size;

#define MAX_PARTITION 32
static int NUM_PARTITION = 1;
static cudaEvent_t write_event[MAX_PARTITION];

static char *source_dir = "FT";
static logical use_shared_mem = true;
static logical use_opt_kernel = true;

static dcomplex *workspace[2];

static dcomplex *m_u;
static dcomplex *m_u0[BUFFERING];
static dcomplex *m_u1[BUFFERING];
static double *m_twiddle[BUFFERING];
static dcomplex *m_chk;
static dcomplex *g_chk;

static size_t cffts_lws;
static size_t checksum_lws;
static size_t checksum_gws;
static size_t checksum_wg_num;
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
static void init_ui(void *ou0, void *ou1, void *ot, int d1, int d2, int d3);
static void evolve(int d1, int d2, int d3);
static void compute_initial_conditions(int d1, int d2, int d3);
static double ipow46(double a, int exponent);
static void setup();
static void setup_cuda(int argc, char *argv[]);
static void release_cuda();
static void parse_arg(int argc, char *argv[]);
static void print_timers();
static void compute_indexmap(int d1, int d2, int d3);
static void fft_init(int n);
static void fft(int d1, int d2, int d3);
static void ifft(int d1, int d2, int d3);
static int ilog2(int n);
static void checksum(int i, int d1, int d2, int d3);
static void verify(int d1, int d2, int d3, int nt, logical *verified, char *Class);
//---------------------------------------------------------------------------

static inline dcomplex dcmplx_div(dcomplex z1, dcomplex z2) {
  double a = z1.real;
  double b = z1.imag;
  double c = z2.real;
  double d = z2.imag;

  double divisor = c*c + d*d;
  double real = (a*c + b*d) / divisor;
  double imag = (b*c - a*d) / divisor;
  dcomplex result = (dcomplex){real, imag};
  return result;
}

int main(int argc, char *argv[])
{
  int i;
  int iter;
  double total_time, mflops;
  logical verified;
  char Class;

  //---------------------------------------------------------------------
  // Start over from the beginning. Note that all operations must
  // be timed, in contrast to other benchmarks.
  //---------------------------------------------------------------------
  for (i = 1; i <= T_max; i++) {
    timer_clear(i);
  }

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timers_enabled = true;
    fclose(fp);
  }
  else {
    timers_enabled = false;
  }

  setup();
  parse_arg(argc, argv);
  setup_cuda(argc, argv);
  init_ui(u0, u1, twiddle, dims[0], dims[1], dims[2]);

  timer_start(T_total);
  cuda_ProfilerStart();

  if (timers_enabled) timer_start(T_setup);

  if (timers_enabled) timer_start(T_compute_im);
  compute_indexmap(dims[0], dims[1], dims[2]);
  if (timers_enabled) timer_stop(T_compute_im);

  if (timers_enabled) timer_start(T_compute_ics);
  compute_initial_conditions(dims[0], dims[1], dims[2]);
  if (timers_enabled) timer_stop(T_compute_ics);

  if (timers_enabled) timer_start(T_fft_init);
  fft_init(dims[0]);
  if (timers_enabled) timer_stop(T_fft_init);


  CUCHK(cudaMemcpyAsync(m_u, u, NXP * sizeof(dcomplex),
                        cudaMemcpyHostToDevice, cmd_queue[0]));
  CUCHK(cudaStreamSynchronize(cmd_queue[0]));

  if (timers_enabled) timer_stop(T_setup);

  if (timers_enabled) timer_start(T_fft);
  fft(dims[0], dims[1], dims[2]);
  if (timers_enabled) timer_stop(T_fft);

  for (iter = 1; iter <= niter; iter++) {
    if (timers_enabled) timer_start(T_evolve);
    evolve(dims[0], dims[1], dims[2]);
    if (timers_enabled) timer_stop(T_evolve);

    if (timers_enabled) timer_start(T_fft);
    ifft(dims[0], dims[1], dims[2]);
    if (timers_enabled) timer_stop(T_fft);

    if (timers_enabled) timer_start(T_checksum);
    checksum(iter, dims[0], dims[1], dims[2]);
    if (timers_enabled) timer_stop(T_checksum);
  }

  verify(NX, NY, NZ, niter, &verified, &Class);

  cuda_ProfilerStop();
  timer_stop(T_total);
  total_time = timer_read(T_total);

  if (total_time != 0.0) {
    mflops = 1.0e-6 * (double)NTOTAL *
      (14.8157 + 7.19641 * log((double)NTOTAL)
       + (5.23518 + 7.21113 * log((double)NTOTAL)) * niter)
      / total_time;
  }
  else {
    mflops = 0.0;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  c_print_results("FT", Class, NX, NY, NZ, niter,
                  total_time, mflops, "          floating point", verified,
                  NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, CS5, CS6, CS7,
                  (const char *)prop.name);
  if (timers_enabled) print_timers();

  release_cuda();

  fflush(stdout);

  return 0;
}


//---------------------------------------------------------------------
// touch all the big data
//---------------------------------------------------------------------
static void init_ui(void *ou0, void *ou1, void *ot, int d1, int d2, int d3)
{
  memset(ou0, 0, sizeof(dcomplex) * d3 * d2 * (d1+1));
  memset(ou1, 0, sizeof(dcomplex) * d3 * d2 * (d1+1));
  memset(ot, 0, sizeof(double) * d3 * d2 * (d1+1));
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
static void evolve_nb(int d1, int d2, int d3)
{
  size_t lws[3], gws[3];
  int limit = work_item_sizes[0];
  lws[0] = MIN(d1, limit);
  limit = max_work_group_size / lws[0];
  lws[1] = MIN(d2, limit);
  limit = limit / lws[1];
  lws[2] = MIN(d3, limit);

  gws[0] = RoundWorkSize(d1, lws[0]);
  gws[1] = RoundWorkSize(d2, lws[1]);
  gws[2] = RoundWorkSize(d3, lws[2]);

  dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1], gws[2]/lws[2]);
  dim3 threadSize(lws[0], lws[1], lws[2]);

  cuda_ProfilerStartEventRecord("k_evolve", cmd_queue[0] );
  k_evolve<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (m_u0[0], m_u1[0], m_twiddle[0], d1, d2, d3);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_evolve", cmd_queue[0] );

  if (timers_enabled) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
static void evolve_db(int d1, int d2, int d3)
{
  int i, ofs3, len3 = d3 / NUM_PARTITION + (0 < d3 % NUM_PARTITION);
  size_t lws[3], gws[3];

  int limit = work_item_sizes[0];
  lws[0] = MIN(d1, limit);
  limit = max_work_group_size / lws[0];
  lws[1] = MIN(d2, limit);
  limit = limit / lws[1];

  gws[0] = RoundWorkSize(d1, lws[0]);
  gws[1] = RoundWorkSize(d2, lws[1]);

  CUCHK(cudaMemcpyAsync(m_u0[0], &u0[0], 
                        len3 * d2 * (d1+1) * sizeof(dcomplex),
                        cudaMemcpyHostToDevice, cmd_queue[0]));

  CUCHK(cudaMemcpyAsync(m_twiddle[0], &twiddle[0],
                        len3 * d2 * (d1+1) * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[0]));
  CUCHK(cudaEventRecord(write_event[0], cmd_queue[0]));

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[2] = MIN(len3, limit);
    gws[2] = RoundWorkSize((size_t) len3, lws[2]);

    dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1], gws[2]/lws[2]);
    dim3 threadSize(lws[0], lws[1], lws[2]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i], 0));
    cuda_ProfilerStartEventRecord("k_evolve", cmd_queue[1] );
    k_evolve<<< blockSize, threadSize, 0, cmd_queue[1] >>>
            (m_u0[i%2], m_u1[i%2], m_twiddle[i%2], d1, d2, len3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_evolve", cmd_queue[1] );

    if (i+1 < NUM_PARTITION) {
      int ofs3 = d3 / NUM_PARTITION * (i+1) + MIN(i+1, d3 % NUM_PARTITION);
      int len3 = d3 / NUM_PARTITION + (i+1 < d3 % NUM_PARTITION);

      CUCHK(cudaMemcpyAsync(m_u0[(i+1)%2], &u0[ofs3 * d2 * (d1+1)],
                            len3 * d2 * (d1+1) * sizeof(dcomplex),
                            cudaMemcpyHostToDevice, cmd_queue[0]));

      CUCHK(cudaMemcpyAsync(m_twiddle[(i+1)%2], &twiddle[ofs3 * d2 * (d1+1)],
                            len3 * d2 * (d1+1) * sizeof(double),
                            cudaMemcpyHostToDevice, cmd_queue[0]));

      CUCHK(cudaEventRecord(write_event[i+1], cmd_queue[0]));
    }

    CUCHK(cudaMemcpyAsync(&u0[ofs3 * d2 * (d1+1)], m_u0[i%2], 
                          len3 * d2 * (d1+1) * sizeof(dcomplex),
                          cudaMemcpyDeviceToHost, cmd_queue[1]));

    CUCHK(cudaMemcpyAsync(&u1[ofs3 * d2 * (d1+1)], m_u1[i%2],
                          len3 * d2 * (d1+1) * sizeof(dcomplex),
                          cudaMemcpyDeviceToHost, cmd_queue[1]));

    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  }

  if (timers_enabled) {
    for (i = 0; i < BUFFERING; i++) {
      CUCHK(cudaStreamSynchronize(cmd_queue[i]));
    }
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
static void evolve(int d1, int d2, int d3) {
  if (NUM_PARTITION == 1) {
    evolve_nb(d1, d2, d3);
  }
  else {
    evolve_db(d1, d2, d3);
  }
}


//---------------------------------------------------------------------
// Fill in array u0 with initial conditions from
// random number generator
//---------------------------------------------------------------------
static void compute_initial_conditions(int d1, int d2, int d3)
{
  int i, k, ofs2, len2;
  double start, an, dummy, starts[NZ];
  size_t lws, gws;
  size_t m_org[3], h_org[3], region[3];
  double *m_starts;

  start = SEED;
  //---------------------------------------------------------------------
  // Jump to the starting element for our first plane.
  //---------------------------------------------------------------------
  an = ipow46(A, 0);
  dummy = randlc(&start, an);
  an = ipow46(A, 2*NX*NY);

  starts[0] = start;
  for (k = 1; k < d3; k++) {
    dummy = randlc(&start, an);
    starts[k] = start;
  }

  CUCHK(cudaMalloc(&m_starts, d3 * sizeof(double)));
  CUCHK(cudaMemcpyAsync(m_starts, starts, d3 * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[0]));

  //---------------------------------------------------------------------
  // Go through by z planes filling in one square at a time.
  //---------------------------------------------------------------------
  for (i = 0; i < NUM_PARTITION; i++) {
    ofs2 = d2 / NUM_PARTITION * i + MIN(i, d2 % NUM_PARTITION);
    len2 = d2 / NUM_PARTITION + (i < d2 % NUM_PARTITION);

    lws = MIN(d3, work_item_sizes[0]);
    gws = RoundWorkSize(d3, lws);

    cuda_ProfilerStartEventRecord("k_compute_initial_conditions", cmd_queue[0] );
    k_compute_initial_conditions<<< gws / lws, lws, 0, cmd_queue[0] >>>
                                (m_u1[0], m_starts, d1, len2, d3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_compute_initial_conditions", cmd_queue[0] );

    if (NUM_PARTITION > 1) {
      m_org[0] = 0;
      m_org[1] = 0;
      m_org[2] = 0;

      h_org[0] = 0;
      h_org[1] = ofs2;
      h_org[2] = 0;

      region[0] = (d1+1) * sizeof(dcomplex);
      region[1] = len2;
      region[2] = d3;

      struct cudaMemcpy3DParms p = { 0 };
      p.srcPtr = make_cudaPitchedPtr(m_u1[0], (d1+1) * sizeof(dcomplex), (d1+1), len2);
      p.srcPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
      p.dstPtr = make_cudaPitchedPtr(u1, (d1+1) * sizeof(dcomplex), (d1+1), d2);
      p.dstPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
      p.extent = make_cudaExtent(region[0], region[1], region[2]);
      p.kind = cudaMemcpyDeviceToHost;
      CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[0]));
    }
  }

  if (timers_enabled) {
    for (i = 0; i < BUFFERING; i++) {
      CUCHK(cudaStreamSynchronize(cmd_queue[i]));
    }
  }

  CUCHK(cudaStreamSynchronize(cmd_queue[0]));

  cudaFree(m_starts);
}


//---------------------------------------------------------------------
// compute a^exponent mod 2^46
//---------------------------------------------------------------------
static double ipow46(double a, int exponent)
{
  double result, dummy, q, r;
  int n, n2;

  //---------------------------------------------------------------------
  // Use
  //   a^n = a^(n/2)*a^(n/2) if n even else
  //   a^n = a*a^(n-1)       if n odd
  //---------------------------------------------------------------------
  result = 1;
  if (exponent == 0) return result;
  q = a;
  r = 1;
  n = exponent;

  while (n > 1) {
    n2 = n / 2;
    if (n2 * 2 == n) {
      dummy = randlc(&q, q);
      n = n2;
    } else {
      dummy = randlc(&r, q);
      n = n-1;
    }
  }
  dummy = randlc(&r, q);
  result = r;
  return result;
}


static void setup()
{
  debug = false;

  niter = NITER_DEFAULT;

  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-CUDA) - FT Benchmark\n\n");
  printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
  printf(" Iterations                  :%7d\n", niter);
  printf("\n");

  dims[0] = NX;
  dims[1] = NY;
  dims[2] = NZ;
}


//---------------------------------------------------------------------
// compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
// for time evolution exponent.
//---------------------------------------------------------------------
static void compute_indexmap(int d1, int d2, int d3)
{
  static double ap = -4.0 * ALPHA * PI * PI;
  size_t lws[3], gws[3];

  int i, ofs3, len3;

  size_t limit = work_item_sizes[0];
  lws[0] = MIN(d1, limit);
  limit = max_work_group_size / lws[0];
  lws[1] = MIN(d2, limit);
  limit = limit / lws[1];

  gws[0] = RoundWorkSize(d1, lws[0]);
  gws[1] = RoundWorkSize(d2, lws[1]);

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[2] = MIN(len3, limit);
    gws[2] = RoundWorkSize(len3, lws[2]);

    dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1], gws[2]/lws[2]);
    dim3 threadSize(lws[0], lws[1], lws[2]);

    cuda_ProfilerStartEventRecord("k_compute_indexmap", cmd_queue[i%2] );
    k_compute_indexmap<<< blockSize, threadSize, 0, cmd_queue[i%2] >>>
                      (m_twiddle[i%2], d1, d2, ofs3, len3, ap);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_compute_indexmap", cmd_queue[i%2] );

    if (NUM_PARTITION > 1) {
      CUCHK(cudaMemcpyAsync(&twiddle[ofs3 * d2 * (d1+1)],
                            m_twiddle[i%2],
                            len3 * d2 * (d1+1) * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            cmd_queue[i%2]));
    }
  }

  if (timers_enabled) {
    for (i = 0; i < BUFFERING; i++) {
      CUCHK(cudaStreamSynchronize(cmd_queue[i]));
    }
  }
}


static void fft_xy_nb(int d1, int d2, int d3) {
  int is = 1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];

  lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = RoundWorkSize(d2 * lws[0], lws[0]);
  gws[1] = RoundWorkSize(d3, lws[1]);

  dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1]);
  dim3 threadSize(lws[0], lws[1]);

  if (use_shared_mem && use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts1_base_3", cmd_queue[0] );
    k_cffts1_base_3<<< blockSize, threadSize,
                       sizeof(dcomplex) * d1 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_3", cmd_queue[0] );
  }
  else if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("k_cffts1_base_2", cmd_queue[0] );
    k_cffts1_base_2<<< blockSize, threadSize,
                       sizeof(dcomplex) * d1 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_2", cmd_queue[0] );
  }
  else if (use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts1_base_1", cmd_queue[0] );
    k_cffts1_base_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_1", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("k_cffts1_base_0", cmd_queue[0] );
    k_cffts1_base_0<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_0", cmd_queue[0] );
  }

  lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
  gws[1] = RoundWorkSize(d3, lws[1]);

  threadSize.x = lws[0];
  threadSize.y = lws[1];
  threadSize.z = 1;

  blockSize.x = gws[0] / lws[0];
  blockSize.y = gws[1] / lws[1];
  blockSize.z = 1;

  if (use_shared_mem && use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts2_base_3", cmd_queue[0] );
    k_cffts2_base_3<<< blockSize, threadSize,
                       sizeof(dcomplex) * d2 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_3", cmd_queue[0] );
  }
  else if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("k_cffts2_base_2", cmd_queue[0] );
    k_cffts2_base_2<<< blockSize, threadSize,
                       sizeof(dcomplex) * d2 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_2", cmd_queue[0] );
  }
  else if (use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts2_base_1", cmd_queue[0] );
    k_cffts2_base_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_1", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("k_cffts2_base_0", cmd_queue[0] );
    k_cffts2_base_0<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_0", cmd_queue[0] );
  }
}

static void fft_xy_db(int d1, int d2, int d3) {
  int i, ofs3, len3 = d3 / NUM_PARTITION + (0 < d3 % NUM_PARTITION);
  int is = 1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];

  CUCHK(cudaMemcpyAsync(m_u1[0], &u1[0], 
                        len3 * d2 * (d1+1) * sizeof(dcomplex),
                        cudaMemcpyHostToDevice, cmd_queue[0]));
  CUCHK(cudaEventRecord(write_event[0], cmd_queue[0]));

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = RoundWorkSize(d2 * lws[0], lws[0]);
    gws[1] = RoundWorkSize(len3, lws[1]);

    dim3 blockSize(gws[0] / lws[0], gws[1] / lws[1]);
    dim3 threadSize(lws[0], lws[1]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i], 0));
    if (use_shared_mem && use_opt_kernel) {
      if (sizeof(dcomplex) * d1 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts1_base_3", cmd_queue[1] );
        k_cffts1_base_3<<< blockSize, threadSize,
                           sizeof(dcomplex) * d1 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_base_3", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts1_inplace", cmd_queue[1] );
        k_cffts1_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d1, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_inplace", cmd_queue[1] );
      }
    }
    else if (use_shared_mem) {
      if (sizeof(dcomplex) * d1 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts1_base_2", cmd_queue[1] );
        k_cffts1_base_2<<< blockSize, threadSize,
                           sizeof(dcomplex) * d1 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_base_2", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts1_inplace", cmd_queue[1] );
        k_cffts1_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d1, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_inplace", cmd_queue[1] );
      }
    }
    else if (use_opt_kernel) {
      cuda_ProfilerStartEventRecord("k_cffts1_base_1", cmd_queue[1] );
      k_cffts1_base_1<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, d2, d3, logd1);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts1_base_1", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("k_cffts1_base_0", cmd_queue[1] );
      k_cffts1_base_0<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, d2, d3, logd1);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts1_base_0", cmd_queue[1] );
    }

    lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
    gws[1] = RoundWorkSize(len3, lws[1]);

    threadSize.x = lws[0];
    threadSize.y = lws[1];
    threadSize.z = 1;

    blockSize.x = gws[0] / lws[0];
    blockSize.y = gws[1] / lws[1];
    blockSize.z = 1;

    if (use_shared_mem && use_opt_kernel) {
      if (sizeof(dcomplex) * d2 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts2_base_3", cmd_queue[1] );
        k_cffts2_base_3<<< blockSize, threadSize,
                           sizeof(dcomplex) * d2 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_base_3", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts2_inplace", cmd_queue[1] );
        k_cffts2_inplace<<< blockSize, threadSize, sizeof(dcomplex) * d2, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_inplace", cmd_queue[1] );
      }
    }
    else if (use_shared_mem) {
      if (sizeof(dcomplex) * d2 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts2_base_2", cmd_queue[1] );
        k_cffts2_base_2<<< blockSize, threadSize,
                           sizeof(dcomplex) * d2 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_base_2", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts2_inplace", cmd_queue[1] );
        k_cffts2_inplace<<< blockSize, threadSize, sizeof(dcomplex) * d2, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_inplace", cmd_queue[1] );
      }
    }
    else if (use_opt_kernel) {
      cuda_ProfilerStartEventRecord("k_cffts2_base_1", cmd_queue[1] );
      k_cffts2_base_1<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, d2, d3, logd2);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts2_base_1", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("k_cffts2_base_0", cmd_queue[1] );
      k_cffts2_base_0<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, d2, d3, logd2);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts2_base_0", cmd_queue[1] );
    }

    if (i+1 < NUM_PARTITION) {
      int ofs3 = d3 / NUM_PARTITION * (i+1) + MIN(i+1, d3 % NUM_PARTITION);
      int len3 = d3 / NUM_PARTITION + (i+1 < d3 % NUM_PARTITION);

      CUCHK(cudaMemcpyAsync(m_u1[(i+1)%2], &u1[ofs3 * d2 * (d1+1)],
                            len3 * d2 * (d1+1) * sizeof(dcomplex),
                            cudaMemcpyHostToDevice, cmd_queue[0]));
      CUCHK(cudaEventRecord(write_event[i+1], cmd_queue[0]));
    }

    CUCHK(cudaMemcpyAsync(&u1[ofs3 * d2 * (d1+1)], m_u1[i%2],
                          len3 * d2 * (d1+1) * sizeof(dcomplex),
                          cudaMemcpyDeviceToHost, cmd_queue[1]));
    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  }
}

static void fft_z0_nb(int d1, int d2, int d3) {
  int is = 1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];

  lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
  gws[1] = RoundWorkSize(d2, lws[1]);

  dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1]);
  dim3 threadSize(lws[0], lws[1]);

  if (use_shared_mem && use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts3_base_3", cmd_queue[0] );
    k_cffts3_base_3<<< blockSize, threadSize,
                     sizeof(dcomplex) * d3 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u0[0], m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_3", cmd_queue[0] );
  }
  else if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("k_cffts3_base_2", cmd_queue[0] );
    k_cffts3_base_2<<< blockSize, threadSize,
                     sizeof(dcomplex) * d3 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u0[0], m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_2", cmd_queue[0] );
  }
  else if (use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts3_base_1", cmd_queue[0] );
    k_cffts3_base_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u0[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_1", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("k_cffts3_base_0", cmd_queue[0] );
    k_cffts3_base_0<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u0[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_0", cmd_queue[0] );
  }
}

static void fft_z0_db(int d1, int d2, int d3) {
  int i, ofs2, len2 = d2 / NUM_PARTITION + (0 < d2 % NUM_PARTITION);
  int is = 1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];
  size_t m_org[3], h_org[3], region[3];

  m_org[0] = 0;
  m_org[1] = 0;
  m_org[2] = 0;

  h_org[0] = 0;
  h_org[1] = 0;
  h_org[2] = 0;

  region[0] = (d1+1) * sizeof(dcomplex);
  region[1] = len2;
  region[2] = d3;

  {
    struct cudaMemcpy3DParms p = { 0 };
    p.srcPtr = make_cudaPitchedPtr(u1, (d1+1) * sizeof(dcomplex), (d1+1), d2);
    p.srcPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
    p.dstPtr = make_cudaPitchedPtr(m_u1[0], (d1+1) * sizeof(dcomplex), (d1+1), len2);
    p.dstPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
    p.extent = make_cudaExtent(region[0], region[1], region[2]);
    p.kind = cudaMemcpyHostToDevice;

    CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[0]));
    CUCHK(cudaEventRecord(write_event[0], cmd_queue[0]));
  }

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs2 = d2 / NUM_PARTITION * i + MIN(i, d2 % NUM_PARTITION);
    len2 = d2 / NUM_PARTITION + (i < d2 % NUM_PARTITION);

    lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
    gws[1] = RoundWorkSize(len2, lws[1]);

    dim3 blockSize(gws[0] / lws[0], gws[1] / lws[1]);
    dim3 threadSize(lws[0], lws[1]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i], 0));
    if (use_shared_mem && use_opt_kernel) {
      if (sizeof(dcomplex) * d3 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts3_base_3", cmd_queue[1] );
        k_cffts3_base_3<<< blockSize, threadSize,
                           sizeof(dcomplex) * d3 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u0[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_base_3", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts3_inplace", cmd_queue[1] );
        k_cffts3_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d3, cmd_queue[1] >>>
                        (m_u1[i%2], m_u0[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_inplace", cmd_queue[1] );
      }
    }
    else if (use_shared_mem) {
      if (sizeof(dcomplex) * d3 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts3_base_2", cmd_queue[1] );
        k_cffts3_base_2<<< blockSize, threadSize,
                           sizeof(dcomplex) * d3 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u0[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_base_2", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts3_inplace", cmd_queue[1] );
        k_cffts3_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d3, cmd_queue[1] >>>
                        (m_u1[i%2], m_u0[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_inplace", cmd_queue[1] );
      }
    }
    else if (use_opt_kernel) {
      cuda_ProfilerStartEventRecord("k_cffts3_base_1", cmd_queue[1] );
      k_cffts3_base_1<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u0[i%2], workspace[0], workspace[1],
                      m_u, is, d1, len2, d3, logd3);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts3_base_1", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("k_cffts3_base_0", cmd_queue[1] );
      k_cffts3_base_0<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u0[i%2], workspace[0], workspace[1],
                      m_u, is, d1, len2, d3, logd3);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts3_base_0", cmd_queue[1] );
    }

    if (i+1 < NUM_PARTITION) {
      int ofs2 = d2 / NUM_PARTITION * (i+1) + MIN(i+1, d2 % NUM_PARTITION);
      int len2 = d2 / NUM_PARTITION + (i+1 < d2 % NUM_PARTITION);

      h_org[1] = ofs2;
      region[1] = len2;

      struct cudaMemcpy3DParms p = { 0 };
      p.srcPtr = make_cudaPitchedPtr(u1, (d1+1) * sizeof(dcomplex), (d1+1), d2);
      p.srcPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
      p.dstPtr = make_cudaPitchedPtr(m_u1[(i+1)%2], (d1+1) * sizeof(dcomplex), (d1+1), len2);
      p.dstPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
      p.extent = make_cudaExtent(region[0], region[1], region[2]);
      p.kind = cudaMemcpyHostToDevice;

      CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[0]));
      CUCHK(cudaEventRecord(write_event[i+1], cmd_queue[0]));
    }

    h_org[1] = ofs2;
    region[1] = len2;

    struct cudaMemcpy3DParms p = { 0 };
    p.srcPtr = make_cudaPitchedPtr(m_u0[i%2], (d1+1) * sizeof(dcomplex), (d1+1), len2);
    p.srcPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
    p.dstPtr = make_cudaPitchedPtr(u0, (d1+1) * sizeof(dcomplex), (d1+1), d2);
    p.dstPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
    p.extent = make_cudaExtent(region[0], region[1], region[2]);
    p.kind = cudaMemcpyDeviceToHost;

    CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[1]));
    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  }
}

static void fft(int d1, int d2, int d3)
{
  if (NUM_PARTITION == 1) {
    fft_xy_nb(d1, d2, d3);
    fft_z0_nb(d1, d2, d3);
  }
  else {
    fft_xy_db(d1, d2, d3);
    fft_z0_db(d1, d2, d3);
  }

  if (timers_enabled) {
    int i;
    for (i = 0; i < BUFFERING; i++) {
      CUCHK(cudaStreamSynchronize(cmd_queue[i]));
    }
  }
}

static void ifft_z1_nb(int d1, int d2, int d3) {
  int is = -1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];

  lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
  gws[1] = RoundWorkSize(d2, lws[1]);

  dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1]);
  dim3 threadSize(lws[0], lws[1]);

  if (use_shared_mem && use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts3_base_3", cmd_queue[0] );
    k_cffts3_base_3<<< blockSize, threadSize,
                       sizeof(dcomplex) * d3 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_3", cmd_queue[0] );
  }
  else if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("k_cffts3_base_2", cmd_queue[0] );
    k_cffts3_base_2<<< blockSize, threadSize,
                       sizeof(dcomplex) * d3 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_2", cmd_queue[0] );
  }
  else if (use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts3_base_1", cmd_queue[0] );
    k_cffts3_base_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_1", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("k_cffts3_base_0", cmd_queue[0] );
    k_cffts3_base_0<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd3);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts3_base_0", cmd_queue[0] );
  }
}

static void ifft_z1_db(int d1, int d2, int d3) {
  int i, ofs2, len2 = d2 / NUM_PARTITION + (0 < d2 % NUM_PARTITION);
  int is = -1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];
  size_t m_org[3], h_org[3], region[3];

  m_org[0] = 0;
  m_org[1] = 0;
  m_org[2] = 0;

  h_org[0] = 0;
  h_org[1] = 0;
  h_org[2] = 0;

  region[0] = (d1+1) * sizeof(dcomplex);
  region[1] = len2;
  region[2] = d3;

  {
    struct cudaMemcpy3DParms p = { 0 };
    p.srcPtr = make_cudaPitchedPtr(u1, (d1+1) * sizeof(dcomplex), (d1+1), d2);
    p.srcPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
    p.dstPtr = make_cudaPitchedPtr(m_u1[0], (d1+1) * sizeof(dcomplex), (d1+1), len2);
    p.dstPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
    p.extent = make_cudaExtent(region[0], region[1], region[2]);
    p.kind = cudaMemcpyHostToDevice;

    CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[0]));
    CUCHK(cudaEventRecord(write_event[0], cmd_queue[0]));
  }

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs2 = d2 / NUM_PARTITION * i + MIN(i, d2 % NUM_PARTITION);
    len2 = d2 / NUM_PARTITION + (i < d2 % NUM_PARTITION);

    lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
    gws[1] = RoundWorkSize(len2, lws[1]);

    dim3 blockSize(gws[0] / lws[0], gws[1] / lws[1]);
    dim3 threadSize(lws[0], lws[1]);

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i], 0));
    if (use_shared_mem && use_opt_kernel) {
      if (sizeof(dcomplex) * d3 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts3_base_3", cmd_queue[1] );
        k_cffts3_base_3<<< blockSize, threadSize,
                           sizeof(dcomplex) * d3 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_base_3", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts3_inplace", cmd_queue[1] );
        k_cffts3_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d3, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_inplace", cmd_queue[1] );
      }
    }
    else if (use_shared_mem) {
      if (sizeof(dcomplex) * d3 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts3_base_2", cmd_queue[1] );
        k_cffts3_base_2<<< blockSize, threadSize,
                           sizeof(dcomplex) * d3 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_base_2", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts3_inplace", cmd_queue[1] );
        k_cffts3_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d3, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, len2, d3, logd3);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts3_inplace", cmd_queue[1] );
      }
    }
    else if (use_opt_kernel) {
      cuda_ProfilerStartEventRecord("k_cffts3_base_1", cmd_queue[1] );
      k_cffts3_base_1<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, len2, d3, logd3);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts3_base_1", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("k_cffts3_base_0", cmd_queue[1] );
      k_cffts3_base_0<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, len2, d3, logd3);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts3_base_0", cmd_queue[1] );
    }

    if (i+1 < NUM_PARTITION) {
      int ofs2 = d2 / NUM_PARTITION * (i+1) + MIN(i+1, d2 % NUM_PARTITION);
      int len2 = d2 / NUM_PARTITION + (i+1 < d2 % NUM_PARTITION);

      h_org[1] = ofs2;
      region[1] = len2;

      struct cudaMemcpy3DParms p = { 0 };
      p.srcPtr = make_cudaPitchedPtr(u1, (d1+1) * sizeof(dcomplex), (d1+1), d2);
      p.srcPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
      p.dstPtr = make_cudaPitchedPtr(m_u1[(i+1)%2], (d1+1) * sizeof(dcomplex), (d1+1), len2);
      p.dstPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
      p.extent = make_cudaExtent(region[0], region[1], region[2]);
      p.kind = cudaMemcpyHostToDevice;

      CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[0]));
      CUCHK(cudaEventRecord(write_event[i+1], cmd_queue[0]));
    }

    h_org[1] = ofs2;
    region[1] = len2;

    {
      struct cudaMemcpy3DParms p = {0};
      p.srcPtr = make_cudaPitchedPtr(m_u1[i%2], (d1+1) * sizeof(dcomplex), (d1+1), len2);
      p.srcPos = make_cudaPos(m_org[0], m_org[1], m_org[2]);
      p.dstPtr = make_cudaPitchedPtr(u1, (d1+1) * sizeof(dcomplex), (d1+1), d2);
      p.dstPos = make_cudaPos(h_org[0], h_org[1], h_org[2]);
      p.extent = make_cudaExtent(region[0], region[1], region[2]);
      p.kind = cudaMemcpyDeviceToHost;

      CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[1]));
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
    }
  }
}

static void ifft_yx_nb(int d1, int d2, int d3) {
  int is = -1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];

  lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
  gws[1] = RoundWorkSize(d3, lws[1]);

  dim3 blockSize(gws[0] / lws[0], gws[1] / lws[1]);
  dim3 threadSize(lws[0], lws[1]);

  if (use_shared_mem && use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts2_base_3", cmd_queue[0] );
    k_cffts2_base_3<<< blockSize, threadSize,
                       sizeof(dcomplex) * d2 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_3", cmd_queue[0] );
  }
  else if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("k_cffts2_base_2", cmd_queue[0] );
    k_cffts2_base_2<<< blockSize, threadSize,
                       sizeof(dcomplex) * d2 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_2", cmd_queue[0] );
  }
  else if (use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts2_base_1", cmd_queue[0] );
    k_cffts2_base_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_1", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("k_cffts2_base_0", cmd_queue[0] );
    k_cffts2_base_0<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts2_base_0", cmd_queue[0] );
  }

  lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = RoundWorkSize(d2 * lws[0], lws[0]);
  gws[1] = RoundWorkSize(d3, lws[1]);

  threadSize.x = lws[0];
  threadSize.y = lws[1];
  threadSize.z = 1;

  blockSize.x = gws[0] / lws[0];
  blockSize.y = gws[1] / lws[1];
  blockSize.z = 1;

  if (use_shared_mem && use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts1_base_3", cmd_queue[0] );
    k_cffts1_base_3<<< blockSize, threadSize,
                       sizeof(dcomplex) * d1 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_3", cmd_queue[0] );
  }
  else if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("k_cffts1_base_2", cmd_queue[0] );
    k_cffts1_base_2<<< blockSize, threadSize,
                       sizeof(dcomplex) * d1 * 2, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_2", cmd_queue[0] );
  }
  else if (use_opt_kernel) {
    cuda_ProfilerStartEventRecord("k_cffts1_base_1", cmd_queue[0] );
    k_cffts1_base_1<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_1", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("k_cffts1_base_0", cmd_queue[0] );
    k_cffts1_base_0<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                   (m_u1[0], m_u1[0], workspace[0], workspace[1],
                    m_u, is, d1, d2, d3, logd1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_cffts1_base_0", cmd_queue[0] );
  }
}

static void ifft_yx_db(int d1, int d2, int d3) {
  int i, ofs3, len3 = d3 / NUM_PARTITION + (0 < d3 % NUM_PARTITION);
  int is = -1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];

  CUCHK(cudaMemcpyAsync(m_u1[0], &u1[0],
                        len3 * d2 * (d1+1) * sizeof(dcomplex),
                        cudaMemcpyHostToDevice, cmd_queue[0]));
  CUCHK(cudaEventRecord(write_event[0], cmd_queue[0]));

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = RoundWorkSize(d1 * lws[0], lws[0]);
    gws[1] = RoundWorkSize(len3, lws[1]);

    dim3 blockSize(gws[0] / lws[0], gws[1] / lws[1]);
    dim3 threadSize(lws[0], lws[1]);
    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i], 0));

    if (use_shared_mem && use_opt_kernel) {
      if (sizeof(dcomplex) * d2 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts2_base_3", cmd_queue[1] );
        k_cffts2_base_3<<< blockSize, threadSize,
                           sizeof(dcomplex) * d2 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_base_3", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts2_inplace", cmd_queue[1] );
        k_cffts2_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d2, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_inplace", cmd_queue[1] );
      }
    }
    else if (use_shared_mem) {
      if (sizeof(dcomplex) * d2 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts2_base_2", cmd_queue[1] );
        k_cffts2_base_2<<< blockSize, threadSize,
                           sizeof(dcomplex) * d2 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_base_2", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts2_inplace", cmd_queue[1] );
        k_cffts2_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d2, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd2);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts2_inplace", cmd_queue[1] );
      }
    }
    else if (use_opt_kernel) {
      cuda_ProfilerStartEventRecord("k_cffts2_base_1", cmd_queue[1] );
      k_cffts2_base_1<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, d2, d3, logd2);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts2_base_1", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("k_cffts2_base_0", cmd_queue[1] );
      k_cffts2_base_0<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                     (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                      m_u, is, d1, d2, d3, logd2);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts2_base_0", cmd_queue[1] );
    }

    lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = RoundWorkSize(d2 * lws[0], lws[0]);
    gws[1] = RoundWorkSize(len3, lws[1]);

    threadSize.x = lws[0];
    threadSize.y = lws[1];
    threadSize.z = 1;

    blockSize.x = gws[0] / lws[0];
    blockSize.y = gws[1] / lws[1];
    blockSize.z = 1;

    CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i], 0));
    if (use_shared_mem && use_opt_kernel) {
      if (sizeof(dcomplex) * d1 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts1_base_3", cmd_queue[1] );
        k_cffts1_base_3<<< blockSize, threadSize,
                           sizeof(dcomplex) * d1 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_base_3", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts1_inplace", cmd_queue[1] );
        k_cffts1_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d1, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_inplace", cmd_queue[1] );
      }
    }
    else if (use_shared_mem) {
      if (sizeof(dcomplex) * d1 * 2 <= shared_mem_size) {
        cuda_ProfilerStartEventRecord("k_cffts1_base_2", cmd_queue[1] );
        k_cffts1_base_2<<< blockSize, threadSize,
                           sizeof(dcomplex) * d1 * 2, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_base_2", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("k_cffts1_inplace", cmd_queue[1] );
        k_cffts1_inplace<<< blockSize, threadSize,
                            sizeof(dcomplex) * d1, cmd_queue[1] >>>
                        (m_u1[i%2], m_u1[i%2], m_u, is, d1, d2, d3, logd1);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("k_cffts1_inplace", cmd_queue[1] );
      }
    }
    else if (use_opt_kernel) {
      cuda_ProfilerStartEventRecord("k_cffts1_base_1  ", cmd_queue[1] );
      k_cffts1_base_1<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                        m_u, is, d1, d2, d3, logd1);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts1_base_1", cmd_queue[1] );
    }
    else {
      cuda_ProfilerStartEventRecord("k_cffts1_base_0  ", cmd_queue[1] );
      k_cffts1_base_0<<< blockSize, threadSize, 0, cmd_queue[1] >>>
                       (m_u1[i%2], m_u1[i%2], workspace[0], workspace[1],
                        m_u, is, d1, d2, d3, logd1);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_cffts1_base_0", cmd_queue[1] );
    }

    if (i+1 < NUM_PARTITION) {
      int ofs3 = d3 / NUM_PARTITION * (i+1) + MIN(i+1, d3 % NUM_PARTITION);
      int len3 = d3 / NUM_PARTITION + (i+1 < d3 % NUM_PARTITION);

      CUCHK(cudaMemcpyAsync(m_u1[(i+1)%2], &u1[ofs3 * d2 * (d1+1)],
                            len3 * d2 * (d1+1) * sizeof(dcomplex),
                            cudaMemcpyHostToDevice, cmd_queue[0]));
    }

    CUCHK(cudaMemcpyAsync(&u1[ofs3 * d2 * (d1+1)], m_u1[i%2],
                          len3 * d2 * (d1+1) * sizeof(dcomplex),
                          cudaMemcpyDeviceToHost, cmd_queue[1]));
    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  }
}

static void ifft(int d1, int d2, int d3)
{
  if (NUM_PARTITION == 1) {
    ifft_z1_nb(d1, d2, d3);
    ifft_yx_nb(d1, d2, d3);
  }
  else {
    ifft_z1_db(d1, d2, d3);
    ifft_yx_db(d1, d2, d3);
  }

  if (timers_enabled) {
    int i;

    for (i = 0; i < BUFFERING; i++) {
      CUCHK(cudaStreamSynchronize(cmd_queue[i]));
    }
  }
}

//---------------------------------------------------------------------
// compute the roots-of-unity array that will be used for subsequent FFTs.
//---------------------------------------------------------------------
static void fft_init(int n)
{
  int m, ku, i, j, ln;
  double t, ti;

  //---------------------------------------------------------------------
  // Initialize the U array with sines and cosines in a manner that permits
  // stride one access at each FFT iteration.
  //---------------------------------------------------------------------
  m = ilog2(n);
  u[0] = dcmplx((double)m, 0.0);
  ku = 2;
  ln = 1;

  for (j = 1; j <= m; j++) {
    t = PI / ln;

    for (i = 0; i <= ln - 1; i++) {
      ti = i * t;
      u[i+ku-1] = dcmplx(cos(ti), sin(ti));
    }

    ku = ku + ln;
    ln = 2 * ln;
  }
}


static int ilog2(int n)
{
  int nn, lg;
  if (n == 1) return 0;
  lg = 1;
  nn = 2;
  while (nn < n) {
    nn = nn*2;
    lg = lg+1;
  }
  return lg;
}


static void checksum(int i, int d1, int d2, int d3)
{
  dcomplex chk = dcmplx(0.0, 0.0);

  if (NUM_PARTITION == 1) {
    cuda_ProfilerStartEventRecord("k_checksum", cmd_queue[0] );
    k_checksum<<< checksum_gws / checksum_lws, checksum_lws, 
                  sizeof(dcomplex) * checksum_lws, cmd_queue[0] >>>
              (m_u1[0], m_chk, d1, d2);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_checksum", cmd_queue[0] );

    CUCHK(cudaMemcpyAsync(g_chk, m_chk, 
                          checksum_wg_num * sizeof(dcomplex),
                          cudaMemcpyDeviceToHost, cmd_queue[0]));
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));

    size_t k;
    for (k = 0; k < checksum_wg_num; k++) {
      chk = dcmplx_add(chk, g_chk[k]);
    }
  }
  else {
    int x, ii, ji, ki;
    for (x = 1; x <= 1024; x++) {
      ii = x % d1;
      ji = 3 * x % d2;
      ki = 5 * x % d3;
      chk = dcmplx_add(chk, u1[ki*d2*(d1+1) + ji*(d1+1) + ii]);
    }
  }

  chk = dcmplx_div2(chk, (double)(NTOTAL));

  printf(" T =%5d     Checksum =%22.12E%22.12E\n", i, chk.real, chk.imag);
  sums[i] = chk;
}


static void verify(int d1, int d2, int d3, int nt,
    logical *verified, char *Class)
{
  int i;
  double err, epsilon;

  //---------------------------------------------------------------------
  // Reference checksums
  //---------------------------------------------------------------------
  dcomplex csum_ref[25+1];

  *Class = 'U';

  epsilon = 1.0e-12;
  *verified = false;

  if (d1 == 64 && d2 == 64 && d3 == 64 && nt == 6) {
    //---------------------------------------------------------------------
    //   Sample size reference checksums
    //---------------------------------------------------------------------
    *Class = 'S';
    csum_ref[1] = dcmplx(5.546087004964E+02, 4.845363331978E+02);
    csum_ref[2] = dcmplx(5.546385409189E+02, 4.865304269511E+02);
    csum_ref[3] = dcmplx(5.546148406171E+02, 4.883910722336E+02);
    csum_ref[4] = dcmplx(5.545423607415E+02, 4.901273169046E+02);
    csum_ref[5] = dcmplx(5.544255039624E+02, 4.917475857993E+02);
    csum_ref[6] = dcmplx(5.542683411902E+02, 4.932597244941E+02);

  } else if (d1 == 128 && d2 == 128 && d3 == 32 && nt == 6) {
    //---------------------------------------------------------------------
    //   Class W size reference checksums
    //---------------------------------------------------------------------
    *Class = 'W';
    csum_ref[1] = dcmplx(5.673612178944E+02, 5.293246849175E+02);
    csum_ref[2] = dcmplx(5.631436885271E+02, 5.282149986629E+02);
    csum_ref[3] = dcmplx(5.594024089970E+02, 5.270996558037E+02);
    csum_ref[4] = dcmplx(5.560698047020E+02, 5.260027904925E+02);
    csum_ref[5] = dcmplx(5.530898991250E+02, 5.249400845633E+02);
    csum_ref[6] = dcmplx(5.504159734538E+02, 5.239212247086E+02);

  } else if (d1 == 256 && d2 == 256 && d3 == 128 && nt == 6) {
    //---------------------------------------------------------------------
    //   Class A size reference checksums
    //---------------------------------------------------------------------
    *Class = 'A';
    csum_ref[1] = dcmplx(5.046735008193E+02, 5.114047905510E+02);
    csum_ref[2] = dcmplx(5.059412319734E+02, 5.098809666433E+02);
    csum_ref[3] = dcmplx(5.069376896287E+02, 5.098144042213E+02);
    csum_ref[4] = dcmplx(5.077892868474E+02, 5.101336130759E+02);
    csum_ref[5] = dcmplx(5.085233095391E+02, 5.104914655194E+02);
    csum_ref[6] = dcmplx(5.091487099959E+02, 5.107917842803E+02);

  } else if (d1 == 512 && d2 == 256 && d3 == 256 && nt == 20) {
    //---------------------------------------------------------------------
    //   Class B size reference checksums
    //---------------------------------------------------------------------
    *Class = 'B';
    csum_ref[1]  = dcmplx(5.177643571579E+02, 5.077803458597E+02);
    csum_ref[2]  = dcmplx(5.154521291263E+02, 5.088249431599E+02);
    csum_ref[3]  = dcmplx(5.146409228649E+02, 5.096208912659E+02);
    csum_ref[4]  = dcmplx(5.142378756213E+02, 5.101023387619E+02);
    csum_ref[5]  = dcmplx(5.139626667737E+02, 5.103976610617E+02);
    csum_ref[6]  = dcmplx(5.137423460082E+02, 5.105948019802E+02);
    csum_ref[7]  = dcmplx(5.135547056878E+02, 5.107404165783E+02);
    csum_ref[8]  = dcmplx(5.133910925466E+02, 5.108576573661E+02);
    csum_ref[9]  = dcmplx(5.132470705390E+02, 5.109577278523E+02);
    csum_ref[10] = dcmplx(5.131197729984E+02, 5.110460304483E+02);
    csum_ref[11] = dcmplx(5.130070319283E+02, 5.111252433800E+02);
    csum_ref[12] = dcmplx(5.129070537032E+02, 5.111968077718E+02);
    csum_ref[13] = dcmplx(5.128182883502E+02, 5.112616233064E+02);
    csum_ref[14] = dcmplx(5.127393733383E+02, 5.113203605551E+02);
    csum_ref[15] = dcmplx(5.126691062020E+02, 5.113735928093E+02);
    csum_ref[16] = dcmplx(5.126064276004E+02, 5.114218460548E+02);
    csum_ref[17] = dcmplx(5.125504076570E+02, 5.114656139760E+02);
    csum_ref[18] = dcmplx(5.125002331720E+02, 5.115053595966E+02);
    csum_ref[19] = dcmplx(5.124551951846E+02, 5.115415130407E+02);
    csum_ref[20] = dcmplx(5.124146770029E+02, 5.115744692211E+02);

  } else if (d1 == 512 && d2 == 512 && d3 == 512 && nt == 20) {
    //---------------------------------------------------------------------
    //   Class C size reference checksums
    //---------------------------------------------------------------------
    *Class = 'C';
    csum_ref[1]  = dcmplx(5.195078707457E+02, 5.149019699238E+02);
    csum_ref[2]  = dcmplx(5.155422171134E+02, 5.127578201997E+02);
    csum_ref[3]  = dcmplx(5.144678022222E+02, 5.122251847514E+02);
    csum_ref[4]  = dcmplx(5.140150594328E+02, 5.121090289018E+02);
    csum_ref[5]  = dcmplx(5.137550426810E+02, 5.121143685824E+02);
    csum_ref[6]  = dcmplx(5.135811056728E+02, 5.121496764568E+02);
    csum_ref[7]  = dcmplx(5.134569343165E+02, 5.121870921893E+02);
    csum_ref[8]  = dcmplx(5.133651975661E+02, 5.122193250322E+02);
    csum_ref[9]  = dcmplx(5.132955192805E+02, 5.122454735794E+02);
    csum_ref[10] = dcmplx(5.132410471738E+02, 5.122663649603E+02);
    csum_ref[11] = dcmplx(5.131971141679E+02, 5.122830879827E+02);
    csum_ref[12] = dcmplx(5.131605205716E+02, 5.122965869718E+02);
    csum_ref[13] = dcmplx(5.131290734194E+02, 5.123075927445E+02);
    csum_ref[14] = dcmplx(5.131012720314E+02, 5.123166486553E+02);
    csum_ref[15] = dcmplx(5.130760908195E+02, 5.123241541685E+02);
    csum_ref[16] = dcmplx(5.130528295923E+02, 5.123304037599E+02);
    csum_ref[17] = dcmplx(5.130310107773E+02, 5.123356167976E+02);
    csum_ref[18] = dcmplx(5.130103090133E+02, 5.123399592211E+02);
    csum_ref[19] = dcmplx(5.129905029333E+02, 5.123435588985E+02);
    csum_ref[20] = dcmplx(5.129714421109E+02, 5.123465164008E+02);

  } else if (d1 == 2048 && d2 == 1024 && d3 == 1024 && nt == 25) {
    //---------------------------------------------------------------------
    //   Class D size reference checksums
    //---------------------------------------------------------------------
    *Class = 'D';
    csum_ref[1]  = dcmplx(5.122230065252E+02, 5.118534037109E+02);
    csum_ref[2]  = dcmplx(5.120463975765E+02, 5.117061181082E+02);
    csum_ref[3]  = dcmplx(5.119865766760E+02, 5.117096364601E+02);
    csum_ref[4]  = dcmplx(5.119518799488E+02, 5.117373863950E+02);
    csum_ref[5]  = dcmplx(5.119269088223E+02, 5.117680347632E+02);
    csum_ref[6]  = dcmplx(5.119082416858E+02, 5.117967875532E+02);
    csum_ref[7]  = dcmplx(5.118943814638E+02, 5.118225281841E+02);
    csum_ref[8]  = dcmplx(5.118842385057E+02, 5.118451629348E+02);
    csum_ref[9]  = dcmplx(5.118769435632E+02, 5.118649119387E+02);
    csum_ref[10] = dcmplx(5.118718203448E+02, 5.118820803844E+02);
    csum_ref[11] = dcmplx(5.118683569061E+02, 5.118969781011E+02);
    csum_ref[12] = dcmplx(5.118661708593E+02, 5.119098918835E+02);
    csum_ref[13] = dcmplx(5.118649768950E+02, 5.119210777066E+02);
    csum_ref[14] = dcmplx(5.118645605626E+02, 5.119307604484E+02);
    csum_ref[15] = dcmplx(5.118647586618E+02, 5.119391362671E+02);
    csum_ref[16] = dcmplx(5.118654451572E+02, 5.119463757241E+02);
    csum_ref[17] = dcmplx(5.118665212451E+02, 5.119526269238E+02);
    csum_ref[18] = dcmplx(5.118679083821E+02, 5.119580184108E+02);
    csum_ref[19] = dcmplx(5.118695433664E+02, 5.119626617538E+02);
    csum_ref[20] = dcmplx(5.118713748264E+02, 5.119666538138E+02);
    csum_ref[21] = dcmplx(5.118733606701E+02, 5.119700787219E+02);
    csum_ref[22] = dcmplx(5.118754661974E+02, 5.119730095953E+02);
    csum_ref[23] = dcmplx(5.118776626738E+02, 5.119755100241E+02);
    csum_ref[24] = dcmplx(5.118799262314E+02, 5.119776353561E+02);
    csum_ref[25] = dcmplx(5.118822370068E+02, 5.119794338060E+02);

  } else if (d1 == 4096 && d2 == 2048 && d3 == 2048 && nt == 25) {
    //---------------------------------------------------------------------
    //   Class E size reference checksums
    //---------------------------------------------------------------------
    *Class = 'E';
    csum_ref[1]  = dcmplx(5.121601045346E+02, 5.117395998266E+02);
    csum_ref[2]  = dcmplx(5.120905403678E+02, 5.118614716182E+02);
    csum_ref[3]  = dcmplx(5.120623229306E+02, 5.119074203747E+02);
    csum_ref[4]  = dcmplx(5.120438418997E+02, 5.119345900733E+02);
    csum_ref[5]  = dcmplx(5.120311521872E+02, 5.119551325550E+02);
    csum_ref[6]  = dcmplx(5.120226088809E+02, 5.119720179919E+02);
    csum_ref[7]  = dcmplx(5.120169296534E+02, 5.119861371665E+02);
    csum_ref[8]  = dcmplx(5.120131225172E+02, 5.119979364402E+02);
    csum_ref[9]  = dcmplx(5.120104767108E+02, 5.120077674092E+02);
    csum_ref[10] = dcmplx(5.120085127969E+02, 5.120159443121E+02);
    csum_ref[11] = dcmplx(5.120069224127E+02, 5.120227453670E+02);
    csum_ref[12] = dcmplx(5.120055158164E+02, 5.120284096041E+02);
    csum_ref[13] = dcmplx(5.120041820159E+02, 5.120331373793E+02);
    csum_ref[14] = dcmplx(5.120028605402E+02, 5.120370938679E+02);
    csum_ref[15] = dcmplx(5.120015223011E+02, 5.120404138831E+02);
    csum_ref[16] = dcmplx(5.120001570022E+02, 5.120432068837E+02);
    csum_ref[17] = dcmplx(5.119987650555E+02, 5.120455615860E+02);
    csum_ref[18] = dcmplx(5.119973525091E+02, 5.120475499442E+02);
    csum_ref[19] = dcmplx(5.119959279472E+02, 5.120492304629E+02);
    csum_ref[20] = dcmplx(5.119945006558E+02, 5.120506508902E+02);
    csum_ref[21] = dcmplx(5.119930795911E+02, 5.120518503782E+02);
    csum_ref[22] = dcmplx(5.119916728462E+02, 5.120528612016E+02);
    csum_ref[23] = dcmplx(5.119902874185E+02, 5.120537101195E+02);
    csum_ref[24] = dcmplx(5.119889291565E+02, 5.120544194514E+02);
    csum_ref[25] = dcmplx(5.119876028049E+02, 5.120550079284E+02);
  }

  if (*Class != 'U') {
    *verified = true;
    for (i = 1; i <= nt; i++) {
      err = dcmplx_abs(dcmplx_div(dcmplx_sub(sums[i], csum_ref[i]),
            csum_ref[i]));
      if (!(err <= epsilon)) {
        *verified = false;
        break;
      }
    }
  }

  if (*Class != 'U') {
    if (*verified) {
      printf(" Result verification successful\n");
    } else {
      printf(" Result verification failed\n");
    }
  }
  printf(" class = %c\n", *Class);
}

static void setup_cuda(int argc, char *argv[])
{
  cuda_ProfilerSetup();

  int i;

  CUCHK(cudaGetDevice(&device));
  device_name = "";

  cudaDeviceProp deviceProp;
  CUCHK(cudaGetDeviceProperties(&deviceProp, device));

  work_item_sizes[0] = deviceProp.maxThreadsDim[0];
  work_item_sizes[1] = deviceProp.maxThreadsDim[1];
  work_item_sizes[2] = deviceProp.maxThreadsDim[2];

  max_work_group_size = deviceProp.maxThreadsPerBlock;
  max_mem_alloc_size = deviceProp.totalGlobalMem / 4;
  global_mem_size = deviceProp.totalGlobalMem;
  shared_mem_size = deviceProp.sharedMemPerBlock;


  // FIXME: The below values are experimental.
#define LIMIT(x) (((x) > DEFAULT_SIZE) ? DEFAULT_SIZE : (x))
#define DEFAULT_SIZE 64
  if (max_work_group_size > DEFAULT_SIZE) {
    max_work_group_size = DEFAULT_SIZE;
    work_item_sizes[0] = LIMIT(work_item_sizes[0]);
    work_item_sizes[1] = LIMIT(work_item_sizes[1]);
    work_item_sizes[2] = LIMIT(work_item_sizes[2]);
  }
#undef DEFAULT_SIZE
#undef LIMIT

  size_t required = sizeof(dcomplex) * NTOTALP;

  if (required > max_mem_alloc_size * 2) {
    while (required / NUM_PARTITION > max_mem_alloc_size) {
      NUM_PARTITION *= 2;
    }

    NUM_PARTITION = (NUM_PARTITION < MAX_PARTITION)
                  ? NUM_PARTITION
                  : MAX_PARTITION;
  }

  //fprintf(stderr, " The number of partition: %d\n", (int) NUM_PARTITION);
  //fprintf(stderr, " Global memory size: %d MB\n", (int) (global_mem_size >> 20));
  //fprintf(stderr, " Shared memory size: %d KB\n", (int) (shared_mem_size >> 10));

  for (i = 0; i < BUFFERING; i++) {
    CUCHK(cudaStreamCreate(&cmd_queue[i]));
  }

  // 5. Create buffers
  CUCHK(cudaMalloc(&m_u, sizeof(dcomplex) * NXP));

  size_t size_per_partition = (NTOTALP + NUM_PARTITION - 1) / NUM_PARTITION;

  for (i = 0; i < BUFFERING; i++) {
    CUCHK(cudaMalloc(&m_u0[i], sizeof(dcomplex) * size_per_partition));
    CUCHK(cudaMalloc(&m_u1[i], sizeof(dcomplex) * size_per_partition));
    CUCHK(cudaMalloc(&m_twiddle[i], sizeof(double) * size_per_partition));
  }

  if (!use_shared_mem) {
    CUCHK(cudaMalloc(&workspace[0], sizeof(dcomplex) * size_per_partition));
    CUCHK(cudaMalloc(&workspace[1], sizeof(dcomplex) * size_per_partition));
  }

  cffts_lws = 256;
  checksum_lws = 32;
  checksum_gws = RoundWorkSize(1024, checksum_lws);
  checksum_wg_num = checksum_gws / checksum_lws;

  CUCHK(cudaMalloc(&m_chk, sizeof(dcomplex) * checksum_wg_num));

  g_chk = (dcomplex *) malloc(sizeof(dcomplex) * checksum_wg_num);

  for (int i = 0; i < MAX_PARTITION; ++i) {
    CUCHK(cudaEventCreate(&write_event[i]));
  }
}


static void release_cuda()
{
  free(g_chk);
  int i;
  for (i = 0; i < BUFFERING; i++) {
    cudaFree(m_u0[i]);
    cudaFree(m_u1[i]);
    cudaFree(m_twiddle[i]);
    cudaStreamDestroy(cmd_queue[i]);
  }
  cudaFree(m_u);
  cudaFree(m_chk);
  for (int i = 0; i < MAX_PARTITION; ++i) {
    cudaEventDestroy(write_event[i]);
  }

  cuda_ProfilerRelease();
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
          use_opt_kernel = false;
        }
        else if (l == 1) {
          use_shared_mem = false;
          use_opt_kernel = true;
        }
        else if (l == 2) {
          use_shared_mem = true;
          use_opt_kernel = false;
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


static void print_timers()
{
  int i;
  double t, t_m;
  char *tstrings[T_max];
  tstrings[ 0] = "           total";
  tstrings[ 1] = "           setup";
  tstrings[ 2] = "      compute_im";
  tstrings[ 3] = " compute_im_kern";
  tstrings[ 4] = " compute_im_comm";
  tstrings[ 5] = "     compute_ics";
  tstrings[ 6] = "compute_ics_kern";
  tstrings[ 7] = "compute_ics_comm";
  tstrings[ 8] = "        fft_init";
  tstrings[ 9] = "          evolve";
  tstrings[10] = "     evolve_kern";
  tstrings[11] = "     evolve_comm";
  tstrings[12] = "             fft";
  tstrings[13] = "      fft_x_kern";
  tstrings[14] = "      fft_y_kern";
  tstrings[15] = "     fft_xy_comm";
  tstrings[16] = "      fft_z_kern";
  tstrings[17] = "      fft_z_comm";
  tstrings[18] = "        checksum";
  tstrings[19] = "   checksum_kern";
  tstrings[20] = "   checksum_comm";
  tstrings[21] = "   checksum_host";

  t_m = timer_read(T_total);
  if (t_m <= 0.0) t_m = 1.00;
  for (i = 0; i < T_max; i++) {
    t = timer_read(i);
    printf(" timer %2d(%16s) :%12.6f (%6.3f%%)\n",
           i, tstrings[i], t, t*100.0/t_m);
  }

  cuda_ProfilerPrintResult();
}
