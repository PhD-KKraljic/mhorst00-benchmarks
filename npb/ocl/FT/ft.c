//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB SP code. This OpenCL C  //
//  version is a part of SNU-NPB 2019 developed by the Center for Manycore   //
//  Programming at Seoul National University and derived from the serial     //
//  Fortran versions in "NPB3.3.1-SER" developed by NAS.                     //
//                                                                           //
//  Permission to use, copy, distribute and modify this software for any     //
//  purpose with or without fee is hereby granted. This software is          //
//  provided "as is" without express or implied warranty.                    //
//                                                                           //
//  Information on original NPB 3.3.1, including the technical report, the   //
//  original specifications, source code, results and information on how     //
//  to submit new results, is available at:                                  //
//                                                                           //
//           http://www.nas.nasa.gov/Software/NPB/                           //
//                                                                           //
//  Information on SNU-NPB 2019, including the conference paper and source   //
//  code, is available at:                                                   //
//                                                                           //
//           http://aces.snu.ac.kr                                           //
//                                                                           //
//  Send comments or suggestions for this OpenCL C version to                //
//  snunpb@aces.snu.ac.kr                                                    //
//                                                                           //
//          Center for Manycore Programming                                  //
//          School of Computer Science and Engineering                       //
//          Seoul National University                                        //
//          Seoul 08826, Korea                                               //
//                                                                           //
//          E-mail: snunpb@aces.snu.ac.kr                                    //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Authors: Youngdong Do, Hyung Mo Kim, Pyeongseok Oh, Daeyoung Park,        //
//          and Jaejin Lee                                                   //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------
// FT benchmark
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "global.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include <CL/cl.h>
#include "cl_util.h"

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
#define MIN(x, y) ((x) < (y) ? (x) : (y))
static cl_device_type device_type;
static cl_device_id device;
static char *device_name;

#define BUFFERING 2
static cl_context context;
static cl_command_queue cmd_queue[BUFFERING];
static cl_program program;

static size_t work_item_sizes[3];
static size_t max_work_group_size;
static size_t max_mem_alloc_size;
static size_t global_mem_size;
static size_t local_mem_size;

static cl_kernel k_evolve;
static cl_kernel k_checksum;
static cl_kernel k_cffts1_base;
static cl_kernel k_cffts2_base;
static cl_kernel k_cffts3_base;
static cl_kernel k_cffts1_inplace;
static cl_kernel k_cffts2_inplace;
static cl_kernel k_cffts3_inplace;
static cl_kernel k_compute_idm;
static cl_kernel k_compute_ics;

#define MAX_PARTITION 32
static size_t NUM_PARTITION = 1;
static cl_event write_event[MAX_PARTITION];

static char *source_dir = "../FT";
static logical use_local_mem = true;
static logical use_opt_kernel = true;

static cl_mem workspace[2];

static cl_mem m_u;
static cl_mem m_u0[BUFFERING];
static cl_mem m_u1[BUFFERING];
static cl_mem m_twiddle[BUFFERING];
static cl_mem m_chk;
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
static void setup(int argc, char *argv[]);
static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
static void compute_indexmap(int d1, int d2, int d3);
static void print_timers();
static void fft_init(int n);
static void fft(int d1, int d2, int d3);
static void ifft(int d1, int d2, int d3);
static int ilog2(int n);
static void checksum(int i, int d1, int d2, int d3);
static void verify(int d1, int d2, int d3, int nt, logical *verified, char *Class);
static void generate_summary(dcomplex *h, double *rsum, double *rmax, long long *zeros);
static void print_summary(cl_mem d);
//---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  int i;
  int iter;
  double total_time, mflops;
  logical verified;
  char Class;
  cl_int ecode;

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

  setup(argc, argv);
  setup_opencl(argc, argv);
  init_ui(u0, u1, twiddle, dims[0], dims[1], dims[2]);

  timer_start(T_total);
  clu_ProfilerStart();

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

  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u,
                               CL_TRUE, 0,
                               NXP * sizeof(dcomplex), u,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");

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

  clu_ProfilerStop();
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

  c_print_results("FT", Class, NX, NY, NZ, niter,
                  total_time, mflops, "          floating point", verified,
                  NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, CS5, CS6, CS7,
                  clu_GetDeviceTypeName(device_type),
                  device_name);
  if (timers_enabled) print_timers();

  release_opencl();

  fflush(stdout);

  return 0;
}


//---------------------------------------------------------------------
// touch all the big data
//---------------------------------------------------------------------
void init_ui(void *ou0, void *ou1, void *ot, int d1, int d2, int d3)
{
  memset(ou0, 0, sizeof(dcomplex) * d3 * d2 * (d1+1));
  memset(ou1, 0, sizeof(dcomplex) * d3 * d2 * (d1+1));
  memset(ot, 0, sizeof(double) * d3 * d2 * (d1+1));
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
void evolve_nb(int d1, int d2, int d3)
{
  size_t lws[3], gws[3];
  cl_int ecode;

  size_t limit = work_item_sizes[0];
  lws[0] = MIN(d1, limit);
  limit = max_work_group_size / lws[0];
  lws[1] = MIN(d2, limit);
  limit = limit / lws[1];
  lws[2] = MIN(d3, limit);

  gws[0] = clu_RoundWorkSize((size_t) d1, lws[0]);
  gws[1] = clu_RoundWorkSize((size_t) d2, lws[1]);
  gws[2] = clu_RoundWorkSize((size_t) d3, lws[2]);

  ecode  = clSetKernelArg(k_evolve, 0, sizeof(cl_mem), &m_u0[0]);
  ecode |= clSetKernelArg(k_evolve, 1, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_evolve, 2, sizeof(cl_mem), &m_twiddle[0]);
  ecode |= clSetKernelArg(k_evolve, 3, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_evolve, 4, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_evolve, 5, sizeof(int), &d3);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_evolve_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_evolve,
                                 3, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_evolve_kern);
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
void evolve_db(int d1, int d2, int d3)
{
  int i, ofs3, len3 = d3 / NUM_PARTITION + (0 < d3 % NUM_PARTITION);
  size_t lws[3], gws[3];
  cl_int ecode;

  size_t limit = work_item_sizes[0];
  lws[0] = MIN(d1, limit);
  limit = max_work_group_size / lws[0];
  lws[1] = MIN(d2, limit);
  limit = limit / lws[1];

  gws[0] = clu_RoundWorkSize((size_t) d1, lws[0]);
  gws[1] = clu_RoundWorkSize((size_t) d2, lws[1]);

  if (timers_enabled) timer_start(T_evolve_comm);
  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u0[0],
                               CL_FALSE, 0,
                               len3 * d2 * (d1+1) * sizeof(dcomplex),
                               &u0[0],
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");

  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_twiddle[0],
                               CL_FALSE, 0,
                               len3 * d2 * (d1+1) * sizeof(double),
                               &twiddle[0],
                               0, NULL, &write_event[0]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_evolve_comm);
  }

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[2] = MIN(len3, limit);
    gws[2] = clu_RoundWorkSize((size_t) len3, lws[2]);

    ecode  = clSetKernelArg(k_evolve, 0, sizeof(cl_mem), &m_u0[i%2]);
    ecode |= clSetKernelArg(k_evolve, 1, sizeof(cl_mem), &m_u1[i%2]);
    ecode |= clSetKernelArg(k_evolve, 2, sizeof(cl_mem), &m_twiddle[i%2]);
    ecode |= clSetKernelArg(k_evolve, 3, sizeof(int), &d1);
    ecode |= clSetKernelArg(k_evolve, 4, sizeof(int), &d2);
    ecode |= clSetKernelArg(k_evolve, 5, sizeof(int), &len3);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (timers_enabled) timer_start(T_evolve_kern);
    ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                   k_evolve,
                                   3, NULL,
                                   gws,
                                   lws,
                                   1, &write_event[i], NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    if (timers_enabled) {
      ecode = clFinish(cmd_queue[1]);
      clu_CheckError(ecode, "clFinish()");
      timer_stop(T_evolve_kern);
    }

    if (i+1 < NUM_PARTITION) {
      int ofs3 = d3 / NUM_PARTITION * (i+1) + MIN(i+1, d3 % NUM_PARTITION);
      int len3 = d3 / NUM_PARTITION + (i+1 < d3 % NUM_PARTITION);

      if (timers_enabled) timer_start(T_evolve_comm);
      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u0[(i+1)%2],
                                   CL_FALSE, 0,
                                   len3 * d2 * (d1+1) * sizeof(dcomplex),
                                   &u0[ofs3 * d2 * (d1+1)],
                                   0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");

      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_twiddle[(i+1)%2],
                                   CL_FALSE, 0,
                                   len3 * d2 * (d1+1) * sizeof(double),
                                   &twiddle[ofs3 * d2 * (d1+1)],
                                   0, NULL, &write_event[i+1]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_evolve_comm);
      }
    }

    if (timers_enabled) timer_start(T_evolve_comm);
    ecode = clEnqueueReadBuffer(cmd_queue[1], m_u0[i%2],
                                CL_FALSE, 0,
                                len3 * d2 * (d1+1) * sizeof(dcomplex),
                                &u0[ofs3 * d2 * (d1+1)],
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");

    ecode = clEnqueueReadBuffer(cmd_queue[1], m_u1[i%2],
                                CL_TRUE, 0,
                                len3 * d2 * (d1+1) * sizeof(dcomplex),
                                &u1[ofs3 * d2 * (d1+1)],
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    if (timers_enabled) timer_stop(T_evolve_comm);
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
void evolve(int d1, int d2, int d3) {
  if (NUM_PARTITION == 1) {
    evolve_nb(d1, d2, d3);
    if (debug) printf("\n[%s:%u] u1 after evolve\n", __FILE__, __LINE__);
    if (debug) print_summary(m_u1[0]);
  }
  else {
    evolve_db(d1, d2, d3);
  }
}


//---------------------------------------------------------------------
// Fill in array u0 with initial conditions from
// random number generator
//---------------------------------------------------------------------
void compute_initial_conditions(int d1, int d2, int d3)
{
  int i, k, ofs2, len2;
  double start, an, dummy, starts[NZ];
  size_t lws, gws;
  size_t m_org[3], h_org[3], region[3];
  cl_mem m_starts;
  cl_int ecode;

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

  if (timers_enabled) timer_start(T_compute_ics_comm);
  m_starts = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            d3 * sizeof(double),
                            NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_starts,
                               CL_FALSE, 0,
                               d3 * sizeof(double),
                               starts,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_compute_ics_comm);
  }

  //---------------------------------------------------------------------
  // Go through by z planes filling in one square at a time.
  //---------------------------------------------------------------------
  for (i = 0; i < NUM_PARTITION; i++) {
    ofs2 = d2 / NUM_PARTITION * i + MIN(i, d2 % NUM_PARTITION);
    len2 = d2 / NUM_PARTITION + (i < d2 % NUM_PARTITION);

    lws = MIN(d3, work_item_sizes[0]);
    gws = clu_RoundWorkSize((size_t) d3, lws);

    ecode  = clSetKernelArg(k_compute_ics, 0, sizeof(cl_mem), &m_u1[0]);
    ecode |= clSetKernelArg(k_compute_ics, 1, sizeof(cl_mem), &m_starts);
    ecode |= clSetKernelArg(k_compute_ics, 2, sizeof(int), &d1);
    ecode |= clSetKernelArg(k_compute_ics, 3, sizeof(int), &len2);
    ecode |= clSetKernelArg(k_compute_ics, 4, sizeof(int), &d3);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (timers_enabled) timer_start(T_compute_ics_kern);
    ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                   k_compute_ics,
                                   1, NULL,
                                   &gws,
                                   &lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    if (timers_enabled) {
      ecode = clFinish(cmd_queue[0]);
      clu_CheckError(ecode, "clFinish()");
      timer_stop(T_compute_ics_kern);
    }

    if (timers_enabled) timer_start(T_compute_ics_comm);
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

      ecode = clEnqueueReadBufferRect(cmd_queue[0], m_u1[0],
                                      CL_FALSE,
                                      m_org, h_org, region,
                                      (d1+1) * sizeof(dcomplex),
                                      (d1+1) * len2 * sizeof(dcomplex),
                                      (d1+1) * sizeof(dcomplex),
                                      (d1+1) * d2 * sizeof(dcomplex),
                                      u1,
                                      0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBufferRect()");
    }
    if (timers_enabled) {
      ecode = clFinish(cmd_queue[0]);
      clu_CheckError(ecode, "clFinish()");
      timer_stop(T_compute_ics_comm);
    }
  }

  ecode = clFinish(cmd_queue[0]);
  clu_CheckError(ecode, "clFinish()");

  clReleaseMemObject(m_starts);
}


//---------------------------------------------------------------------
// compute a^exponent mod 2^46
//---------------------------------------------------------------------
double ipow46(double a, int exponent)
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


void setup(int argc, char *argv[])
{
  debug = false;

  niter = NITER_DEFAULT;

  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-OCL) - FT Benchmark\n\n");
  printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
  printf(" Iterations                  :%7d\n", niter);
  printf("\n");

  dims[0] = NX;
  dims[1] = NY;
  dims[2] = NZ;

  int c, l;
  char opt_level[100];

  while ((c = getopt(argc, argv, "o:s:")) != -1) {
    switch (c) {
      case 's':
        source_dir = (char*)malloc(strlen(optarg) + 1);
        memcpy(source_dir, optarg, strlen(optarg) + 1);
        break;

      case 'o':
        memcpy(opt_level, optarg, strlen(optarg) + 1);
        l = atoi(opt_level);

        if (l == 0) {
          use_local_mem = false;
          use_opt_kernel = false;
        }
        else if (l == 1) {
          use_local_mem = false;
          use_opt_kernel = true;
        }
        else if (l == 2) {
          use_local_mem = true;
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

//---------------------------------------------------------------------
// compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
// for time evolution exponent.
//---------------------------------------------------------------------
void compute_indexmap(int d1, int d2, int d3)
{
  static double ap = -4.0 * ALPHA * PI * PI;
  size_t lws[3], gws[3];
  cl_int ecode;

  int i, ofs3, len3;

  size_t limit = work_item_sizes[0];
  lws[0] = MIN(d1, limit);
  limit = max_work_group_size / lws[0];
  lws[1] = MIN(d2, limit);
  limit = limit / lws[1];

  gws[0] = clu_RoundWorkSize((size_t) d1, lws[0]);
  gws[1] = clu_RoundWorkSize((size_t) d2, lws[1]);
  
  for (i = 0; i < NUM_PARTITION; i++) {
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[2] = MIN(len3, limit);
    gws[2] = clu_RoundWorkSize((size_t) len3, lws[2]);

    ecode  = clSetKernelArg(k_compute_idm, 0, sizeof(cl_mem), &m_twiddle[i%2]);
    ecode |= clSetKernelArg(k_compute_idm, 1, sizeof(int), &d1);
    ecode |= clSetKernelArg(k_compute_idm, 2, sizeof(int), &d2);
    ecode |= clSetKernelArg(k_compute_idm, 3, sizeof(int), &ofs3);
    ecode |= clSetKernelArg(k_compute_idm, 4, sizeof(int), &len3);
    ecode |= clSetKernelArg(k_compute_idm, 5, sizeof(double), &ap);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (timers_enabled) timer_start(T_compute_im_kern);
    ecode = clEnqueueNDRangeKernel(cmd_queue[i%2],
                                   k_compute_idm,
                                   3, NULL,
                                   gws, lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    if (timers_enabled) {
      ecode = clFinish(cmd_queue[i%2]);
      clu_CheckError(ecode, "clFinish()");
      timer_stop(T_compute_im_kern);
    }

    if (timers_enabled) timer_start(T_compute_im_comm);
    if (NUM_PARTITION > 1) {
      ecode = clEnqueueReadBuffer(cmd_queue[i%2], m_twiddle[i%2],
                                  CL_FALSE, 0,
                                  len3 * d2 * (d1+1) * sizeof(double),
                                  &twiddle[ofs3 * d2 * (d1+1)],
                                  0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");
    }
    if (timers_enabled) {
      ecode = clFinish(cmd_queue[i%2]);
      clu_CheckError(ecode, "clFinish()");
      timer_stop(T_compute_im_comm);
    }
  }

  if (timers_enabled) {
    for (i = 0; i < BUFFERING; i++) {
      ecode = clFinish(cmd_queue[i]);
      clu_CheckError(ecode, "clFinish()");
    }
  }
}


void fft_xy_nb(int d1, int d2, int d3) {
  int is = 1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];
  cl_int ecode;

  lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = d2 * lws[0];
  gws[1] = d3;

  ecode  = clSetKernelArg(k_cffts1_base, 0, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts1_base, 1, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts1_base, 2, sizeof(cl_mem), &m_u);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(dcomplex) * d1, NULL);
    ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(dcomplex) * d1, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(cl_mem), &workspace[0]);
    ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(cl_mem), &workspace[1]);
  }
  ecode |= clSetKernelArg(k_cffts1_base, 5, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts1_base, 6, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts1_base, 7, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts1_base, 8, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts1_base, 9, sizeof(int), &logd1);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_fft_x_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_cffts1_base,
                                 2, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_x_kern);
  }

  lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = d1 * lws[0];
  gws[1] = d3;

  ecode  = clSetKernelArg(k_cffts2_base, 0, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts2_base, 1, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts2_base, 2, sizeof(cl_mem), &m_u);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(dcomplex) * d2, NULL);
    ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(dcomplex) * d2, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(cl_mem), &workspace[0]);
    ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(cl_mem), &workspace[1]);
  }
  ecode |= clSetKernelArg(k_cffts2_base, 5, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts2_base, 6, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts2_base, 7, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts2_base, 8, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts2_base, 9, sizeof(int), &logd2);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_fft_y_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_cffts2_base,
                                 2, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_y_kern);
  }
}

void fft_xy_db(int d1, int d2, int d3) {
  int i, ofs3, len3 = d3 / NUM_PARTITION + (0 < d3 % NUM_PARTITION);
  int is = 1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];
  cl_int ecode;

  if (timers_enabled) timer_start(T_fft_xy_comm);
  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u1[0],
                               CL_FALSE, 0,
                               len3 * d2 * (d1+1) * sizeof(dcomplex),
                               &u1[0],
                               0, NULL, &write_event[0]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_xy_comm);
  }

  for (i = 0; i < NUM_PARTITION; i++) { 
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = d2 * lws[0];
    gws[1] = len3;

    if (sizeof(dcomplex) * d1 * 2 <= local_mem_size) {
      ecode  = clSetKernelArg(k_cffts1_base, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_base, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_base, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(dcomplex) * d1, NULL);
        ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(dcomplex) * d1, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(cl_mem), &workspace[0]);
        ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(cl_mem), &workspace[1]);
      }
      ecode |= clSetKernelArg(k_cffts1_base, 5, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts1_base, 6, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts1_base, 7, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts1_base, 8, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts1_base, 9, sizeof(int), &logd1);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_x_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts1_base,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_x_kern);
      }
    }
    else {
      ecode  = clSetKernelArg(k_cffts1_inplace, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_inplace, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_inplace, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts1_inplace, 3, sizeof(dcomplex) * d1, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts1_inplace, 3, sizeof(cl_mem), &workspace[0]);
      }
      ecode |= clSetKernelArg(k_cffts1_inplace, 4, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts1_inplace, 5, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts1_inplace, 6, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts1_inplace, 7, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts1_inplace, 8, sizeof(int), &logd1);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_x_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts1_inplace,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_x_kern);
      }
    }

    lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = d1 * lws[0];
    gws[1] = len3;

    if (sizeof(dcomplex) * d2 * 2 <= local_mem_size) {
      ecode  = clSetKernelArg(k_cffts2_base, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_base, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_base, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(dcomplex) * d2, NULL);
        ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(dcomplex) * d2, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(cl_mem), &workspace[0]);
        ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(cl_mem), &workspace[1]);
      }
      ecode |= clSetKernelArg(k_cffts2_base, 5, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts2_base, 6, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts2_base, 7, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts2_base, 8, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts2_base, 9, sizeof(int), &logd2);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_y_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts2_base,
                                     2, NULL,
                                     gws,
                                     lws,
                                     0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_y_kern);
      }
    }
    else {
      ecode  = clSetKernelArg(k_cffts2_inplace, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_inplace, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_inplace, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts2_inplace, 3, sizeof(dcomplex) * d2, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts2_inplace, 3, sizeof(dcomplex) * d2, NULL);
      }
      ecode |= clSetKernelArg(k_cffts2_inplace, 4, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts2_inplace, 5, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts2_inplace, 6, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts2_inplace, 7, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts2_inplace, 8, sizeof(int), &logd2);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_y_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts2_inplace,
                                     2, NULL,
                                     gws,
                                     lws,
                                     0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_y_kern);
      }
    }

    if (i+1 < NUM_PARTITION) {
      int ofs3 = d3 / NUM_PARTITION * (i+1) + MIN(i+1, d3 % NUM_PARTITION);
      int len3 = d3 / NUM_PARTITION + (i+1 < d3 % NUM_PARTITION);

      if (timers_enabled) timer_start(T_fft_xy_comm);
      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u1[(i+1)%2],
                                   CL_FALSE, 0,
                                   len3 * d2 * (d1+1) * sizeof(dcomplex),
                                   &u1[ofs3 * d2 * (d1+1)],
                                   0, NULL, &write_event[i+1]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_xy_comm);
      }
    }

    if (timers_enabled) timer_start(T_fft_xy_comm);
    ecode = clEnqueueReadBuffer(cmd_queue[1], m_u1[i%2],
                                CL_TRUE, 0,
                                len3 * d2 * (d1+1) * sizeof(dcomplex),
                                &u1[ofs3 * d2 * (d1+1)],
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    if (timers_enabled) timer_stop(T_fft_xy_comm);
  }
}

void fft_z0_nb(int d1, int d2, int d3) {
  int is = 1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];
  cl_int ecode;

  lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = d1 * lws[0];
  gws[1] = d2;

  ecode  = clSetKernelArg(k_cffts3_base, 0, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts3_base, 1, sizeof(cl_mem), &m_u0[0]);
  ecode |= clSetKernelArg(k_cffts3_base, 2, sizeof(cl_mem), &m_u);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(dcomplex) * d3, NULL);
    ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(dcomplex) * d3, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(cl_mem), &workspace[0]);
    ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(cl_mem), &workspace[1]);
  }
  ecode |= clSetKernelArg(k_cffts3_base, 5, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts3_base, 6, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts3_base, 7, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts3_base, 8, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts3_base, 9, sizeof(int), &logd3);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_fft_z_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_cffts3_base,
                                 2, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_z_kern);
  }
}

void fft_z0_db(int d1, int d2, int d3) {
  int i, ofs2, len2 = d2 / NUM_PARTITION + (0 < d2 % NUM_PARTITION);
  int is = 1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];
  size_t m_org[3], h_org[3], region[3];
  cl_int ecode;

  m_org[0] = 0;
  m_org[1] = 0;
  m_org[2] = 0;

  h_org[0] = 0;
  h_org[1] = 0;
  h_org[2] = 0;

  region[0] = (d1+1) * sizeof(dcomplex);
  region[1] = len2;
  region[2] = d3;

  if (timers_enabled) timer_start(T_fft_z_comm);
  ecode = clEnqueueWriteBufferRect(cmd_queue[0], m_u1[0],
                                   CL_FALSE,
                                   m_org, h_org, region,
                                   (d1+1) * sizeof(dcomplex),
                                   (d1+1) * len2 * sizeof(dcomplex),
                                   (d1+1) * sizeof(dcomplex),
                                   (d1+1) * d2 * sizeof(dcomplex),
                                   u1,
                                   0, NULL, &write_event[0]);
  clu_CheckError(ecode, "clEnqueueWriteBufferRect()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_z_comm);
  }

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs2 = d2 / NUM_PARTITION * i + MIN(i, d2 % NUM_PARTITION);
    len2 = d2 / NUM_PARTITION + (i < d2 % NUM_PARTITION);

    lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = d1 * lws[0];
    gws[1] = len2;

    if (sizeof(dcomplex) * d3 * 2 <= local_mem_size) {
      ecode  = clSetKernelArg(k_cffts3_base, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts3_base, 1, sizeof(cl_mem), &m_u0[i%2]);
      ecode |= clSetKernelArg(k_cffts3_base, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(dcomplex) * d3, NULL);
        ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(dcomplex) * d3, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(cl_mem), &workspace[0]);
        ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(cl_mem), &workspace[1]);
      }
      ecode |= clSetKernelArg(k_cffts3_base, 5, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts3_base, 6, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts3_base, 7, sizeof(int), &len2);
      ecode |= clSetKernelArg(k_cffts3_base, 8, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts3_base, 9, sizeof(int), &logd3);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_z_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts3_base,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_z_kern);
      }
    }
    else {
      ecode  = clSetKernelArg(k_cffts3_inplace, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts3_inplace, 1, sizeof(cl_mem), &m_u0[i%2]);
      ecode |= clSetKernelArg(k_cffts3_inplace, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts3_inplace, 3, sizeof(dcomplex) * d3, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts3_inplace, 3, sizeof(cl_mem), &workspace[0]);
      }
      ecode |= clSetKernelArg(k_cffts3_inplace, 4, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts3_inplace, 5, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts3_inplace, 6, sizeof(int), &len2);
      ecode |= clSetKernelArg(k_cffts3_inplace, 7, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts3_inplace, 8, sizeof(int), &logd3);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_z_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts3_inplace,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_z_kern);
      }
    }

    if (i+1 < NUM_PARTITION) {
      int ofs2 = d2 / NUM_PARTITION * (i+1) + MIN(i+1, d2 % NUM_PARTITION);
      int len2 = d2 / NUM_PARTITION + (i+1 < d2 % NUM_PARTITION);

      h_org[1] = ofs2;
      region[1] = len2;

      if (timers_enabled) timer_start(T_fft_z_comm);
      ecode = clEnqueueWriteBufferRect(cmd_queue[0], m_u1[(i+1)%2],
                                       CL_FALSE,
                                       m_org, h_org, region,
                                       (d1+1) * sizeof(dcomplex),
                                       (d1+1) * len2 * sizeof(dcomplex),
                                       (d1+1) * sizeof(dcomplex),
                                       (d1+1) * d2 * sizeof(dcomplex),
                                       u1,
                                       0, NULL, &write_event[i+1]);
      clu_CheckError(ecode, "clEnqueueWriteBufferRect()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_z_comm);
      }
    }

    h_org[1] = ofs2;
    region[1] = len2;

    if (timers_enabled) timer_start(T_fft_z_comm);
    ecode = clEnqueueReadBufferRect(cmd_queue[1], m_u0[i%2],
                                    CL_TRUE,
                                    m_org, h_org, region,
                                    (d1+1) * sizeof(dcomplex),
                                    (d1+1) * len2 * sizeof(dcomplex),
                                    (d1+1) * sizeof(dcomplex),
                                    (d1+1) * d2 * sizeof(dcomplex),
                                    u0,
                                    0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBufferRect()");
    if (timers_enabled) timer_stop(T_fft_z_comm);
  }
}

void fft(int d1, int d2, int d3)
{
  if (NUM_PARTITION == 1) {
    fft_xy_nb(d1, d2, d3);
    if (debug) printf("\n[%s:%u] u1 after fft_xy\n", __FILE__, __LINE__);
    if (debug) print_summary(m_u1[0]);
    fft_z0_nb(d1, d2, d3);
    if (debug) printf("\n[%s:%u] u0 after fft_z0\n", __FILE__, __LINE__);
    if (debug) print_summary(m_u0[0]);
  }
  else {
    fft_xy_db(d1, d2, d3);
    fft_z0_db(d1, d2, d3);
  }
}

void ifft_z1_nb(int d1, int d2, int d3) {
  int is = -1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];
  cl_int ecode;

  lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = d1 * lws[0];
  gws[1] = d2;

  ecode  = clSetKernelArg(k_cffts3_base, 0, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts3_base, 1, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts3_base, 2, sizeof(cl_mem), &m_u);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(dcomplex) * d3, NULL);
    ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(dcomplex) * d3, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(cl_mem), &workspace[0]);
    ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(cl_mem), &workspace[1]);
  }
  ecode |= clSetKernelArg(k_cffts3_base, 5, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts3_base, 6, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts3_base, 7, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts3_base, 8, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts3_base, 9, sizeof(int), &logd3);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_fft_z_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_cffts3_base,
                                 2, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_z_kern);
  }
}

void ifft_z1_db(int d1, int d2, int d3) {
  int i, ofs2, len2 = d2 / NUM_PARTITION + (0 < d2 % NUM_PARTITION);
  int is = -1, logd3 = ilog2(d3);
  size_t lws[2], gws[2];
  size_t m_org[3], h_org[3], region[3];
  cl_int ecode;

  m_org[0] = 0;
  m_org[1] = 0;
  m_org[2] = 0;

  h_org[0] = 0;
  h_org[1] = 0;
  h_org[2] = 0;

  region[0] = (d1+1) * sizeof(dcomplex);
  region[1] = len2;
  region[2] = d3;

  if (timers_enabled) timer_start(T_fft_z_comm);
  ecode = clEnqueueWriteBufferRect(cmd_queue[0], m_u1[0],
                                   CL_FALSE,
                                   m_org, h_org, region,
                                   (d1+1) * sizeof(dcomplex),
                                   (d1+1) * len2 * sizeof(dcomplex),
                                   (d1+1) * sizeof(dcomplex),
                                   (d1+1) * d2 * sizeof(dcomplex),
                                   u1,
                                   0, NULL, &write_event[0]);
  clu_CheckError(ecode, "clEnqueueWriteBufferRect()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_z_comm);
  }

  for (i = 0; i < NUM_PARTITION; i++) {
    ofs2 = d2 / NUM_PARTITION * i + MIN(i, d2 % NUM_PARTITION);
    len2 = d2 / NUM_PARTITION + (i < d2 % NUM_PARTITION);

    lws[0] = (cffts_lws > d3/2) ? d3/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = d1 * lws[0];
    gws[1] = len2;

    if (sizeof(dcomplex) * d3 * 2 <= local_mem_size) {
      ecode  = clSetKernelArg(k_cffts3_base, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts3_base, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts3_base, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(dcomplex) * d3, NULL);
        ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(dcomplex) * d3, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts3_base, 3, sizeof(cl_mem), &workspace[0]);
        ecode |= clSetKernelArg(k_cffts3_base, 4, sizeof(cl_mem), &workspace[1]);
      }
      ecode |= clSetKernelArg(k_cffts3_base, 5, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts3_base, 6, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts3_base, 7, sizeof(int), &len2);
      ecode |= clSetKernelArg(k_cffts3_base, 8, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts3_base, 9, sizeof(int), &logd3);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_z_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts3_base,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_z_kern);
      }
    }
    else {
      ecode  = clSetKernelArg(k_cffts3_inplace, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts3_inplace, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts3_inplace, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts3_inplace, 3, sizeof(dcomplex) * d3, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts3_inplace, 3, sizeof(cl_mem), &workspace[0]);
      }
      ecode |= clSetKernelArg(k_cffts3_inplace, 4, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts3_inplace, 5, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts3_inplace, 6, sizeof(int), &len2);
      ecode |= clSetKernelArg(k_cffts3_inplace, 7, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts3_inplace, 8, sizeof(int), &logd3);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_z_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts3_inplace,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_z_kern);
      }
    }

    if (i+1 < NUM_PARTITION) {
      int ofs2 = d2 / NUM_PARTITION * (i+1) + MIN(i+1, d2 % NUM_PARTITION);
      int len2 = d2 / NUM_PARTITION + (i+1 < d2 % NUM_PARTITION);

      h_org[1] = ofs2;
      region[1] = len2;

      if (timers_enabled) timer_start(T_fft_z_comm);
      ecode = clEnqueueWriteBufferRect(cmd_queue[0], m_u1[(i+1)%2],
                                       CL_FALSE,
                                       m_org, h_org, region,
                                       (d1+1) * sizeof(dcomplex),
                                       (d1+1) * len2 * sizeof(dcomplex),
                                       (d1+1) * sizeof(dcomplex),
                                       (d1+1) * d2 * sizeof(dcomplex),
                                       u1,
                                       0, NULL, &write_event[i+1]);
      clu_CheckError(ecode, "clEnqueueWriteBufferRect()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_z_comm);
      }
    }

    h_org[1] = ofs2;
    region[1] = len2;

    if (timers_enabled) timer_start(T_fft_z_comm);
    ecode = clEnqueueReadBufferRect(cmd_queue[1], m_u1[i%2],
                                    CL_TRUE,
                                    m_org, h_org, region,
                                    (d1+1) * sizeof(dcomplex),
                                    (d1+1) * len2 * sizeof(dcomplex),
                                    (d1+1) * sizeof(dcomplex),
                                    (d1+1) * d2 * sizeof(dcomplex),
                                    u1,
                                    0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBufferRect()");
    if (timers_enabled) timer_stop(T_fft_z_comm);
  }
}

void ifft_yx_nb(int d1, int d2, int d3) {
  int is = -1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];
  cl_int ecode;

  lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = d1 * lws[0];
  gws[1] = d3;

  ecode  = clSetKernelArg(k_cffts2_base, 0, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts2_base, 1, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts2_base, 2, sizeof(cl_mem), &m_u);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(dcomplex) * d2, NULL);
    ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(dcomplex) * d2, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(cl_mem), &workspace[0]);
    ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(cl_mem), &workspace[1]);
  }
  ecode |= clSetKernelArg(k_cffts2_base, 5, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts2_base, 6, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts2_base, 7, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts2_base, 8, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts2_base, 9, sizeof(int), &logd2);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_fft_x_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_cffts2_base,
                                 2, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_x_kern);
  }

  lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
  lws[1] = 1;

  gws[0] = d2 * lws[0];
  gws[1] = d3;

  ecode  = clSetKernelArg(k_cffts1_base, 0, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts1_base, 1, sizeof(cl_mem), &m_u1[0]);
  ecode |= clSetKernelArg(k_cffts1_base, 2, sizeof(cl_mem), &m_u);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(dcomplex) * d1, NULL);
    ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(dcomplex) * d1, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(cl_mem), &workspace[0]);
    ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(cl_mem), &workspace[1]);
  }
  ecode |= clSetKernelArg(k_cffts1_base, 5, sizeof(int), &is);
  ecode |= clSetKernelArg(k_cffts1_base, 6, sizeof(int), &d1);
  ecode |= clSetKernelArg(k_cffts1_base, 7, sizeof(int), &d2);
  ecode |= clSetKernelArg(k_cffts1_base, 8, sizeof(int), &d3);
  ecode |= clSetKernelArg(k_cffts1_base, 9, sizeof(int), &logd1);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timers_enabled) timer_start(T_fft_y_kern);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_cffts1_base,
                                 2, NULL,
                                 gws,
                                 lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_y_kern);
  }
}

void ifft_yx_db(int d1, int d2, int d3) {
  int i, ofs3, len3 = d3 / NUM_PARTITION + (0 < d3 % NUM_PARTITION);
  int is = -1, logd1 = ilog2(d1), logd2 = ilog2(d2);
  size_t lws[2], gws[2];
  cl_int ecode;

  if (timers_enabled) timer_start(T_fft_xy_comm);
  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u1[0],
                               CL_FALSE, 0,
                               len3 * d2 * (d1+1) * sizeof(dcomplex),
                               &u1[0],
                               0, NULL, &write_event[0]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");
  if (timers_enabled) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_fft_xy_comm);
  }

  for (i = 0; i < NUM_PARTITION; i++) { 
    ofs3 = d3 / NUM_PARTITION * i + MIN(i, d3 % NUM_PARTITION);
    len3 = d3 / NUM_PARTITION + (i < d3 % NUM_PARTITION);

    lws[0] = (cffts_lws > d2/2) ? d2/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = d1 * lws[0];
    gws[1] = len3;

    if (sizeof(dcomplex) * d2 * 2 <= local_mem_size) {
      ecode  = clSetKernelArg(k_cffts2_base, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_base, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_base, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(dcomplex) * d2, NULL);
        ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(dcomplex) * d2, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts2_base, 3, sizeof(cl_mem), &workspace[0]);
        ecode |= clSetKernelArg(k_cffts2_base, 4, sizeof(cl_mem), &workspace[1]);
      }
      ecode |= clSetKernelArg(k_cffts2_base, 5, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts2_base, 6, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts2_base, 7, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts2_base, 8, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts2_base, 9, sizeof(int), &logd2);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_y_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts2_base,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_y_kern);
      }
    }
    else {
      ecode  = clSetKernelArg(k_cffts2_inplace, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_inplace, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts2_inplace, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts2_inplace, 3, sizeof(dcomplex) * d2, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts2_inplace, 3, sizeof(cl_mem), &workspace[0]);
      }
      ecode |= clSetKernelArg(k_cffts2_inplace, 4, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts2_inplace, 5, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts2_inplace, 6, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts2_inplace, 7, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts2_inplace, 8, sizeof(int), &logd2);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_y_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts2_inplace,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_y_kern);
      }
    }

    lws[0] = (cffts_lws > d1/2) ? d1/2 : cffts_lws;
    lws[1] = 1;

    gws[0] = d2 * lws[0];
    gws[1] = len3;

    if (sizeof(dcomplex) * d1 * 2 <= local_mem_size) {
      ecode  = clSetKernelArg(k_cffts1_base, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_base, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_base, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(dcomplex) * d1, NULL);
        ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(dcomplex) * d1, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts1_base, 3, sizeof(cl_mem), &workspace[0]);
        ecode |= clSetKernelArg(k_cffts1_base, 4, sizeof(cl_mem), &workspace[1]);
      }
      ecode |= clSetKernelArg(k_cffts1_base, 5, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts1_base, 6, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts1_base, 7, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts1_base, 8, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts1_base, 9, sizeof(int), &logd1);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_x_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts1_base,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_x_kern);
      }
    }
    else {
      ecode  = clSetKernelArg(k_cffts1_inplace, 0, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_inplace, 1, sizeof(cl_mem), &m_u1[i%2]);
      ecode |= clSetKernelArg(k_cffts1_inplace, 2, sizeof(cl_mem), &m_u);
      if (use_local_mem) {
        ecode |= clSetKernelArg(k_cffts1_inplace, 3, sizeof(dcomplex) * d1, NULL);
      }
      else {
        ecode |= clSetKernelArg(k_cffts1_inplace, 3, sizeof(cl_mem), &workspace[0]);
      }
      ecode |= clSetKernelArg(k_cffts1_inplace, 4, sizeof(int), &is);
      ecode |= clSetKernelArg(k_cffts1_inplace, 5, sizeof(int), &d1);
      ecode |= clSetKernelArg(k_cffts1_inplace, 6, sizeof(int), &d2);
      ecode |= clSetKernelArg(k_cffts1_inplace, 7, sizeof(int), &d3);
      ecode |= clSetKernelArg(k_cffts1_inplace, 8, sizeof(int), &logd1);
      clu_CheckError(ecode, "clSetKernelArg()");

      if (timers_enabled) timer_start(T_fft_x_kern);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_cffts1_inplace,
                                     2, NULL,
                                     gws,
                                     lws,
                                     1, &write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_x_kern);
      }
    }

    if (i+1 < NUM_PARTITION) {
      int ofs3 = d3 / NUM_PARTITION * (i+1) + MIN(i+1, d3 % NUM_PARTITION);
      int len3 = d3 / NUM_PARTITION + (i+1 < d3 % NUM_PARTITION);

      if (timers_enabled) timer_start(T_fft_xy_comm);
      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_u1[(i+1)%2],
                                   CL_FALSE, 0,
                                   len3 * d2 * (d1+1) * sizeof(dcomplex),
                                   &u1[ofs3 * d2 * (d1+1)],
                                   0, NULL, &write_event[i+1]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");
      if (timers_enabled) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_fft_xy_comm);
      }
    }

    if (timers_enabled) timer_start(T_fft_xy_comm);
    ecode = clEnqueueReadBuffer(cmd_queue[1], m_u1[i%2],
                                CL_TRUE, 0,
                                len3 * d2 * (d1+1) * sizeof(dcomplex),
                                &u1[ofs3 * d2 * (d1+1)],
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    if (timers_enabled) timer_stop(T_fft_xy_comm);
  }
}

void ifft(int d1, int d2, int d3)
{
  if (NUM_PARTITION == 1) {
    ifft_z1_nb(d1, d2, d3);
    if (debug) printf("\n[%s:%u] u1 after ifft_z1\n", __FILE__, __LINE__);
    if (debug) print_summary(m_u1[0]);
    ifft_yx_nb(d1, d2, d3);
    if (debug) printf("\n[%s:%u] u1 after ifft_yx\n", __FILE__, __LINE__);
    if (debug) print_summary(m_u1[0]);
  }
  else {
    ifft_z1_db(d1, d2, d3);
    ifft_yx_db(d1, d2, d3);
  }
}

//---------------------------------------------------------------------
// compute the roots-of-unity array that will be used for subsequent FFTs.
//---------------------------------------------------------------------
void fft_init(int n)
{
  int m, ku, i, j, ln;
  double t, ti;

  //---------------------------------------------------------------------
  // Initialize the U array with sines and cosines in a manner that permits
  // stride one access at each FFT iteration.
  //---------------------------------------------------------------------
  m = ilog2(n);
  u[0] = dcmplx(m, 0.0);
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


int ilog2(int n)
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


void checksum(int i, int d1, int d2, int d3)
{
  dcomplex chk = dcmplx(0.0, 0.0);
  cl_int ecode;

  if (NUM_PARTITION == 1) {
    ecode  = clSetKernelArg(k_checksum, 0, sizeof(cl_mem), &m_u1[0]);
    ecode |= clSetKernelArg(k_checksum, 1, sizeof(cl_mem), &m_chk);
    ecode |= clSetKernelArg(k_checksum, 2, sizeof(dcomplex) * checksum_lws, NULL);
    ecode |= clSetKernelArg(k_checksum, 3, sizeof(int), &d1);
    ecode |= clSetKernelArg(k_checksum, 4, sizeof(int), &d2);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (timers_enabled) timer_start(T_checksum_kern);
    ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                   k_checksum,
                                   1, NULL,
                                   &checksum_gws,
                                   &checksum_lws,
                                   0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
    if (timers_enabled) {
      ecode = clFinish(cmd_queue[0]);
      clu_CheckError(ecode, "clFinish()");
      timer_stop(T_checksum_kern);
    }

    if (timers_enabled) timer_start(T_checksum_comm);
    ecode = clEnqueueReadBuffer(cmd_queue[0], m_chk,
                                CL_TRUE, 0,
                                checksum_wg_num * sizeof(dcomplex), g_chk,
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer()");
    if (timers_enabled) timer_stop(T_checksum_comm);

    if (timers_enabled) timer_start(T_checksum_host);
    int k;
    for (k = 0; k < checksum_wg_num; k++) {
      chk = dcmplx_add(chk, g_chk[k]);
    }
    if (timers_enabled) timer_stop(T_checksum_host);
  }
  else {
    if (timers_enabled) timer_start(T_checksum_host);
    int x, ii, ji, ki;
    for (x = 1; x <= 1024; x++) {
      ii = x % d1;
      ji = 3 * x % d2;
      ki = 5 * x % d3;
      chk = dcmplx_add(chk, u1[ki*d2*(d1+1) + ji*(d1+1) + ii]);
    }
    if (timers_enabled) timer_stop(T_checksum_host);
  }

  chk = dcmplx_div2(chk, (double)(NTOTAL));

  printf(" T =%5d     Checksum =%22.12E%22.12E\n", i, chk.real, chk.imag);
  sums[i] = chk;
}


void verify(int d1, int d2, int d3, int nt, logical *verified, char *Class)
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

void setup_opencl(int argc, char *argv[])
{
  clu_ProfilerSetup();

  int i;

  device_type = clu_GetDefaultDeviceType();
  device = clu_GetAvailableDevice(device_type);
  device_name = clu_GetDeviceName(device);

  cl_int ecode;
  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(work_item_sizes),
                          &work_item_sizes,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(size_t),
                          &max_work_group_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(size_t),
                          &max_mem_alloc_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(size_t),
                          &global_mem_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_LOCAL_MEM_SIZE,
                          sizeof(size_t),
                          &local_mem_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

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

  //fprintf(stderr, " The number of partitions: %d\n", (int) NUM_PARTITION);
  //fprintf(stderr, " Global memory size: %d MB\n", (int) (global_mem_size >> 20));
  //fprintf(stderr, " Local memory size: %d KB\n", (int) (local_mem_size >> 10));

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  for (i = 0; i < BUFFERING; i++) {
    cmd_queue[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ecode);
    clu_CheckError(ecode, "clCreateCommandQueue()");
  }

  char *source_file;
  char build_option[100];

  if (device_type == CL_DEVICE_TYPE_GPU) {
    char vendor[50];

    ecode = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 50, vendor, NULL);
    clu_CheckError(ecode, "clGetDeviceInfo()");

    if (use_local_mem && use_opt_kernel) {
      source_file = "ft_gpu.cl";
    }
    else if (use_local_mem && !use_opt_kernel) {
      source_file = "ft_gpu_local_mem.cl";
    }
    else if (!use_local_mem && use_opt_kernel) {
      source_file = "ft_gpu_opt_kernel.cl";
    }
    else {
      source_file = "ft_gpu_baseline.cl";
    }
    sprintf(build_option, "-I. -DA=%lf -DNX=%d -DNY=%d -DNZ=%d",
            A, NX, NY, NZ);
  }
  else {
    fprintf(stderr, "Set the environment variable OPENCL_DEVICE_TYPE!\n");
    exit(EXIT_FAILURE);
  }

  program = clu_MakeProgram(context, device, source_dir, source_file, build_option);

  // 5. Create buffers
  m_u = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       sizeof(dcomplex) * NXP,
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  size_t size_per_partition = (NTOTALP + NUM_PARTITION - 1) / NUM_PARTITION;

  for (i = 0; i < BUFFERING; i++) {
    m_u0[i] = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             sizeof(dcomplex) * size_per_partition,
                             NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_u1[i] = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             sizeof(dcomplex) * size_per_partition,
                             NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    m_twiddle[i] = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(double) * size_per_partition,
                                  NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  }

  if (!use_local_mem) {
    workspace[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(dcomplex) * size_per_partition,
                                  NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");

    workspace[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(dcomplex) * size_per_partition,
                                  NULL, &ecode);
    clu_CheckError(ecode, "clCreateBuffer()");
  }

  cffts_lws = 256;
  checksum_lws = 32;
  checksum_gws = clu_RoundWorkSize((size_t) 1024, checksum_lws);
  checksum_wg_num = checksum_gws / checksum_lws;

  m_chk = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         sizeof(dcomplex) * checksum_wg_num,
                         NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer()");

  g_chk = (dcomplex *) malloc(sizeof(dcomplex) * checksum_wg_num);

  k_evolve = clCreateKernel(program, "evolve", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_checksum = clCreateKernel(program, "checksum", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_cffts1_base = clCreateKernel(program, "cffts1_base", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_cffts2_base = clCreateKernel(program, "cffts2_base", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_cffts3_base = clCreateKernel(program, "cffts3_base", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_cffts1_inplace = clCreateKernel(program, "cffts1_inplace", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_cffts2_inplace = clCreateKernel(program, "cffts2_inplace", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_cffts3_inplace = clCreateKernel(program, "cffts3_inplace", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_compute_idm = clCreateKernel(program, "compute_indexmap", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_compute_ics = clCreateKernel(program, "compute_initial_conditions", &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
}


void release_opencl()
{
  free(g_chk);

  int i;
  for (i = 0; i < BUFFERING; i++) {
    clReleaseMemObject(m_u0[i]);
    clReleaseMemObject(m_u1[i]);
    clReleaseMemObject(m_twiddle[i]);
    clReleaseCommandQueue(cmd_queue[i]);
  }

  if (!use_local_mem) {
    clReleaseMemObject(workspace[0]);
    clReleaseMemObject(workspace[1]);
  }

  clReleaseMemObject(m_u);
  clReleaseMemObject(m_chk);

  clReleaseKernel(k_evolve);
  clReleaseKernel(k_checksum);
  clReleaseKernel(k_cffts1_base);
  clReleaseKernel(k_cffts2_base);
  clReleaseKernel(k_cffts3_base);
  clReleaseKernel(k_cffts1_inplace);
  clReleaseKernel(k_cffts2_inplace);
  clReleaseKernel(k_cffts3_inplace);
  clReleaseKernel(k_compute_idm);
  clReleaseKernel(k_compute_ics);

  clReleaseProgram(program);
  clReleaseContext(context);

  clu_ProfilerRelease();
}


void print_timers()
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

  clu_ProfilerPrintResult();
}

/////////////////////////////////////////////////////////////////
// Debugging
/////////////////////////////////////////////////////////////////
void print_summary(cl_mem d)
{
  cl_int ecode;

  int d3 = dims[2];
  int d2 = dims[1];
  int d1 = dims[0];

  ecode = clFinish(cmd_queue[0]);
  clu_CheckError(ecode, "clFinish()");

  ecode = clFinish(cmd_queue[1]);
  clu_CheckError(ecode, "clFinish()");

  printf("BUFFER[%d][%d][%d]\n", d3, d2, d1);

  dcomplex *h = (dcomplex *) malloc(sizeof(dcomplex) * d3 * d2 * (d1+1));
  double sum = 0.0;
  double max = 0.0;
  long long num_zero = 0;

  ecode = clEnqueueReadBuffer(cmd_queue[0], d,
                              CL_TRUE, 0,
                              d3 * d2 * (d1+1) * sizeof(dcomplex),
                              h,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");

  generate_summary(h, &sum, &max, &num_zero);

  printf(" * sum of sqare = %010.6lf\n", sum);
  printf(" * max of sqare = %010.6lf\n", max);
  printf(" * num of zeros = %lld\n", num_zero);
}

void generate_summary(dcomplex *h, double *rsum, double *rmax, long long *zeros)
{
  int x, y, z;

  int d3 = dims[2];
  int d2 = dims[1];
  int d1 = dims[0];

  double (*data)[d2][d1+1] = (double (*)[d2][d1+1]) h;

  double sum = 0.0;
  double max = 0.0;
  long long num_zero = 0;

  for (z = 0; z < d3; z++) {
    for (y = 0; y < d2; y++) {
      for (x = 0; x < d1; x++) {
        double sqr = data[z][y][x] * data[z][y][x];

        sum += sqr;
        max = (max < sqr) ? sqr : max;
        if (sqr == 0.0) num_zero++;
      }
    }
  }

  *rsum += sum;
  *rmax = (*rmax < max) ? max : *rmax;
  *zeros += num_zero;
}
