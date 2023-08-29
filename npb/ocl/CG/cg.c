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
// program cg
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include <CL/cl.h>
#include "cl_util.h"

//---------------------------------------------------------------------
/* common / main_int_mem / */
static int colidx[NZ];
static int rowstr[NA + 1];
static int iv[NZ + 1 + NA];
static int arow[NA + 1];
static int acol[NAZ];

/* common / main_flt_mem / */
static double aelt[NAZ];
static double a[NZ];
static double x[NA + 2];
static double z[NA + 2];
static double p[NA + 2];
static double q[NA + 2];
static double r[NA + 2];

/* common / partit_size / */
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;

/* common /urando/ */
static double amult;
static double tran;

/* common /timers/ */
static char *t_names[T_last];
static logical timeron;
//---------------------------------------------------------------------


//---------------------------------------------------------------------
#define MAX_DEVICE 32
static cl_device_type device_type;
static cl_device_id device;
static char *device_name;

#define BUFFERING 2
static cl_platform_id platform;
static cl_context context;
static cl_command_queue cmd_queue[BUFFERING];
static cl_program program;

static size_t work_item_sizes[3];
static size_t max_work_group_size;
static size_t max_mem_alloc_size;;
static size_t global_mem_size;
static size_t local_mem_size;

#define NUM_K_MAIN 2
static cl_kernel k_main[NUM_K_MAIN];
#define NUM_K_CONJ_GRAD 8
static cl_kernel k_conj_grad[NUM_K_CONJ_GRAD];

#define MAX_TILES 32
static size_t NUM_TILES = 1;
static int vec_lst;
static int vec_ofs[MAX_TILES+1];

static cl_event write_event[MAX_TILES][2];

static char *source_dir = "../CG";
static logical use_local_mem = true;

/* common / main_int_mem / */
static cl_mem m_colidx[BUFFERING];
static cl_mem m_rowstr;

/* common / main_flt_mem / */
static cl_mem m_a[BUFFERING];
static cl_mem m_x;
static cl_mem m_z;
static cl_mem m_p;
static cl_mem m_q;
static cl_mem m_r;

static cl_mem m_norm_temp1;
static cl_mem m_norm_temp2;
static cl_mem m_rho;
static cl_mem m_d;

static cl_mem m_main[2];
static cl_mem m_conj[2];

static double *g_norm_temp1;
static double *g_norm_temp2;
static double *g_rho;
static double *g_d;

static size_t norm_temp_size;
static size_t rho_size;
static size_t d_size;

static size_t MAIN_LWS[2];
static size_t CG_LWS[3];
//---------------------------------------------------------------------


//---------------------------------------------------------------------
static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm);

static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER + 1],
                  double aelt[][NONZER + 1],
                  int iv[]);

static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER + 1],
                   double aelt[][NONZER + 1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift);

static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);

static int icnvrt(double x, int ipwr2);

static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);

static void setup(int argc, char *argv[]);

static void setup_timers();

static void setup_opencl(int argc, char *argv[], char Class);

static void release_opencl();
//---------------------------------------------------------------------

static void main_0(double *norm1, double *norm2);
static void main_1(double norm2);

int main(int argc, char *argv[])
{
  int i, j, k, it;

  double zeta;
  double rnorm;
  double norm_temp1, norm_temp2;

  double t, mflops, tmax;
  char Class;
  logical verified;
  double zeta_verify_value, epsilon, err;

  setup(argc, argv);

  timer_start(T_init);

  firstrow = 0;
  lastrow  = NA-1;
  firstcol = 0;
  lastcol  = NA-1;

  if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10) {
    Class = 'S';
    zeta_verify_value = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12) {
    Class = 'W';
    zeta_verify_value = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20) {
    Class = 'A';
    zeta_verify_value = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60) {
    Class = 'B';
    zeta_verify_value = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110) {
    Class = 'C';
    zeta_verify_value = 28.973605592845;
  } else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500) {
    Class = 'D';
    zeta_verify_value = 52.514532105794;
  } else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500) {
    Class = 'E';
    zeta_verify_value = 77.522164599383;
  } else {
    Class = 'U';
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-OCL) - CG Benchmark\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations:                  %5d\n", NITER);
  printf("\n");

  naa = NA;
  nzz = NZ;

  //---------------------------------------------------------------------
  // Inialize random number generator
  //---------------------------------------------------------------------
  tran    = 314159265.0;
  amult   = 1220703125.0;
  zeta    = randlc(&tran, amult);

  //---------------------------------------------------------------------
  //
  //---------------------------------------------------------------------
  makea(naa, nzz, a, colidx, rowstr,
        firstrow, lastrow, firstcol, lastcol,
        arow,
        (int (*)[NONZER + 1])(void *) acol,
        (double (*)[NONZER + 1])(void *) aelt,
        iv);

  //---------------------------------------------------------------------
  // Note: as a result of the above call to makea:
  //    values of j used in indexing rowstr go from 0 --> lastrow-firstrow
  //    values of colidx which are col indexes go from firstcol --> lastcol
  //    So:
  //    Shift the col index vals from actual (firstcol --> lastcol )
  //    to local, i.e., (0 --> lastcol-firstcol)
  //---------------------------------------------------------------------
  for (j = 0; j < lastrow - firstrow + 1; j++) {
    for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
      colidx[k] = colidx[k] - firstcol;
    }
  }

  //---------------------------------------------------------------------
  // set starting vector to (1, 1, .... 1)
  //---------------------------------------------------------------------
  for (i = 0; i < NA + 1; i++) {
    x[i] = 1.0;
  }
  for (j = 0; j < lastcol - firstcol + 1; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = 0.0;
    p[j] = 0.0;
  }

  zeta = 0.0;

  setup_opencl(argc, argv, Class);

  vec_lst = 0;

  timer_stop(T_init);

  printf(" Initialization time = %15.6f seconds\n", timer_read(T_init));

  timer_start(T_bench);
  clu_ProfilerStart();

  //---------------------------------------------------------------------
  //---->
  // Main Iteration for inverse power method
  //---->
  //---------------------------------------------------------------------
  for (it = 1; it <= NITER; it++) {
    //---------------------------------------------------------------------
    // The call to the conjugate gradient routine:
    //---------------------------------------------------------------------
    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    main_0(&norm_temp1, &norm_temp2);

    norm_temp2 = 1.0 / sqrt(norm_temp2);
    zeta = SHIFT + 1.0 / norm_temp1;

    if (it == 1)
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);
    fflush(stdout);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    if (it < NITER) {
      main_1(norm_temp2);
    }
  } // end of main iter inv pow meth

  clu_ProfilerStop();
  timer_stop(T_bench);

  //---------------------------------------------------------------------
  // End of timed section
  //---------------------------------------------------------------------

  t = timer_read(T_bench);

  printf(" Benchmark completed\n");

  epsilon = 1.0e-10;
  if (Class != 'U') {
    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.13E\n", zeta);
      printf(" Error is   %20.13E\n", err);
    }
    else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13E\n", zeta);
      printf(" The correct zeta is %20.13E\n", zeta_verify_value);
    }
  }
  else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (t != 0.0) {
    mflops = (double)(2*NITER*NA)
           * (3.0+(double)(NONZER*(NONZER+1))
             + 25.0*(5.0+(double)(NONZER*(NONZER+1)))
             + 3.0) / t / 1000000.0;
  }
  else {
    mflops = 0.0;
  }

  c_print_results("CG", Class, NA, 0, 0,
                  NITER, t,
                  mflops, "          floating point",
                  verified, NPBVERSION, COMPILETIME,
                  CS1, CS2, CS3, CS4, CS5, CS6, CS7,
                  clu_GetDeviceTypeName(device_type), device_name);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    tmax = timer_read(T_bench);
    if (tmax == 0.0) tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = 0; i < T_last; i++) {
      t = timer_read(i);
      if (i != T_init) {
        printf("  %8s:%12.6f\t(%6.3f%%)\n", t_names[i], t, t*100.0/tmax);
      }
    }
    fflush(stdout);

    clu_ProfilerPrintResult();
  }

  release_opencl();

  return 0;
}


static void main_0(double *norm1, double *norm2)
{
  int j;
  cl_int ecode;

  size_t main_lws = MAIN_LWS[0];
  size_t main_gws = clu_RoundWorkSize((size_t) naa, main_lws);

  ecode  = clSetKernelArg(k_main[0], 0, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_main[0], 1, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_main[0], 2, sizeof(cl_mem), &m_norm_temp1);
  ecode |= clSetKernelArg(k_main[0], 3, sizeof(cl_mem), &m_norm_temp2);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_main[0], 4, sizeof(double) * main_lws, NULL);
    ecode |= clSetKernelArg(k_main[0], 5, sizeof(double) * main_lws, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_main[0], 4, sizeof(cl_mem), &m_main[0]);
    ecode |= clSetKernelArg(k_main[0], 5, sizeof(cl_mem), &m_main[1]);
  }
  ecode |= clSetKernelArg(k_main[0], 6, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_main_0);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_main[0],
                                 1, NULL,
                                 &main_gws,
                                 &main_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_main_0);
  }

  if (timeron) timer_start(T_comm_main_0);
  ecode = clEnqueueReadBuffer(cmd_queue[0],
                              m_norm_temp1,
                              CL_FALSE, 0,
                              norm_temp_size,
                              g_norm_temp1,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");

  ecode = clEnqueueReadBuffer(cmd_queue[0],
                              m_norm_temp2,
                              CL_TRUE, 0,
                              norm_temp_size,
                              g_norm_temp2,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");
  if (timeron) timer_stop(T_comm_main_0);

  double norm_temp1 = 0.0;
  double norm_temp2 = 0.0;

  if (timeron) timer_start(T_host_main_0);
  for (j = 0; j < main_gws / main_lws; j++) {
    norm_temp1 += g_norm_temp1[j];
    norm_temp2 += g_norm_temp2[j];
  }
  if (timeron) timer_stop(T_host_main_0);

  *norm1 = norm_temp1;
  *norm2 = norm_temp2;
}


static void main_1(double norm2)
{
  cl_int ecode;

  size_t main_lws = MAIN_LWS[1];
  size_t main_gws = clu_RoundWorkSize((size_t) naa, main_lws);

  ecode  = clSetKernelArg(k_main[1], 0, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_main[1], 1, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_main[1], 2, sizeof(double), &norm2);
  ecode |= clSetKernelArg(k_main[1], 3, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_main_1);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_main[1],
                                 1, NULL,
                                 &main_gws,
                                 &main_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_main_1);
  }
}


//---------------------------------------------------------------------
// Floaging point arrays here are named as in NPB1 spec discussion of
// CG algorithm
//---------------------------------------------------------------------
static void conj_grad_0();
static void conj_grad_1(double *rho);
static void conj_grad_2();
static void conj_grad_3(double *d);
static void conj_grad_4(double alpha, double *rho);
static void conj_grad_5(double beta);
static void conj_grad_6();
static void conj_grad_7(double *sum);

static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm)
{
  int cgit, cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta;

  //---------------------------------------------------------------------
  // Initialize the CG algorithm:
  //---------------------------------------------------------------------
  conj_grad_0();

  //---------------------------------------------------------------------
  // rho = r.r
  // Now, obtain the norm of r: First, sum squares of r elements locally...
  //---------------------------------------------------------------------
  conj_grad_1(&rho);

  //---------------------------------------------------------------------
  //---->
  // The conj grad iteration loop
  //---->
  //---------------------------------------------------------------------
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    //---------------------------------------------------------------------
    // Save a temporary of rho
    //---------------------------------------------------------------------
    rho0 = rho;

    //---------------------------------------------------------------------
    // q = A.p
    // The partition submatrix-vector multiply: use workspace w
    //---------------------------------------------------------------------
    conj_grad_2();

    //---------------------------------------------------------------------
    // Obtain p.q
    //---------------------------------------------------------------------
    conj_grad_3(&d);

    //---------------------------------------------------------------------
    // Obtain alpha = rho / (p.q)
    //---------------------------------------------------------------------
    alpha = rho0 / d;

    //---------------------------------------------------------------------
    // Obtain z = z + alpha*p
    // and    r = r - alpha*q
    // then   rho = r.r
    // Now, obtain the norm of r: First, sum squares of r elements locally..
    //---------------------------------------------------------------------
    conj_grad_4(alpha, &rho);

    //---------------------------------------------------------------------
    // Obtain beta:
    //---------------------------------------------------------------------
    beta = rho / rho0;

    //---------------------------------------------------------------------
    // p = r + beta*p
    //---------------------------------------------------------------------
    conj_grad_5(beta);
  } // end of do cgit=1,cgitmax

  //---------------------------------------------------------------------
  // Compute residual norm explicitly:  ||r|| = ||x - A.z||
  // First, form A.z
  // The partition submatrix-vector multiply
  //---------------------------------------------------------------------
  conj_grad_6();

  //---------------------------------------------------------------------
  // At this point, r contains A.z
  //---------------------------------------------------------------------
  conj_grad_7(&sum);

  *rnorm = sqrt(sum);
}


static void conj_grad_0()
{
  cl_int ecode;

  int n = naa + 1;
  size_t cg_lws = CG_LWS[0];
  size_t cg_gws = clu_RoundWorkSize((size_t) n, cg_lws);

  ecode  = clSetKernelArg(k_conj_grad[0], 0, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_conj_grad[0], 1, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_conj_grad[0], 2, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[0], 3, sizeof(cl_mem), &m_x);
  ecode |= clSetKernelArg(k_conj_grad[0], 4, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[0], 5, sizeof(int), &n);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_conj_0);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_conj_grad[0],
                                 1, NULL,
                                 &cg_gws,
                                 &cg_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_conj_0);
  }
}


static void conj_grad_1(double *rho)
{
  int j;
  cl_int ecode;

  size_t cg_lws = CG_LWS[1];
  size_t cg_gws = clu_RoundWorkSize((size_t) naa, cg_lws);

  ecode  = clSetKernelArg(k_conj_grad[1], 0, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[1], 1, sizeof(cl_mem), &m_rho);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_conj_grad[1], 2, sizeof(double) * cg_lws, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_conj_grad[1], 2, sizeof(cl_mem), &m_conj[0]);
  }
  ecode |= clSetKernelArg(k_conj_grad[1], 3, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_conj_1);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_conj_grad[1],
                                 1, NULL,
                                 &cg_gws,
                                 &cg_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_conj_1);
  }

  if (timeron) timer_start(T_comm_conj_1);
  ecode = clEnqueueReadBuffer(cmd_queue[0], m_rho,
                              CL_TRUE, 0,
                              rho_size, g_rho,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");
  if (timeron) timer_stop(T_comm_conj_1);

  double rho_temp = 0.0;

  if (timeron) timer_start(T_host_conj_1);
  for (j = 0; j < cg_gws / cg_lws; j++) {
    rho_temp += g_rho[j];
  }
  if (timeron) timer_stop(T_host_conj_1);

  *rho = rho_temp;
}


static void conj_grad_2()
{
  int i, n, blk;
  cl_int ecode;
  size_t cg_lws, cg_gws;

  int i_step = (vec_lst == 0) ? 1 : -1;

  for (i = vec_lst; 0 <= i && i < NUM_TILES; i += i_step) {
    n = vec_ofs[i+1] - vec_ofs[i];
    cg_lws = CG_LWS[2];
    cg_gws = n * cg_lws;

    ecode  = clSetKernelArg(k_conj_grad[2], 0, sizeof(cl_mem), &m_rowstr);
    ecode |= clSetKernelArg(k_conj_grad[2], 1, sizeof(cl_mem), &m_colidx[i%2]);
    ecode |= clSetKernelArg(k_conj_grad[2], 2, sizeof(cl_mem), &m_a[i%2]);
    ecode |= clSetKernelArg(k_conj_grad[2], 3, sizeof(cl_mem), &m_p);
    ecode |= clSetKernelArg(k_conj_grad[2], 4, sizeof(cl_mem), &m_q);
    if (use_local_mem) {
      ecode |= clSetKernelArg(k_conj_grad[2], 5, sizeof(double) * cg_lws, NULL);
    }
    else {
      ecode |= clSetKernelArg(k_conj_grad[2], 5, sizeof(cl_mem), &m_conj[1]);
    }
    ecode |= clSetKernelArg(k_conj_grad[2], 6, sizeof(int), &vec_ofs[i]);
    ecode |= clSetKernelArg(k_conj_grad[2], 7, sizeof(int), &rowstr[vec_ofs[i]]);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (NUM_TILES == 1) {
      if (timeron) timer_start(T_kern_conj_2);
      ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                     k_conj_grad[2],
                                     1, NULL,
                                     &cg_gws,
                                     &cg_lws,
                                     0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timeron) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_kern_conj_2);
      }
    }
    else {
      if (timeron) timer_start(T_kern_conj_2);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_conj_grad[2],
                                     1, NULL,
                                     &cg_gws,
                                     &cg_lws,
                                     2, write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timeron) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_kern_conj_2);
      }
    }

    if (0 <= (i+i_step) && (i+i_step) < NUM_TILES) {
      blk = rowstr[vec_ofs[i+i_step+1]] - rowstr[vec_ofs[i+i_step]];

      if (timeron) timer_start(T_comm_conj_2);
      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_colidx[(i+i_step)%2],
                                   CL_FALSE, 0,
                                   blk * sizeof(int),
                                   colidx + rowstr[vec_ofs[i+i_step]],
                                   0, NULL, &write_event[i+i_step][0]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer");

      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_a[(i+i_step)%2],
                                   CL_FALSE, 0,
                                   blk * sizeof(double),
                                   a + rowstr[vec_ofs[i+i_step]],
                                   0, NULL, &write_event[i+i_step][1]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer");
      if (timeron) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_comm_conj_2);
      }

      vec_lst = i+i_step;
    }
  }

  if (NUM_TILES > 1) {
    ecode = clFinish(cmd_queue[1]);
    clu_CheckError(ecode, "clFinish()");
  }
}


static void conj_grad_3(double *d)
{
  int j;
  cl_int ecode;

  size_t cg_lws = CG_LWS[1];
  size_t cg_gws = clu_RoundWorkSize((size_t) naa, cg_lws);

  ecode  = clSetKernelArg(k_conj_grad[3], 0, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[3], 1, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_conj_grad[3], 2, sizeof(cl_mem), &m_d);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_conj_grad[3], 3, sizeof(double) * cg_lws, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_conj_grad[3], 3, sizeof(cl_mem), &m_conj[0]);
  }
  ecode |= clSetKernelArg(k_conj_grad[3], 4, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_conj_3);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_conj_grad[3],
                                 1, NULL,
                                 &cg_gws,
                                 &cg_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_conj_3);
  }

  if (timeron) timer_start(T_comm_conj_3);
  ecode = clEnqueueReadBuffer(cmd_queue[0], m_d,
                              CL_TRUE, 0,
                              d_size, g_d,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");
  if (timeron) timer_stop(T_comm_conj_3);

  double d_temp = 0.0;

  if (timeron) timer_start(T_host_conj_3);
  for (j = 0; j < cg_gws / cg_lws; j++) {
    d_temp += g_d[j];
  }
  if (timeron) timer_stop(T_host_conj_3);

  *d = d_temp;
}


static void conj_grad_4(double alpha, double *rho)
{
  int j;
  cl_int ecode;

  size_t cg_lws = CG_LWS[1];
  size_t cg_gws = clu_RoundWorkSize((size_t) naa, cg_lws);

  ecode  = clSetKernelArg(k_conj_grad[4], 0, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[4], 1, sizeof(cl_mem), &m_q);
  ecode |= clSetKernelArg(k_conj_grad[4], 2, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[4], 3, sizeof(cl_mem), &m_z);
  ecode |= clSetKernelArg(k_conj_grad[4], 4, sizeof(cl_mem), &m_rho);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_conj_grad[4], 5, sizeof(double) * cg_lws, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_conj_grad[4], 5, sizeof(cl_mem), &m_conj[0]);
  }
  ecode |= clSetKernelArg(k_conj_grad[4], 6, sizeof(double), &alpha);
  ecode |= clSetKernelArg(k_conj_grad[4], 7, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_conj_4);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_conj_grad[4],
                                 1, NULL,
                                 &cg_gws,
                                 &cg_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_conj_4);
  }

  if (timeron) timer_start(T_comm_conj_4);
  ecode = clEnqueueReadBuffer(cmd_queue[0], m_rho,
                              CL_TRUE, 0,
                              rho_size, g_rho,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");
  if (timeron) timer_stop(T_comm_conj_4);

  double rho_temp = 0.0;

  if (timeron) timer_start(T_host_conj_4);
  for (j = 0; j < cg_gws / cg_lws; j++) {
    rho_temp += g_rho[j];
  }
  if (timeron) timer_stop(T_host_conj_4);

  *rho = rho_temp;
}


static void conj_grad_5(double beta)
{
  cl_int ecode;

  size_t cg_lws = CG_LWS[0];
  size_t cg_gws = clu_RoundWorkSize((size_t) naa, cg_lws);

  ecode  = clSetKernelArg(k_conj_grad[5], 0, sizeof(cl_mem), &m_p);
  ecode |= clSetKernelArg(k_conj_grad[5], 1, sizeof(cl_mem), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[5], 2, sizeof(double), &beta);
  ecode |= clSetKernelArg(k_conj_grad[5], 3, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_conj_5);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_conj_grad[5],
                                 1, NULL,
                                 &cg_gws,
                                 &cg_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_conj_5);
  }
}


static void conj_grad_6()
{
  int i, n, blk;
  cl_int ecode;
  size_t cg_lws, cg_gws;

  int i_step = (vec_lst == 0) ? 1 : -1;

  for (i = vec_lst; 0 <= i && i < NUM_TILES; i += i_step) {
    n = vec_ofs[i+1] - vec_ofs[i];
    cg_lws = CG_LWS[2];
    cg_gws = n * cg_lws;

    ecode  = clSetKernelArg(k_conj_grad[6], 0, sizeof(cl_mem), &m_rowstr);
    ecode |= clSetKernelArg(k_conj_grad[6], 1, sizeof(cl_mem), &m_colidx[i%2]);
    ecode |= clSetKernelArg(k_conj_grad[6], 2, sizeof(double), &m_a[i%2]);
    ecode |= clSetKernelArg(k_conj_grad[6], 3, sizeof(double), &m_r);
    ecode |= clSetKernelArg(k_conj_grad[6], 4, sizeof(double), &m_z);
    if (use_local_mem) {
      ecode |= clSetKernelArg(k_conj_grad[6], 5, sizeof(double) * cg_lws, NULL);
    }
    else {
      ecode |= clSetKernelArg(k_conj_grad[6], 5, sizeof(cl_mem), &m_conj[1]);
    }
    ecode |= clSetKernelArg(k_conj_grad[6], 6, sizeof(int), &vec_ofs[i]);
    ecode |= clSetKernelArg(k_conj_grad[6], 7, sizeof(int), &rowstr[vec_ofs[i]]);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (NUM_TILES == 1) {
      if (timeron) timer_start(T_kern_conj_6);
      ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                     k_conj_grad[6],
                                     1, NULL,
                                     &cg_gws,
                                     &cg_lws,
                                     0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timeron) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_kern_conj_6);
      }
    }
    else {
      if (timeron) timer_start(T_kern_conj_6);
      ecode = clEnqueueNDRangeKernel(cmd_queue[1],
                                     k_conj_grad[6],
                                     1, NULL,
                                     &cg_gws,
                                     &cg_lws,
                                     2, write_event[i], NULL);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
      if (timeron) {
        ecode = clFinish(cmd_queue[1]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_kern_conj_6);
      }
    }

    if (0 <= (i+i_step) && (i+i_step) < NUM_TILES) {
      blk = rowstr[vec_ofs[i+i_step+1]] - rowstr[vec_ofs[i+i_step]];

      if (timeron) timer_start(T_comm_conj_6);
      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_colidx[(i+i_step)%2],
                                   CL_FALSE, 0,
                                   blk * sizeof(int),
                                   colidx + rowstr[vec_ofs[i+i_step]],
                                   0, NULL, &write_event[i+i_step][0]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer");

      ecode = clEnqueueWriteBuffer(cmd_queue[0], m_a[(i+i_step)%2],
                                   CL_FALSE, 0,
                                   blk * sizeof(double),
                                   a + rowstr[vec_ofs[i+i_step]],
                                   0, NULL, &write_event[i+i_step][1]);
      clu_CheckError(ecode, "clEnqueueWriteBuffer");
      if (timeron) {
        ecode = clFinish(cmd_queue[0]);
        clu_CheckError(ecode, "clFinish()");
        timer_stop(T_comm_conj_6);
      }

      vec_lst = i+i_step;
    }
  }

  if (NUM_TILES > 1) {
    ecode = clFinish(cmd_queue[1]);
    clu_CheckError(ecode, "clFinish()");
  }
}


static void conj_grad_7(double *sum)
{
  int j;
  cl_int ecode;

  size_t cg_lws = CG_LWS[1];
  size_t cg_gws = clu_RoundWorkSize((size_t) naa, cg_lws);

  ecode  = clSetKernelArg(k_conj_grad[7], 0, sizeof(double), &m_r);
  ecode |= clSetKernelArg(k_conj_grad[7], 1, sizeof(double), &m_x);
  ecode |= clSetKernelArg(k_conj_grad[7], 2, sizeof(double), &m_d);
  if (use_local_mem) {
    ecode |= clSetKernelArg(k_conj_grad[7], 3, sizeof(double) * cg_lws, NULL);
  }
  else {
    ecode |= clSetKernelArg(k_conj_grad[7], 3, sizeof(cl_mem), &m_conj[0]);
  }
  ecode |= clSetKernelArg(k_conj_grad[7], 4, sizeof(int), &naa);
  clu_CheckError(ecode, "clSetKernelArg()");

  if (timeron) timer_start(T_kern_conj_7);
  ecode = clEnqueueNDRangeKernel(cmd_queue[0],
                                 k_conj_grad[7],
                                 1, NULL,
                                 &cg_gws,
                                 &cg_lws,
                                 0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");
  if (timeron) {
    ecode = clFinish(cmd_queue[0]);
    clu_CheckError(ecode, "clFinish()");
    timer_stop(T_kern_conj_7);
  }

  if (timeron) timer_start(T_comm_conj_7);
  ecode = clEnqueueReadBuffer(cmd_queue[0], m_d,
                              CL_TRUE, 0,
                              d_size, g_d,
                              0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueReadBuffer()");
  if (timeron) timer_stop(T_comm_conj_7);

  double sum_temp = 0.0;

  if (timeron) timer_start(T_host_conj_7);
  for (j = 0; j < cg_gws / cg_lws; j++) {
    sum_temp += g_d[j];
  }
  if (timeron) timer_stop(T_host_conj_7);

  *sum = sum_temp;
}


//---------------------------------------------------------------------
// generate the test problem for benchmark 6
// makea generates a sparse matrix with a
// prescribed sparsity distribution
//
// parameter    type        usage
//
// input
//
// n            i           number of cols/rows of matrix
// nz           i           nonzeros as declared array size
// rcond        r*8         condition number
// shift        r*8         main diagonal shift
//
// output
//
// a            r*8         array for nonzeros
// colidx       i           col indices
// rowstr       i           row pointers
//
// workspace
//
// iv, arow, acol i
// aelt           r*8
//---------------------------------------------------------------------
static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER + 1],
                  double aelt[][NONZER + 1],
                  int iv[])
{
  int iouter, ivelt, nzv, nn1;
  int ivc[NONZER + 1];
  double vc[NONZER + 1];

  //---------------------------------------------------------------------
  // nonzer is approximately  (int(sqrt(nnza /n)));
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // nn1 is the smallest power of two not less than n
  //---------------------------------------------------------------------
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  //---------------------------------------------------------------------
  // Generate nonzero positions and save for the use in sparse.
  //---------------------------------------------------------------------
  for (iouter = 0; iouter < n; iouter++) {
    nzv = NONZER;
    sprnvc(n, nzv, nn1, vc, ivc);
    vecset(n, vc, ivc, &nzv, iouter + 1, 0.5);
    arow[iouter] = nzv;

    for (ivelt = 0; ivelt < nzv; ivelt++) {
      acol[iouter][ivelt] = ivc[ivelt] - 1;
      aelt[iouter][ivelt] = vc[ivelt];
    }
  }

  //---------------------------------------------------------------------
  // ... make the sparse matrix from list of elements with duplicates
  //     (v and iv are used as  workspace)
  //---------------------------------------------------------------------
  sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol,
         aelt, firstrow, lastrow,
         iv, RCOND, SHIFT);
}


//---------------------------------------------------------------------
// rows range from firstrow to lastrow
// the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
//---------------------------------------------------------------------
static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER + 1],
                   double aelt[][NONZER + 1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift)
{
  int nrows;

  //---------------------------------------------------
  // generate a sparse matrix from a list of
  // [col, row, element] tri
  //---------------------------------------------------
  int i, j, nzrow, jcol;
  int j1, j2, k, kk, nza;
  double size, scale, ratio, va;
  logical cont40;

  //---------------------------------------------------------------------
  // how many rows of result
  //---------------------------------------------------------------------
  nrows = lastrow - firstrow + 1;

  //---------------------------------------------------------------------
  // ...count the number of triples in each row
  //---------------------------------------------------------------------
  for (j = 0; j < nrows+1; j++) {
    rowstr[j] = 0;
  }

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza] + 1;
      rowstr[j] = rowstr[j] + arow[i];
    }
  }

  rowstr[0] = 0;
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j-1];
  }
  nza = rowstr[nrows] - 1;

  //---------------------------------------------------------------------
  // ... rowstr(j) now is the location of the first nonzero
  //     of row j of a
  //---------------------------------------------------------------------
  if (nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", nza, nz);
    exit(EXIT_FAILURE);
  }

  //---------------------------------------------------------------------
  // ... preload data pages
  //---------------------------------------------------------------------
  for (j = 0; j < nrows; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      a[k] = 0.0;
      colidx[k] = -1;
    }
    nzloc[j] = 0;
  }

  //---------------------------------------------------------------------
  // ... generate actual values by summing duplicates
  //---------------------------------------------------------------------
  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];

      scale = size * aelt[i][nza];
      for (nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = acol[i][nzrow];
        va = aelt[i][nzrow] * scale;

        //--------------------------------------------------------------------
        // ... add the identity * rcond to the generated matrix to bound
        //     the smallest eigenvalue from below by rcond
        //--------------------------------------------------------------------
        if (jcol == j && j == i) {
          va = va + rcond - shift;
        }

        cont40 = false;
        for (k = rowstr[j]; k < rowstr[j+1]; k++) {
          if (colidx[k] > jcol) {
            //----------------------------------------------------------------
            // ... insert colidx here orderly
            //----------------------------------------------------------------
            for (kk = rowstr[j+1]-2; kk >= k; kk--) {
              if (colidx[kk] > -1) {
                a[kk+1]  = a[kk];
                colidx[kk+1] = colidx[kk];
              }
            }
            colidx[k] = jcol;
            a[k]  = 0.0;
            cont40 = true;
            break;
          } else if (colidx[k] == -1) {
            colidx[k] = jcol;
            cont40 = true;
            break;
          } else if (colidx[k] == jcol) {
            //--------------------------------------------------------------
            // ... mark the duplicated entry
            //--------------------------------------------------------------
            nzloc[j] = nzloc[j] + 1;
            cont40 = true;
            break;
          }
        }
        if (cont40 == false) {
          printf("internal error in sparse: i=%d\n", i);
          exit(EXIT_FAILURE);
        }
        a[k] = a[k] + va;
      }
    }
    size = size * ratio;
  }

  //---------------------------------------------------------------------
  // ... remove empty entries and generate final results
  //---------------------------------------------------------------------
  for (j = 1; j < nrows; j++) {
    nzloc[j] = nzloc[j] + nzloc[j-1];
  }

  for (j = 0; j < nrows; j++) {
    if (j > 0) {
      j1 = rowstr[j] - nzloc[j-1];
    } else {
      j1 = 0;
    }
    j2 = rowstr[j+1] - nzloc[j];
    nza = rowstr[j];
    for (k = j1; k < j2; k++) {
      a[k] = a[nza];
      colidx[k] = colidx[nza];
      nza = nza + 1;
    }
  }
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] - nzloc[j-1];
  }
  nza = rowstr[nrows] - 1;
}


//---------------------------------------------------------------------
// generate a sparse n-vector (v, iv)
// having nzv nonzeros
//
// mark(i) is set to 1 if position i is nonzero.
// mark is all zero on entry and is reset to all zero before exit
// this corrects a performance bug found by John G. Lewis, caused by
// reinitialization of mark on every one of the n calls to sprnvc
//---------------------------------------------------------------------
static void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
  int i, ii, nzv = 0;
  double vecelt, vecloc;

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    //---------------------------------------------------------------------
    // generate an integer between 1 and n in a portable manner
    //---------------------------------------------------------------------
    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n)
      continue;

    //---------------------------------------------------------------------
    // was this integer generated already?
    //---------------------------------------------------------------------
    logical was_gen = false;

    for (ii = 0; ii < nzv; ii++) {
      if (iv[ii] == i) {
        was_gen = true;
        break;
      }
    }

    if (was_gen)
      continue;

    v[nzv] = vecelt;
    iv[nzv] = i;
    nzv++;
  }
}


//---------------------------------------------------------------------
// scale a double precision number x in (0,1) by a power of 2 and chop it
//---------------------------------------------------------------------
static int icnvrt(double x, int ipwr2)
{
  return (int) (ipwr2 * x);
}


//---------------------------------------------------------------------
// set ith element of sparse vector (v, iv) with
// nzv nonzeros to val
//---------------------------------------------------------------------
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
  int k;
  logical set = false;

  for (k = 0; k < *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set = true;
    }
  }
  if (!set) {
    v[*nzv] = val;
    iv[*nzv] = i;
    *nzv = *nzv + 1;
  }
}

static void setup(int argc, char *argv[])
{
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
        }
        else if (l == 1) {
          exit(0);
        }
        else if (l == 2) {
          use_local_mem = true;
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

  setup_timers();
}

static void setup_timers()
{
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[T_init] = "init";
    t_names[T_bench] = "benchmk";

    t_names[T_kern_main_0] = "kern main 0";
    t_names[T_comm_main_0] = "comm main 0";
    t_names[T_host_main_0] = "host main 0";

    t_names[T_kern_main_1] = "kern main 1";

    t_names[T_kern_conj_0] = "kern conj 0";

    t_names[T_kern_conj_1] = "kern conj 1";
    t_names[T_comm_conj_1] = "comm conj 1";
    t_names[T_host_conj_1] = "host conj 1";

    t_names[T_kern_conj_2] = "kern conj 2";
    t_names[T_comm_conj_2] = "comm conj 2";

    t_names[T_kern_conj_3] = "kern conj 3";
    t_names[T_comm_conj_3] = "comm conj 3";
    t_names[T_host_conj_3] = "host conj 3";

    t_names[T_kern_conj_4] = "kern conj 4";
    t_names[T_comm_conj_4] = "comm conj 4";
    t_names[T_host_conj_4] = "host conj 4";

    t_names[T_kern_conj_5] = "kern conj 5";

    t_names[T_kern_conj_6] = "kern conj 6";
    t_names[T_comm_conj_6] = "comm conj 6";

    t_names[T_kern_conj_7] = "kern conj 7";
    t_names[T_comm_conj_7] = "comm conj 7";
    t_names[T_host_conj_7] = "host conj 7";

    fclose(fp);
  }
  else {
    timeron = false;
  }

  int i;
  for (i = 0; i < T_last; i++) {
    timer_clear(i);
  }
}

static void setup_opencl(int argc, char *argv[], char Class)
{
  clu_ProfilerSetup();

  int i;
  cl_int ecode;

  //-----------------------------------------------------------------------
  // 1. Find all available GPU devices and get a device
  //-----------------------------------------------------------------------
  device_type = CL_DEVICE_TYPE_GPU;

  ecode = clGetPlatformIDs(1, &platform, NULL);
  clu_CheckError(ecode, "clGetPlatformIDs()");

  unsigned NUM_DEVICE = 1;
  cl_device_id devices[MAX_DEVICE];

  ecode = clGetDeviceIDs(platform, device_type, 0, NULL, &NUM_DEVICE);
  clu_CheckError(ecode, "clGetDeviceIDs()");

  ecode = clGetDeviceIDs(platform, device_type, NUM_DEVICE, devices, NULL);
  clu_CheckError(ecode, "clGetDeviceIDs()");

  device = devices[0];
  device_name = clu_GetDeviceName(device);

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

  while (max_mem_alloc_size * NUM_TILES < NZ * sizeof(int)) {
    NUM_TILES <<= 1;
  }

  while (global_mem_size * NUM_TILES < NZ * sizeof(double)) {
    NUM_TILES <<= 1;
  }

  //fprintf(stderr, " NUM_TILES = %d\n", (int) NUM_TILES);
  //fprintf(stderr, " Global memory size: %d MB\n", (int) (global_mem_size >> 20));
  //fprintf(stderr, " Local memory size: %d KB\n", (int) (local_mem_size >> 10));

  // 2. Create a context for the specified device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  // 3. Create a command queue
  cmd_queue[0] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");

  cmd_queue[1] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");

  // 4. Build the program
  char *source_file;
  char build_option[100];

  if (device_type == CL_DEVICE_TYPE_GPU) {
    MAIN_LWS[0] = 128;
    MAIN_LWS[1] = 128;

    CG_LWS[0] = 128;
    CG_LWS[1] = 128;
    CG_LWS[2] = (Class < 'D') ? 64 : 128;

    source_file = (use_local_mem) ? "cg_gpu_opt.cl" : "cg_gpu_base.cl";
    sprintf(build_option, "-I. -cl-mad-enable");
  }
  else {
    fprintf(stderr, "%s: not supported.", clu_GetDeviceTypeName(device_type));
    exit(EXIT_FAILURE);
  }

  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);

  // 5. Create kernels
  char kname[50];

  for (i = 0; i < NUM_K_MAIN; i++) {
    sprintf(kname, "main_%d", i);
    k_main[i] = clCreateKernel(program, kname, &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }

  for (i = 0; i < NUM_K_CONJ_GRAD; i++) {
    sprintf(kname, "conj_grad_%d", i);
    k_conj_grad[i] = clCreateKernel(program, kname, &ecode);
    clu_CheckError(ecode, "clCreateKernel()");
  }

  // 6. Create buffers
  int blk, max_blk, trg_val, min_idx, mid_idx, max_idx;

  vec_ofs[0] = 0;
  max_blk = 0;

  for (i = 0; i < NUM_TILES; i++) {
    trg_val = (NZ - rowstr[vec_ofs[i]])
            / (NUM_TILES - i) + rowstr[vec_ofs[i]];

    min_idx = vec_ofs[i];
    max_idx = NA;
    mid_idx = (min_idx + max_idx) / 2;

    while (min_idx < max_idx) {
      if (rowstr[mid_idx] < trg_val) {
        min_idx = mid_idx + 1;
      }
      else {
        max_idx = mid_idx;
      }

      mid_idx = (min_idx + max_idx) / 2;
    }

    vec_ofs[i+1] = mid_idx;

    blk = rowstr[vec_ofs[i+1]] - rowstr[vec_ofs[i]];
    max_blk = (blk > max_blk) ? blk : max_blk;
  }

  m_colidx[0] = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               max_blk * sizeof(int),
                               NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_colidx");

  m_colidx[1] = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               max_blk * sizeof(int),
                               NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_colidx");

  m_a[0] = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          max_blk * sizeof(double),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_a");

  m_a[1] = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          max_blk * sizeof(double),
                          NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_a");

  m_rowstr = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            (NA + 1) * sizeof(int),
                            NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rowstr");

  m_x = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA + 2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_x");

  m_z = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA + 2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_z");

  m_p = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA + 2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_p");

  m_q = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA + 2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_q");

  m_r = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       (NA + 2) * sizeof(double),
                       NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_r");

  norm_temp_size = ((NA + MAIN_LWS[0] - 1) / MAIN_LWS[0]) * sizeof(double);
  rho_size = ((NA + CG_LWS[1] - 1) / CG_LWS[1]) * sizeof(double);
  d_size = ((NA + CG_LWS[1] - 1) / CG_LWS[1]) * sizeof(double);

  g_norm_temp1 = (double *) malloc(norm_temp_size);
  g_norm_temp2 = (double *) malloc(norm_temp_size);
  g_rho = (double *) malloc(rho_size);
  g_d = (double *) malloc(d_size);

  m_norm_temp1 = clCreateBuffer(context,
                                CL_MEM_READ_WRITE,
                                norm_temp_size,
                                NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_norm_temp1");

  m_norm_temp2 = clCreateBuffer(context,
                                CL_MEM_READ_WRITE,
                                norm_temp_size,
                                NULL, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_norm_temp2");

  m_rho = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         rho_size,
                         0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_rho");

  m_d = clCreateBuffer(context,
                       CL_MEM_READ_WRITE,
                       d_size,
                       0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_d");

  m_main[0] = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             (naa + MAIN_LWS[0]) * sizeof(double),
                             0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_main[0]");

  m_main[1] = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             (naa + MAIN_LWS[0]) * sizeof(double),
                             0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_main[1]");

  m_conj[0] = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             (naa + CG_LWS[1]) * sizeof(double),
                             0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_conj[0]");

  m_conj[1] = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             naa * CG_LWS[2] * sizeof(double),
                             0, &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_conj[1]");

  // 7. Initialize buffers
  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_x,
                               CL_FALSE, 0,
                               (NA + 2) * sizeof(double), x,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");

  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_rowstr,
                               CL_FALSE, 0,
                               (NA + 1) * sizeof(int), rowstr,
                               0, NULL, NULL);
  clu_CheckError(ecode, "clEnqueueWriteBuffer()");

  blk = rowstr[vec_ofs[1]] - rowstr[vec_ofs[0]];

  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_colidx[0],
                               CL_FALSE, 0,
                               blk * sizeof(int), colidx,
                               0, NULL, &write_event[0][0]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer");

  ecode = clEnqueueWriteBuffer(cmd_queue[0], m_a[0],
                               CL_TRUE, 0,
                               blk * sizeof(double), a,
                               0, NULL, &write_event[0][1]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer");
}

static void release_opencl()
{
  clReleaseMemObject(m_colidx[0]);
  clReleaseMemObject(m_colidx[1]);
  clReleaseMemObject(m_rowstr);
  clReleaseMemObject(m_a[0]);
  clReleaseMemObject(m_a[1]);
  clReleaseMemObject(m_q);
  clReleaseMemObject(m_z);
  clReleaseMemObject(m_r);
  clReleaseMemObject(m_p);
  clReleaseMemObject(m_x);

  clReleaseMemObject(m_norm_temp1);
  clReleaseMemObject(m_norm_temp2);
  clReleaseMemObject(m_rho);
  clReleaseMemObject(m_d);

  clReleaseMemObject(m_main[0]);
  clReleaseMemObject(m_main[1]);
  clReleaseMemObject(m_conj[0]);
  clReleaseMemObject(m_conj[1]);

  free(g_norm_temp1);
  free(g_norm_temp2);
  free(g_rho);
  free(g_d);

  int i;
  for (i = 0; i < NUM_K_MAIN; i++) {
    clReleaseKernel(k_main[i]);
  }
  for (i = 0; i < NUM_K_CONJ_GRAD; i++) {
    clReleaseKernel(k_conj_grad[i]);
  }

  clReleaseProgram(program);
  clReleaseCommandQueue(cmd_queue[0]);
  clReleaseCommandQueue(cmd_queue[1]);
  clReleaseContext(context);

  clu_ProfilerRelease();
}
