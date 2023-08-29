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
// program cg
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
extern "C" {
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
}

#include "cuda_util.h"


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
//---------------------------------------------------------------------


//---------------------------------------------------------------------
#define BUFFERING 2
static cudaStream_t cmd_queue[BUFFERING];

static size_t work_item_sizes[3];
static size_t max_work_group_size;
static size_t max_mem_alloc_size;;
static size_t global_mem_size;
static size_t shared_mem_size;

#define NUM_K_MAIN 2
#define NUM_K_CONJ_GRAD 8

#define MAX_PARTITION 32
static size_t NUM_TILES = 1;
static int vec_lst;
static int vec_ofs[MAX_PARTITION+1];

static cudaEvent_t write_event[MAX_PARTITION][BUFFERING];

static logical use_shared_mem = true;

/* common / main_int_mem / */
static int *m_colidx[BUFFERING];
static int *m_rowstr;

/* common / main_flt_mem / */
static double *m_a[BUFFERING];
static double *m_x;
static double *m_z;
static double *m_p;
static double *m_q;
static double *m_r;

static double *m_norm_temp1;
static double *m_norm_temp2;
static double *m_rho;
static double *m_d;

static double *m_main[2];
static double *m_conj[2];

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

static void setup_cuda(int argc, char *argv[], char Class);

static void release_cuda();
//---------------------------------------------------------------------


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

  int gws;
  int blk;
  size_t main_lws[NUM_K_MAIN], main_gws[NUM_K_MAIN];

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

  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-CUDA) - CG Benchmark\n\n");
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

  setup_cuda(argc, argv, Class);

  CUCHK(cudaMemcpyAsync(m_x, x, (naa + 2) * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[0]));

  CUCHK(cudaMemcpyAsync(m_rowstr, rowstr, (naa + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, cmd_queue[0]));

  blk = rowstr[vec_ofs[1]] - rowstr[vec_ofs[0]];

  CUCHK(cudaMemcpyAsync(m_colidx[0], colidx, blk * sizeof(int),
                        cudaMemcpyHostToDevice, cmd_queue[0]));
  CUCHK(cudaEventRecord(write_event[0][0], cmd_queue[0]));

  CUCHK(cudaMemcpyAsync(m_a[0], a, blk * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[0]));
  CUCHK(cudaEventRecord(write_event[0][1], cmd_queue[0]));

  CUCHK(cudaStreamSynchronize(cmd_queue[0]));

  timer_stop(T_init);

  printf(" Initialization time = %15.6f seconds\n", timer_read(T_init));

  timer_start(T_bench);
  cuda_ProfilerStart();

  vec_lst = 0;

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
    gws = lastcol - firstcol + 1;
    main_lws[0] = MAIN_LWS[0];
    main_gws[0] = RoundWorkSize((size_t) naa, main_lws[0]);

    if (timeron) timer_start(T_kern_main_0);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("main_0_opt", cmd_queue[0] );
      main_0_opt<<< main_gws[0] / main_lws[0],
                    main_lws[0],
                    sizeof(double) * main_lws[0] * 2,
                    cmd_queue[0] >>>
            (m_x, m_z, m_norm_temp1, m_norm_temp2, gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("main_0_opt", cmd_queue[0] );
    }
    else {
      cuda_ProfilerStartEventRecord("main_0_base", cmd_queue[0] );
      main_0_base<<< main_gws[0] / main_lws[0],
                     main_lws[0],
                     0,
                     cmd_queue[0] >>>
            (m_x, m_z, m_norm_temp1, m_norm_temp2, m_main[0], m_main[1], gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("main_0_base", cmd_queue[0] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
      timer_stop(T_kern_main_0);
    }

    if (timeron) timer_start(T_comm_main_0);
    CUCHK(cudaMemcpyAsync(g_norm_temp1, m_norm_temp1, norm_temp_size,
                          cudaMemcpyDeviceToHost, cmd_queue[0]));
    CUCHK(cudaMemcpyAsync(g_norm_temp2, m_norm_temp2, norm_temp_size,
                          cudaMemcpyDeviceToHost, cmd_queue[0]));
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    if (timeron) timer_stop(T_comm_main_0);

    norm_temp1 = 0.0;
    norm_temp2 = 0.0;

    if (timeron) timer_start(T_host_main_0);
    for (j = 0; j < main_gws[0] / main_lws[0]; j++) {
      norm_temp1 += g_norm_temp1[j];
      norm_temp2 += g_norm_temp2[j];
    }
    if (timeron) timer_stop(T_host_main_0);

    norm_temp2 = 1.0 / sqrt(norm_temp2);
    zeta = SHIFT + 1.0 / norm_temp1;

    if (it == 1)
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    if (it < NITER) {
      gws = lastcol - firstcol + 1;

      main_lws[1] = MAIN_LWS[1];
      main_gws[1] = RoundWorkSize((size_t) gws, main_lws[1]);

      if (timeron) timer_start(T_kern_main_1);
      cuda_ProfilerStartEventRecord("main_1", cmd_queue[0] );
      main_1<<< main_gws[1] / main_lws[1],
                main_lws[1], 0, cmd_queue[0] >>>
            (m_x, m_z, norm_temp2, gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("main_1", cmd_queue[0] );
      if (timeron) {
        CUCHK(cudaStreamSynchronize(cmd_queue[0]));
        timer_stop(T_kern_main_1);
      }
    }
  } // end of main iter inv pow meth

  cuda_ProfilerStop();
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
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13E\n", zeta);
      printf(" The correct zeta is %20.13E\n", zeta_verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (t != 0.0) {
    mflops = (double)(2*NITER*NA)
                   * (3.0+(double)(NONZER*(NONZER+1))
                     + 25.0*(5.0+(double)(NONZER*(NONZER+1)))
                     + 3.0) / t / 1000000.0;
  } else {
    mflops = 0.0;
  }


  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  c_print_results("CG", Class, NA, 0, 0,
                  NITER, t,
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
    for (i = 0; i < T_last; i++) {
      t = timer_read(i);
      if (i != T_init) {
        printf("  %8s:%12.6f\t(%6.3f%%)\n", t_names[i], t, t*100.0/tmax);
      }
    }
    fflush(stdout);
    
    cuda_ProfilerPrintResult();
  }

  release_cuda();

  return 0;
}


//---------------------------------------------------------------------
// Floaging point arrays here are named as in NPB1 spec discussion of
// CG algorithm
//---------------------------------------------------------------------
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
  int i, i_step, j, cgit, cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta;

  int gws, blk;
  size_t cg_lws[NUM_K_CONJ_GRAD], cg_gws[NUM_K_CONJ_GRAD];

  rho = 0.0;
  sum = 0.0;

  //---------------------------------------------------------------------
  // Initialize the CG algorithm:
  //---------------------------------------------------------------------
  gws = naa + 1;
  cg_lws[0] = CG_LWS[0];
  cg_gws[0] = RoundWorkSize((size_t) gws, cg_lws[0]);

  if (timeron) timer_start(T_kern_conj_0);
  cuda_ProfilerStartEventRecord("conj_grad_0", cmd_queue[0] );
  conj_grad_0<<< cg_gws[0] / cg_lws[0], cg_lws[0], 0, cmd_queue[0] >>>
             (m_q, m_z, m_r, m_x, m_p, gws);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("conj_grad_0", cmd_queue[0] );
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    timer_stop(T_kern_conj_0);
  }

  //---------------------------------------------------------------------
  // rho = r.r
  // Now, obtain the norm of r: First, sum squares of r elements locally...
  //---------------------------------------------------------------------
  gws = lastcol - firstcol + 1;
  cg_lws[1] = CG_LWS[1];
  cg_gws[1] = RoundWorkSize((size_t) gws, cg_lws[1]);

  if (timeron) timer_start(T_kern_conj_1);
  if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("conj_grad_1_opt", cmd_queue[0] );
    conj_grad_1_opt<<< cg_gws[1] / cg_lws[1],
                       cg_lws[1],
                       sizeof(double) * cg_lws[1],
                       cmd_queue[0] >>>
               (m_r, m_rho, gws);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("conj_grad_1_opt", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("conj_grad_1_base", cmd_queue[0] );
    conj_grad_1_base<<< cg_gws[1] / cg_lws[1],
                        cg_lws[1],
                        0,
                        cmd_queue[0] >>>
               (m_r, m_rho, m_conj[0], gws);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("conj_grad_1_base", cmd_queue[0] );
  }
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    timer_stop(T_kern_conj_1);
  }

  if (timeron) timer_start(T_comm_conj_1);
  CUCHK(cudaMemcpyAsync(g_rho, m_rho, rho_size, cudaMemcpyDeviceToHost, cmd_queue[0]));
  CUCHK(cudaStreamSynchronize(cmd_queue[0]));
  if (timeron) timer_stop(T_comm_conj_1);

  if (timeron) timer_start(T_host_conj_1);
  for (j = 0; j < cg_gws[1] / cg_lws[1]; j++) {
    rho += g_rho[j];
  }
  if (timeron) timer_stop(T_host_conj_1);

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
    rho = 0.0;
    d = 0.0;

    //---------------------------------------------------------------------
    // q = A.p
    // The partition submatrix-vector multiply: use workspace w
    //---------------------------------------------------------------------
    i_step = (vec_lst == 0) ? 1 : -1;

    for (i = vec_lst; 0 <= i && i < NUM_TILES; i += i_step) {
      gws = vec_ofs[i+1] - vec_ofs[i];
      cg_lws[2] = CG_LWS[2];
      cg_gws[2] = gws * cg_lws[2];

      if (NUM_TILES == 1) {
        if (timeron) timer_start(T_kern_conj_2);
        if (use_shared_mem) {
          cuda_ProfilerStartEventRecord("conj_grad_2_opt", cmd_queue[0] );
          conj_grad_2_opt<<< cg_gws[2] / cg_lws[2],
                             cg_lws[2],
                             sizeof(double) * cg_lws[2],
                             cmd_queue[0] >>>
                     (m_rowstr, m_colidx[i%2], m_a[i%2], m_p, m_q,
                      vec_ofs[i], rowstr[vec_ofs[i]]);
          CUCHK(cudaGetLastError());
          cuda_ProfilerEndEventRecord("conj_grad_2_opt", cmd_queue[0] );
        }
        else {
          cuda_ProfilerStartEventRecord("conj_grad_2_base", cmd_queue[0] );
          conj_grad_2_base<<< cg_gws[2] / cg_lws[2],
                              cg_lws[2],
                              0,
                              cmd_queue[0] >>>
                     (m_rowstr, m_colidx[i%2], m_a[i%2], m_p, m_q, m_conj[1],
                      vec_ofs[i], rowstr[vec_ofs[i]]);
          CUCHK(cudaGetLastError());
          cuda_ProfilerEndEventRecord("conj_grad_2_base", cmd_queue[0] );
        }
        if (timeron) {
          CUCHK(cudaStreamSynchronize(cmd_queue[0]));
          timer_stop(T_kern_conj_2);
        }
      }
      else {
        CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i][0], 0));
        CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i][1], 0));
        if (timeron) timer_start(T_kern_conj_2);
        if (use_shared_mem) {
          cuda_ProfilerStartEventRecord("conj_grad_2_opt", cmd_queue[1] );
          conj_grad_2_opt<<< cg_gws[2] / cg_lws[2],
                             cg_lws[2],
                             sizeof(double) * cg_lws[2],
                             cmd_queue[1] >>>
                     (m_rowstr, m_colidx[i%2], m_a[i%2], m_p, m_q,
                      vec_ofs[i], rowstr[vec_ofs[i]]);
          CUCHK(cudaGetLastError());
          cuda_ProfilerEndEventRecord("conj_grad_2_opt", cmd_queue[1] );
        }
        else {
          cuda_ProfilerStartEventRecord("conj_grad_2_base", cmd_queue[1] );
          conj_grad_2_base<<< cg_gws[2] / cg_lws[2],
                              cg_lws[2],
                              0,
                              cmd_queue[1] >>>
                     (m_rowstr, m_colidx[i%2], m_a[i%2], m_p, m_q, m_conj[1],
                      vec_ofs[i], rowstr[vec_ofs[i]]);
          CUCHK(cudaGetLastError());
          cuda_ProfilerEndEventRecord("conj_grad_2_base", cmd_queue[1] );
        }
        if (timeron) {
          CUCHK(cudaStreamSynchronize(cmd_queue[1]));
          timer_stop(T_kern_conj_2);
        }
      }

      if (0 <= (i+i_step) && (i+i_step) < NUM_TILES) {
        blk = rowstr[vec_ofs[i+i_step+1]] - rowstr[vec_ofs[i+i_step]];

        if (timeron) timer_start(T_comm_conj_2);
        CUCHK(cudaMemcpyAsync(m_colidx[(i+i_step)%2],
            colidx + rowstr[vec_ofs[i+i_step]],
            blk * sizeof(int),
            cudaMemcpyHostToDevice, cmd_queue[0]));
        CUCHK(cudaEventRecord(write_event[i+i_step][0]));

        CUCHK(cudaMemcpyAsync(m_a[(i+i_step)%2],
            a + rowstr[vec_ofs[i+i_step]],
            blk * sizeof(double),
            cudaMemcpyHostToDevice, cmd_queue[0]));
        CUCHK(cudaEventRecord(write_event[i+i_step][1]));
        if (timeron) {
          CUCHK(cudaStreamSynchronize(cmd_queue[0]));
          timer_stop(T_comm_conj_2);
        }

        vec_lst = i+i_step;
      }
    }

    if (NUM_TILES > 1) {
      CUCHK(cudaStreamSynchronize(cmd_queue[1]));
    }

    //---------------------------------------------------------------------
    // Obtain p.q
    //---------------------------------------------------------------------
    gws = lastcol - firstcol + 1;
    cg_lws[3] = CG_LWS[1];
    cg_gws[3] = RoundWorkSize((size_t) gws, cg_lws[3]);

    if (timeron) timer_start(T_kern_conj_3);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("conj_grad_3_opt", cmd_queue[0] );
      conj_grad_3_opt<<< cg_gws[3] / cg_lws[3],
                         cg_lws[3],
                         sizeof(double) * cg_lws[3],
                         cmd_queue[0] >>>
                 (m_p, m_q, m_d, gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("conj_grad_3_opt", cmd_queue[0] );
    }
    else {
      cuda_ProfilerStartEventRecord("conj_grad_3_base", cmd_queue[0] );
      conj_grad_3_base<<< cg_gws[3] / cg_lws[3],
                          cg_lws[3],
                          0,
                          cmd_queue[0] >>>
                 (m_p, m_q, m_d, m_conj[0], gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("conj_grad_3_base", cmd_queue[0] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
      timer_stop(T_kern_conj_3);
    }

    CUCHK(cudaMemcpyAsync(g_d, m_d, d_size, cudaMemcpyDeviceToHost, cmd_queue[0]));
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));

    if (timeron) timer_start(T_host_conj_3);
    for (j = 0; j < cg_gws[3] / cg_lws[3]; j++) {
      d += g_d[j];
    }
    if (timeron) timer_stop(T_host_conj_3);

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
    gws = lastcol - firstcol + 1;
    cg_lws[4] = CG_LWS[1];
    cg_gws[4] = RoundWorkSize((size_t) gws, cg_lws[4]);

    if (timeron) timer_start(T_kern_conj_4);
    if (use_shared_mem) {
      cuda_ProfilerStartEventRecord("conj_grad_4_opt", cmd_queue[0] );
      conj_grad_4_opt<<< cg_gws[4] / cg_lws[4],
                         cg_lws[4],
                         sizeof(double) * cg_lws[4],
                         cmd_queue[0] >>>
                 (m_p, m_q, m_r, m_z, m_rho, alpha, gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("conj_grad_4_opt", cmd_queue[0] );
    }
    else {
      cuda_ProfilerStartEventRecord("conj_grad_4_base", cmd_queue[0] );
      conj_grad_4_base<<< cg_gws[4] / cg_lws[4],
                          cg_lws[4],
                          0,
                          cmd_queue[0] >>>
                 (m_p, m_q, m_r, m_z, m_rho, m_conj[0], alpha, gws);
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("conj_grad_4_base", cmd_queue[0] );
    }
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
      timer_stop(T_kern_conj_4);
    }

    if (timeron) timer_start(T_comm_conj_4);
    CUCHK(cudaMemcpyAsync(g_rho, m_rho, rho_size, cudaMemcpyDeviceToHost, cmd_queue[0]));
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    if (timeron) timer_stop(T_comm_conj_4);

    if (timeron) timer_start(T_host_conj_4);
    for (j = 0; j < cg_gws[4] / cg_lws[4]; j++) {
      rho += g_rho[j];
    }
    if (timeron) timer_stop(T_host_conj_4);

    //---------------------------------------------------------------------
    // Obtain beta:
    //---------------------------------------------------------------------
    beta = rho / rho0;

    //---------------------------------------------------------------------
    // p = r + beta*p
    //---------------------------------------------------------------------
    gws = lastcol - firstcol + 1;
    cg_lws[5] = CG_LWS[0];
    cg_gws[5] = RoundWorkSize((size_t) gws, cg_lws[5]);

  if (timeron) timer_start(T_kern_conj_5);
  cuda_ProfilerStartEventRecord("conj_grad_5", cmd_queue[0] );
  conj_grad_5<<< cg_gws[5] / cg_lws[5], cg_lws[5], 0, cmd_queue[0] >>>
               (m_p, m_r, beta, gws);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("conj_grad_5", cmd_queue[0] );
    if (timeron) {
      CUCHK(cudaStreamSynchronize(cmd_queue[0]));
      timer_stop(T_kern_conj_5);
    }
  } // end of do cgit=1,cgitmax

  //---------------------------------------------------------------------
  // Compute residual norm explicitly:  ||r|| = ||x - A.z||
  // First, form A.z
  // The partition submatrix-vector multiply
  //---------------------------------------------------------------------
  i_step = (vec_lst == 0) ? 1 : -1;

  for (i = vec_lst; 0 <= i && i < NUM_TILES; i += i_step) {
    gws = vec_ofs[i+1] - vec_ofs[i];
    cg_lws[6] = CG_LWS[2];
    cg_gws[6] = gws * cg_lws[6];

    if (NUM_TILES == 1) {
      if (timeron) timer_start(T_kern_conj_6);
      if (use_shared_mem) {
        cuda_ProfilerStartEventRecord("conj_grad_6_opt", cmd_queue[0] );
        conj_grad_6_opt<<< cg_gws[6] / cg_lws[6],
                           cg_lws[6],
                           sizeof(double) * cg_lws[6],
                           cmd_queue[0] >>>
                   (m_rowstr, m_colidx[i%2], m_a[i%2], m_r, m_z,
                    vec_ofs[i], rowstr[vec_ofs[i]]);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("conj_grad_6_opt", cmd_queue[0] );
      }
      else {
        cuda_ProfilerStartEventRecord("conj_grad_6_base", cmd_queue[0] );
        conj_grad_6_base<<< cg_gws[6] / cg_lws[6],
                            cg_lws[6],
                            0,
                            cmd_queue[0] >>>
                   (m_rowstr, m_colidx[i%2], m_a[i%2], m_r, m_z, m_conj[1],
                    vec_ofs[i], rowstr[vec_ofs[i]]);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("conj_grad_6_base", cmd_queue[0] );
      }
      if (timeron) {
        CUCHK(cudaStreamSynchronize(cmd_queue[0]));
        timer_stop(T_kern_conj_6);
      }
    }
    else {
      CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i][0], 0));
      CUCHK(cudaStreamWaitEvent(cmd_queue[1], write_event[i][1], 0));
      if (timeron) timer_start(T_kern_conj_6);
      if (use_shared_mem) {
        cuda_ProfilerStartEventRecord("conj_grad_6_opt", cmd_queue[1] );
        conj_grad_6_opt<<< cg_gws[6] / cg_lws[6],
                           cg_lws[6],
                           sizeof(double) * cg_lws[6],
                           cmd_queue[1] >>>
                   (m_rowstr, m_colidx[i%2], m_a[i%2], m_r, m_z,
                    vec_ofs[i], rowstr[vec_ofs[i]]);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("conj_grad_6_opt", cmd_queue[1] );
      }
      else {
        cuda_ProfilerStartEventRecord("conj_grad_6_base", cmd_queue[1] );
        conj_grad_6_base<<< cg_gws[6] / cg_lws[6],
                            cg_lws[6],
                            0,
                            cmd_queue[1] >>>
                   (m_rowstr, m_colidx[i%2], m_a[i%2], m_r, m_z, m_conj[1],
                    vec_ofs[i], rowstr[vec_ofs[i]]);
        CUCHK(cudaGetLastError());
        cuda_ProfilerEndEventRecord("conj_grad_6_base", cmd_queue[1] );
      }
      if (timeron) {
        CUCHK(cudaStreamSynchronize(cmd_queue[1]));
        timer_stop(T_kern_conj_6);
      }
    }

    if (0 <= (i+i_step) && (i+i_step) < NUM_TILES) {
      blk = rowstr[vec_ofs[i+i_step+1]] - rowstr[vec_ofs[i+i_step]];

      if (timeron) timer_start(T_comm_conj_6);
      CUCHK(cudaMemcpyAsync(m_colidx[(i+i_step)%2], colidx + rowstr[vec_ofs[i+i_step]],
                            blk * sizeof(int),
                            cudaMemcpyHostToDevice, cmd_queue[0]));
      CUCHK(cudaEventRecord(write_event[i+i_step][0], cmd_queue[0]));

      CUCHK(cudaMemcpyAsync(m_a[(i+i_step)%2], a + rowstr[vec_ofs[i+i_step]],
                            blk * sizeof(double),
                            cudaMemcpyHostToDevice, cmd_queue[0]));
      CUCHK(cudaEventRecord(write_event[i+i_step][1], cmd_queue[0]));
      if (timeron) {
        CUCHK(cudaStreamSynchronize(cmd_queue[0]));
        timer_stop(T_comm_conj_6);
      }

      vec_lst = i+i_step;
    }
  }

  if (NUM_TILES > 1) {
    CUCHK(cudaStreamSynchronize(cmd_queue[1]));
  }

  //---------------------------------------------------------------------
  // At this point, r contains A.z
  //---------------------------------------------------------------------
  sum = 0.0;

  gws = lastcol - firstcol + 1;
  cg_lws[7] = CG_LWS[1];
  cg_gws[7] = RoundWorkSize((size_t) gws, cg_lws[7]);

  if (timeron) timer_start(T_kern_conj_7);
  if (use_shared_mem) {
    cuda_ProfilerStartEventRecord("conj_grad_7_opt", cmd_queue[0] );
    conj_grad_7_opt<<< cg_gws[7] / cg_lws[7],
                       cg_lws[7],
                       sizeof(double) * cg_lws[7],
                       cmd_queue[0] >>>
               (m_r, m_x, m_d, gws);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("conj_grad_7_opt", cmd_queue[0] );
  }
  else {
    cuda_ProfilerStartEventRecord("conj_grad_7_base", cmd_queue[0] );
    conj_grad_7_base<<< cg_gws[7] / cg_lws[7],
                        cg_lws[7],
                        0,
                        cmd_queue[0] >>>
               (m_r, m_x, m_d, m_conj[0], gws);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("conj_grad_7_base", cmd_queue[0] );
  }
  if (timeron) {
    CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    timer_stop(T_kern_conj_7);
  }

  CUCHK(cudaMemcpyAsync(g_d, m_d, d_size,
                        cudaMemcpyDeviceToHost, cmd_queue[0]));
  CUCHK(cudaStreamSynchronize(cmd_queue[0]));

  if (timeron) timer_start(T_host_conj_7);
  for (j = 0; j < cg_gws[7] / cg_lws[7];j++) {
    sum += g_d[j];
  }
  if (timeron) timer_stop(T_host_conj_7);

  *rnorm = sqrt(sum);
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
    printf("nza, nzmax = %lld, %lld\n", nza, nz);
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

static void setup_cuda(int argc, char *argv[], char Class)
{
  cuda_ProfilerSetup();

  int i;

  // 1. Find the default device type and get a device for the device type
  int device;
  CUCHK(cudaGetDevice(&device));

  cudaDeviceProp deviceProp;
  CUCHK(cudaGetDeviceProperties(&deviceProp, device));

  work_item_sizes[0] = deviceProp.maxThreadsDim[0];
  work_item_sizes[1] = deviceProp.maxThreadsDim[1];
  work_item_sizes[2] = deviceProp.maxThreadsDim[2];
  max_work_group_size = deviceProp.maxThreadsPerBlock;
  max_mem_alloc_size = deviceProp.totalGlobalMem / 4;
  global_mem_size = deviceProp.totalGlobalMem;
  shared_mem_size = deviceProp.sharedMemPerBlock;

  while (max_mem_alloc_size * NUM_TILES < NZ * sizeof(int)) {
    NUM_TILES <<= 1;
  }

  while (global_mem_size * NUM_TILES < NZ * sizeof(double)) {
    NUM_TILES <<= 1;
  }

  //fprintf(stderr, " NUM_TILES = %d\n", (int) NUM_TILES);
  //fprintf(stderr, " Global memory size: %d MB\n", (int) (global_mem_size >> 20));
  //fprintf(stderr, " Shared memory size: %d KB\n", (int) (shared_mem_size >> 10));

  // 3. Create a command queue
  CUCHK(cudaStreamCreate(&cmd_queue[0]));
  CUCHK(cudaStreamCreate(&cmd_queue[1]));

  MAIN_LWS[0] = 128;
  MAIN_LWS[1] = 128;

  CG_LWS[0] = 128;
  CG_LWS[1] = 128;
  CG_LWS[2] = (Class < 'D') ? 64 : 128;

  // 6. Create buffers
  int blk, max_blk, trg_val, min_idx, mid_idx, max_idx;

  vec_ofs[0] = 0;
  max_blk = 0;

  for (i = 0; i < NUM_TILES; i++) {
    trg_val = (NZ - rowstr[vec_ofs[i]]) / (NUM_TILES - i) + rowstr[vec_ofs[i]];

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

  CUCHK(cudaMalloc(&m_colidx[0], max_blk * sizeof(int)));
  CUCHK(cudaMalloc(&m_a[0], max_blk * sizeof(double)));
  if (NUM_TILES > 1) {
    CUCHK(cudaMalloc(&m_colidx[1], max_blk * sizeof(int)));
    CUCHK(cudaMalloc(&m_a[1], max_blk * sizeof(double)));
  }
  CUCHK(cudaMalloc(&m_rowstr, (NA + 1) * sizeof(int)));
  CUCHK(cudaMalloc(&m_x, (NA + 2) * sizeof(double)));
  CUCHK(cudaMalloc(&m_z, (NA + 2) * sizeof(double)));
  CUCHK(cudaMalloc(&m_p, (NA + 2) * sizeof(double)));
  CUCHK(cudaMalloc(&m_q, (NA + 2) * sizeof(double)));
  CUCHK(cudaMalloc(&m_r, (NA + 2) * sizeof(double)));

  norm_temp_size = ((NA + MAIN_LWS[0] - 1) / MAIN_LWS[0]) * sizeof(double);
  rho_size = ((NA + CG_LWS[1] - 1) / CG_LWS[1]) * sizeof(double);
  d_size = ((NA + CG_LWS[1] - 1) / CG_LWS[1]) * sizeof(double);

  g_norm_temp1 = (double *) malloc(norm_temp_size);
  g_norm_temp2 = (double *) malloc(norm_temp_size);
  g_rho = (double *) malloc(rho_size);
  g_d = (double *) malloc(d_size);

  CUCHK(cudaMalloc(&m_norm_temp1, norm_temp_size));
  CUCHK(cudaMalloc(&m_norm_temp2, norm_temp_size));
  CUCHK(cudaMalloc(&m_rho, rho_size));
  CUCHK(cudaMalloc(&m_d, d_size));

  CUCHK(cudaMalloc(&m_main[0], sizeof(double) * (NA + MAIN_LWS[0])));
  CUCHK(cudaMalloc(&m_main[1], sizeof(double) * (NA + MAIN_LWS[0])));
  CUCHK(cudaMalloc(&m_conj[0], sizeof(double) * (NA + CG_LWS[1])));
  CUCHK(cudaMalloc(&m_conj[1], sizeof(double) * NA * CG_LWS[2]));

  for (int i = 0; i < MAX_PARTITION; i++) {
    for (int j = 0; j < BUFFERING; j++) {
       CUCHK(cudaEventCreate(&write_event[i][j]));
    }
  }
}

static void release_cuda()
{
  cudaFree(m_colidx[0]);
  cudaFree(m_a[0]);
  if (NUM_TILES > 1) {
    cudaFree(m_a[1]);
    cudaFree(m_colidx[1]);
  }
  cudaFree(m_rowstr);
  cudaFree(m_q);
  cudaFree(m_z);
  cudaFree(m_r);
  cudaFree(m_p);
  cudaFree(m_x);

  cudaFree(m_norm_temp1);
  cudaFree(m_norm_temp2);
  cudaFree(m_rho);
  cudaFree(m_d);

  cudaFree(m_main[0]);
  cudaFree(m_main[1]);
  cudaFree(m_conj[0]);
  cudaFree(m_conj[1]);

  free(g_norm_temp1);
  free(g_norm_temp2);
  free(g_rho);
  free(g_d);

  cudaStreamDestroy(cmd_queue[0]);
  cudaStreamDestroy(cmd_queue[1]);

  cuda_ProfilerRelease();
}
