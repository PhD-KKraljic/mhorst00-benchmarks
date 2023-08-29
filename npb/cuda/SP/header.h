//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB SP code. This CUDA® C  //
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
// The following include file is generated automatically by the
// "setparams" utility. It defines 
//      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
//      dt_default:    default time step for this problem size if no
//                     config file
//      niter_default: default number of iterations for this problem size
//---------------------------------------------------------------------

#ifndef __HEADER_H_
#define __HEADER_H_

#include <cuda_runtime.h>
#include "cuda_util.h"
#include "npbparams.h"
extern "C" {
#include "type.h"
#include "timers.h"
}

/* common /global/ */
extern int grid_points[3], nx2, ny2, nz2;
extern logical timeron;

/* common /constants/ */
extern double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
              dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
              dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
              ce[5][13], dxmax, dymax, dzmax, xxcon1, xxcon2, 
              xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
              dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
              yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
              zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
              dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
              dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
              c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
              dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
              c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
              c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

#define IMAX    PROBLEM_SIZE
#define JMAX    PROBLEM_SIZE
#define KMAX    PROBLEM_SIZE
#define IMAXP   (IMAX/2*2)
#define JMAXP   (JMAX/2*2)

//---------------------------------------------------------------------
// To improve cache performance, grid dimensions padded by 1 
// for even number sizes only
//---------------------------------------------------------------------
/* common /fields/ */
extern double u      [KMAX][JMAXP+1][IMAXP+1][5];
extern double us     [KMAX][JMAXP+1][IMAXP+1];
extern double vs     [KMAX][JMAXP+1][IMAXP+1];
extern double ws     [KMAX][JMAXP+1][IMAXP+1];
extern double qs     [KMAX][JMAXP+1][IMAXP+1];
extern double rho_i  [KMAX][JMAXP+1][IMAXP+1];
extern double speed  [KMAX][JMAXP+1][IMAXP+1];
extern double square [KMAX][JMAXP+1][IMAXP+1];
extern double rhs    [KMAX][JMAXP+1][IMAXP+1][5];
extern double forcing[KMAX][JMAXP+1][IMAXP+1][5];

/* common /work_1d/ */
extern double cv  [PROBLEM_SIZE];
extern double rhon[PROBLEM_SIZE];
extern double rhos[PROBLEM_SIZE];
extern double rhoq[PROBLEM_SIZE];
extern double cuf [PROBLEM_SIZE];
extern double q   [PROBLEM_SIZE];
extern double ue [PROBLEM_SIZE][5];
extern double buf[PROBLEM_SIZE][5];

/* common /work_lhs/ */
extern double lhs [IMAXP+1][IMAXP+1][5];
extern double lhsp[IMAXP+1][IMAXP+1][5];
extern double lhsm[IMAXP+1][IMAXP+1][5];

//-----------------------------------------------------------------------
// CUDA Variables
//-----------------------------------------------------------------------
#define MAX_PARTITONS 240
extern cudaStream_t cmd_queue[2];

extern double *buf_u[2];
extern double *buf_us;
extern double *buf_vs;
extern double *buf_ws;
extern double *buf_qs;
extern double *buf_rho_i;
extern double *buf_speed;
extern double *buf_square;
extern double *buf_rhs[2];
extern double *buf_forcing[2];

extern double *buf_lhs;
extern double *buf_lhsp;
extern double *buf_lhsm;

extern double *buf_u0, *buf_u1, *buf_u2, *buf_u3, *buf_u4;
extern double *buf_rhs0, *buf_rhs1, *buf_rhs2, *buf_rhs3, *buf_rhs4;
extern double *buf_forcing0, *buf_forcing1, *buf_forcing2, *buf_forcing3, *buf_forcing4;

extern double *buf_lhs0, *buf_lhs1, *buf_lhs2, *buf_lhs3, *buf_lhs4;
extern double *buf_lhsp0, *buf_lhsp1, *buf_lhsp2, *buf_lhsp3, *buf_lhsp4;
extern double *buf_lhsm0, *buf_lhsm1, *buf_lhsm2, *buf_lhsm3, *buf_lhsm4;
extern double *buf_temp;

extern cudaEvent_t write_event[MAX_PARTITONS];

extern int KMAXP_D, JMAXP_D, WORK_NUM_ITEM_K, WORK_NUM_ITEM_J;
extern int NUM_PARTITIONS;

extern int opt_level_t;

//-----------------------------------------------------------------------
// Timer constants
//-----------------------------------------------------------------------
#define t_total     1
#define t_rhsx      2
#define t_rhsy      3
#define t_rhsz      4
#define t_rhs       5
#define t_xsolve    6
#define t_ysolve    7
#define t_zsolve    8
#define t_rdis1     9
#define t_rdis2     10
#define t_txinvr    11
#define t_pinvr     12
#define t_ninvr     13
#define t_tzetar    14
#define t_add       15
#define t_last      15


//-----------------------------------------------------------------------
void initialize();
void lhsinit(int ni, int nj);
void lhsinitj(int nj, int ni);
void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
void exact_rhs();
void set_constants();
void adi();
void compute_rhs();
void x_solve();
void ninvr();
void y_solve();
void pinvr();
void z_solve();
void tzetar();
void add();
void txinvr();
void error_norm(double rms[5]);
void rhs_norm(double rms[5]);
void verify(int no_time_steps, char *Class, logical *verified);

void adi_gpu();
void compute_rhs_gpu(int, int, int, int, int);
void x_solve_gpu(int, int, int, int, int);
void ninvr_gpu(int, int, int, int, int);
void y_solve_gpu(int, int, int, int, int);
void pinvr_gpu(int, int, int, int, int);
void z_solve_gpu(int, int, int, int, int);
void tzetar_gpu(int, int, int, int, int);
void add_gpu(int, int, int, int, int);
void txinvr_gpu(int, int, int, int, int);

void transpose_x_gpu();
void detranspose_x_gpu();
void scatter_gpu();
void gather_gpu();

__global__ void k_compute_rhs0( double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs0_base( double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs0_opt( double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs1(
   double *g_forcing0,
   double *g_forcing1,
   double *g_forcing2,
   double *g_forcing3,
   double *g_forcing4,
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs1_base(
   double *g_forcing0,
   double *g_forcing1,
   double *g_forcing2,
   double *g_forcing3,
   double *g_forcing4,
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs1_opt(
   double *g_forcing0,
   double *g_forcing1,
   double *g_forcing2,
   double *g_forcing3,
   double *g_forcing4,
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs2(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs2_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs2_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs3(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs3_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs3_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs4(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs4_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs4_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs5(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs5_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_compute_rhs5_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);

__global__ void k_x_solve0(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve0_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve0_parallel(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve0_layout(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve0_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve1(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve1_base(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve1_parallel(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve1_layout(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve1_opt(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve2(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve2_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve2_parallel(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve2_layout(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve2_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve3(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve3_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve3_parallel(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve3_layout(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve3_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve4(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve4_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve4_parallel(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve4_layout(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);
__global__ void k_x_solve4_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K);

__global__ void k_ninvr(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_ninvr_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_ninvr_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve0(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_vs,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve0_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_vs,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve0_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_vs,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve1(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve1_base(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve1_opt(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve2(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve2_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve2_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve3(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve3_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve3_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve4(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve4_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_y_solve4_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2);
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__global__ void k_pinvr(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_pinvr_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_pinvr_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_z_solve0(
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve0_base(
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve0_opt(
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve1(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_ws,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve1_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_ws,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve1_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_ws,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve2(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve2_base(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve2_opt(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve3(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve3_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve3_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve4(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve4_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve4_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve5(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve5_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_z_solve5_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);

__global__ void k_tzetar(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_tzetar_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_tzetar_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_txinvr(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_txinvr_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_txinvr_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
   double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2);
__global__ void k_add(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_add_base(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_add_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J);
__global__ void k_transpose( double *A, double *B, const int P, const int Q);
__global__ void k_transpose_base( double *A, double *B, const int P, const int Q);
__global__ void k_transpose_opt( double *A, double *B, const int P, const int Q);
__global__ void k_scatter( double *g_A, double *g_A0, double *g_A1, double *g_A2, double *g_A3, double *g_A4,
        const int nx, const int ny, const int nz);
__global__ void k_gather(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz);
__global__ void k_scatter_j(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz, const int WORK_NUM_ITEM_J);
__global__ void k_gather_j(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz, const int WORK_NUM_ITEM_J);

#endif
