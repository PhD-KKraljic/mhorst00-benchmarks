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
//---------------------------------------------------------------------
//
//  header.h
//
//---------------------------------------------------------------------
//---------------------------------------------------------------------
 
//---------------------------------------------------------------------
// The following include file is generated automatically by the
// "setparams" utility. It defines 
//      maxcells:      the square root of the maximum number of processors
//      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
//      dt_default:    default time step for this problem size if no
//                     config file
//      niter_default: default number of iterations for this problem size
//---------------------------------------------------------------------

#ifndef __HEADER_H_
#define __HEADER_H_

#include "npbparams.h"
#include <cuda_runtime.h>
#include "cuda_util.h"

extern "C" {
#include "timers.h"
#include "type.h"
}



#ifndef max
#define max(x, y) ((x > y) ? x : y)
#endif

#define min(x,y) ((x < y) ? x : y)

#define AA            0
#define BB            1
#define CC            2
#define BLOCK_SIZE    5

/* common /global/ */
extern double elapsed_time;
extern int grid_points[3];
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
              c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1,
              dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
              c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
              c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

#define IMAX      PROBLEM_SIZE
#define JMAX      PROBLEM_SIZE
#define KMAX      PROBLEM_SIZE
#define IMAXP     IMAX/2*2
#define JMAXP     JMAX/2*2


// to improve cache performance, grid dimensions padded by 1 
// for even number sizes only.
/* common /fields/ */
extern double us     [KMAX][JMAXP+1][IMAXP+1];
extern double vs     [KMAX][JMAXP+1][IMAXP+1];
extern double ws     [KMAX][JMAXP+1][IMAXP+1];
extern double qs     [KMAX][JMAXP+1][IMAXP+1];
extern double rho_i  [KMAX][JMAXP+1][IMAXP+1];
extern double square [KMAX][JMAXP+1][IMAXP+1];
extern double forcing[KMAX][JMAXP+1][IMAXP+1][5];
extern double u      [KMAX][JMAXP+1][IMAXP+1][5];
extern double rhs    [KMAX][JMAXP+1][IMAXP+1][5];

/* common /work_1d/ */
extern double cuf[PROBLEM_SIZE+1];
extern double q  [PROBLEM_SIZE+1];
extern double ue [PROBLEM_SIZE+1][5];
extern double buf[PROBLEM_SIZE+1][5];
      

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
#define t_add       11
#define t_last      11


void initialize();
void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
void exact_rhs();
void set_constants();
void adi();
void adi_init();
void adi_free();

void compute_rhs();
void compute_rhs_init(int iter);
void compute_rhs_release(int iter);
void compute_rhs_body(int work_step, 
                      int work_base, 
                      int work_num_item, 
                      int copy_buffer_base, 
                      int copy_num_item, 
                      int buf_idx,
                      cudaEvent_t* ev_wb_end_ptr);

void x_solve_init(int iter);
void x_solve_release(int iter);
void x_solve(int work_step, 
             int work_base, 
             int work_num_item,
             int buf_idx);

void y_solve_init(int iter);
void y_solve_release(int iter);
cudaEvent_t* y_solve(int work_step, 
                     int work_base, 
                     int work_num_item, 
                     int buf_idx);

void z_solve_init(int iter);
void z_solve_release(int iter);
void z_solve(int work_step, 
             int work_base, 
             int work_num_item, 
             int buf_idx,
             cudaEvent_t* ev_wb_end_ptr);

void add(int work_step, 
         int work_base, 
         int work_num_item, 
         int buf_idx);

void error_norm(double rms[5]);
void rhs_norm(double rms[5]);
void verify(int no_time_steps, char *_class, logical *verified);

//-----------------------------------------------------------------------
// CUDA - Macros
//-----------------------------------------------------------------------
#define t_memcpy_pre  20
#define t_memcpy_post 21
#define t_prof_cont   22
#define t_prof_rect   23

// Command Queue Types - default queue is kernel queue 
#define KERNEL_Q  0
#define DATA_Q    1
#define NUM_Q     2

//#define DETAIL_INFO

#ifdef DETAIL_INFO
#define DETAIL_LOG(fmt, ...) fprintf(stdout, " [CUDA Detailed Info] " fmt "\n", ## __VA_ARGS__)
#else
#define DETAIL_LOG(fmt, ...) 
#endif

//-----------------------------------------------------------------------
// CUDA - Variables
//-----------------------------------------------------------------------

/* CUDA environment variables */
extern int            device;
extern cudaStream_t   cmd_q[NUM_Q];

extern double         *m_us, 
                      *m_vs, 
                      *m_ws, 
                      *m_qs,
                      *m_rho_i, 
                      *m_square,
                      *m_forcing[2], 
                      *m_u[2], 
                      *m_rhs[2],
                      *m_lhsA, 
                      *m_lhsB, 
                      *m_lhsC;

extern double         *m_fjac,
                      *m_njac,
                      *m_lhs;

extern cudaEvent_t    *loop1_ev_wb_start, 
                      *loop1_ev_wb_end,
                      *loop1_ev_rb_start, 
                      *loop1_ev_rb_end,
                      *loop2_ev_wb_start, 
                      *loop2_ev_wb_end,
                      *loop2_ev_rb_start, 
                      *loop2_ev_rb_end,
                      (*loop2_ev_kernel_add)[2];

/* CUDA dynamic configuration flags */
extern int            split_flag,
                      buffering_flag;

/* CUDA optimization levels enumerate */
enum OptLevel {
  OPT_BASELINE=0,
  OPT_PARALLEL,
  OPT_GLOBALMEM,
  OPT_MEMLAYOUT,
  OPT_SYNC,
  OPT_FULL,
};

enum DataTransferType {
  DT_CONT = 0,
  DT_RECT,
};

/* CUDA optimization level */
extern enum OptLevel  g_opt_level;


/* CUDA device dependent variables */
extern size_t         max_compute_unit;
extern size_t         max_work_item_sizes[3];
extern size_t         max_work_group_size;
extern unsigned long  local_mem_size,
                      max_mem_alloc_size,
                      gmem_size;
extern int            work_num_item_default,
                      work_num_item_default_j;
extern int            loop1_work_max_iter, 
                      loop1_work_num_item_default,
                      loop2_work_max_iter, 
                      loop2_work_num_item_default;

enum Kernel         { RHSDATAGEN, RHS1, RHS2,
                      RHSX1, RHSX2, RHSY1, RHSY2, RHSZ1, RHSZ2,
                      XSOL1, XSOL2, XSOL3, 
                      YSOL1, YSOL2, YSOL3,
                      ZSOLDATAGEN,
                      ZSOL1, ZSOL2, ZSOL3,
                      ADD,
                      OMP1, OMP2 };

/* CUDUA Helper Functions */
int get_loop1_copy_num_item(int work_base, int work_num_item);
int get_loop1_copy_buffer_base(int work_base);
int get_loop1_copy_host_base(int work_base);

/* CUDA Kernel Functions */

// -----------------------------------------------
// Kernel Functions for x solve
// -----------------------------------------------
// baseline
__global__
void k_x_solve_baseline(double *m_qs,
                       double *m_rho_i,
                       double *m_square,
                       double *m_u,
                       double *m_rhs,
                       double *m_lhs, 
                       double *m_fjac,
                       double *m_njac,
                       int gp0, int gp1, int gp2,
                       double dx1, double dx2, 
                       double dx3, double dx4, 
                       double dx5,
                       double c1, double c2,
                       double tx1, double tx2,
                       double con43, double c3c4,
                       double c1345, double dt,
                       int work_base, 
                       int work_num_item, 
                       int split_flag);
// parallel
__global__
void k_x_solve1_parallel(double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         int gp0, int gp1, int gp2,
                         int work_base, 
                         int work_num_item, 
                         int split_flag);
__global__
void k_x_solve2_parallel(double *m_qs, 
                         double *m_rho_i,
                         double *m_square, 
                         double *m_u, 
                         double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         double *m_fjac,
                         double *m_njac,
                         int gp0, int gp1, int gp2,
                         double dx1, double dx2, 
                         double dx3, double dx4, 
                         double dx5,
                         double c1, double c2, 
                         double tx1, double tx2, 
                         double con43,
                         double c3c4, double c1345, 
                         double dt,
                         int work_base, 
                         int work_num_item, 
                         int split_flag);
__global__
void k_x_solve3_parallel(double *m_rhs,
                         double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         int gp0, int gp1, int gp2,
                         int work_base, 
                         int work_num_item, 
                         int split_flag);
// memlayout
__global__
void k_x_solve_memlayout(double *m_qs,
                         double *m_rho_i,
                         double *m_square,
                         double *m_u,
                         double *m_rhs,
                         double *m_lhs, 
                         double *m_fjac,
                         double *m_njac,
                         int gp0, int gp1, int gp2,
                         double dx1, double dx2, 
                         double dx3, double dx4, 
                         double dx5,
                         double c1, double c2,
                         double tx1, double tx2,
                         double con43, double c3c4,
                         double c1345, double dt,
                         int work_base, 
                         int work_num_item, 
                         int split_flag,
                         int WORK_NUM_ITEM_DEFAULT);

// fullopt
__global__ 
void k_x_solve1_fullopt(double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT);
__global__ 
void k_x_solve2_fullopt(double *m_qs, double *m_rho_i,
                        double *m_square, double *m_u, 
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        double dx1, double dx2, 
                        double dx3, double dx4, 
                        double dx5,
                        double c1, double c2, 
                        double tx1, double tx2, 
                        double con43,
                        double c3c4, double c1345, 
                        double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT);
__global__ 
void k_x_solve3_fullopt(double *m_rhs,
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT);

// -----------------------------------------------
// Kernel Functions for y solve
// -----------------------------------------------

// baseline
__global__
void k_y_solve_baseline(double *m_qs,
                        double *m_rho_i,
                        double *m_square,
                        double *m_u,
                        double *m_rhs,
                        double *m_lhs, 
                        double *m_fjac,
                        double *m_njac,
                        int gp0, int gp1, int gp2,
                        double dy1, double dy2,
                        double dy3, double dy4,
                        double dy5,
                        double c1, double c2,
                        double ty1, double ty2,
                        double con43, double c3c4,
                        double c1345, double dt,
                        int work_base,
                        int work_num_item,
                        int split_flag);

// parallel
__global__
void k_y_solve1_parallel(double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         int gp0, int gp1, int gp2,
                         int work_base, 
                         int work_num_item, 
                         int split_flag);
__global__ 
void k_y_solve2_parallel(double *m_qs, 
                         double *m_rho_i,
                         double *m_square, 
                         double *m_u, 
                         double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         double *m_fjac,
                         double *m_njac,
                         int gp0, int gp1, int gp2,
                         double dy1, double dy2, 
                         double dy3, double dy4, 
                         double dy5,
                         double c1, double c2, 
                         double ty1, double ty2, 
                         double con43, double c3c4, 
                         double c1345, double dt,
                         int work_base, 
                         int work_num_item, 
                         int split_flag);
__global__
void k_y_solve3_parallel(double *m_rhs,
                         double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         int gp0, int gp1, int gp2,
                         int work_base, 
                         int work_num_item, 
                         int split_flag);
// memlayout
__global__ 
void k_y_solve_memlayout(double *m_qs,
                         double *m_rho_i,
                         double *m_square,
                         double *m_u,
                         double *m_rhs,
                         double *m_lhs, 
                         double *m_fjac,
                         double *m_njac,
                         int gp0, int gp1, int gp2,
                         double dy1, double dy2,
                         double dy3, double dy4,
                         double dy5,
                         double c1, double c2,
                         double ty1, double ty2,
                         double con43, double c3c4,
                         double c1345, double dt,
                         int work_base,
                         int work_num_item,
                         int split_flag,
                         int WORK_NUM_ITEM_DEFAULT);


// fullopt
__global__ 
void k_y_solve1_fullopt(double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT);
__global__ 
void k_y_solve2_fullopt(double *m_qs, double *m_rho_i,
                        double *m_square, double *m_u, 
                        double *m_lhsA, double * m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        double dy1, double dy2, 
                        double dy3, double dy4, 
                        double dy5,
                        double c1, double c2, 
                        double ty1, double ty2, 
                        double con43,
                        double c3c4, double c1345, 
                        double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT);
__global__ 
void k_y_solve3_fullopt(double *m_rhs,
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT);

// -----------------------------------------------
// Kernel Functions for z solve
// -----------------------------------------------

// baseline
__global__
void k_z_solve_data_gen_baseline(double *m_u,
                                 double *m_square,
                                 double *m_qs,
                                 int gp0, int gp1, int gp2,
                                 int work_base,
                                 int work_num_item,
                                 int split_flag,
                                 int WORK_NUM_ITEM_DEFAULT_J);
__global__
void k_z_solve_baseline(double *m_qs,
                        double *m_square,
                        double *m_u,
                        double *m_rhs,
                        double *m_lhs,
                        double *m_fjac,
                        double *m_njac,
                        int gp0, int gp1, int gp2,
                        double dz1, double dz2,
                        double dz3, double dz4,
                        double dz5,
                        double c1, double c2,
                        double c3, double c4,
                        double tz1, double tz2,
                        double con43, double c3c4,
                        double c1345, double dt,
                        int work_base,
                        int work_num_item,
                        int split_flag,
                        int WORK_NUM_ITEM_DEFAULT_J);

// parallel
__global__ 
void k_z_solve_data_gen_parallel(double *m_u,
                                 double *m_square, double *m_qs,
                                 int gp0, int gp1, int gp2,
                                 int work_base, 
                                 int work_num_item, 
                                 int split_flag, 
                                 int WORK_NUM_ITEM_DEFAULT_J);

__global__
void k_z_solve1_parallel(double *m_lhsA, 
                                double *m_lhsB, 
                                double *m_lhsC,
                                int gp0, int gp1, int gp2,
                                int work_base, 
                                int work_num_item, 
                                int split_flag,
                                int WORK_NUM_ITEM_DEFAULT_J);
__global__
void k_z_solve2_parallel(double *m_qs, 
                        double *m_square, 
                        double *m_u,
                        double *m_lhsA, 
                        double *m_lhsB, 
                        double *m_lhsC,
                        double *m_fjac,
                        double *m_njac,
                        int gp0, int gp1, int gp2,
                        double dz1, double dz2, 
                        double dz3, double dz4, 
                        double dz5,
                        double c1, double c2, 
                        double c3, double c4,
                        double tz1, double tz2, 
                        double con43, double c3c4, 
                        double c1345, double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag,
                        int WORK_NUM_ITEM_DEFAULT_J);
__global__
void k_z_solve3_parallel(double *m_rhs,
                         double *m_lhsA, 
                         double *m_lhsB, 
                         double *m_lhsC,
                         int gp0, int gp1, int gp2,
                         int work_base, 
                         int work_num_item, 
                         int split_flag,
                         int WORK_NUM_ITEM_DEFAULT_J);

// memlayout
__global__
void k_z_solve_data_gen_memlayout(double *m_u,
                                  double *m_square,
                                  double *m_qs,
                                  int gp0, int gp1, int gp2,
                                  int work_base,
                                  int work_num_item,
                                  int split_flag,
                                  int WORK_NUM_ITEM_DEFAULT_J);
__global__
void k_z_solve_memlayout(double *m_qs,
                         double *m_square,
                         double *m_u,
                         double *m_rhs,
                         double *m_lhs,
                         double *m_fjac,
                         double *m_njac,
                         int gp0, int gp1, int gp2,
                         double dz1, double dz2,
                         double dz3, double dz4,
                         double dz5,
                         double c1, double c2,
                         double c3, double c4,
                         double tz1, double tz2,
                         double con43, double c3c4,
                         double c1345, double dt,
                         int work_base,
                         int work_num_item,
                         int split_flag,
                         int WORK_NUM_ITEM_DEFAULT_J);




// fullopt
__global__ 
void k_z_solve_data_gen_fullopt(double *m_u,
                                double *m_square, double *m_qs,
                                int gp0, int gp1, int gp2,
                                int work_base, 
                                int work_num_item, 
                                int split_flag, 
                                int WORK_NUM_ITEM_DEFAULT_J);
__global__ 
void k_z_solve1_fullopt(double *m_lhsA, 
                        double *m_lhsB, double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT_J);
__global__ 
void k_z_solve2_fullopt(double *m_qs, double *m_square, 
                        double *m_u,
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        double dz1, double dz2, 
                        double dz3, double dz4, 
                        double dz5,
                        double c1, double c2, 
                        double c3, double c4,
                        double tz1, double tz2, 
                        double con43,
                        double c3c4, double c1345, 
                        double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT_J);
__global__ 
void k_z_solve3_fullopt(double *m_rhs,
                        double *m_lhsA, double *m_lhsB,  
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT_J);



// -----------------------------------------------
// Kernel Functions for compute rhs 
// -----------------------------------------------

// baseline
__global__ 
void k_compute_rhs_data_gen_baseline(double *m_rho_i, 
                                     double *m_us, 
                                     double *m_vs, 
                                     double *m_ws, 
                                     double *m_qs, 
                                     double *m_square,
                                     double *m_u,
                                     int gp0, int gp1, int gp2,
                                     int copy_buffer_base, 
                                     int copy_num_item, 
                                     int split_flag);
__global__
void k_compute_rhs1_baseline(double *m_rhs, 
                             double *m_forcing,
                             int gp0, int gp1, int gp2,
                             int work_base, 
                             int work_num_item, 
                             int split_flag);
__global__
void k_compute_rhsx_baseline(double *m_us, 
                             double *m_vs,
                             double *m_ws, 
                             double *m_qs,
                             double *m_rho_i, 
                             double *m_square,
                             double *m_u, 
                             double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dx1tx1, double dx2tx1, 
                             double dx3tx1, double dx4tx1, 
                             double dx5tx1,
                             double xxcon2, double xxcon3, 
                             double xxcon4, double xxcon5,
                             double c1, double c2, 
                             double tx2, double con43, 
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag);
__global__
void k_compute_rhsy_baseline(double *m_us, 
                             double *m_vs,
                             double *m_ws, 
                             double *m_qs,
                             double *m_rho_i, 
                             double *m_square,
                             double *m_u, 
                             double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dy1ty1, double dy2ty1, 
                             double dy3ty1, double dy4ty1, 
                             double dy5ty1,
                             double yycon2, double yycon3, 
                             double yycon4, double yycon5,
                             double c1, double c2, 
                             double ty2, double con43, 
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag);
 
__global__
void k_compute_rhsz1_baseline(double *m_us, 
                              double *m_vs,
                              double *m_ws, 
                              double *m_qs,
                              double *m_rho_i, 
                              double *m_square,
                              double *m_u, 
                              double *m_rhs,
                              int gp0, int gp1, int gp2,
                              double dz1tz1, double dz2tz1, 
                              double dz3tz1, double dz4tz1, 
                              double dz5tz1,
                              double zzcon2, double zzcon3, 
                              double zzcon4, double zzcon5,
                              double c1, double c2, 
                              double tz2, double con43,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhsz2_baseline(double *m_u, 
                              double *m_rhs,
                              int gp0, int gp1, int gp2, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhsz3_baseline(double *m_u, 
                              double *m_rhs,
                              int gp0, int gp1, int gp2, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhsz4_baseline(double *m_u, 
                              double *m_rhs,
                              int gp0, int gp1, int gp2, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhsz5_baseline(double *m_u, 
                              double *m_rhs,
                              int gp0, int gp1, int gp2, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhsz6_baseline(double *m_u, 
                              double *m_rhs,
                              int gp0, int gp1, int gp2, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhs2_baseline(double *m_rhs,
                             int gp0, int gp1, int gp2, 
                             double dt,
                             int work_base, 
                             int work_num_item, 
                             int split_flag);

// parallel
__global__ 
void k_compute_rhs_data_gen_parallel(double *m_rho_i, double *m_us, 
                                     double *m_vs, double *m_ws, 
                                     double *m_qs, double *m_square,
                                     double *m_u,
                                     int gp0, int gp1, int gp2,
                                     int copy_buffer_base, 
                                     int copy_num_item, 
                                     int split_flag);
__global__ 
void k_compute_rhs1_parallel(double *m_rhs,  double *m_forcing,
                             int gp0, int gp1, int gp2, 
                             int work_base, 
                             int work_num_item, 
                             int split_flag);
__global__ 
void k_compute_rhsx1_parallel(double *m_us, double *m_vs,
                              double *m_ws, double * m_qs,
                              double *m_rho_i, double *m_square,
                              double *m_u, double *m_rhs,
                              int gp0, int gp1, int gp2,
                              double dx1tx1, double dx2tx1, 
                              double dx3tx1, double dx4tx1, 
                              double dx5tx1,
                              double xxcon2, double xxcon3, 
                              double xxcon4, double xxcon5,
                              double c1, double c2, 
                              double tx2, double con43, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__ 
void k_compute_rhsx2_parallel(double *m_u,  double *m_rhs,
                              int gp0, int gp1, int gp2,
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__ 
void k_compute_rhsy1_parallel(double *m_us, double *m_vs,
                              double *m_ws, double * m_q,
                              double *m_rho_i, double *m_square,
                              double *m_u, double *m_rhs,
                              int gp0, int gp1, int gp2,
                              double dy1ty1, double dy2ty1, 
                              double dy3ty1, double dy4ty1, 
                              double dy5ty1,
                              double yycon2, double yycon3, 
                              double yycon4, double yycon5,
                              double c1, double c2, 
                              double ty2, double con43, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__ 
void k_compute_rhsy2_parallel(double *m_u, double *m_rhs,
                              int gp0, int gp1, int gp2,
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__
void k_compute_rhsz1_parallel(double *m_us, double *m_vs,
                              double *m_ws, double *m_qs,
                              double *m_rho_i, double *m_square,
                              double *m_u, double *m_rhs,
                              int gp0, int gp1, int gp2,
                              double dz1tz1, double dz2tz1, 
                              double dz3tz1, double dz4tz1, 
                              double dz5tz1,
                              double zzcon2, double zzcon3, 
                              double zzcon4, double zzcon5,
                              double c1, double c2, 
                              double tz2, double con43,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__ 
void k_compute_rhsz2_parallel(double *m_u, double *m_rhs,
                              int gp0, int gp1, int gp2, 
                              double dssp,
                              int work_base, 
                              int work_num_item, 
                              int split_flag);
__global__ 
void k_compute_rhs2_parallel(double *m_rhs,
                             int gp0, int gp1, int gp2, 
                             double dt,
                             int work_base, 
                             int work_num_item, 
                             int split_flag);


__global__ 
void k_add( double * m_u,  double * m_rhs,
    int gp0, int gp1, int gp2,
    int work_base, int work_num_item, int split_flag, int WORK_NUM_ITEM_DEFAULT_J);
#endif
