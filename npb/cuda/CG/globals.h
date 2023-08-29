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

#include "npbparams.h"


#ifdef __cplusplus
typedef bool logical;
#endif

#define min(x,y)    ((x) < (y) ? (x) : (y))
#define max(x,y)    ((x) > (y) ? (x) : (y))


//---------------------------------------------------------------------
//  Note: please observe that in the routine conj_grad three
//  implementations of the sparse matrix-vector multiply have
//  been supplied.  The default matrix-vector multiply is not
//  loop unrolled.  The alternate implementations are unrolled
//  to a depth of 2 and unrolled to a depth of 8.  Please
//  experiment with these to find the fastest for your particular
//  architecture.  If reporting timing results, any of these three may
//  be used without penalty.
//---------------------------------------------------------------------


//---------------------------------------------------------------------
//  Class specific parameters:
//  It appears here for reference only.
//  These are their values, however, this info is imported in the npbparams.h
//  include file, which is written by the sys/setparams.c program.
//---------------------------------------------------------------------

//----------
//  Class S:
//----------
//#define NA        1400
//#define NONZER    7
//#define SHIFT     10
//#define NITER     15
//#define RCOND     1.0e-1

//----------
//  Class W:
//----------
//#define NA        7000
//#define NONZER    8
//#define SHIFT     12
//#define NITER     15
//#define RCOND     1.0e-1

//----------
//  Class A:
//----------
//#define NA        14000
//#define NONZER    11
//#define SHIFT     20
//#define NITER     15
//#define RCOND     1.0e-1

//----------
//  Class B:
//----------
//#define NA        75000
//#define NONZER    13
//#define SHIFT     60
//#define NITER     75
//#define RCOND     1.0e-1

//----------
//  Class C:
//----------
//#define NA        150000
//#define NONZER    15
//#define SHIFT     110
//#define NITER     75
//#define RCOND     1.0e-1

//----------
//  Class D:
//----------
//#define NA        1500000
//#define NONZER    21
//#define SHIFT     500
//#define NITER     100
//#define RCOND     1.0e-1

//----------
//  Class E:
//----------
//#define NA        9000000
//#define NONZER    26
//#define SHIFT     1500
//#define NITER     100
//#define RCOND     1.0e-1

#define NZ    ((size_t) NA*(NONZER+1)*(NONZER+1))
#define NAZ   ((size_t) NA*(NONZER+1))

static logical timeron;

enum {
  T_init,
  T_bench,

  T_kern_main_0,
  T_comm_main_0,
  T_host_main_0,

  T_kern_main_1,

  T_kern_conj_0,

  T_kern_conj_1,
  T_comm_conj_1,
  T_host_conj_1,

  T_kern_conj_2,
  T_comm_conj_2,

  T_kern_conj_3,
  T_comm_conj_3,
  T_host_conj_3,

  T_kern_conj_4,
  T_comm_conj_4,
  T_host_conj_4,

  T_kern_conj_5,

  T_kern_conj_6,
  T_comm_conj_6,

  T_kern_conj_7,
  T_comm_conj_7,
  T_host_conj_7,

  T_last
};

__global__ void main_0_base(double *x,
                            double *z,
                            double *g_norm_temp1,
                            double *g_norm_temp2,
                            double *l_norm_temp1,
                            double *l_norm_temp2,
                            int n);
__global__ void main_0_opt(double *x,
                           double *z,
                           double *g_norm_temp1,
                           double *g_norm_temp2,
                           int n);
__global__ void main_1(double *x,
                       double *z,
                       double norm_temp2,
                       int n);
__global__ void conj_grad_0(double *q,
                            double *z,
                            double *r,
                            double *x,
                            double *p,
                            int n);
__global__ void conj_grad_1_base(double *r,
                                 double *g_rho,
                                 double *l_rho,
                                 int n);
__global__ void conj_grad_1_opt(double *r,
                                double *g_rho,
                                int n);
__global__ void conj_grad_2_base(int *rowstr,
                                 int *colidx,
                                 double *a,
                                 double *p,
                                 double *q,
                                 double *l_sum,
                                 int vec_ofs,
                                 int mat_ofs);
__global__ void conj_grad_2_opt(int *rowstr,
                                int *colidx,
                                double *a,
                                double *p,
                                double *q,
                                int vec_ofs,
                                int mat_ofs);
__global__ void conj_grad_3_base(double *p,
                                 double *q,
                                 double *g_d,
                                 double *l_d,
                                 int n);
__global__ void conj_grad_3_opt(double *p,
                                double *q,
                                double *g_d,
                                int n);
__global__ void conj_grad_4_base(double *p,
                                 double *q,
                                 double *r,
                                 double *z,
                                 double *g_rho,
                                 double *l_rho,
                                 double alpha,
                                 int n);
__global__ void conj_grad_4_opt(double *p,
                                double *q,
                                double *r,
                                double *z,
                                double *g_rho,
                                double alpha,
                                int n);
__global__ void conj_grad_5(double *p,
                            double *r,
                            const double beta,
                            int n);
__global__ void conj_grad_6_base(int *rowstr,
                                 int *colidx,
                                 double *a,
                                 double *r,
                                 double *z,
                                 double *l_sum,
                                 int vec_ofs,
                                 int mat_ofs);
__global__ void conj_grad_6_opt(int *rowstr,
                                int *colidx,
                                double *a,
                                double *r,
                                double *z,
                                int vec_ofs,
                                int mat_ofs);
__global__ void conj_grad_7_base(double *r,
                                 double *x,
                                 double *g_d,
                                 double *l_d,
                                 int n);
__global__ void conj_grad_7_opt(double *r,
                                double *x,
                                double *g_d,
                                int n);
