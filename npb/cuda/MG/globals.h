//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB MG code. This CUDA® C  //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3.1-SER" developed by NAS.                                       //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3.1, including the technical report, the original //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
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
//  Parameter lm (declared and set in "npbparams.h") is the log-base2 of
//  the edge size max for the partition on a given node, so must be changed
//  either to save space (if running a small case) or made bigger for larger
//  cases, for example, 512^3. Thus lm=7 means that the largest dimension
//  of a partition that can be solved on a node is 2^7 = 128. lm is set
//  automatically in npbparams.h
//  Parameters ndim1, ndim2, ndim3 are the local problem dimensions.
//---------------------------------------------------------------------

#include "npbparams.h"
#include "type.h"

// actual dimension including ghost cells for communications
#define NM          (2+(1<<LM))

// size of rhs array
#define NV          ((size_t)ONE*(2+(1<<NDIM1))*(2+(1<<NDIM2))*(2+(1<<NDIM3)))

// size of residual array
#define NR          (((NV+(size_t)NM*NM+5*NM+7*LM+6)/7)*8)

// maximum number of levels
#define MAXLEVEL    (LT_DEFAULT+1)


//---------------------------------------------------------------------
/* common /mg3/ */
static int nx[MAXLEVEL+1];
static int ny[MAXLEVEL+1];
static int nz[MAXLEVEL+1];

/* common /ClassType/ */
static char Class;

/* common /fap/ */
static long long m1[MAXLEVEL+1];
static long long m2[MAXLEVEL+1];
static long long m3[MAXLEVEL+1];
static long long ir[MAXLEVEL+1];
static int lt, lb;


//---------------------------------------------------------------------
//  Set at m=1024, can handle cases up to 1024^3 case
//---------------------------------------------------------------------
#define M   (NM+1)


/* common /timers/ */
static logical timeron;

enum {
  T_init,
  T_bench,

  T_mg3P,

  T_psinv,
  T_psinv_kern,
  T_psinv_comm,

  T_resid,
  T_resid_kern,
  T_resid_comm,

  T_rprj3,
  T_rprj3_kern,
  T_rprj3_comm,

  T_interp,
  T_interp_kern,
  T_interp_comm,

  T_norm2u3,
  T_norm2u3_kern,
  T_norm2u3_comm,

  T_comm3,
  T_comm3_kern,
  T_comm3_host,

  T_zero3,

  T_last
};

#define NORM2U3_LWS 128

__global__ void kernel_zero3(double *u, int n1, int n2, int n3);
__global__ void kernel_psinv_base(double *_r, double *_u, double *c, int n1, int n2, int len);
__global__ void kernel_psinv_opt(double *r, double *u, double *c, int n1, int n2, int len);
__global__ void kernel_resid_base(double *r, double *u, double *v, double *a, int n1, int n2, int len);
__global__ void kernel_resid_opt(double *r, double *u, double *v, double *a, int n1, int n2, int len);
__global__ void kernel_rprj3(double *rk, double *rj, int m1k, int m2k, int m1j, int m2j, long long ofs_j);
__global__ void kernel_interp(double *uj, double *uk, int mm1, int mm2, int n1, int n2, long long ofs_j);
__global__ void kernel_norm2u3_base(double *or_, int n1, int n2, double *g_sum, double *g_max, double *l_sum, double *l_max);
__global__ void kernel_norm2u3_opt(double *or_, int n1, int n2, double *g_sum, double *g_max);
__global__ void kernel_comm3_1(double *u, int n1, int n2, int n3);
__global__ void kernel_comm3_2(double *u, int n1, int n2, int n3);
__global__ void kernel_comm3_3(double *u, int n1, int n2, int n3);
