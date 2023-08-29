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

/* common /debug/ */
static logical debug;

/* common /fap/ */
static size_t m1[MAXLEVEL+1];
static size_t m2[MAXLEVEL+1];
static size_t m3[MAXLEVEL+1];
static size_t ir[MAXLEVEL+1];
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
