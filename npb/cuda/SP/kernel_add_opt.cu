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

#include "npbparams.h"
#include "kernel_header.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
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
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define u0(a, b, c) g_u0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u1(a, b, c) g_u1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u2(a, b, c) g_u2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u3(a, b, c) g_u3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u4(a, b, c) g_u4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
    
  int i, j, k, m;
  k = 1 + blockDim.z * blockIdx.z + threadIdx.z;
  j = offset_j + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (k > nz2 || base_j + j > ny2 || i > nx2) return;
  if (j >= offset_j + gws_j) return;

  u0(k, j, i) = u0(k, j, i) + rhs0(k, j, i);
  u1(k, j, i) = u1(k, j, i) + rhs1(k, j, i);
  u2(k, j, i) = u2(k, j, i) + rhs2(k, j, i);
  u3(k, j, i) = u3(k, j, i) + rhs3(k, j, i);
  u4(k, j, i) = u4(k, j, i) + rhs4(k, j, i);

#undef u0
#undef u1
#undef u2
#undef u3
#undef u4
#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
}
