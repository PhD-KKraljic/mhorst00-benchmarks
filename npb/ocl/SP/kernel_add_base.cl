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

#include "kernel_header.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
__kernel void add(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2)
{
  __global double (*u0)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_u0;
  __global double (*u1)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_u1;
  __global double (*u2)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_u2;
  __global double (*u3)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_u3;
  __global double (*u4)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_u4;
  __global double (*rhs0)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_rhs0;
  __global double (*rhs1)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_rhs1;
  __global double (*rhs2)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_rhs2;
  __global double (*rhs3)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_rhs3;
  __global double (*rhs4)[WORK_NUM_ITEM_J][IMAXP+1] = 
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_rhs4;

  int i, j, k, m;
  k = 1 + get_global_id(0);
  if (k > nz2) return;

  for (j = offset_j; j <= min(ny2 - base_j, offset_j + gws_j - 1); j++) {
    for (i = 1; i <= nx2; i++) {
      u0[k][j][i] = u0[k][j][i] + rhs0[k][j][i];
      u1[k][j][i] = u1[k][j][i] + rhs1[k][j][i];
      u2[k][j][i] = u2[k][j][i] + rhs2[k][j][i];
      u3[k][j][i] = u3[k][j][i] + rhs3[k][j][i];
      u4[k][j][i] = u4[k][j][i] + rhs4[k][j][i];
    }
  }
}
