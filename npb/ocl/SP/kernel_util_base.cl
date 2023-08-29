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
// transpose matrix A to B : B = A^t
// Input  => A : P * Q matrix
// Output => B : Q * P matrix
//---------------------------------------------------------------------
__kernel void transpose(
  __global double *A,
  __global double *B,
  const int P, const int Q)
{
  int i = get_global_id(1);
  int j = get_global_id(0);

  if (i < P && j < Q) {
    B[j * P + i]= A[i * Q + j];
  }
}

__kernel void scatter(
  __global double *g_A,
  __global double *g_A0,
  __global double *g_A1,
  __global double *g_A2,
  __global double *g_A3,
  __global double *g_A4,
  const int nx, const int ny, const int nz)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);

  __global double (*A)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_A;

  __global double (*A0)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A0;
  __global double (*A1)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A1;
  __global double (*A2)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A2;
  __global double (*A3)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A3;
  __global double (*A4)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A4;

  if (i < nx && j < ny && k < nz) {
    A0[k][j][i] = A[k][j][i][0];
    A1[k][j][i] = A[k][j][i][1];
    A2[k][j][i] = A[k][j][i][2];
    A3[k][j][i] = A[k][j][i][3];
    A4[k][j][i] = A[k][j][i][4];
  }
}

__kernel void gather(
  __global double *g_A,
  __global double *g_A0,
  __global double *g_A1,
  __global double *g_A2,
  __global double *g_A3,
  __global double *g_A4,
  const int nx, const int ny, const int nz)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);

  __global double (*A)[JMAXP+1][IMAXP+1][5] =
    (__global double (*)[JMAXP+1][IMAXP+1][5])g_A;

  __global double (*A0)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A0;
  __global double (*A1)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A1;
  __global double (*A2)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A2;
  __global double (*A3)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A3;
  __global double (*A4)[JMAXP+1][IMAXP+1] =
    (__global double (*)[JMAXP+1][IMAXP+1])g_A4;

  if (i < nx && j < ny && k < nz) {
    A[k][j][i][0] = A0[k][j][i];
    A[k][j][i][1] = A1[k][j][i];
    A[k][j][i][2] = A2[k][j][i];
    A[k][j][i][3] = A3[k][j][i];
    A[k][j][i][4] = A4[k][j][i];
  }
}

__kernel void scatter_j(
  __global double *g_A,
  __global double *g_A0,
  __global double *g_A1,
  __global double *g_A2,
  __global double *g_A3,
  __global double *g_A4,
  const int nx, const int ny, const int nz)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);

  __global double (*A)[WORK_NUM_ITEM_J][IMAXP+1][5] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1][5])g_A;

  __global double (*A0)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A0;
  __global double (*A1)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A1;
  __global double (*A2)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A2;
  __global double (*A3)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A3;
  __global double (*A4)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A4;

  if (i < nx && j < ny && k < nz) {
    A0[k][j][i] = A[k][j][i][0];
    A1[k][j][i] = A[k][j][i][1];
    A2[k][j][i] = A[k][j][i][2];
    A3[k][j][i] = A[k][j][i][3];
    A4[k][j][i] = A[k][j][i][4];
  }
}

__kernel void gather_j(
  __global double *g_A,
  __global double *g_A0,
  __global double *g_A1,
  __global double *g_A2,
  __global double *g_A3,
  __global double *g_A4,
  const int nx, const int ny, const int nz)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);

  __global double (*A)[WORK_NUM_ITEM_J][IMAXP+1][5] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1][5])g_A;

  __global double (*A0)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A0;
  __global double (*A1)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A1;
  __global double (*A2)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A2;
  __global double (*A3)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A3;
  __global double (*A4)[WORK_NUM_ITEM_J][IMAXP+1] =
    (__global double (*)[WORK_NUM_ITEM_J][IMAXP+1])g_A4;

  if (i < nx && j < ny && k < nz) {
    A[k][j][i][0] = A0[k][j][i];
    A[k][j][i][1] = A1[k][j][i];
    A[k][j][i][2] = A2[k][j][i];
    A[k][j][i][3] = A3[k][j][i];
    A[k][j][i][4] = A4[k][j][i];
  }
}
