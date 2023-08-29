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
// transpose matrix A to B : B = A^t
// Input  => A : P * Q matrix
// Output => B : Q * P matrix
//---------------------------------------------------------------------
__global__ void k_transpose(
   double *A,
   double *B,
  const int P, const int Q)
{
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  int gi = blockIdx.y, gj = blockIdx.x;
  int li = threadIdx.y, lj = threadIdx.x;

  __shared__ double X[16][16 + 1];

  if (i < P && j < Q) {
    X[li][lj] = A[i * Q + j];
  }

  __syncthreads();

  int ni = gj * 16 + li;
  int nj = gi * 16 + lj;
  if (ni < Q && nj < P) {
    B[ni * P + nj]= X[lj][li];
  }
}

__global__ void k_scatter(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

   double (*A)[JMAXP+1][IMAXP+1][5] =
    ( double (*)[JMAXP+1][IMAXP+1][5])g_A;

   double (*A0)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A0;
   double (*A1)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A1;
   double (*A2)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A2;
   double (*A3)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A3;
   double (*A4)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A4;

  if (i < nx && j < ny && k < nz) {
    A0[k][j][i] = A[k][j][i][0];
    A1[k][j][i] = A[k][j][i][1];
    A2[k][j][i] = A[k][j][i][2];
    A3[k][j][i] = A[k][j][i][3];
    A4[k][j][i] = A[k][j][i][4];
  }
}

__global__ void k_gather(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

   double (*A)[JMAXP+1][IMAXP+1][5] =
    ( double (*)[JMAXP+1][IMAXP+1][5])g_A;

   double (*A0)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A0;
   double (*A1)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A1;
   double (*A2)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A2;
   double (*A3)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A3;
   double (*A4)[JMAXP+1][IMAXP+1] =
    ( double (*)[JMAXP+1][IMAXP+1])g_A4;

  if (i < nx && j < ny && k < nz) {
    A[k][j][i][0] = A0[k][j][i];
    A[k][j][i][1] = A1[k][j][i];
    A[k][j][i][2] = A2[k][j][i];
    A[k][j][i][3] = A3[k][j][i];
    A[k][j][i][4] = A4[k][j][i];
  }
}

__global__ void k_scatter_j(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz, const int WORK_NUM_ITEM_J)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

#define A(a, b, c, d) g_A[(((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)) * 5 + (d)]
#define A0(a, b, c) g_A0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A1(a, b, c) g_A1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A2(a, b, c) g_A2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A3(a, b, c) g_A3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A4(a, b, c) g_A4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

  if (i < nx && j < ny && k < nz) {
    A0(k, j, i) = A(k, j, i, 0);
    A1(k, j, i) = A(k, j, i, 1);
    A2(k, j, i) = A(k, j, i, 2);
    A3(k, j, i) = A(k, j, i, 3);
    A4(k, j, i) = A(k, j, i, 4);
  }

#undef A
#undef A0
#undef A1
#undef A2
#undef A3
#undef A4
#undef A0
}

__global__ void k_gather_j(
   double *g_A,
   double *g_A0,
   double *g_A1,
   double *g_A2,
   double *g_A3,
   double *g_A4,
  const int nx, const int ny, const int nz, const int WORK_NUM_ITEM_J)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

#define A(a, b, c, d) g_A[(((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)) * 5 + (d)]
#define A0(a, b, c) g_A0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A1(a, b, c) g_A1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A2(a, b, c) g_A2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A3(a, b, c) g_A3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define A4(a, b, c) g_A4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

  if (i < nx && j < ny && k < nz) {
    A(k, j, i, 0) = A0(k, j, i);
    A(k, j, i, 1) = A1(k, j, i);
    A(k, j, i, 2) = A2(k, j, i);
    A(k, j, i, 3) = A3(k, j, i);
    A(k, j, i, 4) = A4(k, j, i);
  }

#undef A
#undef A0
#undef A1
#undef A2
#undef A3
#undef A4
#undef A0
}
