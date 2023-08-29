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
__global__ void k_transpose_opt(
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
