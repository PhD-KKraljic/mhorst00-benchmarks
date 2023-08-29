//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB LU code. This CUDA® C  //
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

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "applu.incl"
extern "C" {
#include "timers.h"
#include "npbparams.h"
}

  __global__ 
void k_ssor1_baseline(double *m_rsd,
                      int nz,
                      int jst,
                      int jend,
                      int ist,
                      int iend,
                      double dt,
                      int work_base,
                      int work_num_item,
                      int split_flag)
{

  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y + jst;
  int t_i = blockDim.x * blockIdx.x + threadIdx.x;
  int i = ( t_i / 5 ) + ist;
  int m = t_i % 5; 
  if (k+work_base >= nz - 1 || k >= work_num_item || j >= jend || i >= iend) return; 

  if (!split_flag) k += work_base;
  else k += 1; // for alignment


  double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] 
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;


  rsd[k][j][i][m] = dt * rsd[k][j][i][m];
}


__global__ 
void k_ssor2_baseline(double *m_u,
                      double *m_rsd,
                      double tmp2,
                      int nz,
                      int jst,
                      int jend,
                      int ist,
                      int iend,
                      int temp_kst,
                      int work_base,
                      int work_num_item,
                      int split_flag)
{

  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y + jst;
  int t_i = blockDim.x * blockIdx.x + threadIdx.x;
  int i = ( t_i / 5 ) + ist;
  int m = t_i % 5; 

  if (k+work_base >= nz - 1 || k+work_base < 1 || k >= work_num_item || j >= jend || i >= iend) return; 

  if (!split_flag) k += work_base;
  else k += temp_kst;

  double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] 
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;


  u[k][j][i][m] = u[k][j][i][m] + tmp2 * rsd[k][j][i][m];

}
