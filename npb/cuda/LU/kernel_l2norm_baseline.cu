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
void k_l2norm_baseline(double *m_v,
                       double *m_sum,
                       double *m_tmp_sum,
                       int nz0,
                       int jst, int jend,
                       int ist, int iend,
                       int work_base,
                       int work_num_item, 
                       int split_flag,
                       int buffer_base)
{
  int m;
  int temp = blockDim.x * blockIdx.x + threadIdx.x;
  int i = temp % (iend - ist) + ist;
  temp = temp / (iend-ist);
  int j = temp % (jend-jst) + jst;
  int k = temp / (jend-jst);
  int l_id = threadIdx.x;
  int l_size = blockDim.x;
  int wg_id = blockIdx.x;

  int step;
  int dummy = 0;

  if (k + work_base >= nz0-1 || k+work_base < 1 || k >= work_num_item || j >= jend || i >= iend)
    dummy = 1;

  if (split_flag) 
    k += buffer_base;
  else
    k += work_base;

  double (* v)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_v;
  double (* tmp_sum)[5]
    = (double (*)[5])m_tmp_sum;
  double (* sum)[5]
    = (double (*)[5])m_sum;

  double (* t_sum)[5] = &tmp_sum[wg_id * l_size];

  for (m = 0; m < 5; m++)
    t_sum[l_id][m] = 0.0;

  if (!dummy) {
    for (m = 0; m < 5; m++)
      t_sum[l_id][m] = v[k][j][i][m] * v[k][j][i][m];
  }

  __syncthreads();

  for (step = l_size/2; step > 0; step = step >> 1) {
    if (l_id < step) {
      for (m = 0; m < 5; m++)
        t_sum[l_id][m] += t_sum[l_id+step][m];
    }
    __syncthreads();
  }

  if (l_id == 0) {
    for (m = 0; m < 5; m++)
      sum[wg_id][m] += t_sum[l_id][m];
  }
}
