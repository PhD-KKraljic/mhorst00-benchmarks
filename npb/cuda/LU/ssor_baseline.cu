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
#ifdef _OPENMP
#include <omp.h>
#endif
#include "applu.incl"
#include "timers.h"
#include <math.h>

#include "applu.incl"
extern "C" {
#include "timers.h"
#include "npbparams.h"
}

cudaEvent_t     (*ev_k_ssor1_baseline)[2],
                (*ev_k_ssor2_baseline)[2];

void ssor1_init_baseline(int iter)
{
  int i;

  ev_k_ssor1_baseline = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_ssor1_baseline[i][0]);

    cudaEventCreate(&ev_k_ssor1_baseline[i][1]);
  }
}

void ssor1_release_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_ssor1_baseline[i][0]);
    cudaEventDestroy(ev_k_ssor1_baseline[i][1]);
  }
}

void ssor1_baseline(int item, int base,
                    int step, int buf_idx,
                    cudaEvent_t *ev_wb_ptr)
{

  dim3 numBlocks;
  dim3 numThreads;
  size_t lws[3];
  size_t gws[3];

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], (int)max_work_group_size);

  gws[2] = (size_t) item;
  gws[1] = (size_t) jend - jst;
  gws[0] = (size_t) iend - ist;
  gws[0] *= 5; 

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  if (buffering_flag)
    CUCHK(cudaStreamWaitEvent(cmd_q[KERNEL_Q], *ev_wb_ptr, 0));

  CUCHK(cudaEventRecord(ev_k_ssor1_baseline[step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_ssor1_baseline",  cmd_q[KERNEL_Q] );
  k_ssor1_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q] >>>
    (m_rsd[buf_idx],
     nz, jst, jend, ist, iend, dt,
     base, item, split_flag);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_ssor1_baseline",  cmd_q[KERNEL_Q] );

  CUCHK(cudaEventRecord(ev_k_ssor1_baseline[step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

}

void ssor2_init_baseline(int iter)
{
  int i;

  ev_k_ssor2_baseline = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_ssor2_baseline[i][0]);

    cudaEventCreate(&ev_k_ssor2_baseline[i][1]);
  }

}

void ssor2_release_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_ssor2_baseline[i][0]);
    cudaEventDestroy(ev_k_ssor2_baseline[i][1]);
  }
}

void ssor2_baseline(int item, int base,
                    int step, int buf_idx,
                    int temp_kst, double tmp2)
{
  dim3 numBlocks;
  dim3 numThreads;
  size_t lws[3];
  size_t gws[3];

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = (size_t) item;
  gws[1] = (size_t) jend - jst;
  gws[0] = (size_t) iend - ist;
  gws[0] *= 5; 

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_ssor2_baseline[step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_ssor2_baseline",  cmd_q[KERNEL_Q]);
  k_ssor2_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (m_u[buf_idx],
     m_rsd[buf_idx],
     tmp2, nz, jst, jend, ist, iend, temp_kst,
     base, item, split_flag);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_ssor2_baseline",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_ssor2_baseline[step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

}

