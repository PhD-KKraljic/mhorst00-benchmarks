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

#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "applu.incl"
extern "C" {
#include "timers.h"
}

cudaEvent_t   (*ev_k_rhs1_parallel)[2],
              (*ev_k_rhs1_datagen_parallel)[2],
              (*ev_k_rhsx1_parallel)[2],
              (*ev_k_rhsx2_parallel)[2],
              (*ev_k_rhsx3_parallel)[2],
              (*ev_k_rhsy_parallel)[2],
              (*ev_k_rhsz_parallel)[2];

void rhs_init_parallel(int iter)
{
  int i;

  ev_k_rhs1_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhs1_datagen_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsx1_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsx2_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsx3_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsy_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsz_parallel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_rhs1_parallel[i][0]);
    cudaEventCreate(&ev_k_rhs1_datagen_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsx1_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsx2_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsx3_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsy_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsz_parallel[i][0]);

    cudaEventCreate(&ev_k_rhs1_parallel[i][1]);
    cudaEventCreate(&ev_k_rhs1_datagen_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsx1_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsx2_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsx3_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsy_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsz_parallel[i][1]);
  }

}

void rhs_release_parallel(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_rhs1_parallel[i][0]);
    cudaEventDestroy(ev_k_rhs1_datagen_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsx1_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsx2_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsx3_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsy_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsz_parallel[i][0]);

    cudaEventDestroy(ev_k_rhs1_parallel[i][1]);
    cudaEventDestroy(ev_k_rhs1_datagen_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsx1_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsx2_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsx3_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsy_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsz_parallel[i][1]);
  }

  free(ev_k_rhs1_parallel);
  free(ev_k_rhs1_datagen_parallel);
  free(ev_k_rhsx1_parallel);
  free(ev_k_rhsx2_parallel);
  free(ev_k_rhsx3_parallel);
  free(ev_k_rhsy_parallel);
  free(ev_k_rhsz_parallel);
}

cudaEvent_t* rhs_body_parallel(int work_step, 
                               int work_base, 
                               int work_num_item, 
                               int copy_buffer_base, 
                               int copy_num_item, 
                               cudaEvent_t* ev_wb_ptr)
{
  size_t lws[3], gws[3];
  dim3 numBlocks, numThreads;

  int buf_idx = (work_step%2)*buffering_flag;

  if (timeron) timer_start(t_rhs);

  // ################
  // kernel execution
  // ################
  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], (int)max_work_group_size);

  gws[2] = (size_t) work_num_item;
  gws[1] = (size_t) ny;
  gws[0] = (size_t) nx*5;

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

  CUCHK(cudaEventRecord(ev_k_rhs1_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhs1_parallel",  cmd_q[KERNEL_Q]);
  k_rhs1_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_rsd[buf_idx], m_frct[buf_idx],
     nx, ny, nz,
     work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhs1_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhs1_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], (int)max_work_group_size);

  gws[2] = (size_t) copy_num_item;
  gws[1] = (size_t) ny;
  gws[0] = (size_t) nx;
 
  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhs1_datagen_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhs1_datagen_parallel",  cmd_q[KERNEL_Q]);
  k_rhs1_datagen_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_u[buf_idx], m_rho_i[buf_idx],
     m_qs[buf_idx],
     nx, ny, nz,
     copy_buffer_base, copy_num_item
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhs1_datagen_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhs1_datagen_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));


  if (timeron) timer_start(t_rhsx);
  //---------------------------------------------------------------------
  // xi-direction flux differences
  //---------------------------------------------------------------------
  lws[2] = 1; 
  lws[1] = 1; 
  lws[0] = max_work_group_size;
  lws[0] = min(iend-ist, (int)lws[0]);

  gws[2] = (size_t) work_num_item;
  gws[1] = (size_t) (jend - jst);
  gws[0] = (size_t) (iend - ist);

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsx1_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhsx1_parallel",  cmd_q[KERNEL_Q]);
  k_rhsx1_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_flux, m_u[buf_idx],
     m_rho_i[buf_idx], m_qs[buf_idx],
     m_rsd[buf_idx],
     jst, jend,
     ist, iend,
     tx2, nz,
     work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhsx1_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhsx1_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;
  lws[0] = min(nx-ist, (int)lws[0]);

  gws[2] = (size_t) work_num_item;
  gws[1] = (size_t) (jend - jst);
  gws[0] = (size_t) (nx - ist);

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsx2_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhsx2_parallel",  cmd_q[KERNEL_Q]);
  k_rhsx2_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_flux, m_u[buf_idx],
     m_rsd[buf_idx], m_rho_i[buf_idx],
     jst, jend,
     ist, iend,
     tx1, tx2, tx3,
     dx1, dx2, dx3, dx4, dx5,
     nx, nz,
     work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhsx2_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhsx2_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;
  lws[0] = min((nx-2)*5, (int)lws[0]);

  gws[2] = (size_t) work_num_item;
  gws[1] = (size_t) (jend - jst);
  gws[0] = (size_t) (nx-2)*5;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsx3_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhsx3_parallel",  cmd_q[KERNEL_Q]);
  k_rhsx3_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_u[buf_idx], m_rsd[buf_idx],
     jst, jend,
     dssp, nx, nz,
     work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhsx3_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhsx3_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));


  if (timeron) timer_stop(t_rhsx);


  if (timeron) timer_start(t_rhsy);
  //---------------------------------------------------------------------
  // eta-direction flux differences
  //---------------------------------------------------------------------

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(work_num_item, (int)max_work_group_size);

  gws[2] = 1;
  gws[1] = 1;
  gws[0] = work_num_item;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsy_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhsy_parallel",  cmd_q[KERNEL_Q]);
  k_rhsy_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_flux, m_u[buf_idx],
     m_rho_i[buf_idx], m_qs[buf_idx],
     m_rsd[buf_idx],
     ty1, ty2, ty3,
     dy1, dy2, dy3, dy4, dy5,
     dssp, nz, ny,
     jst, jend,
     ist, iend,
     work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhsy_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhsy_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  if (timeron) timer_stop(t_rhsy);



  if (timeron) timer_start(t_rhsz);
  //---------------------------------------------------------------------
  // zeta-direction flux differences
  //---------------------------------------------------------------------

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], iend-ist);
  lws[0] = min(lws[0], max_work_group_size);

  gws[2] = 1;
  gws[1] = (size_t) (jend - jst);
  gws[0] = (size_t) (iend - ist);

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsz_parallel[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_rhsz_parallel",  cmd_q[KERNEL_Q]);
  k_rhsz_parallel<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_flux, m_utmp, m_rtmp,
     m_u[buf_idx], m_rho_i[buf_idx],
     m_qs[buf_idx], m_rsd[buf_idx],
     tz1, tz2, tz3,
     dz1, dz2, dz3, dz4, dz5,
     dssp, nz,
     jst, jend,
     ist, iend,
     work_base, work_num_item, 
     split_flag, work_num_item_default
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_rhsz_parallel",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_rhsz_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  if (timeron) timer_stop(t_rhsz);

  if (timeron) timer_stop(t_rhs);

  return &ev_k_rhsz_parallel[work_step][1];
}
