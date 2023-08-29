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
#include "applu.incl"

extern "C" {
#include "timers.h"
}
//---------------------------------------------------------------------
// compute the upper triangular part of the jacobian matrix
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block upper triangular solution:
// 
// v <-- ( U-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------


cudaEvent_t       (*ev_k_jacu_baseline)[2],
                  (*ev_k_buts_KL_baseline)[2],
                  (*ev_k_jbu_datagen_baseline)[2];

static void jacu_baseline(int work_step,
                          int temp_kst,
                          int temp_kend);

static void buts_KL_baseline(int work_step,
                             int temp_kst,
                             int temp_kend);

void jacu_buts_init_baseline(int iter, int item_default,
                             int blk_size_k, int blk_size)
{
  int i;

  ev_k_jacu_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_buts_KL_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbu_datagen_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_jacu_baseline[i][0]);
    cudaEventCreate(&ev_k_buts_KL_baseline[i][0]);
    cudaEventCreate(&ev_k_jbu_datagen_baseline[i][0]);

    cudaEventCreate(&ev_k_jacu_baseline[i][1]);
    cudaEventCreate(&ev_k_buts_KL_baseline[i][1]);
    cudaEventCreate(&ev_k_jbu_datagen_baseline[i][1]);
  }
}

void jacu_buts_release_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_jacu_baseline[i][0]);
    cudaEventDestroy(ev_k_buts_KL_baseline[i][0]);
    cudaEventDestroy(ev_k_jbu_datagen_baseline[i][0]);

    cudaEventDestroy(ev_k_jacu_baseline[i][1]);
    cudaEventDestroy(ev_k_buts_KL_baseline[i][1]);
    cudaEventDestroy(ev_k_jbu_datagen_baseline[i][1]);
  }

  free(ev_k_jacu_baseline);
  free(ev_k_buts_KL_baseline);
  free(ev_k_jbu_datagen_baseline);
}

void jacu_buts_body_baseline(int work_step, 
                             int work_max_iter, 
                             int work_num_item, 
                             int next_work_num_item, 
                             int temp_kst, int temp_kend,
                             cudaEvent_t *ev_wb_ptr)
{
  dim3 numBlocks, numThreads;
  size_t lws[3], gws[3];
  int buf_idx = (work_step%2)*buffering_flag;

  // recover "u" value before update
  if (split_flag && work_step > 0) {

    CUCHK(cudaEventRecord(loop2_ev_copy_u_prev[work_step][0], cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(((char*)m_u[buf_idx])+sizeof(double)*temp_kend*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          m_u_prev, 
                          sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          cudaMemcpyDeviceToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaEventRecord(loop2_ev_copy_u_prev[work_step][1], cmd_q[KERNEL_Q]));

    CUCHK(cudaEventRecord(loop2_ev_copy_r_prev[work_step][0], cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(((char*)m_rsd[buf_idx])+sizeof(double)*temp_kend*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          m_r_prev, 
                          sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          cudaMemcpyDeviceToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaEventRecord(loop2_ev_copy_r_prev[work_step][1], cmd_q[KERNEL_Q]));


    if (!buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  if (split_flag) {

    lws[2] = 1;
    lws[1] = 1;
    lws[0] = max_work_group_size;

    gws[2] = temp_kend - temp_kst + 1;
    gws[1] = jend - jst + 1;
    gws[0] = iend - ist + 1;

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

    CUCHK(cudaEventRecord(ev_k_jbu_datagen_baseline[work_step][0], cmd_q[KERNEL_Q]));

    cuda_ProfilerStartEventRecord("k_jbu_datagen_baseline",  cmd_q[KERNEL_Q]);
    k_jbu_datagen_baseline<<<numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
      (
       m_u[buf_idx],
       m_qs[buf_idx],
       m_rho_i[buf_idx],
       jst, jend, ist, iend, 
       temp_kst, temp_kend, 
       work_num_item
       );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jbu_datagen_baseline",  cmd_q[KERNEL_Q]);

    CUCHK(cudaEventRecord(ev_k_jbu_datagen_baseline[work_step][1], cmd_q[KERNEL_Q]));

    if (split_flag && !buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }


  jacu_baseline(work_step, temp_kst, temp_kend);

  // Propagation Kernel call
  buts_KL_baseline(work_step, temp_kst, temp_kend);
 
  if (!buffering_flag && split_flag) 
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  // store temporal data
  // store "u" value before update
  if (split_flag && work_step < loop2_work_max_iter - 1) {

    CUCHK(cudaEventRecord(loop2_ev_copy_u[work_step][0], cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_u_prev, 
                          ((char*)m_u[buf_idx])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5, 
                          sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          cudaMemcpyDeviceToDevice, 
                          cmd_q[KERNEL_Q]));
    CUCHK(cudaEventRecord(loop2_ev_copy_u[work_step][1], cmd_q[KERNEL_Q]));

    CUCHK(cudaEventRecord(loop2_ev_copy_rsd[work_step][0], cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_r_prev, 
                          ((char*)m_rsd[buf_idx])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                          cudaMemcpyDeviceToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaEventRecord(loop2_ev_copy_rsd[work_step][1], cmd_q[KERNEL_Q]));

    if (!buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

}


static void jacu_baseline(int work_step,
                          int temp_kst,
                          int temp_kend)
{

  dim3 numBlocks, numThreads;
  size_t gws[3], lws[3];
  int buf_idx = (work_step%2)*buffering_flag;

  if (temp_kend - temp_kst == 0)
    return;

  gws[2] = temp_kend - temp_kst;
  gws[1] = jend - jst;
  gws[0] = iend - ist;

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  numBlocks.x = gws[0] / lws[0];
  numBlocks.y = gws[1] / lws[1];
  numBlocks.z = gws[2] / lws[2];
  numThreads.x = lws[0];
  numThreads.y = lws[1];
  numThreads.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_jacu_baseline[work_step][0], cmd_q[KERNEL_Q]));
    
  cuda_ProfilerStartEventRecord("k_jacu_baseline",  cmd_q[KERNEL_Q]);
  k_jacu_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
    (
     m_rsd[buf_idx], m_u[buf_idx],
     m_qs[buf_idx], m_rho_i[buf_idx],
     m_a, m_b, m_c, m_d,
     nz, ny, nx,
     jst, jend,
     ist, iend,
     temp_kst, temp_kend
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_jacu_baseline",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_jacu_baseline[work_step][1], cmd_q[KERNEL_Q]));

  if (split_flag && !buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

}

static void buts_KL_baseline(int work_step,
                             int temp_kst,
                             int temp_kend)
{

  dim3 numBlocks, numThreads;
  int lbk, ubk, lbj, ubj, temp, k;
  size_t buts_max_work_group_size = 64;
  size_t buts_max_work_item_sizes0 = 64;
  size_t buts_gws[3], buts_lws[3];
  int buf_idx = (work_step%2)*buffering_flag;

  CUCHK(cudaEventRecord(ev_k_buts_KL_baseline[work_step][0], cmd_q[KERNEL_Q]));

  for (k = (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1); k >= 0; k--) {
    lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
    ubk = k < (temp_kend-temp_kst-1) ? k : (temp_kend-temp_kst);
    lbj = (k-(iend-ist-1)-(temp_kend-temp_kst)) >= 0 ? (k-(iend-ist-1)-(temp_kend-temp_kst)) : 0;
    ubj = k < (jend-jst-1) ? k : (jend-jst-1);

    buts_lws[0] = (ubj-lbj+1) < (int)buts_max_work_item_sizes0 ? (ubj-lbj+1) : buts_max_work_item_sizes0;
    temp = buts_max_work_group_size / buts_lws[0];
    buts_lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
    buts_gws[0] = (size_t)(ubj-lbj+1);
    buts_gws[1] = (size_t)(ubk-lbk+1);

    buts_gws[1] = RoundWorkSize(buts_gws[1], buts_lws[1]);
    buts_gws[0] = RoundWorkSize(buts_gws[0], buts_lws[0]);

    numBlocks.x = buts_gws[0] / buts_lws[0];
    numBlocks.y = buts_gws[1] / buts_lws[1];
    numBlocks.z = 1;
    numThreads.x = buts_lws[0];
    numThreads.y = buts_lws[1];
    numThreads.z = 1;

    cuda_ProfilerStartEventRecord("k_buts_KL_baseline",  cmd_q[KERNEL_Q]);
    k_buts_KL_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
      (
       m_rsd[buf_idx], m_u[buf_idx],
       m_qs[buf_idx], m_rho_i[buf_idx],
       m_a, m_b, m_c, m_d,
       nz, ny, nx,
       k, lbk,
       lbj, jst, jend,
       ist, iend,
       temp_kst, temp_kend
      );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_buts_KL_baseline",  cmd_q[KERNEL_Q]);
  }

  CUCHK(cudaEventRecord(ev_k_buts_KL_baseline[work_step][1], cmd_q[KERNEL_Q]));
}

