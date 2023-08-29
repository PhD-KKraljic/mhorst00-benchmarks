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

cudaEvent_t                 (*ev_k_jbu_datagen_fullopt)[2],
                            (*ev_k_jbu_BR_fullopt)[2],
                            (*ev_k_jbu_KL_fullopt)[2];

// This is for BR version
size_t                      *jacu_buts_prop_iter_fullopt,
                            *jacu_buts_work_num_item_fullopt;

// This is for KL version
size_t                      *jacu_buts_prop_iter_KL_fullopt;

static enum PropSyncAlgo    jbu_prop_algo_fullopt;


static void jacu_buts_prop_barrier(int work_step, int temp_kst, int temp_kend);
static void jacu_buts_prop_kernel(int work_step, int temp_kst, int temp_kend);

void jacu_buts_init_fullopt(int iter, int item_default,
                            int blk_size_k, int blk_size)
{
  int tmp_work_end, 
      tmp_work_num_item,
      tmp_work_base,
      tmp_kend, tmp_kst,
      kst, kend,
      num_k, num_j, num_i,
      num_blk_k, num_blk_j, num_blk_i,
      i;
  double start_t, end_t, KL_time, BR_time;

  // This is for BR version
  jacu_buts_prop_iter_fullopt = (size_t*)malloc(sizeof(size_t)*iter);
  jacu_buts_work_num_item_fullopt = (size_t*)malloc(sizeof(size_t)*iter);

  // This is for KL version
  jacu_buts_prop_iter_KL_fullopt = (size_t*)malloc(sizeof(size_t)*iter);

  kst = 1; kend = nz-1;

  for (i = 0; i < iter; i++) {
    tmp_work_end = nz - i*item_default;

    if (split_flag) {
      tmp_work_num_item = min(item_default, tmp_work_end);
      tmp_work_base = tmp_work_end - tmp_work_num_item;

      jacu_buts_work_num_item_fullopt[i] = min(tmp_work_num_item, kend - tmp_work_base);
      tmp_kst = 2;

      jacu_buts_work_num_item_fullopt[i] += min(2, tmp_work_base - 1);
      tmp_kst -= min(2, tmp_work_base - 1);
      tmp_kend = tmp_kst + jacu_buts_work_num_item_fullopt[i];
    }
    else {
      tmp_work_num_item = nz;
      tmp_work_base = 0;

      jacu_buts_work_num_item_fullopt[i] = kend - kst;
      tmp_kst = kst;
      tmp_kend = kend;
    }

    num_k = tmp_kend - tmp_kst;
    num_j = jend - jst;
    num_i = iend - jst;

    num_blk_k = ((num_k - 1)/blk_size_k) + 1;
    num_blk_j = ((num_j - 1)/blk_size) + 1;
    num_blk_i = ((num_i - 1)/blk_size) + 1;

    // THis is for BR version
    jacu_buts_prop_iter_fullopt[i] = num_blk_k + num_blk_j + num_blk_i - 2;

    // This is for KL version
    jacu_buts_prop_iter_KL_fullopt[i] = num_k + num_j + num_i - 2;
  }

  ev_k_jbu_datagen_fullopt  = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbu_BR_fullopt       = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbu_KL_fullopt       = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_jbu_datagen_fullopt[i][0]);
    cudaEventCreate(&ev_k_jbu_BR_fullopt[i][0]);
    cudaEventCreate(&ev_k_jbu_KL_fullopt[i][0]);

    cudaEventCreate(&ev_k_jbu_datagen_fullopt[i][1]);
    cudaEventCreate(&ev_k_jbu_BR_fullopt[i][1]);
    cudaEventCreate(&ev_k_jbu_KL_fullopt[i][1]);
  }
 
  // Propagation Strategy Selection

  // warm up before profiling
  for (i = 0; i < iter; i++) {
    tmp_work_end = nz - i*item_default;

    if (split_flag) {
      tmp_work_num_item = min(item_default, tmp_work_end);
      tmp_work_base = tmp_work_end - tmp_work_num_item;
      tmp_kst = 2;
      tmp_kst -= min(2, tmp_work_base - 1);
      tmp_kend = tmp_kst + jacu_buts_work_num_item_fullopt[i];
    }
    else {
      tmp_work_num_item = nz;
      tmp_work_base = 0;

      tmp_kst = kst;
      tmp_kend = kend;
    }

    jacu_buts_prop_kernel(i, tmp_kst, tmp_kend);
    jacu_buts_prop_barrier(i, tmp_kst, tmp_kend);

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  // profiling for KL
  timer_clear(t_jbu_KL_prof);
  start_t = timer_read(t_jbu_KL_prof);
  timer_start(t_jbu_KL_prof);

  for (i = 0; i < iter; i++) {
    tmp_work_end = nz - i*item_default;

    if (split_flag) {
      tmp_work_num_item = min(item_default, tmp_work_end);
      tmp_work_base = tmp_work_end - tmp_work_num_item;

      tmp_kst = 2;
      tmp_kst -= min(2, tmp_work_base - 1);
      tmp_kend = tmp_kst + jacu_buts_work_num_item_fullopt[i];
    }
    else {
      tmp_work_num_item = nz;
      tmp_work_base = 0;

      tmp_kst = kst;
      tmp_kend = kend;
    }

    jacu_buts_prop_kernel(i, tmp_kst, tmp_kend);

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  timer_stop(t_jbu_KL_prof);
  end_t = timer_read(t_jbu_KL_prof);
  KL_time = end_t - start_t;

  DETAIL_LOG("KL time : %f", KL_time);

  // profiling for BR
  timer_clear(t_jbu_BR_prof);
  start_t = timer_read(t_jbu_BR_prof);
  timer_start(t_jbu_BR_prof);

  for (i = 0; i < iter; i++) {
    tmp_work_end = nz - i*item_default;

    if (split_flag) {
      tmp_work_num_item = min(item_default, tmp_work_end);
      tmp_work_base = tmp_work_end - tmp_work_num_item;
      tmp_kst = 2;
      tmp_kst -= min(2, tmp_work_base - 1);
      tmp_kend = tmp_kst + jacu_buts_work_num_item_fullopt[i];
    }
    else {
      tmp_work_num_item = nz;
      tmp_work_base = 0;

      tmp_kst = kst;
      tmp_kend = kend;
    }

    jacu_buts_prop_barrier(i, tmp_kst, tmp_kend);

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  timer_stop(t_jbu_BR_prof);
  end_t = timer_read(t_jbu_BR_prof);
  BR_time = end_t - start_t;

  DETAIL_LOG("BR time : %f", BR_time);

  if (KL_time < BR_time)
    jbu_prop_algo_fullopt = KERNEL_LAUNCH;
  else
    jbu_prop_algo_fullopt = BARRIER;

  if (jbu_prop_algo_fullopt == KERNEL_LAUNCH)
    DETAIL_LOG("jacu buts computation policy : Kernel launch");
  else
    DETAIL_LOG("jacu buts computation policy : Kernel launch + barrier");

}

void jacu_buts_release_fullopt(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_jbu_datagen_fullopt[i][0]);
    cudaEventDestroy(ev_k_jbu_BR_fullopt[i][0]);
    cudaEventDestroy(ev_k_jbu_KL_fullopt[i][0]);

    cudaEventDestroy(ev_k_jbu_datagen_fullopt[i][1]);
    cudaEventDestroy(ev_k_jbu_BR_fullopt[i][1]);
    cudaEventDestroy(ev_k_jbu_KL_fullopt[i][1]);
  }

  free(ev_k_jbu_datagen_fullopt);
  free(ev_k_jbu_BR_fullopt);
  free(ev_k_jbu_KL_fullopt);

  free(jacu_buts_prop_iter_fullopt);
  free(jacu_buts_work_num_item_fullopt);
  free(jacu_buts_prop_iter_KL_fullopt);
}

void jacu_buts_body_fullopt(int work_step, 
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

  // ##################
  //  Kernel Execution
  // ##################

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

    CUCHK(cudaEventRecord(ev_k_jbu_datagen_fullopt[work_step][0], cmd_q[KERNEL_Q]));

    cuda_ProfilerStartEventRecord("k_jacu_buts_data_gen_fullopt",  cmd_q[KERNEL_Q]);
    k_jacu_buts_data_gen_fullopt<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
      (m_u[buf_idx],
       m_qs[buf_idx],
       m_rho_i[buf_idx],
       jst, jend, ist, iend, temp_kst, temp_kend, work_num_item);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jacu_buts_data_gen_fullopt",  cmd_q[KERNEL_Q]);
    
    CUCHK(cudaEventRecord(ev_k_jbu_datagen_fullopt[work_step][1], cmd_q[KERNEL_Q]));

    if (split_flag && !buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }


  if (jbu_prop_algo_fullopt == KERNEL_LAUNCH)
    jacu_buts_prop_kernel(work_step, temp_kst, temp_kend);
  else
    jacu_buts_prop_barrier(work_step, temp_kst, temp_kend);

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

static void jacu_buts_prop_barrier(int work_step, int temp_kst, int temp_kend)
{

  dim3 numBlocks, numThreads;
  size_t lws[3], gws[3];
  int num_k, num_j, num_i;
  int num_block_k, num_block_j, num_block_i;
  int wg00_tail_k, wg00_tail_j, wg00_tail_i;
  int wg00_block_k, wg00_block_j, wg00_block_i;
  int iter;
  int diagonals;
  int depth;
  int num_wg;
  int buf_idx = (work_step%2)*buffering_flag;


  wg00_tail_k = temp_kend-1;
  wg00_tail_j = jend-1;
  wg00_tail_i = iend-1;

  num_k = temp_kend - temp_kst;
  num_j = jend - jst;
  num_i = iend - ist;

  num_block_k = ( ( num_k - 1) / block_size_k ) + 1;
  num_block_j = ( ( num_j - 1) / block_size ) + 1;
  num_block_i = ( ( num_i - 1) / block_size ) + 1;



  CUCHK(cudaEventRecord(ev_k_jbu_BR_fullopt[work_step][0], cmd_q[KERNEL_Q]));

  //blocking iteration

  for (iter = 0; iter < num_block_k + num_block_j + num_block_i - 2; iter++) {

    wg00_block_k = (temp_kend-1 - wg00_tail_k) / block_size_k;
    wg00_block_j = (jend-1 - wg00_tail_j) / block_size;
    wg00_block_i = (iend-1 - wg00_tail_i) / block_size;

    //CAUTION !!!!!
    //block indexing direction and element indexing(in one block) 
    //direction is opposite!!!
    num_wg = 0;

    // diagonals = the number of active diagonals
    diagonals = min(wg00_block_k+1, num_block_j+num_block_i-1 - (wg00_block_j+wg00_block_i));

    for (depth = 0 ; depth < diagonals; depth++) {
      num_wg += min(wg00_block_j+1, num_block_i - wg00_block_i);

      wg00_block_j++;
      if (wg00_block_j >= num_block_j) {
        wg00_block_j--;
        wg00_block_i++;
      }
    }

    wg00_block_k = (temp_kend-1 - wg00_tail_k) / block_size_k;
    wg00_block_j = (jend-1 - wg00_tail_j) / block_size;
    wg00_block_i = (iend-1 - wg00_tail_i) / block_size;

    lws[1] = (size_t)block_size;
    lws[0] = (size_t)block_size;

    gws[1] = (size_t)block_size*num_wg;
    gws[0] = (size_t)block_size;

    lws[0] = jacu_buts_lws;
    gws[0] = lws[0]*num_wg;

    gws[1] = RoundWorkSize(gws[1], lws[1]);
    gws[0] = RoundWorkSize(gws[0], lws[0]);

    numBlocks.x = gws[0] / lws[0];
    numBlocks.y = 1;
    numBlocks.z = 1;
    numThreads.x = lws[0];
    numThreads.y = 1;
    numThreads.z = 1;

    cuda_ProfilerStartEventRecord("k_jacu_buts_BR_fullopt",  cmd_q[KERNEL_Q]);
    k_jacu_buts_BR_fullopt<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
      (m_rsd[buf_idx],
       m_rho_i[buf_idx],
       m_u[buf_idx],
       m_qs[buf_idx],
       temp_kst, temp_kend, jst, jend, ist, iend,
       wg00_tail_k, wg00_tail_j, wg00_tail_i,
       wg00_block_k, wg00_block_j, wg00_block_i,
       num_block_k, num_block_j, num_block_i,
       block_size, block_size_k
      );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jacu_buts_BR_fullopt",  cmd_q[KERNEL_Q]);


    wg00_tail_k -= block_size_k;
    if (wg00_tail_k < temp_kst) {
      wg00_tail_k += block_size_k;
      wg00_tail_j -= block_size;
      if (wg00_tail_j < jst) {
        wg00_tail_j += block_size;
        wg00_tail_i -= block_size;
      }
    }
  }

  CUCHK(cudaEventRecord(ev_k_jbu_BR_fullopt[work_step][1], cmd_q[KERNEL_Q]));
}

static void jacu_buts_prop_kernel(int work_step, int temp_kst, int temp_kend)
{

  dim3 numBlocks, numThreads;
  int lbk, ubk, lbj, ubj, temp, k;
  size_t buts_max_work_group_size = 64;
  size_t buts_max_work_item_sizes0 = 64;
  size_t buts_gws[3], buts_lws[3];
  int buf_idx = (work_step%2)*buffering_flag;

  CUCHK(cudaEventRecord(ev_k_jbu_KL_fullopt[work_step][0], cmd_q[KERNEL_Q]));

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

    cuda_ProfilerStartEventRecord("k_jacu_buts_KL_fullopt",  cmd_q[KERNEL_Q]);
    k_jacu_buts_KL_fullopt<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>(
        m_rsd[buf_idx],
        m_u[buf_idx],
        m_qs[buf_idx],
        m_rho_i[buf_idx],
        nz, ny, nx,
        k, lbk, lbj, 
        jst, jend, ist, iend,
        temp_kst, temp_kend
        );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jacu_buts_KL_fullopt",  cmd_q[KERNEL_Q]);
  }

  CUCHK(cudaEventRecord(ev_k_jbu_KL_fullopt[work_step][1], cmd_q[KERNEL_Q]));
}

