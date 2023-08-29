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
// compute the lower triangular part of the jacobian matrix
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block lower triangular solution:
// 
// v <-- ( L-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------


cudaEvent_t         (*ev_k_jbl_datagen_fullopt)[2],
                    (*ev_k_jbl_datacopy_fullopt)[2],
                    (*ev_k_jbl_BR_fullopt)[2],
                    (*ev_k_jbl_KL_fullopt)[2];

// This is for BR version
size_t              *jacld_blts_prop_iter_fullopt,
                    *jacld_blts_work_num_item_fullopt;

// This is for KL version
size_t              *jacld_blts_prop_iter_KL_fullopt;

static enum PropSyncAlgo   jbl_prop_algo_fullopt;

static cudaEvent_t* jacld_blts_prop_barrier(int work_step, 
                                            int work_num_item);
static cudaEvent_t* jacld_blts_prop_kernel(int work_step, 
                                           int work_num_item);

void jacld_blts_init_fullopt(int iter, int item_default, 
                             int blk_size_k, int blk_size)
{
  int i, tmp_work_base, tmp_work_item,
      num_k, num_j, num_i,
      num_blk_k, num_blk_j, num_blk_i;
  double start_t, end_t, KL_time, BR_time;

  // This if for BR version
  jacld_blts_prop_iter_fullopt = (size_t*)malloc(sizeof(size_t)*iter);
  jacld_blts_work_num_item_fullopt = (size_t*)malloc(sizeof(size_t)*iter);

  // This is for KL version
  jacld_blts_prop_iter_KL_fullopt = (size_t*)malloc(sizeof(size_t)*iter);

  for (i = 0; i < iter; i++) {
    tmp_work_base = i*item_default + 1;
    jacld_blts_work_num_item_fullopt[i] = min(item_default, nz-1 - tmp_work_base);

    num_k = jacld_blts_work_num_item_fullopt[i];
    num_j = jend - jst;
    num_i = iend - ist;

    num_blk_k = ((num_k - 1)/blk_size_k) + 1;
    num_blk_j = ((num_j - 1)/blk_size) + 1;
    num_blk_i = ((num_i - 1)/blk_size) + 1;

    // This is for BR version
    jacld_blts_prop_iter_fullopt[i] = num_blk_k + num_blk_j + num_blk_i - 2;

    // This is for KL version
    jacld_blts_prop_iter_KL_fullopt[i] = jacld_blts_work_num_item_fullopt[i] + num_j + num_i - 2;
  }

  /* memory allocation for event objects */
  ev_k_jbl_datagen_fullopt = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbl_datacopy_fullopt = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbl_BR_fullopt = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbl_KL_fullopt = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_jbl_datagen_fullopt[i][0]);
    cudaEventCreate(&ev_k_jbl_datacopy_fullopt[i][0]);
    cudaEventCreate(&ev_k_jbl_BR_fullopt[i][0]);
    cudaEventCreate(&ev_k_jbl_KL_fullopt[i][0]);

    cudaEventCreate(&ev_k_jbl_datagen_fullopt[i][1]);
    cudaEventCreate(&ev_k_jbl_datacopy_fullopt[i][1]);
    cudaEventCreate(&ev_k_jbl_BR_fullopt[i][1]);
    cudaEventCreate(&ev_k_jbl_KL_fullopt[i][1]);
  }

  // Propagation Strategy Selection

  // warm up before profiling
  for (i = 0; i < iter; i++) {
    tmp_work_base = i*item_default + 1;
    tmp_work_item = min(item_default, nz-1 - tmp_work_base);

    jacld_blts_prop_kernel(i, tmp_work_item);
    jacld_blts_prop_barrier(i, tmp_work_item);

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  // profiling for KL
  timer_clear(t_jbl_KL_prof);
  start_t = timer_read(t_jbl_KL_prof);
  timer_start(t_jbl_KL_prof);

  for (i = 0; i < iter; i++) {
    tmp_work_base = i*item_default + 1;
    tmp_work_item = min(item_default, nz-1 - tmp_work_base);

    jacld_blts_prop_kernel(i, tmp_work_item);

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  timer_stop(t_jbl_KL_prof);
  end_t = timer_read(t_jbl_KL_prof);
  KL_time = end_t - start_t;

  DETAIL_LOG("KL time : %f", KL_time);

  // profiling for BR
  timer_clear(t_jbl_BR_prof);
  start_t = timer_read(t_jbl_BR_prof);
  timer_start(t_jbl_BR_prof);

  for (i = 0; i < iter; i++) {
    tmp_work_base = i*item_default + 1;
    tmp_work_item = min(item_default, nz-1 - tmp_work_base);

    jacld_blts_prop_barrier(i, tmp_work_item);

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  timer_stop(t_jbl_BR_prof);
  end_t = timer_read(t_jbl_BR_prof);
  BR_time = end_t - start_t;

  DETAIL_LOG("BR time : %f", BR_time);

  if (KL_time < BR_time)
    jbl_prop_algo_fullopt = KERNEL_LAUNCH;
  else
    jbl_prop_algo_fullopt = BARRIER;


  if (jbl_prop_algo_fullopt == KERNEL_LAUNCH)
    DETAIL_LOG("jacld blts computation policy : Kernel launch");
  else
    DETAIL_LOG("jacld blts computation policy : Kernel launch + barrier");

}

void jacld_blts_release_fullopt(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_jbl_datagen_fullopt[i][0]);
    cudaEventDestroy(ev_k_jbl_datacopy_fullopt[i][0]);
    cudaEventDestroy(ev_k_jbl_BR_fullopt[i][0]);
    cudaEventDestroy(ev_k_jbl_KL_fullopt[i][0]);

    cudaEventDestroy(ev_k_jbl_datagen_fullopt[i][1]);
    cudaEventDestroy(ev_k_jbl_datacopy_fullopt[i][1]);
    cudaEventDestroy(ev_k_jbl_BR_fullopt[i][1]);
    cudaEventDestroy(ev_k_jbl_KL_fullopt[i][1]);
  }

  free(ev_k_jbl_datagen_fullopt);
  free(ev_k_jbl_datacopy_fullopt);
  free(ev_k_jbl_BR_fullopt);
  free(ev_k_jbl_KL_fullopt);

  // This is for BR version
  free(jacld_blts_prop_iter_fullopt);
  free(jacld_blts_work_num_item_fullopt);

  // This is for KL version
  free(jacld_blts_prop_iter_KL_fullopt);
}

cudaEvent_t* jacld_blts_body_fullopt(int work_step, 
                                     int work_max_iter, 
                                     int work_base, 
                                     int work_num_item)
{
  cudaEvent_t* ev_prop_end;
  dim3 numBlocks, numThreads;
  size_t lws[3], gws[3];
  int kend = (int)nz-1;
  int buf_idx = (work_step%2)*buffering_flag;
  int next_buf_idx = ((work_step+1)%2)*buffering_flag;



  // ##################
  //  Kernel Execution
  // ##################

  if(split_flag){

    lws[2] = 1;
    lws[1] = 1;
    lws[0] = max_work_group_size;

    gws[2] = work_num_item + 1;
    gws[1] = jend;
    gws[0] = iend;

    gws[2] = RoundWorkSize(gws[2], lws[2]);
    gws[1] = RoundWorkSize(gws[1], lws[1]);
    gws[0] = RoundWorkSize(gws[0], lws[0]);

    numBlocks.x = gws[0] / lws[0];
    numBlocks.y = gws[1] / lws[1];
    numBlocks.z = gws[2] / lws[2];
    numThreads.x = lws[0];
    numThreads.y = lws[1];
    numThreads.z = lws[2];

    CUCHK(cudaEventRecord(ev_k_jbl_datagen_fullopt[work_step][0], cmd_q[KERNEL_Q]));

    cuda_ProfilerStartEventRecord("k_jacld_blts_data_gen_fullopt",  cmd_q[KERNEL_Q]);
    k_jacld_blts_data_gen_fullopt<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
      (m_u[buf_idx],
       m_qs[buf_idx],
       m_rho_i[buf_idx],
       kend, jend, iend, work_base, work_num_item);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jacld_blts_data_gen_fullopt",  cmd_q[KERNEL_Q]);

    CUCHK(cudaEventRecord(ev_k_jbl_datagen_fullopt[work_step][1], cmd_q[KERNEL_Q]));

    if (!buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  }




  if (jbl_prop_algo_fullopt == KERNEL_LAUNCH)
    ev_prop_end = jacld_blts_prop_kernel(work_step, work_num_item);
  else
    ev_prop_end = jacld_blts_prop_barrier(work_step, work_num_item);

  if (!buffering_flag && split_flag) 
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));


  // THIS SHOULD BE enqueue in kernel command queue
  if (work_step < work_max_iter-1) {
    if (split_flag) {

      lws[1] = 1;
      lws[0] = max_work_group_size;

      gws[1] = jend - jst;
      gws[0] = iend - ist;
      gws[0] *= 5;

      gws[1] = RoundWorkSize(gws[1], lws[1]);
      gws[0] = RoundWorkSize(gws[0], lws[0]);

      numBlocks.x = gws[0] / lws[0];
      numBlocks.y = gws[1] / lws[1];
      numBlocks.z = 1;
      numThreads.x = lws[0];
      numThreads.y = lws[1];
      numThreads.z = 1;

      CUCHK(cudaEventRecord(ev_k_jbl_datacopy_fullopt[work_step][0], cmd_q[KERNEL_Q]));

      cuda_ProfilerStartEventRecord("k_jacld_blts_data_copy_fullopt",  cmd_q[KERNEL_Q]);
      k_jacld_blts_data_copy_fullopt<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
        (m_rsd[buf_idx], m_rsd[next_buf_idx],
         m_u[buf_idx], m_u[next_buf_idx],
         jst, jend, ist, iend, work_num_item
        );
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_jacld_blts_data_copy_fullopt",  cmd_q[KERNEL_Q]);

      CUCHK(cudaEventRecord(ev_k_jbl_datacopy_fullopt[work_step][1], cmd_q[KERNEL_Q]));

      if (!buffering_flag)
        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    }
  }


  if (work_step == work_max_iter - 1)
    return ev_prop_end;
  else
    return &ev_k_jbl_datacopy_fullopt[work_step][1];

}

static cudaEvent_t* jacld_blts_prop_barrier(int work_step, int work_num_item)
{

  size_t lws[3], gws[3];
  int temp_kst, temp_kend;
  int num_k, num_j, num_i;
  int num_block_k, num_block_j, num_block_i;
  int wg00_block_k, wg00_block_j, wg00_block_i;
  int wg00_head_k, wg00_head_j, wg00_head_i;
  int depth, diagonals;
  int iter, max_iter;
  int num_wg;
  int buf_idx = (work_step%2)*buffering_flag;

  temp_kst = 1; temp_kend = work_num_item + 1;

  num_k = temp_kend - temp_kst; 
  num_j = jend - jst;
  num_i = iend - ist;

  num_block_k = ( ( num_k - 1) / block_size_k ) + 1;
  num_block_j = ( ( num_j - 1) / block_size ) + 1;
  num_block_i = ( ( num_i - 1) / block_size ) + 1;

  wg00_head_k = temp_kst;
  wg00_head_j = jst;
  wg00_head_i = ist;


  max_iter = num_block_k + num_block_j + num_block_i - 2;

  CUCHK(cudaEventRecord(ev_k_jbl_BR_fullopt[work_step][0], cmd_q[KERNEL_Q]));

  // blocking iteration
  for (iter = 0; iter < max_iter ; iter++) {

    num_wg = 0;

    wg00_block_k = (wg00_head_k-temp_kst) / block_size_k;
    wg00_block_j = (wg00_head_j-jst) / block_size;
    wg00_block_i = (wg00_head_i-ist) / block_size;


    // calculate the number of work-group
    // diagonals = the number of active diagonals
    diagonals  = min(wg00_block_k+1, num_block_j+num_block_i-1 - (wg00_block_j+wg00_block_i));

    for(depth = 0; depth < diagonals; depth++){
      // addition factor is the number of blocks in current diagonal
      num_wg += min(wg00_block_j+1, num_block_i - wg00_block_i);

      wg00_block_j++;
      if(wg00_block_j >= num_block_j){
        wg00_block_j--;
        wg00_block_i++;
      }

    }

    // reset the current block position
    wg00_block_k = (wg00_head_k-temp_kst) / block_size_k;
    wg00_block_j = (wg00_head_j-jst) / block_size;
    wg00_block_i = (wg00_head_i-ist) / block_size;


    lws[0] = jacld_blts_lws;
    gws[0] = lws[0]*num_wg;

    gws[0] = RoundWorkSize(gws[0], lws[0]);

    cuda_ProfilerStartEventRecord("k_jacld_blts_BR_fullopt",  cmd_q[KERNEL_Q]);
    k_jacld_blts_BR_fullopt<<< gws[0] / lws[0], lws[0], 0, cmd_q[KERNEL_Q]>>>
      (m_rsd[buf_idx],
       m_rho_i[buf_idx],
       m_u[buf_idx],
       m_qs[buf_idx],
       temp_kst, temp_kend, jst, jend, ist, iend, wg00_head_k,
       wg00_head_j, wg00_head_i, wg00_block_k, wg00_block_j,
       wg00_block_i, num_block_k, num_block_j, num_block_i,
       block_size, block_size_k
      );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jacld_blts_BR_fullopt",  cmd_q[KERNEL_Q]);

    wg00_head_k += block_size_k;
    if (wg00_head_k >= temp_kend) {
      wg00_head_k -= block_size_k;
      wg00_head_j += block_size;
      if (wg00_head_j >= jend) {
        wg00_head_j -= block_size;
        wg00_head_i += block_size;
      }
    }

  }

  CUCHK(cudaEventRecord(ev_k_jbl_BR_fullopt[work_step][1], cmd_q[KERNEL_Q]));


  return &ev_k_jbl_BR_fullopt[work_step][1];



}
static cudaEvent_t* jacld_blts_prop_kernel(int work_step, int work_num_item)
{

  int k;
  int temp;
  int temp_kend, temp_kst;
  int buf_idx = (work_step%2)*buffering_flag;

  temp_kst = 1; 
  temp_kend = work_num_item + 1;

  dim3 numBlocks, numThreads;
  size_t lws[3], gws[3];
  int lbk, ubk, lbj, ubj;


  size_t blts_max_work_group_size = 64;
  size_t blts_max_work_items_sizes = 64;

  CUCHK(cudaEventRecord(ev_k_jbl_KL_fullopt[work_step][0], cmd_q[KERNEL_Q]));

  for (k = 0; k <= (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1); k++) {
    lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
    ubk = k < (temp_kend-temp_kst-1) ? k : (temp_kend-temp_kst-1);
    lbj = (k-(iend-ist-1)-(temp_kend-temp_kst)) >= 0 ? (k-(iend-ist-1)-(temp_kend-temp_kst)) : 0;
    ubj = k < (jend-jst-1) ? k : (jend-jst-1);

    lws[0] = (ubj-lbj+1) < (int)blts_max_work_items_sizes? (ubj-lbj+1) : blts_max_work_items_sizes;
    temp = blts_max_work_group_size / lws[0];
    lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;

    gws[0] = (size_t)(ubj-lbj+1);
    gws[1] = (size_t)(ubk-lbk+1);

    gws[1] = RoundWorkSize(gws[1], lws[1]);
    gws[0] = RoundWorkSize(gws[0], lws[0]);

    numBlocks.x = gws[0] / lws[0];
    numBlocks.y = gws[1] / lws[1];
    numBlocks.z = 1;

    numThreads.x = lws[0];
    numThreads.y = lws[1];
    numThreads.z = 1;

    cuda_ProfilerStartEventRecord("k_jacld_blts_KL_fullopt",  cmd_q[KERNEL_Q]);
    k_jacld_blts_KL_fullopt<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>(
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
    cuda_ProfilerEndEventRecord("k_jacld_blts_KL_fullopt",  cmd_q[KERNEL_Q]);

  }

  CUCHK(cudaEventRecord(ev_k_jbl_KL_fullopt[work_step][1], cmd_q[KERNEL_Q]));

  return &ev_k_jbl_KL_fullopt[work_step][1];

}
