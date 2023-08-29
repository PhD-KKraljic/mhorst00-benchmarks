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


cudaEvent_t     (*ev_k_jacld_baseline)[2],
                (*ev_k_blts_baseline)[2],
                (*ev_k_jbl_datagen_baseline)[2],
                (*ev_k_jbl_datacopy_baseline)[2];

static void jacld_baseline(int work_step,
                           int work_num_item);

static cudaEvent_t* blts_KL_baseline(int work_step, 
                                  int work_num_item);

void jacld_blts_init_baseline(int iter, int item_default,
                              int blk_size_k, int blk_size)
{
  int i;

  ev_k_jacld_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_blts_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbl_datagen_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_jbl_datacopy_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_jacld_baseline[i][0]);
    cudaEventCreate(&ev_k_blts_baseline[i][0]);
    cudaEventCreate(&ev_k_jbl_datagen_baseline[i][0]);
    cudaEventCreate(&ev_k_jbl_datacopy_baseline[i][0]);

    cudaEventCreate(&ev_k_jacld_baseline[i][1]);
    cudaEventCreate(&ev_k_blts_baseline[i][1]);
    cudaEventCreate(&ev_k_jbl_datagen_baseline[i][1]);
    cudaEventCreate(&ev_k_jbl_datacopy_baseline[i][1]);
  }
}

void jacld_blts_release_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_jacld_baseline[i][0]);
    cudaEventDestroy(ev_k_blts_baseline[i][0]);
    cudaEventDestroy(ev_k_jbl_datagen_baseline[i][0]);
    cudaEventDestroy(ev_k_jbl_datacopy_baseline[i][0]);

    cudaEventDestroy(ev_k_jacld_baseline[i][1]);
    cudaEventDestroy(ev_k_blts_baseline[i][1]);
    cudaEventDestroy(ev_k_jbl_datagen_baseline[i][1]);
    cudaEventDestroy(ev_k_jbl_datacopy_baseline[i][1]);
  }

  free(ev_k_jacld_baseline);
  free(ev_k_blts_baseline);
  free(ev_k_jbl_datagen_baseline);
  free(ev_k_jbl_datacopy_baseline);
}

cudaEvent_t* jacld_blts_body_baseline(int work_step, 
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

    CUCHK(cudaEventRecord(ev_k_jbl_datagen_baseline[work_step][0], cmd_q[KERNEL_Q]));

    cuda_ProfilerStartEventRecord("k_jbl_datagen_baseline",  cmd_q[KERNEL_Q]);
    k_jbl_datagen_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
      (
       m_u[buf_idx], m_qs[buf_idx],
       m_rho_i[buf_idx],
       kend, jend, iend,
       work_base, work_num_item
      );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_jbl_datagen_baseline",  cmd_q[KERNEL_Q]);

    CUCHK(cudaEventRecord(ev_k_jbl_datagen_baseline[work_step][1], cmd_q[KERNEL_Q]));

    if (!buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }



  jacld_baseline(work_step, work_num_item);

  ev_prop_end = blts_KL_baseline(work_step, work_num_item);

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

      CUCHK(cudaEventRecord(ev_k_jbl_datacopy_baseline[work_step][0], cmd_q[KERNEL_Q]));

      cuda_ProfilerStartEventRecord("k_jbl_datacopy_baseline",  cmd_q[KERNEL_Q]);
      k_jbl_datacopy_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
        (
         m_rsd[buf_idx], m_rsd[next_buf_idx],
         m_u[buf_idx], m_u[next_buf_idx],
         jst, jend,
         ist, iend,
         work_num_item
        );
      CUCHK(cudaGetLastError());
      cuda_ProfilerEndEventRecord("k_jbl_datacopy_baseline",  cmd_q[KERNEL_Q]);
      CUCHK(cudaEventRecord(ev_k_jbl_datacopy_baseline[work_step][1], cmd_q[KERNEL_Q]));

      if (!buffering_flag)
        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
    }
  }


  if (work_step == work_max_iter - 1)
    return ev_prop_end;
  else
    return &ev_k_jbl_datacopy_baseline[work_step][1];

}

static void jacld_baseline(int work_step, int work_num_item)
{
  int temp_kend, temp_kst;
  int buf_idx = (work_step%2)*buffering_flag;
  temp_kst = 1; 
  temp_kend = work_num_item + 1;
  size_t lws[3], gws[3];
  dim3 numBlocks, numThreads;

  gws[2] = work_num_item;
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

  CUCHK(cudaEventRecord(ev_k_jacld_baseline[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_jacld_baseline",  cmd_q[KERNEL_Q]);
  k_jacld_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
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
  cuda_ProfilerEndEventRecord("k_jacld_baseline",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(ev_k_jacld_baseline[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
}

static cudaEvent_t* blts_KL_baseline(int work_step, int work_num_item)
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

  CUCHK(cudaEventRecord(ev_k_blts_baseline[work_step][0], cmd_q[KERNEL_Q]));

  for (k = 0; k <= (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1); k++) {
    lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
    ubk = k < (temp_kend-temp_kst-1) ? k : (temp_kend-temp_kst-1);
    lbj = (k-(iend-ist-1)-(temp_kend-temp_kst)) >= 0 ? (k-(iend-ist-1)-(temp_kend-temp_kst)) : 0;
    ubj = k < (jend-jst-1) ? k : (jend-jst-1);

    lws[0] = (ubj-lbj+1) < (int)blts_max_work_items_sizes? (ubj-lbj+1) : blts_max_work_items_sizes;
    temp = blts_max_work_group_size / lws[0];
    lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
    lws[2] = 1;

    gws[2] = 1;
    gws[1] = (size_t)(ubk-lbk+1);
    gws[0] = (size_t)(ubj-lbj+1);

    gws[2] = RoundWorkSize(gws[2], lws[2]);
    gws[1] = RoundWorkSize(gws[1], lws[1]);
    gws[0] = RoundWorkSize(gws[0], lws[0]);

    numBlocks.x = gws[0] / lws[0];
    numBlocks.y = gws[1] / lws[1];
    numBlocks.z = gws[2] / lws[2];
    numThreads.x = lws[0];
    numThreads.y = lws[1];
    numThreads.z = lws[2];

    cuda_ProfilerStartEventRecord("k_blts_KL_baseline",  cmd_q[KERNEL_Q]);
    k_blts_KL_baseline<<< numBlocks, numThreads, 0, cmd_q[KERNEL_Q]>>>
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
    cuda_ProfilerEndEventRecord("k_blts_KL_baseline",  cmd_q[KERNEL_Q]);
  }

  CUCHK(cudaEventRecord(ev_k_blts_baseline[work_step][1], cmd_q[KERNEL_Q]));

  return &ev_k_blts_baseline[work_step][1];
}



               

