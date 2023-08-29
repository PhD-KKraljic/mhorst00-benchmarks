//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB LU code. This OpenCL C  //
//  version is a part of SNU-NPB 2019 developed by the Center for Manycore   //
//  Programming at Seoul National University and derived from the serial     //
//  Fortran versions in "NPB3.3.1-SER" developed by NAS.                     //
//                                                                           //
//  Permission to use, copy, distribute and modify this software for any     //
//  purpose with or without fee is hereby granted. This software is          //
//  provided "as is" without express or implied warranty.                    //
//                                                                           //
//  Information on original NPB 3.3.1, including the technical report, the   //
//  original specifications, source code, results and information on how     //
//  to submit new results, is available at:                                  //
//                                                                           //
//           http://www.nas.nasa.gov/Software/NPB/                           //
//                                                                           //
//  Information on SNU-NPB 2019, including the conference paper and source   //
//  code, is available at:                                                   //
//                                                                           //
//           http://aces.snu.ac.kr                                           //
//                                                                           //
//  Send comments or suggestions for this OpenCL C version to                //
//  snunpb@aces.snu.ac.kr                                                    //
//                                                                           //
//          Center for Manycore Programming                                  //
//          School of Computer Science and Engineering                       //
//          Seoul National University                                        //
//          Seoul 08826, Korea                                               //
//                                                                           //
//          E-mail: snunpb@aces.snu.ac.kr                                    //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Authors: Youngdong Do, Hyung Mo Kim, Pyeongseok Oh, Daeyoung Park,        //
//          and Jaejin Lee                                                   //
//---------------------------------------------------------------------------//

#include "applu.incl"
#include "timers.h"
#include <stdio.h>
#include <math.h>

cl_kernel k_jbl_datagen_sync,
          k_jacld_sync,
          k_blts_BR_sync,
          k_blts_KL_sync,
          k_jbl_datacopy_sync;

size_t    *blts_BR_prop_iter_sync,
          *blts_KL_prop_iter_sync;

cl_event  *ev_k_jacld_sync,
          (*ev_k_blts_BR_sync)[2],
          (*ev_k_blts_KL_sync)[2],
          *ev_k_jbl_datagen_sync,
          *ev_k_jbl_datacopy_sync;

static enum PropSyncAlgo   jbl_prop_algo_sync;

static void jacld_sync(int work_step, 
                       int work_num_item);

static cl_event* blts_BR_sync(int work_step, 
                              int work_num_item);

static cl_event* blts_KL_sync(int work_step, 
                              int work_num_item);


void jacld_blts_init_sync(int iter, 
                          int item_default, 
                          int blk_size_k, 
                          int blk_size)
{
  int i, 
      tmp_work_base, 
      tmp_work_num_item,
      num_k, num_j, num_i,
      num_blk_k,
      num_blk_j,
      num_blk_i;
  double start_t, end_t, KL_time, BR_time;

  cl_int ecode;

  k_jbl_datagen_sync = clCreateKernel(p_jacld_blts_sync,
                                      "jacld_blts_datagen_sync",
                                      &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_jacld_sync = clCreateKernel(p_jacld_blts_sync,
                                "jacld_sync",
                                &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_blts_BR_sync = clCreateKernel(p_jacld_blts_sync,
                                  "blts_BR_sync",
                                  &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_blts_KL_sync = clCreateKernel(p_jacld_blts_sync,
                                  "blts_KL_sync",
                                  &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_jbl_datacopy_sync = clCreateKernel(p_jacld_blts_sync,
                                       "jacld_blts_datacopy_sync",
                                       &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  blts_BR_prop_iter_sync = (size_t*)malloc(sizeof(size_t)*iter);
  blts_KL_prop_iter_sync = (size_t*)malloc(sizeof(size_t)*iter);

  for (i = 0; i < iter; i++) {
    tmp_work_base     = i*item_default + 1;
    tmp_work_num_item = min(item_default, nz-1 - tmp_work_base);

    num_k = tmp_work_num_item;
    num_j = jend - jst;
    num_i = iend - ist;
    num_blk_k = ((num_k-1)/blk_size_k)+1;
    num_blk_j = ((num_j-1)/blk_size)+1;
    num_blk_i = ((num_i-1)/blk_size)+1;
    blts_BR_prop_iter_sync[i] = num_blk_k + num_blk_j + num_blk_i - 2;
    blts_KL_prop_iter_sync[i] = num_k + num_j + num_i - 2;
  }  

  ev_k_jbl_datagen_sync   = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_jbl_datacopy_sync  = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_jacld_sync         = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_blts_BR_sync       = (cl_event (*)[2])malloc(sizeof(cl_event)*iter*2);
  ev_k_blts_KL_sync       = (cl_event (*)[2])malloc(sizeof(cl_event)*iter*2);

  // Propagation Strategy Selection

  // warm up before profiling
  for (i = 0; i < iter; i++) {
    tmp_work_base = i*item_default + 1;
    tmp_work_num_item = min(item_default, nz-1 - tmp_work_base);

    blts_KL_sync(i, tmp_work_num_item);
    blts_BR_sync(i, tmp_work_num_item);

    clFinish(cmd_q[KERNEL_Q]);
  }

  // profiling for KL
  timer_clear(t_jbl_KL_prof);
  start_t = timer_read(t_jbl_KL_prof);
  timer_start(t_jbl_KL_prof);

  for (i = 0; i < iter; i++) {
    tmp_work_base = i*item_default + 1;
    tmp_work_num_item = min(item_default, nz-1 - tmp_work_base);

    blts_KL_sync(i, tmp_work_num_item);

    clFinish(cmd_q[KERNEL_Q]);
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
    tmp_work_num_item = min(item_default, nz-1 - tmp_work_base);
    blts_BR_sync(i, tmp_work_num_item);

    clFinish(cmd_q[KERNEL_Q]);
  }

  timer_stop(t_jbl_BR_prof);
  end_t = timer_read(t_jbl_BR_prof);
  BR_time = end_t - start_t;

  DETAIL_LOG("BR time : %f", BR_time);

  if (KL_time < BR_time)
    jbl_prop_algo_sync = KERNEL_LAUNCH;
  else
    jbl_prop_algo_sync = BARRIER;

  if (jbl_prop_algo_sync == KERNEL_LAUNCH)
    DETAIL_LOG("jacld blts propagation strategy : Kernel Launch");
  else
    DETAIL_LOG("jacld blts propagation strategy : Kernel Launch + Work Group Barrier");

}

void jacld_blts_release_sync(int iter)
{
  clReleaseKernel(k_jbl_datagen_sync);
  clReleaseKernel(k_jacld_sync);
  clReleaseKernel(k_blts_BR_sync);
  clReleaseKernel(k_blts_KL_sync);
  clReleaseKernel(k_jbl_datacopy_sync);

  free(blts_BR_prop_iter_sync);
  free(blts_KL_prop_iter_sync);

  free(ev_k_jacld_sync);
  free(ev_k_blts_BR_sync);
  free(ev_k_blts_KL_sync);
}

void jacld_blts_release_ev_sync(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    if (split_flag) {
      clReleaseEvent(ev_k_jbl_datagen_sync[i]);

      if (i < iter-1)
        clReleaseEvent(ev_k_jbl_datacopy_sync[i]);
    }

    clReleaseEvent(ev_k_jacld_sync[i]);

    if (jbl_prop_algo_sync == BARRIER) {
      clReleaseEvent(ev_k_blts_BR_sync[i][0]);
      if (blts_BR_prop_iter_sync[i] > 1)
        clReleaseEvent(ev_k_blts_BR_sync[i][1]);
    }
    else {
      clReleaseEvent(ev_k_blts_KL_sync[i][0]);
      if (blts_KL_prop_iter_sync[i] > 1)
        clReleaseEvent(ev_k_blts_KL_sync[i][1]);
    }
  }
}



cl_event* jacld_blts_body_sync(int work_step,
                               int work_max_iter,
                               int work_base,
                               int work_num_item)
{
  cl_int ecode;
  size_t lws[3], gws[3];
  int kend = (int)nz-1;
  cl_event *ev_prop_end;
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

    gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
    gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
    gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

    ecode  = clSetKernelArg(k_jbl_datagen_sync, 0, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 1, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 2, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 3, sizeof(int), &kend);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 4, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 5, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 6, sizeof(int), &work_base);
    ecode |= clSetKernelArg(k_jbl_datagen_sync, 7, sizeof(int), &work_num_item);
    clu_CheckError(ecode, "clSetKernelArg");

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_jbl_datagen_sync,
                                   3, NULL, 
                                   gws, lws,
                                   0, NULL, 
                                   &ev_k_jbl_datagen_sync[work_step]);
    clu_CheckError(ecode, "clEnqueueNDRangeKerenel");

    if (!buffering_flag)
      clFinish(cmd_q[KERNEL_Q]);
    else
      clFlush(cmd_q[KERNEL_Q]);
  }

  jacld_sync(work_step, work_num_item);

  if (jbl_prop_algo_sync == BARRIER)
    ev_prop_end = blts_BR_sync(work_step, work_num_item);
  else 
    ev_prop_end = blts_KL_sync(work_step, work_num_item);

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);

  // THIS SHOULD BE enqueue in kernel command queue
  if (work_step < work_max_iter-1) {

    if (split_flag) {

      lws[1] = 1;
      lws[0] = max_work_group_size;

      gws[1] = jend - jst;
      gws[0] = iend - ist;
      gws[0] *= 5;

      gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
      gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

      ecode  = clSetKernelArg(k_jbl_datacopy_sync, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 1, sizeof(cl_mem), &m_rsd[next_buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 2, sizeof(cl_mem), &m_u[buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 3, sizeof(cl_mem), &m_u[next_buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 4, sizeof(int), &jst);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 5, sizeof(int), &jend);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 6, sizeof(int), &ist);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 7, sizeof(int), &iend);
      ecode |= clSetKernelArg(k_jbl_datacopy_sync, 8, sizeof(int), &work_num_item);
      clu_CheckError(ecode, "clSetKernelArg");

      ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                     k_jbl_datacopy_sync,
                                     2, NULL, 
                                     gws, lws,
                                     0, NULL, 
                                     &ev_k_jbl_datacopy_sync[work_step]);
      clu_CheckError(ecode, "clEnqueueNDRangeKernel");

      if (!buffering_flag)
        clFinish(cmd_q[KERNEL_Q]);
      else
        clFlush(cmd_q[KERNEL_Q]);
    }
  }


  if (work_step == work_max_iter - 1)
    return ev_prop_end;
  else
    return &ev_k_jbl_datacopy_sync[work_step];

}



static void jacld_sync(int work_step, int work_num_item)
{
  int temp_kend, temp_kst;
  int buf_idx = (work_step%2)*buffering_flag;
  temp_kst = 1; 
  temp_kend = work_num_item + 1;
  size_t lws[3], gws[3];
  cl_int       ecode;
  cl_event     *my_ev = NULL;

  gws[2] = work_num_item;
  gws[1] = jend - jst;
  gws[0] = iend - ist;

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_jacld_sync, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_sync, 1, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_sync, 2, sizeof(cl_mem), &m_qs[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_sync, 3, sizeof(cl_mem), &m_rho_i[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_sync, 4, sizeof(cl_mem), &m_a);
  ecode |= clSetKernelArg(k_jacld_sync, 5, sizeof(cl_mem), &m_b);
  ecode |= clSetKernelArg(k_jacld_sync, 6, sizeof(cl_mem), &m_c);
  ecode |= clSetKernelArg(k_jacld_sync, 7, sizeof(cl_mem), &m_d);
  ecode |= clSetKernelArg(k_jacld_sync, 8, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_jacld_sync, 9, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_jacld_sync, 10, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_jacld_sync, 11, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_jacld_sync, 12, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_jacld_sync, 13, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_jacld_sync, 14, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_jacld_sync, 15, sizeof(int), &temp_kst);
  ecode |= clSetKernelArg(k_jacld_sync, 16, sizeof(int), &temp_kend);
  clu_CheckError(ecode, "clSetKernelArg()");

  my_ev = &ev_k_jacld_sync[work_step];

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_jacld_sync, 
                                 3, NULL,
                                 gws, lws,
                                 0, NULL, 
                                 my_ev);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

}


static cl_event* blts_BR_sync(int work_step, int work_num_item)
{
  cl_int ecode;
  size_t lws[3], gws[3];
  int temp_kst, temp_kend,
      num_k, num_j, num_i,
      num_block_k, num_block_j, num_block_i,
      wg00_block_k, wg00_block_j, wg00_block_i,
      wg00_head_k, wg00_head_j, wg00_head_i,
      depth, diagonals,
      num_wg,
      iter, max_iter;

  int buf_idx = (work_step%2)*buffering_flag;
  cl_event *my_ev = NULL;
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
    gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

    ecode  = clSetKernelArg(k_blts_BR_sync, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
    ecode |= clSetKernelArg(k_blts_BR_sync, 1, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_blts_BR_sync, 2, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_blts_BR_sync, 3, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_blts_BR_sync, 4, sizeof(cl_mem), &m_a);
    ecode |= clSetKernelArg(k_blts_BR_sync, 5, sizeof(cl_mem), &m_b);
    ecode |= clSetKernelArg(k_blts_BR_sync, 6, sizeof(cl_mem), &m_c);
    ecode |= clSetKernelArg(k_blts_BR_sync, 7, sizeof(cl_mem), &m_d);
    ecode |= clSetKernelArg(k_blts_BR_sync, 8, sizeof(int), &temp_kst);
    ecode |= clSetKernelArg(k_blts_BR_sync, 9, sizeof(int), &temp_kend);
    ecode |= clSetKernelArg(k_blts_BR_sync, 10, sizeof(int), &jst);
    ecode |= clSetKernelArg(k_blts_BR_sync, 11, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_blts_BR_sync, 12, sizeof(int), &ist);
    ecode |= clSetKernelArg(k_blts_BR_sync, 13, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_blts_BR_sync, 14, sizeof(int), &wg00_head_k);
    ecode |= clSetKernelArg(k_blts_BR_sync, 15, sizeof(int), &wg00_head_j);
    ecode |= clSetKernelArg(k_blts_BR_sync, 16, sizeof(int), &wg00_head_i);
    ecode |= clSetKernelArg(k_blts_BR_sync, 17, sizeof(int), &wg00_block_k);
    ecode |= clSetKernelArg(k_blts_BR_sync, 18, sizeof(int), &wg00_block_j);
    ecode |= clSetKernelArg(k_blts_BR_sync, 19, sizeof(int), &wg00_block_i);
    ecode |= clSetKernelArg(k_blts_BR_sync, 20, sizeof(int), &num_block_k);
    ecode |= clSetKernelArg(k_blts_BR_sync, 21, sizeof(int), &num_block_j);
    ecode |= clSetKernelArg(k_blts_BR_sync, 22, sizeof(int), &num_block_i);
    ecode |= clSetKernelArg(k_blts_BR_sync, 23, sizeof(int), &block_size);
    ecode |= clSetKernelArg(k_blts_BR_sync, 24, sizeof(int), &block_size_k);
    clu_CheckError(ecode, "clSetKernelArg");

    if (iter == 0)
      my_ev = &ev_k_blts_BR_sync[work_step][0];
    else if (iter == max_iter-1)
      my_ev = &ev_k_blts_BR_sync[work_step][1];
    else
      my_ev = NULL;

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_blts_BR_sync,
                                   1, NULL, 
                                   gws, lws,
                                   0, NULL, 
                                   my_ev);
    clu_CheckError(ecode , "clEnqueueNDRangeKerenel");

    if (buffering_flag)
      clFlush(cmd_q[KERNEL_Q]);

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
  return my_ev;
}

static cl_event* blts_KL_sync(int work_step, int work_num_item)
{

  int k;
  int temp;
  int temp_kend, temp_kst;
  int buf_idx = (work_step%2)*buffering_flag;
  temp_kst = 1; 
  temp_kend = work_num_item + 1;

  size_t lws[3], gws[3];
  int lbk, ubk, lbj, ubj;

  cl_int       ecode;
  cl_event     *my_ev = NULL;

  size_t blts_max_work_group_size = 64;
  size_t blts_max_work_items_sizes = 64;

  for (k = 0; k <= (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1); k++) {
    lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
    ubk = k < (temp_kend-temp_kst-1) ? k : (temp_kend-temp_kst-1);
    lbj = (k-(iend-ist-1)-(temp_kend-temp_kst)) >= 0 ? (k-(iend-ist-1)-(temp_kend-temp_kst)) : 0;
    ubj = k < (jend-jst-1) ? k : (jend-jst-1);

    ecode  = clSetKernelArg(k_blts_KL_sync, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_sync, 1, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_sync, 2, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_sync, 3, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_sync, 4, sizeof(cl_mem), &m_a);
    ecode |= clSetKernelArg(k_blts_KL_sync, 5, sizeof(cl_mem), &m_b);
    ecode |= clSetKernelArg(k_blts_KL_sync, 6, sizeof(cl_mem), &m_c);
    ecode |= clSetKernelArg(k_blts_KL_sync, 7, sizeof(cl_mem), &m_d);
    ecode |= clSetKernelArg(k_blts_KL_sync, 8, sizeof(int), &nz);
    ecode |= clSetKernelArg(k_blts_KL_sync, 9, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_blts_KL_sync, 10, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_blts_KL_sync, 11, sizeof(int), &k);
    ecode |= clSetKernelArg(k_blts_KL_sync, 12, sizeof(int), &lbk);
    ecode |= clSetKernelArg(k_blts_KL_sync, 13, sizeof(int), &lbj);
    ecode |= clSetKernelArg(k_blts_KL_sync, 14, sizeof(int), &jst);
    ecode |= clSetKernelArg(k_blts_KL_sync, 15, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_blts_KL_sync, 16, sizeof(int), &ist);
    ecode |= clSetKernelArg(k_blts_KL_sync, 17, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_blts_KL_sync, 18, sizeof(int), &temp_kst);
    ecode |= clSetKernelArg(k_blts_KL_sync, 19, sizeof(int), &temp_kend);
    clu_CheckError(ecode, "clSetKernelArg()");

    lws[0] = (ubj-lbj+1) < blts_max_work_items_sizes? (ubj-lbj+1) : blts_max_work_items_sizes;
    temp = blts_max_work_group_size / lws[0];
    lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
    gws[0] = clu_RoundWorkSize((size_t)(ubj-lbj+1), lws[0]);
    gws[1] = clu_RoundWorkSize((size_t)(ubk-lbk+1), lws[1]);

    if (k == 0)
      my_ev = &ev_k_blts_KL_sync[work_step][0];
    else if (k == (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1))
      my_ev = &ev_k_blts_KL_sync[work_step][1];
    else
      my_ev = NULL;

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_blts_KL_sync, 
                                   2, NULL,
                                   gws, lws,
                                   0, NULL, 
                                   my_ev);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    if (buffering_flag)
      clFlush(cmd_q[KERNEL_Q]);
  }

  return my_ev;
}
