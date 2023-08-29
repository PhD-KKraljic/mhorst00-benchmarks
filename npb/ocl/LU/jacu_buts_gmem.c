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

#include <stdio.h>
#include "applu.incl"
#include "timers.h"
#include <math.h>

cl_kernel k_jbu_datagen_gmem,
          k_jbu_KL_gmem;

cl_event  *ev_k_jbu_datagen_gmem,
          (*ev_k_jbu_KL_gmem)[2];

size_t    *jbu_KL_prop_iter_gmem;

static void jacu_buts_prop_kernel(int work_step,
                                  int temp_kst,
                                  int temp_kend);

void jacu_buts_init_gmem(int iter,
                         int item_default,
                         int blk_size_k,
                         int blk_size)
{
  cl_int ecode;
  int tmp_work_end,
      tmp_work_num_item,
      tmp_work_base,
      tmp_kend, tmp_kst,
      kst, kend,
      i;

  size_t jbu_work_num_item;

  k_jbu_datagen_gmem = clCreateKernel(p_jacu_buts_gmem,
                                      "jacu_buts_datagen_gmem",
                                      &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_jbu_KL_gmem = clCreateKernel(p_jacu_buts_gmem,
                                 "jacu_buts_KL_gmem",
                                 &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  jbu_KL_prop_iter_gmem = (size_t*)malloc(sizeof(size_t)*iter);

  kst = 1; kend = nz-1;

  for (i = 0; i < iter; i++) {
    tmp_work_end = nz - i*item_default;

    if (split_flag) {
      tmp_work_num_item = min(item_default, tmp_work_end);
      tmp_work_base = tmp_work_end - tmp_work_num_item;
      jbu_work_num_item = min(tmp_work_num_item, kend - tmp_work_base);
      tmp_kst = 2 ;

      jbu_work_num_item += min(2, tmp_work_base-1);
      tmp_kst -= min(2, tmp_work_base-1);
      tmp_kend = tmp_kst + jbu_work_num_item;
    }
    else {
      tmp_kst = kst;
      tmp_kend = kend;
    }

    jbu_KL_prop_iter_gmem[i] = tmp_kend-tmp_kst + jend-jst + iend-ist - 2;
  }

  ev_k_jbu_datagen_gmem   = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_jbu_KL_gmem        = (cl_event(*)[2])malloc(sizeof(cl_event)*iter*2);

}

void jacu_buts_release_gmem(int iter)
{ 
  clReleaseKernel(k_jbu_datagen_gmem);
  clReleaseKernel(k_jbu_KL_gmem);

  free(ev_k_jbu_KL_gmem);
  free(ev_k_jbu_datagen_gmem);

  free(jbu_KL_prop_iter_gmem);
}

void jacu_buts_release_ev_gmem(int iter)
{
  int i;
  for (i = 0; i < iter; i++) {
    if (split_flag) {
      clReleaseEvent(ev_k_jbu_datagen_gmem[i]);
    }

    clReleaseEvent(ev_k_jbu_KL_gmem[i][0]);
    if (jbu_KL_prop_iter_gmem[i] > 1)
      clReleaseEvent(ev_k_jbu_KL_gmem[i][1]);
  }
}

void jacu_buts_body_gmem(int work_step,
                         int work_max_iter,
                         int work_num_item,
                         int next_work_num_item,
                         int temp_kst,
                         int temp_kend,
                         cl_event *ev_wb_ptr)
{
  cl_int ecode;
  size_t lws[3], gws[3];

  int num_wait;
  cl_event *wait_ev;
  int buf_idx = (work_step%2)*buffering_flag;

  // ##################
  //  Kernel Execution
  // ##################

  if (split_flag) {

    // recover "u" value before update
    if (work_step > 0) {
      ecode = clEnqueueCopyBuffer(cmd_q[KERNEL_Q], 
                                  m_u_prev, 
                                  m_u[buf_idx],
                                  0, 
                                  u_slice_size*temp_kend,
                                  u_slice_size*2,
                                  0, NULL, 
                                  &loop2_ev_copy_u_prev[work_step]);
      clu_CheckError(ecode, "clEnqueueCopyBuffer ssor2 loop");

      ecode = clEnqueueCopyBuffer(cmd_q[KERNEL_Q], 
                                  m_r_prev, 
                                  m_rsd[buf_idx],
                                  0, 
                                  rsd_slice_size*temp_kend,
                                  rsd_slice_size*2,
                                  0, NULL, 
                                  &loop2_ev_copy_r_prev[work_step]);
      clu_CheckError(ecode, "clEnqueueCopyBuffer ssor2 loop");

      if (buffering_flag)
        clFlush(cmd_q[KERNEL_Q]);
      else
        clFinish(cmd_q[KERNEL_Q]);
    }

    lws[2] = 1;
    lws[1] = 1;
    lws[0] = max_work_group_size;

    gws[2] = temp_kend - temp_kst + 1;
    gws[1] = jend - jst + 1;
    gws[0] = iend - ist + 1;

    gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
    gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
    gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

    ecode  = clSetKernelArg(k_jbu_datagen_gmem, 0, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 1, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 2, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 3, sizeof(int), &jst);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 4, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 5, sizeof(int), &ist);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 6, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 7, sizeof(int), &temp_kst);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 8, sizeof(int), &temp_kend);
    ecode |= clSetKernelArg(k_jbu_datagen_gmem, 9, sizeof(int), &work_num_item);
    clu_CheckError(ecode, "clSetKernelArg");

    num_wait = (buffering_flag) ? 1 : 0;
    wait_ev = (buffering_flag) ? ev_wb_ptr : NULL;

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_jbu_datagen_gmem,
                                   3, NULL, 
                                   gws, lws,
                                   num_wait, wait_ev, 
                                   &ev_k_jbu_datagen_gmem[work_step]);
    clu_CheckError(ecode, "clEnqueueNDRangeKerenel");

    if (!buffering_flag)
      clFinish(cmd_q[KERNEL_Q]);
    else
      clFlush(cmd_q[KERNEL_Q]);
  }

  jacu_buts_prop_kernel(work_step, temp_kst, temp_kend);
 
  if (!buffering_flag) 
    clFinish(cmd_q[KERNEL_Q]);

  if (split_flag) {
    // store temporal data
    if (work_step < work_max_iter - 1) {
      ecode = clEnqueueCopyBuffer(cmd_q[KERNEL_Q], 
                                  m_u[buf_idx], 
                                  m_u_prev,
                                  u_slice_size*2,
                                  0, 
                                  u_slice_size*2,
                                  0, NULL, 
                                  &loop2_ev_copy_u[work_step]);
      clu_CheckError(ecode, "clEnqueueCopyBuffer ssor2 loop");

      if (buffering_flag)
        clFlush(cmd_q[KERNEL_Q]);
      else
        clFinish(cmd_q[KERNEL_Q]);

      ecode = clEnqueueCopyBuffer(cmd_q[KERNEL_Q], 
                                  m_rsd[buf_idx], 
                                  m_r_prev,
                                  rsd_slice_size*2,
                                  0, 
                                  rsd_slice_size*2,
                                  0, NULL, 
                                  &loop2_ev_copy_rsd[work_step]);
      clu_CheckError(ecode, "clEnqueueCopyBuffer ssor2 loop");

      if (buffering_flag)
        clFlush(cmd_q[KERNEL_Q]);
      else
        clFinish(cmd_q[KERNEL_Q]);
    }
  }
}


static void jacu_buts_prop_kernel(int work_step,
                                  int temp_kst,
                                  int temp_kend)
{

  int lbk, ubk, lbj, ubj, temp, k;
  size_t buts_max_work_group_size = 64;
  size_t buts_max_work_item_sizes0 = 64;
  size_t buts_gws[3], buts_lws[3];
  int buf_idx = (work_step%2)*buffering_flag;
  cl_int ecode;
  cl_event *my_ev = NULL;

  for (k = (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1); k >= 0; k--) {
    lbk = (k-(iend-ist-1)-(jend-jst-1)) >= 0 ? (k-(iend-ist-1)-(jend-jst-1)) : 0;
    ubk = k < (temp_kend-temp_kst-1) ? k : (temp_kend-temp_kst);
    lbj = (k-(iend-ist-1)-(temp_kend-temp_kst)) >= 0 ? (k-(iend-ist-1)-(temp_kend-temp_kst)) : 0;
    ubj = k < (jend-jst-1) ? k : (jend-jst-1);

    ecode  = clSetKernelArg(k_jbu_KL_gmem, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 1, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 2, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 3, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 4, sizeof(int), &nz);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 5, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 6, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 7, sizeof(int), &k);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 8, sizeof(int), &lbk);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 9, sizeof(int), &lbj);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 10, sizeof(int), &jst);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 11, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 12, sizeof(int), &ist);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 13, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 14, sizeof(int), &temp_kst);
    ecode |= clSetKernelArg(k_jbu_KL_gmem, 15, sizeof(int), &temp_kend);
    clu_CheckError(ecode, "clSetKernelArg()");

    buts_lws[0] = (ubj-lbj+1) < buts_max_work_item_sizes0 ? (ubj-lbj+1) : buts_max_work_item_sizes0;
    temp = buts_max_work_group_size / buts_lws[0];
    buts_lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
    buts_gws[0] = clu_RoundWorkSize((size_t)(ubj-lbj+1), buts_lws[0]);
    buts_gws[1] = clu_RoundWorkSize((size_t)(ubk-lbk+1), buts_lws[1]);

    if (k == (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1))
      my_ev = &ev_k_jbu_KL_gmem[work_step][0];
    else if (k == 0)
      my_ev = &ev_k_jbu_KL_gmem[work_step][1];
    else
      my_ev = NULL;

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_jbu_KL_gmem, 
                                   2, NULL,
                                   buts_gws, buts_lws,
                                   0, NULL, 
                                   my_ev);
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    if (buffering_flag)
      clFlush(cmd_q[KERNEL_Q]);

  }

}



