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

cl_kernel k_jbl_datagen_baseline,
          k_jacld_baseline,
          k_blts_KL_baseline,
          k_jbl_datacopy_baseline;

size_t    *blts_KL_prop_iter_baseline;

cl_event  *ev_k_jacld_baseline,
          (*ev_k_blts_baseline)[2],
          *ev_k_jbl_datagen_baseline,
          *ev_k_jbl_datacopy_baseline;

static void jacld_baseline(int work_step, 
                           int work_num_item);
static cl_event* blts_KL_baseline(int work_step, 
                                  int work_num_item);

void jacld_blts_init_baseline(int iter, 
                              int item_default, 
                              int blk_size_k, 
                              int blk_size)
{
  int i, tmp_work_base, tmp_work_num_item;;
  cl_int ecode;

  k_jbl_datagen_baseline = clCreateKernel(p_jacld_blts_baseline,
                                          "jacld_blts_datagen_baseline",
                                          &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_jacld_baseline = clCreateKernel(p_jacld_blts_baseline,
                                     "jacld_baseline",
                                     &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_blts_KL_baseline = clCreateKernel(p_jacld_blts_baseline,
                                     "blts_KL_baseline",
                                     &ecode);
  clu_CheckError(ecode, "clCreateKernel");


  k_jbl_datacopy_baseline = clCreateKernel(p_jacld_blts_baseline,
                                           "jacld_blts_datacopy_baseline",
                                           &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  blts_KL_prop_iter_baseline = (size_t*)malloc(sizeof(size_t)*iter);

  for (i = 0; i < iter; i++) {
    tmp_work_base     = i*item_default + 1;
    tmp_work_num_item = min(item_default, nz-1 - tmp_work_base);
    blts_KL_prop_iter_baseline[i] = tmp_work_num_item + jend-jst + iend-ist - 2; 
  }  

  ev_k_jbl_datagen_baseline   = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_jbl_datacopy_baseline  = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_jacld_baseline         = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_blts_baseline          = (cl_event (*)[2])malloc(sizeof(cl_event)*iter*2);
}

void jacld_blts_release_baseline(int iter)
{
  clReleaseKernel(k_jbl_datagen_baseline);
  clReleaseKernel(k_jacld_baseline);
  clReleaseKernel(k_blts_KL_baseline);
  clReleaseKernel(k_jbl_datacopy_baseline);

  free(blts_KL_prop_iter_baseline);

  free(ev_k_jacld_baseline);
  free(ev_k_blts_baseline);
}

void jacld_blts_release_ev_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    if (split_flag) {
      clReleaseEvent(ev_k_jbl_datagen_baseline[i]);

      if (i < iter-1)
        clReleaseEvent(ev_k_jbl_datacopy_baseline[i]);
    }

    clReleaseEvent(ev_k_jacld_baseline[i]);

    clReleaseEvent(ev_k_blts_baseline[i][0]);
    if (blts_KL_prop_iter_baseline[i] > 1)
      clReleaseEvent(ev_k_blts_baseline[i][1]);

  }
}

cl_event* jacld_blts_body_baseline(int work_step,
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

    ecode  = clSetKernelArg(k_jbl_datagen_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 1, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 2, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 3, sizeof(int), &kend);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 4, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 5, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 6, sizeof(int), &work_base);
    ecode |= clSetKernelArg(k_jbl_datagen_baseline, 7, sizeof(int), &work_num_item);
    clu_CheckError(ecode, "clSetKernelArg");

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_jbl_datagen_baseline,
                                   3, NULL, 
                                   gws, lws,
                                   0, NULL, 
                                   &ev_k_jbl_datagen_baseline[work_step]);
    clu_CheckError(ecode, "clEnqueueNDRangeKerenel");

    if (!buffering_flag)
      clFinish(cmd_q[KERNEL_Q]);
    else
      clFlush(cmd_q[KERNEL_Q]);
  }

  jacld_baseline(work_step, work_num_item);

  ev_prop_end = blts_KL_baseline(work_step, work_num_item);

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

      ecode  = clSetKernelArg(k_jbl_datacopy_baseline, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 1, sizeof(cl_mem), &m_rsd[next_buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 2, sizeof(cl_mem), &m_u[buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 3, sizeof(cl_mem), &m_u[next_buf_idx]);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 4, sizeof(int), &jst);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 5, sizeof(int), &jend);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 6, sizeof(int), &ist);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 7, sizeof(int), &iend);
      ecode |= clSetKernelArg(k_jbl_datacopy_baseline, 8, sizeof(int), &work_num_item);
      clu_CheckError(ecode, "clSetKernelArg");

      ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                     k_jbl_datacopy_baseline,
                                     2, NULL, 
                                     gws, lws,
                                     0, NULL, 
                                     &ev_k_jbl_datacopy_baseline[work_step]);
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
    return &ev_k_jbl_datacopy_baseline[work_step];

}

static void jacld_baseline(int work_step, int work_num_item)
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

  ecode  = clSetKernelArg(k_jacld_baseline, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_baseline, 1, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_baseline, 2, sizeof(cl_mem), &m_qs[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_baseline, 3, sizeof(cl_mem), &m_rho_i[buf_idx]);
  ecode |= clSetKernelArg(k_jacld_baseline, 4, sizeof(cl_mem), &m_a);
  ecode |= clSetKernelArg(k_jacld_baseline, 5, sizeof(cl_mem), &m_b);
  ecode |= clSetKernelArg(k_jacld_baseline, 6, sizeof(cl_mem), &m_c);
  ecode |= clSetKernelArg(k_jacld_baseline, 7, sizeof(cl_mem), &m_d);
  ecode |= clSetKernelArg(k_jacld_baseline, 8, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_jacld_baseline, 9, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_jacld_baseline, 10, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_jacld_baseline, 11, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_jacld_baseline, 12, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_jacld_baseline, 13, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_jacld_baseline, 14, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_jacld_baseline, 15, sizeof(int), &temp_kst);
  ecode |= clSetKernelArg(k_jacld_baseline, 16, sizeof(int), &temp_kend);
  clu_CheckError(ecode, "clSetKernelArg()");

  my_ev = &ev_k_jacld_baseline[work_step];

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_jacld_baseline, 
                                 3, NULL,
                                 gws, lws,
                                 0, NULL, 
                                 my_ev);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

}

static cl_event* blts_KL_baseline(int work_step, int work_num_item)
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

    ecode  = clSetKernelArg(k_blts_KL_baseline, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 1, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 2, sizeof(cl_mem), &m_qs[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 3, sizeof(cl_mem), &m_rho_i[buf_idx]);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 4, sizeof(cl_mem), &m_a);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 5, sizeof(cl_mem), &m_b);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 6, sizeof(cl_mem), &m_c);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 7, sizeof(cl_mem), &m_d);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 8, sizeof(int), &nz);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 9, sizeof(int), &ny);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 10, sizeof(int), &nx);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 11, sizeof(int), &k);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 12, sizeof(int), &lbk);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 13, sizeof(int), &lbj);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 14, sizeof(int), &jst);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 15, sizeof(int), &jend);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 16, sizeof(int), &ist);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 17, sizeof(int), &iend);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 18, sizeof(int), &temp_kst);
    ecode |= clSetKernelArg(k_blts_KL_baseline, 19, sizeof(int), &temp_kend);
    clu_CheckError(ecode, "clSetKernelArg()");

    lws[0] = (ubj-lbj+1) < blts_max_work_items_sizes? (ubj-lbj+1) : blts_max_work_items_sizes;
    temp = blts_max_work_group_size / lws[0];
    lws[1] = (ubk-lbk+1) < temp ? (ubk-lbk+1) : temp;
    gws[0] = clu_RoundWorkSize((size_t)(ubj-lbj+1), lws[0]);
    gws[1] = clu_RoundWorkSize((size_t)(ubk-lbk+1), lws[1]);

    if (k == 0)
      my_ev = &ev_k_blts_baseline[work_step][0];
    else if (k == (temp_kend-temp_kst-1)+(iend-ist-1)+(jend-jst-1))
      my_ev = &ev_k_blts_baseline[work_step][1];
    else
      my_ev = NULL;

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_blts_KL_baseline, 
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
