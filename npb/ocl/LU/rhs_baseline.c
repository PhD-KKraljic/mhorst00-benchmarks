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
#include <math.h>
#include <stdio.h>

cl_kernel k_rhs1_baseline,
          k_rhs1_datagen_baseline,
          k_rhsx_baseline,
          k_rhsy_baseline,
          k_rhsz_baseline;

cl_event  *ev_k_rhs1_baseline,
          *ev_k_rhs1_datagen_baseline,
          *ev_k_rhsx_baseline,
          *ev_k_rhsy_baseline,
          *ev_k_rhsz_baseline;

void rhs_init_baseline(int iter)
{
  cl_int ecode;

  k_rhs1_baseline = clCreateKernel(p_rhs_baseline, 
                                   "rhs1_baseline",
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_rhs1_datagen_baseline = clCreateKernel(p_rhs_baseline, 
                                           "rhs1_datagen_baseline",
                                           &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_rhsx_baseline = clCreateKernel(p_rhs_baseline,
                                   "rhsx_baseline",
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_rhsy_baseline = clCreateKernel(p_rhs_baseline,
                                   "rhsy_baseline",
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  k_rhsz_baseline = clCreateKernel(p_rhs_baseline,
                                   "rhsz_baseline",
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel");

  ev_k_rhs1_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhs1_datagen_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsx_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsy_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void rhs_release_baseline()
{
  clReleaseKernel(k_rhs1_baseline);
  clReleaseKernel(k_rhs1_datagen_baseline);
  clReleaseKernel(k_rhsx_baseline);
  clReleaseKernel(k_rhsy_baseline);
  clReleaseKernel(k_rhsz_baseline);

  free(ev_k_rhs1_baseline);
  free(ev_k_rhs1_datagen_baseline);
  free(ev_k_rhsx_baseline);
  free(ev_k_rhsy_baseline);
  free(ev_k_rhsz_baseline);
}

void rhs_release_ev_baseline(int iter)
{
  int i;
  for (i = 0; i < iter; i++) {
    clReleaseEvent(ev_k_rhs1_baseline[i]);
    clReleaseEvent(ev_k_rhs1_datagen_baseline[i]);
    clReleaseEvent(ev_k_rhsx_baseline[i]);
    clReleaseEvent(ev_k_rhsy_baseline[i]);
    clReleaseEvent(ev_k_rhsz_baseline[i]);
  }
}

cl_event* rhs_body_baseline(int work_step,
                            int work_base,
                            int work_num_item,
                            int copy_buffer_base,
                            int copy_num_item,
                            cl_event *ev_wb_ptr)
{
  cl_int ecode;
  size_t lws[3], gws[3];
  int num_wait;
  cl_event *wait_ev;
  int buf_idx = (work_step%2)*buffering_flag;

  if (timeron) timer_start(t_rhs);

  // ################
  // kernel execution
  // ################
  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], max_work_group_size);

  gws[2] = (size_t) work_num_item;
  gws[1] = (size_t) ny;
  gws[0] = (size_t) nx*5;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhs1_baseline, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 1, sizeof(cl_mem), &m_frct[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 2, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_rhs1_baseline, 3, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_rhs1_baseline, 4, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_rhs1_baseline, 5, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhs1_baseline, 6, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhs1_baseline, 7, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg");

  num_wait = (buffering_flag) ? 1 : 0;
  wait_ev = (buffering_flag) ? ev_wb_ptr : NULL;

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs1_baseline,
                                 3, NULL, 
                                 gws, lws,
                                 num_wait, wait_ev, 
                                 &ev_k_rhs1_baseline[work_step]);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel");

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], max_work_group_size);

  gws[2] = (size_t) copy_num_item;
  gws[1] = (size_t) ny;
  gws[0] = (size_t) nx;
 
  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhs1_datagen_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 1, sizeof(cl_mem), &m_rho_i[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 2, sizeof(cl_mem), &m_qs[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 3, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 4, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 5, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 6, sizeof(int), &copy_buffer_base);
  ecode |= clSetKernelArg(k_rhs1_datagen_baseline, 7, sizeof(int), &copy_num_item);
  clu_CheckError(ecode, "clSetKernelArg");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs1_datagen_baseline,
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &ev_k_rhs1_datagen_baseline[work_step]);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel");

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_start(t_rhsx);

  lws[1] = 1;
  lws[0] = min(jend - jst, max_work_item_sizes[0]);
  lws[0] = min(lws[0], max_work_group_size);

  gws[1] = work_num_item;
  gws[0] = jend - jst;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsx_baseline, 0, sizeof(cl_mem), &m_flux);
  ecode |= clSetKernelArg(k_rhsx_baseline, 1, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 2, sizeof(cl_mem), &m_rho_i[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 3, sizeof(cl_mem), &m_qs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 4, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 5, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_rhsx_baseline, 6, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_rhsx_baseline, 7, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_rhsx_baseline, 8, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_rhsx_baseline, 9, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_rhsx_baseline, 10, sizeof(int), &nx);
  ecode |= clSetKernelArg(k_rhsx_baseline, 11, sizeof(double), &tx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 12, sizeof(double), &tx2);
  ecode |= clSetKernelArg(k_rhsx_baseline, 13, sizeof(double), &tx3);
  ecode |= clSetKernelArg(k_rhsx_baseline, 14, sizeof(double), &dx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 15, sizeof(double), &dx2);
  ecode |= clSetKernelArg(k_rhsx_baseline, 16, sizeof(double), &dx3);
  ecode |= clSetKernelArg(k_rhsx_baseline, 17, sizeof(double), &dx4);
  ecode |= clSetKernelArg(k_rhsx_baseline, 18, sizeof(double), &dx5);
  ecode |= clSetKernelArg(k_rhsx_baseline, 19, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsx_baseline, 20, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsx_baseline, 21, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsx_baseline, 22, sizeof(int), &split_flag);

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsx_baseline,
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &ev_k_rhsx_baseline[work_step]);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel");


  if (timeron) timer_stop(t_rhsx);


  if (timeron) timer_start(t_rhsy);
  //---------------------------------------------------------------------
  // eta-direction flux differences
  //---------------------------------------------------------------------

  lws[0] = min(work_num_item, max_work_group_size);

  gws[0] = work_num_item;

  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsy_baseline, 0, sizeof(cl_mem), &m_flux);
  ecode |= clSetKernelArg(k_rhsy_baseline, 1, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 2, sizeof(cl_mem), &m_rho_i[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 3, sizeof(cl_mem), &m_qs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 4, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 5, sizeof(double), &ty1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 6, sizeof(double), &ty2);
  ecode |= clSetKernelArg(k_rhsy_baseline, 7, sizeof(double), &ty3);
  ecode |= clSetKernelArg(k_rhsy_baseline, 8, sizeof(double), &dy1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 9, sizeof(double), &dy2);
  ecode |= clSetKernelArg(k_rhsy_baseline, 10, sizeof(double), &dy3);
  ecode |= clSetKernelArg(k_rhsy_baseline, 11, sizeof(double), &dy4);
  ecode |= clSetKernelArg(k_rhsy_baseline, 12, sizeof(double), &dy5);
  ecode |= clSetKernelArg(k_rhsy_baseline, 13, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsy_baseline, 14, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_rhsy_baseline, 15, sizeof(int), &ny);
  ecode |= clSetKernelArg(k_rhsy_baseline, 16, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_rhsy_baseline, 17, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_rhsy_baseline, 18, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_rhsy_baseline, 19, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_rhsy_baseline, 20, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsy_baseline, 21, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsy_baseline, 22, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsy_baseline,
                                 1, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &ev_k_rhsy_baseline[work_step]);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel");

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_rhsy);



  if (timeron) timer_start(t_rhsz);
  //---------------------------------------------------------------------
  // zeta-direction flux differences
  //---------------------------------------------------------------------

  // Not Implemented for split case

  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], iend-ist);
  lws[0] = min(lws[0], max_work_group_size);

  gws[1] = (size_t) (jend - jst);
  gws[0] = (size_t) (iend - ist);

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz_baseline, 0, sizeof(cl_mem), &m_flux);
  ecode |= clSetKernelArg(k_rhsz_baseline, 1, sizeof(cl_mem), &m_utmp);
  ecode |= clSetKernelArg(k_rhsz_baseline, 2, sizeof(cl_mem), &m_rtmp);
  ecode |= clSetKernelArg(k_rhsz_baseline, 3, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz_baseline, 4, sizeof(cl_mem), &m_rho_i[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz_baseline, 5, sizeof(cl_mem), &m_qs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz_baseline, 6, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz_baseline, 7, sizeof(double), &tz1);
  ecode |= clSetKernelArg(k_rhsz_baseline, 8, sizeof(double), &tz2);
  ecode |= clSetKernelArg(k_rhsz_baseline, 9, sizeof(double), &tz3);
  ecode |= clSetKernelArg(k_rhsz_baseline, 10, sizeof(double), &dz1);
  ecode |= clSetKernelArg(k_rhsz_baseline, 11, sizeof(double), &dz2);
  ecode |= clSetKernelArg(k_rhsz_baseline, 12, sizeof(double), &dz3);
  ecode |= clSetKernelArg(k_rhsz_baseline, 13, sizeof(double), &dz4);
  ecode |= clSetKernelArg(k_rhsz_baseline, 14, sizeof(double), &dz5);
  ecode |= clSetKernelArg(k_rhsz_baseline, 15, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz_baseline, 16, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_rhsz_baseline, 17, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_rhsz_baseline, 18, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_rhsz_baseline, 19, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_rhsz_baseline, 20, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_rhsz_baseline, 21, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz_baseline, 22, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz_baseline, 23, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz_baseline,
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &ev_k_rhsz_baseline[work_step]);
  clu_CheckError(ecode, "clEnqueueNDRangeKernel");

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_rhsz);
  if (timeron) timer_stop(t_rhs);

  return &ev_k_rhsz_baseline[work_step];
}
