//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB BT code. This OpenCL C  //
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

#include "header.h"
#include "timers.h"
#include <math.h>
#include <stdio.h>

cl_kernel k_rhs_datagen_baseline, 
          k_rhs1_baseline,
          k_rhs2_baseline,
          k_rhsx_baseline,
          k_rhsy_baseline,
          k_rhsz1_baseline,
          k_rhsz2_baseline,
          k_rhsz3_baseline,
          k_rhsz4_baseline,
          k_rhsz5_baseline,
          k_rhsz6_baseline;

cl_event  *ev_k_rhs_datagen_baseline,
          *ev_k_rhs1_baseline,
          *ev_k_rhs2_baseline,
          *ev_k_rhsx_baseline,
          *ev_k_rhsy_baseline,
          *ev_k_rhsz1_baseline,
          *ev_k_rhsz2_baseline,
          *ev_k_rhsz3_baseline,
          *ev_k_rhsz4_baseline,
          *ev_k_rhsz5_baseline,
          *ev_k_rhsz6_baseline;

void compute_rhs_init_baseline(int iter)
{
  cl_int ecode;

  k_rhs_datagen_baseline = clCreateKernel(p_rhs_baseline, 
                                          "rhs_datagen_baseline", 
                                          &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhs1_baseline = clCreateKernel(p_rhs_baseline, 
                                   "rhs1_baseline", 
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhs2_baseline = clCreateKernel(p_rhs_baseline, 
                                   "rhs2_baseline", 
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsx_baseline = clCreateKernel(p_rhs_baseline, 
                                   "rhsx_baseline", 
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsy_baseline = clCreateKernel(p_rhs_baseline, 
                                   "rhsy_baseline", 
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz1_baseline = clCreateKernel(p_rhs_baseline, 
                                    "rhsz1_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz2_baseline = clCreateKernel(p_rhs_baseline, 
                                    "rhsz2_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz3_baseline = clCreateKernel(p_rhs_baseline, 
                                    "rhsz3_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz4_baseline = clCreateKernel(p_rhs_baseline, 
                                    "rhsz4_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz5_baseline = clCreateKernel(p_rhs_baseline, 
                                    "rhsz5_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz6_baseline = clCreateKernel(p_rhs_baseline, 
                                    "rhsz6_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_rhs_datagen_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhs1_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhs2_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsx_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsy_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz1_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz2_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz3_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz4_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz5_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz6_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);

}

void compute_rhs_release_baseline()
{
  clReleaseKernel(k_rhs_datagen_baseline);
  clReleaseKernel(k_rhs1_baseline);
  clReleaseKernel(k_rhs2_baseline);
  clReleaseKernel(k_rhsx_baseline);
  clReleaseKernel(k_rhsy_baseline);
  clReleaseKernel(k_rhsz1_baseline);
  clReleaseKernel(k_rhsz2_baseline);
  clReleaseKernel(k_rhsz3_baseline);
  clReleaseKernel(k_rhsz4_baseline);
  clReleaseKernel(k_rhsz5_baseline);
  clReleaseKernel(k_rhsz6_baseline);

  free(ev_k_rhs_datagen_baseline);
  free(ev_k_rhs1_baseline);
  free(ev_k_rhs2_baseline);
  free(ev_k_rhsx_baseline);
  free(ev_k_rhsy_baseline);
  free(ev_k_rhsz1_baseline);
  free(ev_k_rhsz2_baseline);
  free(ev_k_rhsz3_baseline);
  free(ev_k_rhsz4_baseline);
  free(ev_k_rhsz5_baseline);
  free(ev_k_rhsz6_baseline);
}

void compute_rhs_release_ev_baseline(int iter)
{
  int i;
  for (i = 0; i < iter; i++) {
    clReleaseEvent(ev_k_rhs_datagen_baseline[i]);
    clReleaseEvent(ev_k_rhs1_baseline[i]);
    clReleaseEvent(ev_k_rhs2_baseline[i]);
    clReleaseEvent(ev_k_rhsx_baseline[i]);
    clReleaseEvent(ev_k_rhsy_baseline[i]);
    clReleaseEvent(ev_k_rhsz1_baseline[i]);
    clReleaseEvent(ev_k_rhsz2_baseline[i]);
    clReleaseEvent(ev_k_rhsz3_baseline[i]);
    clReleaseEvent(ev_k_rhsz4_baseline[i]);
    clReleaseEvent(ev_k_rhsz5_baseline[i]);
    clReleaseEvent(ev_k_rhsz6_baseline[i]);
  }
}

void compute_rhs_body_baseline(int work_step, int work_base, int work_num_item, 
                               int copy_buffer_base, int copy_num_item, 
                               int buf_idx,
                               int wait,
                               cl_event * ev_wb_end_ptr)
{
  cl_int ecode;
  size_t lws[3];
  size_t gws[3]; 
  int num_wait;
  cl_event *ev_ptr_wait;

  num_wait = (wait) ? 1 : 0;
  ev_ptr_wait = (wait) ? ev_wb_end_ptr : NULL;

  if (timeron) timer_start(t_rhs);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = (size_t)copy_num_item;
  gws[1] = (size_t)grid_points[1];
  gws[0] = (size_t)grid_points[0];

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhs_datagen_baseline, 0, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 7, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 8, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 9, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 10, sizeof(int), &copy_buffer_base);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 11, sizeof(int), &copy_num_item);
  ecode |= clSetKernelArg(k_rhs_datagen_baseline, 12, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs_datagen_baseline,
                                 3, NULL, 
                                 gws, lws,
                                 num_wait, ev_ptr_wait, 
                                 &(ev_k_rhs_datagen_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = (size_t)grid_points[1];
  gws[0] = (size_t)grid_points[0];
  gws[0] *= 5;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhs1_baseline, 0, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 1, sizeof(cl_mem), &m_forcing[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhs1_baseline, 5, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhs1_baseline, 6, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhs1_baseline, 7, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs1_baseline, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhs1_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);


  if (timeron) timer_start(t_rhsx);
  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------

  lws[0] = max_work_group_size;

  gws[0] = work_num_item;

  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsx_baseline, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhsx_baseline, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhsx_baseline, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhsx_baseline, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsx_baseline, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhsx_baseline, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhsx_baseline, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 7, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsx_baseline, 11, sizeof(double), &dx1tx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 12, sizeof(double), &dx2tx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 13, sizeof(double), &dx3tx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 14, sizeof(double), &dx4tx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 15, sizeof(double), &dx5tx1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 16, sizeof(double), &xxcon2);
  ecode |= clSetKernelArg(k_rhsx_baseline, 17, sizeof(double), &xxcon3);
  ecode |= clSetKernelArg(k_rhsx_baseline, 18, sizeof(double), &xxcon4);
  ecode |= clSetKernelArg(k_rhsx_baseline, 19, sizeof(double), &xxcon5);
  ecode |= clSetKernelArg(k_rhsx_baseline, 20, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_rhsx_baseline, 21, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_rhsx_baseline, 22, sizeof(double), &tx2);
  ecode |= clSetKernelArg(k_rhsx_baseline, 23, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_rhsx_baseline, 24, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsx_baseline, 25, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsx_baseline, 26, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsx_baseline, 27, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsx_baseline, 
                                 1, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsx_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_rhsx);


  if (timeron) timer_start(t_rhsy);
  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------
  lws[0] = max_work_group_size;

  gws[0] = work_num_item; 

  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsy_baseline, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhsy_baseline, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhsy_baseline, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhsy_baseline, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsy_baseline, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhsy_baseline, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhsy_baseline, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 7, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsy_baseline, 11, sizeof(double), &dy1ty1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 12, sizeof(double), &dy2ty1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 13, sizeof(double), &dy3ty1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 14, sizeof(double), &dy4ty1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 15, sizeof(double), &dy5ty1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 16, sizeof(double), &yycon2);
  ecode |= clSetKernelArg(k_rhsy_baseline, 17, sizeof(double), &yycon3);
  ecode |= clSetKernelArg(k_rhsy_baseline, 18, sizeof(double), &yycon4);
  ecode |= clSetKernelArg(k_rhsy_baseline, 19, sizeof(double), &yycon5);
  ecode |= clSetKernelArg(k_rhsy_baseline, 20, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_rhsy_baseline, 21, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_rhsy_baseline, 22, sizeof(double), &ty2);
  ecode |= clSetKernelArg(k_rhsy_baseline, 23, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_rhsy_baseline, 24, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsy_baseline, 25, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsy_baseline, 26, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsy_baseline, 27, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsy_baseline, 
                                 1, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsy_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);


  if (timeron) timer_stop(t_rhsy);


  if (timeron) timer_start(t_rhsz);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = (size_t)work_num_item;
  gws[1] = (size_t)grid_points[1]-2;
  gws[0] = (size_t)grid_points[0]-2;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz1_baseline, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 7, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 11, sizeof(double), &dz1tz1);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 12, sizeof(double), &dz2tz1);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 13, sizeof(double), &dz3tz1);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 14, sizeof(double), &dz4tz1);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 15, sizeof(double), &dz5tz1);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 16, sizeof(double), &zzcon2);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 17, sizeof(double), &zzcon3);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 18, sizeof(double), &zzcon4);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 19, sizeof(double), &zzcon5);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 20, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 21, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 22, sizeof(double), &tz2);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 23, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 24, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 25, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz1_baseline, 26, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz1_baseline, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz1_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[1] = grid_points[1]-2;
  gws[0] = grid_points[0]-2;
  gws[0]*= 5;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz2_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz2_baseline, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz2_baseline, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz2_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[1] = grid_points[1]-2;
  gws[0] = grid_points[0]-2;
  gws[0]*= 5;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz3_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz3_baseline, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz3_baseline, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz3_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = grid_points[1]-2;
  gws[0] = grid_points[0]-2;
  gws[0]*= 5;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz4_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz4_baseline, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz4_baseline, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz4_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[1] = grid_points[1]-2;
  gws[0] = grid_points[0]-2;
  gws[0]*= 5;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz5_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz5_baseline, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz5_baseline, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL,
                                 &(ev_k_rhsz5_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[1] = grid_points[1]-2;
  gws[0] = grid_points[0]-2;
  gws[0]*= 5;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsz6_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz6_baseline, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz6_baseline, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz6_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_rhsz);


  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = grid_points[1] - 2;
  gws[0] = grid_points[0] - 2;
  gws[0]*= 5;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhs2_baseline, 0, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhs2_baseline, 1, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhs2_baseline, 2, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhs2_baseline, 3, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhs2_baseline, 4, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_rhs2_baseline, 5, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhs2_baseline, 6, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhs2_baseline, 7, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs2_baseline, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhs2_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_rhs);
}

