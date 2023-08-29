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

cl_kernel k_rhs_datagen_parallel, 
          k_rhs1_parallel, 
          k_rhs2_parallel,
          k_rhsx1_parallel, 
          k_rhsx2_parallel,
          k_rhsy1_parallel, 
          k_rhsy2_parallel,
          k_rhsz1_parallel, 
          k_rhsz2_parallel;

cl_event  *ev_k_rhs_datagen_parallel,
          *ev_k_rhs1_parallel,
          *ev_k_rhs2_parallel,
          *ev_k_rhsx1_parallel,
          *ev_k_rhsx2_parallel,
          *ev_k_rhsy1_parallel,
          *ev_k_rhsy2_parallel,
          *ev_k_rhsz1_parallel,
          *ev_k_rhsz2_parallel;

void compute_rhs_init_parallel(int iter)
{
  cl_int ecode; 

  k_rhs_datagen_parallel = clCreateKernel(p_rhs_parallel, 
                                          "rhs_datagen_parallel", 
                                          &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhs1_parallel = clCreateKernel(p_rhs_parallel, 
                                   "rhs1_parallel", 
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhs2_parallel = clCreateKernel(p_rhs_parallel, 
                                   "rhs2_parallel", 
                                   &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsx1_parallel = clCreateKernel(p_rhs_parallel, 
                                    "rhsx1_parallel", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsx2_parallel = clCreateKernel(p_rhs_parallel, 
                                    "rhsx2_parallel", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsy1_parallel = clCreateKernel(p_rhs_parallel, 
                                    "rhsy1_parallel", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsy2_parallel = clCreateKernel(p_rhs_parallel, 
                                    "rhsy2_parallel", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz1_parallel = clCreateKernel(p_rhs_parallel, 
                                    "rhsz1_parallel", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_rhsz2_parallel = clCreateKernel(p_rhs_parallel, 
                                    "rhsz2_parallel", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_rhs_datagen_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhs1_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhs2_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsx1_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsx2_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsy1_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsy2_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz1_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_rhsz2_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void compute_rhs_release_parallel()
{
  clReleaseKernel(k_rhs_datagen_parallel);
  clReleaseKernel(k_rhs1_parallel); 
  clReleaseKernel(k_rhs2_parallel); 
  clReleaseKernel(k_rhsx1_parallel);
  clReleaseKernel(k_rhsx2_parallel);
  clReleaseKernel(k_rhsy1_parallel);
  clReleaseKernel(k_rhsy2_parallel);
  clReleaseKernel(k_rhsz1_parallel);
  clReleaseKernel(k_rhsz2_parallel);

  free(ev_k_rhs_datagen_parallel);
  free(ev_k_rhs1_parallel);
  free(ev_k_rhs2_parallel);
  free(ev_k_rhsx1_parallel);
  free(ev_k_rhsx2_parallel);
  free(ev_k_rhsy1_parallel);
  free(ev_k_rhsy2_parallel);
  free(ev_k_rhsz1_parallel);
  free(ev_k_rhsz2_parallel);
}

void compute_rhs_release_ev_parallel(int iter)
{
  int i;
  for (i = 0; i < iter; i++) {
    clReleaseEvent(ev_k_rhs_datagen_parallel[i]);
    clReleaseEvent(ev_k_rhs1_parallel[i]);
    clReleaseEvent(ev_k_rhs2_parallel[i]);
    clReleaseEvent(ev_k_rhsx1_parallel[i]);
    clReleaseEvent(ev_k_rhsx2_parallel[i]);
    clReleaseEvent(ev_k_rhsy1_parallel[i]);
    clReleaseEvent(ev_k_rhsy2_parallel[i]);
    clReleaseEvent(ev_k_rhsz1_parallel[i]);
    clReleaseEvent(ev_k_rhsz2_parallel[i]);
  }
}

void compute_rhs_body_parallel(int work_step, 
                               int work_base, 
                               int work_num_item, 
                               int copy_buffer_base, 
                               int copy_num_item, 
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

  ecode  = clSetKernelArg(k_rhs_datagen_parallel, 0, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 1, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 2, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 3, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 4, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 7, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 8, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 9, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 10, sizeof(int), &copy_buffer_base);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 11, sizeof(int), &copy_num_item);
  ecode |= clSetKernelArg(k_rhs_datagen_parallel, 12, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs_datagen_parallel,
                                 3, NULL, 
                                 gws, lws,
                                 num_wait, ev_ptr_wait, 
                                 &(ev_k_rhs_datagen_parallel[work_step]));
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

  ecode  = clSetKernelArg(k_rhs1_parallel, 0, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_parallel, 1, sizeof(cl_mem), &m_forcing[buf_idx]);
  ecode |= clSetKernelArg(k_rhs1_parallel, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhs1_parallel, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhs1_parallel, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhs1_parallel, 5, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhs1_parallel, 6, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhs1_parallel, 7, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs1_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhs1_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);


  if (timeron) timer_start(t_rhsx);
  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = grid_points[1] - 2;
  gws[0] = grid_points[0] - 2;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsx1_parallel, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 7, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 11, sizeof(double), &dx1tx1);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 12, sizeof(double), &dx2tx1);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 13, sizeof(double), &dx3tx1);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 14, sizeof(double), &dx4tx1);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 15, sizeof(double), &dx5tx1);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 16, sizeof(double), &xxcon2);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 17, sizeof(double), &xxcon3);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 18, sizeof(double), &xxcon4);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 19, sizeof(double), &xxcon5);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 20, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 21, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 22, sizeof(double), &tx2);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 23, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 24, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 25, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 26, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsx1_parallel, 27, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsx1_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsx1_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);


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

  ecode  = clSetKernelArg(k_rhsx2_parallel, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsx2_parallel, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsx2_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsx2_parallel[work_step]));
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
  lws[2] = 1; 
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item; 
  gws[1] = grid_points[1] - 2;
  gws[0] = grid_points[0] - 2;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_rhsy1_parallel, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 7, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 11, sizeof(double), &dy1ty1);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 12, sizeof(double), &dy2ty1);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 13, sizeof(double), &dy3ty1);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 14, sizeof(double), &dy4ty1);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 15, sizeof(double), &dy5ty1);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 16, sizeof(double), &yycon2);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 17, sizeof(double), &yycon3);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 18, sizeof(double), &yycon4);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 19, sizeof(double), &yycon5);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 20, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 21, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 22, sizeof(double), &ty2);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 23, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 24, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 25, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 26, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsy1_parallel, 27, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsy1_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsy1_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);


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

  ecode  = clSetKernelArg(k_rhsy2_parallel, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsy2_parallel, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsy2_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsy2_parallel[work_step]));
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

  ecode  = clSetKernelArg(k_rhsz1_parallel, 0, sizeof(cl_mem), &m_us);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 1, sizeof(cl_mem), &m_vs);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 2, sizeof(cl_mem), &m_ws);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 3, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 4, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 5, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 6, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 7, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 11, sizeof(double), &dz1tz1);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 12, sizeof(double), &dz2tz1);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 13, sizeof(double), &dz3tz1);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 14, sizeof(double), &dz4tz1);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 15, sizeof(double), &dz5tz1);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 16, sizeof(double), &zzcon2);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 17, sizeof(double), &zzcon3);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 18, sizeof(double), &zzcon4);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 19, sizeof(double), &zzcon5);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 20, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 21, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 22, sizeof(double), &tz2);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 23, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 24, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 25, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz1_parallel, 26, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz1_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz1_parallel[work_step]));
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

  ecode  = clSetKernelArg(k_rhsz2_parallel, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 5, sizeof(double), &dssp);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhsz2_parallel, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhsz2_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhsz2_parallel[work_step]));
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

  ecode  = clSetKernelArg(k_rhs2_parallel, 0, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_rhs2_parallel, 1, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_rhs2_parallel, 2, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_rhs2_parallel, 3, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_rhs2_parallel, 4, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_rhs2_parallel, 5, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_rhs2_parallel, 6, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_rhs2_parallel, 7, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_rhs2_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_rhs2_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_rhs);
}

