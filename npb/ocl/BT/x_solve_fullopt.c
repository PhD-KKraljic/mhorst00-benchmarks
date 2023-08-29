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

//---------------------------------------------------------------------
// 
// Performs line solves in X direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
// 
//---------------------------------------------------------------------

cl_kernel k_x_solve1_fullopt,
          k_x_solve2_fullopt,
          k_x_solve3_fullopt;

cl_event  *ev_k_x_solve1_fullopt,
          *ev_k_x_solve2_fullopt,
          *ev_k_x_solve3_fullopt;

void x_solve_init_fullopt(int iter)
{
  cl_int ecode;

  k_x_solve1_fullopt = clCreateKernel(p_solve_fullopt, 
                                      "x_solve1_fullopt", 
                                      &ecode);
  clu_CheckError(ecode, "clCreateKernel() for x_solve1");

  k_x_solve2_fullopt = clCreateKernel(p_solve_fullopt, 
                                      "x_solve2_fullopt", 
                                      &ecode);
  clu_CheckError(ecode, "clCreateKernel() for x_solve2");

  k_x_solve3_fullopt = clCreateKernel(p_solve_fullopt, 
                                      "x_solve3_fullopt", 
                                      &ecode);
  clu_CheckError(ecode, "clCreateKernel() for x_solve3");

  ev_k_x_solve1_fullopt = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_x_solve2_fullopt = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_x_solve3_fullopt = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void x_solve_release_fullopt()
{
  clReleaseKernel(k_x_solve1_fullopt);
  clReleaseKernel(k_x_solve2_fullopt);
  clReleaseKernel(k_x_solve3_fullopt);

  free(ev_k_x_solve1_fullopt);
  free(ev_k_x_solve2_fullopt);
  free(ev_k_x_solve3_fullopt);
}

void x_solve_release_ev_fullopt(int iter)
{
  int i;
  for (i = 0; i < iter; i++) {
    clReleaseEvent(ev_k_x_solve1_fullopt[i]);
    clReleaseEvent(ev_k_x_solve2_fullopt[i]);
    clReleaseEvent(ev_k_x_solve3_fullopt[i]);
  }
}

void x_solve_fullopt(int work_step, 
                     int work_base, 
                     int work_num_item, 
                     int buf_idx)
{
  //---------------------------------------------------------------------
  // This function computes the left hand side in the xi-direction
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // determine a (labeled f) and n jacobians
  //---------------------------------------------------------------------
  cl_int ecode;
  size_t lws[3];
  size_t gws[3]; 

  if (timeron) timer_start(t_xsolve);

  gws[1] = work_num_item*5*5;
  gws[0] = grid_points[1]-2;

  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_x_solve1_fullopt, 0, sizeof(cl_mem), &m_lhsA);
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 1, sizeof(cl_mem), &m_lhsB);
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 2, sizeof(cl_mem), &m_lhsC);
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 3, sizeof(int), &grid_points[0] );
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 4, sizeof(int), &grid_points[1] );
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 5, sizeof(int), &grid_points[2] );
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_x_solve1_fullopt, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_x_solve1_fullopt, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_x_solve1_fullopt[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  gws[2] = (size_t) work_num_item;
  gws[1] = (size_t) grid_points[1]-2;
  gws[0] = (size_t) grid_points[0]-2;

  lws[2] = 1;
  lws[1] = max_work_item_sizes[1];
  lws[0] = 1;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_x_solve2_fullopt, 0, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 1, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 2, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 3, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 4, sizeof(cl_mem), &m_lhsA);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 5, sizeof(cl_mem), &m_lhsB);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 6, sizeof(cl_mem), &m_lhsC);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 7, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 8, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 9, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 10, sizeof(double), &dx1);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 11, sizeof(double), &dx2);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 12, sizeof(double), &dx3);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 13, sizeof(double), &dx4);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 14, sizeof(double), &dx5);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 15, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 16, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 17, sizeof(double), &tx1);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 18, sizeof(double), &tx2);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 19, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 20, sizeof(double), &c3c4);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 21, sizeof(double), &c1345);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 22, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 23, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 24, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_x_solve2_fullopt, 25, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_x_solve2_fullopt, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_x_solve2_fullopt[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  size_t max_lws_j = min(max_work_group_size/5, local_mem_size/(sizeof(double)*(3*5*5+2*5)));

  lws[1] = 5; 
  lws[0] = max_lws_j; 

  gws[1] = (size_t) work_num_item;
  gws[1] *= 5;
  gws[0] = (size_t) grid_points[1]-2;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_x_solve3_fullopt, 0, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 1, sizeof(cl_mem), &m_lhsA);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 2, sizeof(cl_mem), &m_lhsB);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 3, sizeof(cl_mem), &m_lhsC);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 4, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 5, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 6, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 7, sizeof(double)*max_lws_j*3*5*5, NULL);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 8, sizeof(double)*max_lws_j*2*5, NULL);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 9, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 10, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_x_solve3_fullopt, 11, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q],
                                 k_x_solve3_fullopt, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_x_solve3_fullopt[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_xsolve);
}

