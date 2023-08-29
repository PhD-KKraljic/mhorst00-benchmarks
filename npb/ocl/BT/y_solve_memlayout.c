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
// Performs line solves in Y direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------

cl_kernel k_y_solve_memlayout;
cl_event  *ev_k_y_solve_memlayout;

void y_solve_init_memlayout(int iter)
{
  cl_int ecode;
  k_y_solve_memlayout = clCreateKernel(p_y_solve_memlayout, 
                                       "y_solve_memlayout", 
                                       &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_y_solve_memlayout = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void y_solve_release_memlayout()
{
  clReleaseKernel(k_y_solve_memlayout);

  free(ev_k_y_solve_memlayout);
}

void y_solve_release_ev_memlayout(int iter)
{
  int i;
  for (i = 0; i < iter; i++)
    clReleaseEvent(ev_k_y_solve_memlayout[i]);
}

cl_event* y_solve_memlayout(int work_step, 
                           int work_base, 
                           int work_num_item, 
                           int buf_idx)
{
  //---------------------------------------------------------------------
  // This function computes the left hand side for the three y-factors   
  //---------------------------------------------------------------------
  //---------------------------------------------------------------------
  // Compute the indices for storing the tri-diagonal matrix;
  // determine a (labeled f) and n jacobians for cell c
  //---------------------------------------------------------------------

  cl_int ecode;

  size_t lws[3];
  size_t gws[3]; 

  if (timeron) timer_start(t_ysolve);

  gws[1] = work_num_item;
  gws[0] = grid_points[0]-2;

  lws[1] = 1;
  lws[0] = min(gws[0], max_work_group_size);

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_y_solve_memlayout, 0, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 1, sizeof(cl_mem), &m_rho_i);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 2, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 3, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 4, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 5, sizeof(cl_mem), &m_lhs);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 6, sizeof(cl_mem), &m_fjac);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 7, sizeof(cl_mem), &m_njac);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 11, sizeof(double), &dy1);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 12, sizeof(double), &dy2);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 13, sizeof(double), &dy3);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 14, sizeof(double), &dy4);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 15, sizeof(double), &dy5);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 16, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 17, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 18, sizeof(double), &ty1);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 19, sizeof(double), &ty2);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 20, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 21, sizeof(double), &c3c4);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 22, sizeof(double), &c1345);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 23, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 24, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 25, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_y_solve_memlayout, 26, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_y_solve_memlayout, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_y_solve_memlayout[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_ysolve);

  return &ev_k_y_solve_memlayout[work_step];
}

