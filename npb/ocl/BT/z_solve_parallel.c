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
// Performs line solves in Z direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------

cl_kernel k_z_solve_datagen_parallel,
          k_z_solve1_parallel,
          k_z_solve2_parallel,
          k_z_solve3_parallel;

cl_event  *ev_k_z_solve_datagen_parallel,
          *ev_k_z_solve1_parallel,
          *ev_k_z_solve2_parallel,
          *ev_k_z_solve3_parallel;

void z_solve_init_parallel(int iter)
{
  cl_int ecode;

  k_z_solve_datagen_parallel = clCreateKernel(p_solve_parallel, 
                                              "z_solve_datagen_parallel", 
                                              &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_z_solve1_parallel = clCreateKernel(p_solve_parallel, 
                                       "z_solve1_parallel", 
                                       &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_z_solve2_parallel = clCreateKernel(p_solve_parallel, 
                                       "z_solve2_parallel", 
                                       &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_z_solve3_parallel = clCreateKernel(p_solve_parallel, 
                                       "z_solve3_parallel", 
                                       &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_z_solve_datagen_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_z_solve1_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_z_solve2_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_z_solve3_parallel = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void z_solve_release_parallel()
{
  clReleaseKernel(k_z_solve_datagen_parallel);
  clReleaseKernel(k_z_solve1_parallel);
  clReleaseKernel(k_z_solve2_parallel);
  clReleaseKernel(k_z_solve3_parallel);

  free(ev_k_z_solve_datagen_parallel);
  free(ev_k_z_solve1_parallel);
  free(ev_k_z_solve2_parallel);
  free(ev_k_z_solve3_parallel);
}

void z_solve_release_ev_parallel(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    if (split_flag)
      clReleaseEvent(ev_k_z_solve_datagen_parallel[i]);
    clReleaseEvent(ev_k_z_solve1_parallel[i]);
    clReleaseEvent(ev_k_z_solve2_parallel[i]);
    clReleaseEvent(ev_k_z_solve3_parallel[i]);
  }
}

void z_solve_parallel(int work_step, 
                      int work_base, 
                      int work_num_item, 
                      int buf_idx, 
                      cl_event* ev_wb_end_ptr)
{
  cl_int ecode;
  size_t lws[3];
  size_t gws[3];
  int num_wait;
  cl_event *ev_ptr_wait;

  if (timeron) timer_start(t_zsolve);

  if (split_flag) {
    lws[2] = 1;
    lws[1] = 1;
    lws[0] = max_work_group_size;

    gws[2] = (size_t) grid_points[2];
    gws[1] = (size_t) work_num_item;
    gws[0] = (size_t) grid_points[0]-2;

    gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
    gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
    gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

    ecode  = clSetKernelArg(k_z_solve_datagen_parallel, 0, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 1, sizeof(cl_mem), &m_square);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 2, sizeof(cl_mem), &m_qs);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 3, sizeof(int), &grid_points[0]);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 4, sizeof(int), &grid_points[1]);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 5, sizeof(int), &grid_points[2]);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 6, sizeof(int), &work_base);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 7, sizeof(int), &work_num_item);
    ecode |= clSetKernelArg(k_z_solve_datagen_parallel, 8, sizeof(int), &split_flag);
    clu_CheckError(ecode, "clSetKernelArg()");

    if (!buffering_flag) {
      num_wait = 0;
      ev_ptr_wait = NULL;
    }
    else {
      num_wait = 1;
      ev_ptr_wait = ev_wb_end_ptr;
    }

    ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                   k_z_solve_datagen_parallel, 
                                   3, NULL, 
                                   gws, lws,
                                   num_wait, ev_ptr_wait, 
                                   &(ev_k_z_solve_datagen_parallel[work_step]));
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    if (!buffering_flag)
      clFinish(cmd_q[KERNEL_Q]);
    else
      clFlush(cmd_q[KERNEL_Q]);

  }


  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[1] = work_num_item;
  gws[0] = grid_points[0]-2;
  gws[0] *= 25;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_z_solve1_parallel, 0, sizeof(cl_mem), &m_lhsA);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 1, sizeof(cl_mem), &m_lhsB);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 2, sizeof(cl_mem), &m_lhsC);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 3, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 4, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 5, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 6, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 7, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_z_solve1_parallel, 8, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_z_solve1_parallel, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_z_solve1_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);


  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_group_size, max_work_item_sizes[0]);

  gws[2] = (size_t) grid_points[2]-2;
  gws[1] = (size_t) work_num_item;
  gws[0] = (size_t) grid_points[0]-2;

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_z_solve2_parallel, 0, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 1, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 2, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 3, sizeof(cl_mem), &m_lhsA);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 4, sizeof(cl_mem), &m_lhsB);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 5, sizeof(cl_mem), &m_lhsC);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 6, sizeof(cl_mem), &m_fjac);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 7, sizeof(cl_mem), &m_njac);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 8, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 9, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 10, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 11, sizeof(double), &dz1);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 12, sizeof(double), &dz2);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 13, sizeof(double), &dz3);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 14, sizeof(double), &dz4);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 15, sizeof(double), &dz5);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 16, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 17, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 18, sizeof(double), &c3);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 19, sizeof(double), &c4);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 20, sizeof(double), &tz1);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 21, sizeof(double), &tz2);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 22, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 23, sizeof(double), &c3c4);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 24, sizeof(double), &c1345);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 25, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 26, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 27, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_z_solve2_parallel, 28, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_z_solve2_parallel, 
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_z_solve2_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  size_t max_lws_i = max_work_group_size/5*5; 

  lws[1] = 1;
  lws[0] = max_lws_i;

  gws[1] = (size_t) work_num_item;
  gws[0] = (size_t) grid_points[0]-2;
  gws[0] *= 5;

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_z_solve3_parallel, 0, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 1, sizeof(cl_mem), &m_lhsA);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 2, sizeof(cl_mem), &m_lhsB);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 3, sizeof(cl_mem), &m_lhsC);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 4, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 5, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 6, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 7, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 8, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_z_solve3_parallel, 9, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_z_solve3_parallel, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_z_solve3_parallel[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_zsolve);
}

