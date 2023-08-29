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

cl_kernel k_z_solve_datagen_baseline,
          k_z_solve_baseline;

cl_event  *ev_k_z_solve_datagen_baseline,
          *ev_k_z_solve_baseline;

void z_solve_init_baseline(int iter)
{
  cl_int ecode;

  k_z_solve_datagen_baseline = clCreateKernel(p_solve_baseline, 
                                               "z_solve_data_gen_baseline", 
                                               &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  k_z_solve_baseline = clCreateKernel(p_solve_baseline, 
                                      "z_solve_baseline", 
                                      &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_z_solve_datagen_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_z_solve_baseline = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void z_solve_release_baseline()
{
  clReleaseKernel(k_z_solve_datagen_baseline);
  clReleaseKernel(k_z_solve_baseline);

  free(ev_k_z_solve_datagen_baseline);
  free(ev_k_z_solve_baseline);
}

void z_solve_release_ev_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    if (split_flag)
      clReleaseEvent(ev_k_z_solve_datagen_baseline[i]);
    clReleaseEvent(ev_k_z_solve_baseline[i]);
  }
}

void z_solve_baseline(int work_step, 
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

    ecode  = clSetKernelArg(k_z_solve_datagen_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 1, sizeof(cl_mem), &m_square);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 2, sizeof(cl_mem), &m_qs);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 3, sizeof(int), &grid_points[0]);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 4, sizeof(int), &grid_points[1]);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 5, sizeof(int), &grid_points[2]);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 6, sizeof(int), &work_base);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 7, sizeof(int), &work_num_item);
    ecode |= clSetKernelArg(k_z_solve_datagen_baseline, 8, sizeof(int), &split_flag);
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
                                   k_z_solve_datagen_baseline, 
                                   3, NULL, 
                                   gws, lws,
                                   num_wait, ev_ptr_wait, 
                                   &(ev_k_z_solve_datagen_baseline[work_step]));
    clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

    if (!buffering_flag)
      clFinish(cmd_q[KERNEL_Q]);
    else
      clFlush(cmd_q[KERNEL_Q]);
  }

  gws[1] = work_num_item;
  gws[0] = grid_points[0]-2;

  lws[1] = 1;
  lws[0] = min(gws[0], max_work_group_size);

  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_z_solve_baseline, 0, sizeof(cl_mem), &m_qs);
  ecode |= clSetKernelArg(k_z_solve_baseline, 1, sizeof(cl_mem), &m_square);
  ecode |= clSetKernelArg(k_z_solve_baseline, 2, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_z_solve_baseline, 3, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_z_solve_baseline, 4, sizeof(cl_mem), &m_lhs);
  ecode |= clSetKernelArg(k_z_solve_baseline, 5, sizeof(cl_mem), &m_fjac);
  ecode |= clSetKernelArg(k_z_solve_baseline, 6, sizeof(cl_mem), &m_njac);
  ecode |= clSetKernelArg(k_z_solve_baseline, 7, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_z_solve_baseline, 8, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_z_solve_baseline, 9, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_z_solve_baseline, 10, sizeof(double), &dz1);
  ecode |= clSetKernelArg(k_z_solve_baseline, 11, sizeof(double), &dz2);
  ecode |= clSetKernelArg(k_z_solve_baseline, 12, sizeof(double), &dz3);
  ecode |= clSetKernelArg(k_z_solve_baseline, 13, sizeof(double), &dz4);
  ecode |= clSetKernelArg(k_z_solve_baseline, 14, sizeof(double), &dz5);
  ecode |= clSetKernelArg(k_z_solve_baseline, 15, sizeof(double), &c1);
  ecode |= clSetKernelArg(k_z_solve_baseline, 16, sizeof(double), &c2);
  ecode |= clSetKernelArg(k_z_solve_baseline, 17, sizeof(double), &c3);
  ecode |= clSetKernelArg(k_z_solve_baseline, 18, sizeof(double), &c4);
  ecode |= clSetKernelArg(k_z_solve_baseline, 19, sizeof(double), &tz1);
  ecode |= clSetKernelArg(k_z_solve_baseline, 20, sizeof(double), &tz2);
  ecode |= clSetKernelArg(k_z_solve_baseline, 21, sizeof(double), &con43);
  ecode |= clSetKernelArg(k_z_solve_baseline, 22, sizeof(double), &c3c4);
  ecode |= clSetKernelArg(k_z_solve_baseline, 23, sizeof(double), &c1345);
  ecode |= clSetKernelArg(k_z_solve_baseline, 24, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_z_solve_baseline, 25, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_z_solve_baseline, 26, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_z_solve_baseline, 27, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_z_solve_baseline, 
                                 2, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(ev_k_z_solve_baseline[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_zsolve);
}

