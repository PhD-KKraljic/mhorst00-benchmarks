//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB BT code. This CUDA® C  //
//  version is a part of SNU-NPB 2019 developed by the Center for Manycore //
//  Programming at Seoul National University and derived from the serial   //
//  Fortran versions in "NPB3.3.1-SER" developed by NAS.                   //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on original NPB 3.3.1, including the technical report, the //
//  original specifications, source code, results and information on how   //
//  to submit new results, is available at:                                //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Information on SNU-NPB 2019, including the conference paper and source //
//  code, is available at:                                                 //
//                                                                         //
//           http://aces.snu.ac.kr                                         //
//                                                                         //
//  Send comments or suggestions for this CUDA® C version to               //
//  snunpb@aces.snu.ac.kr                                                  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 08826, Korea                                             //
//                                                                         //
//          E-mail: snunpb@aces.snu.ac.kr                                  //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Youngdong Do, Hyung Mo Kim, Pyeongseok Oh, Daeyoung Park,      //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "header.h"

//---------------------------------------------------------------------
// Performs line solves in Z direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------

cudaEvent_t   (*ev_k_z_solve_data_gen_baseline)[2],
              (*ev_k_z_solve_baseline)[2];

void z_solve_init_baseline(int iter)
{
  int i;

  ev_k_z_solve_data_gen_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_z_solve_baseline = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_z_solve_data_gen_baseline[i][0]);
    cudaEventCreate(&ev_k_z_solve_baseline[i][0]);

    cudaEventCreate(&ev_k_z_solve_data_gen_baseline[i][1]);
    cudaEventCreate(&ev_k_z_solve_baseline[i][1]);
  }
}

void z_solve_release_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_z_solve_data_gen_baseline[i][0]);
    cudaEventDestroy(ev_k_z_solve_baseline[i][0]);

    cudaEventDestroy(ev_k_z_solve_data_gen_baseline[i][1]);
    cudaEventDestroy(ev_k_z_solve_baseline[i][1]);
  }

  free(ev_k_z_solve_data_gen_baseline);
  free(ev_k_z_solve_baseline);
}

void z_solve_baseline(int work_step, 
                     int work_base, 
                     int work_num_item, 
                     int buf_idx,
                     cudaEvent_t* ev_wb_end_ptr) 
{
  size_t lws[3];
  size_t gws[3];

  if (timeron) timer_start(t_zsolve);

  if (split_flag) {
    lws[2] = 1;
    lws[1] = 1;
    lws[0] = max_work_group_size;

    gws[2] = (size_t) grid_points[2];
    gws[1] = (size_t) work_num_item;
    gws[0] = (size_t) grid_points[0]-2;

    gws[2] = RoundWorkSize(gws[2], lws[2]);
    gws[1] = RoundWorkSize(gws[1], lws[1]);
    gws[0] = RoundWorkSize(gws[0], lws[0]);

    dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1], gws[2]/lws[2]);
    dim3 threadSize(lws[0], lws[1], lws[2]);

    if (buffering_flag)
      CUCHK(cudaStreamWaitEvent(cmd_q[KERNEL_Q], *ev_wb_end_ptr, 0));

    CUCHK(cudaEventRecord(ev_k_z_solve_data_gen_baseline[work_step][0], cmd_q[KERNEL_Q]));
    cuda_ProfilerStartEventRecord("k_z_solve_data_gen_baseline",  cmd_q[KERNEL_Q]);
    k_z_solve_data_gen_baseline<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
      (
       m_u[buf_idx], m_square, m_qs,
       grid_points[0], grid_points[1], grid_points[2],
       work_base, work_num_item, split_flag,
       work_num_item_default_j
      );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_z_solve_data_gen_baseline",  cmd_q[KERNEL_Q]);
    CUCHK(cudaEventRecord(ev_k_z_solve_data_gen_baseline[work_step][1], cmd_q[KERNEL_Q]));

    if (!buffering_flag)
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  gws[1] = work_num_item;
  gws[0] = grid_points[0]-2;

  lws[1] = 1;
  lws[0] = min(gws[0], max_work_group_size);

  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1]);
  dim3 threadSize(lws[0], lws[1]);

  CUCHK(cudaEventRecord(ev_k_z_solve_baseline[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_z_solve_baseline",  cmd_q[KERNEL_Q]);
  k_z_solve_baseline<<<blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_qs, m_square, m_u[buf_idx], m_rhs[buf_idx],
     m_lhs, m_fjac, m_njac,
     grid_points[0], grid_points[1], grid_points[2],
     dz1, dz2, dz3, dz4, dz5,
     c1, c2, c3, c4, tz1,
     tz2, con43, c3c4, c1345, dt,
     work_base, work_num_item, split_flag,
     work_num_item_default_j
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_z_solve_baseline",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_z_solve_baseline[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  if (timeron) timer_stop(t_zsolve);


}

