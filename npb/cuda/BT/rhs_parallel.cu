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
#include "header.h"
#include "cuda_util.h"

cudaEvent_t   (*ev_k_rhs_datagen_parallel)[2],
              (*ev_k_rhs1_parallel)[2],
              (*ev_k_rhs2_parallel)[2],
              (*ev_k_rhsx1_parallel)[2],
              (*ev_k_rhsx2_parallel)[2],
              (*ev_k_rhsy1_parallel)[2],
              (*ev_k_rhsy2_parallel)[2],
              (*ev_k_rhsz1_parallel)[2],
              (*ev_k_rhsz2_parallel)[2];

void compute_rhs_init_parallel(int iter)
{
  int i;

  ev_k_rhs_datagen_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhs1_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhs2_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsx1_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsx2_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsy1_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsy2_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsz1_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_rhsz2_parallel = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_rhs_datagen_parallel[i][0]);
    cudaEventCreate(&ev_k_rhs1_parallel[i][0]);
    cudaEventCreate(&ev_k_rhs2_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsx1_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsx2_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsy1_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsy2_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsz1_parallel[i][0]);
    cudaEventCreate(&ev_k_rhsz2_parallel[i][0]);

    cudaEventCreate(&ev_k_rhs_datagen_parallel[i][1]);
    cudaEventCreate(&ev_k_rhs1_parallel[i][1]);
    cudaEventCreate(&ev_k_rhs2_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsx1_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsx2_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsy1_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsy2_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsz1_parallel[i][1]);
    cudaEventCreate(&ev_k_rhsz2_parallel[i][1]);
  }
}

void compute_rhs_release_parallel(int iter)
{
  int i;

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_rhs_datagen_parallel[i][0]);
    cudaEventDestroy(ev_k_rhs1_parallel[i][0]);
    cudaEventDestroy(ev_k_rhs2_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsx1_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsx2_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsy1_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsy2_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsz1_parallel[i][0]);
    cudaEventDestroy(ev_k_rhsz2_parallel[i][0]);

    cudaEventDestroy(ev_k_rhs_datagen_parallel[i][1]);
    cudaEventDestroy(ev_k_rhs1_parallel[i][1]);
    cudaEventDestroy(ev_k_rhs2_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsx1_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsx2_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsy1_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsy2_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsz1_parallel[i][1]);
    cudaEventDestroy(ev_k_rhsz2_parallel[i][1]);
  }

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

void compute_rhs_body_parallel(int work_step,
                               int work_base,
                               int work_num_item,
                               int copy_buffer_base,
                               int copy_num_item,
                               int buf_idx,
                               cudaEvent_t *ev_wb_end_ptr)
{
  size_t lws[3];
  size_t gws[3]; 
  dim3 blockSize;
  dim3 threadSize;


  if (timeron) timer_start(t_rhs);
  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = (size_t)copy_num_item;
  gws[1] = (size_t)grid_points[1];
  gws[0] = (size_t)grid_points[0];

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  if (buffering_flag && ev_wb_end_ptr != NULL) {
    CUCHK(cudaStreamWaitEvent(cmd_q[KERNEL_Q], *ev_wb_end_ptr, 0));
  }

  CUCHK(cudaEventRecord(ev_k_rhs_datagen_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhs_data_gen_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhs_data_gen_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_rho_i, m_us, m_vs, m_ws, m_qs, m_square, m_u[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     copy_buffer_base, copy_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhs_data_gen_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhs_datagen_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

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

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhs1_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhs1_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhs1_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_rhs[buf_idx],
     m_forcing[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhs1_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhs1_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));





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

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsx1_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhsx1_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhsx1_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_us, m_vs, m_ws, m_qs, m_rho_i, m_square,
     m_u[buf_idx],
     m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, xxcon2, xxcon3, xxcon4, xxcon5,
     c1, c2, tx2, con43, dssp, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhsx1_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhsx1_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item; 
  gws[1] = grid_points[1] - 2;
  gws[0] = grid_points[0] - 2;
  gws[0]*= 5;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsx2_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhsx2_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhsx2_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_u[buf_idx],
     m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     dssp, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhsx2_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhsx2_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));



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

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsy1_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhsy1_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhsy1_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_us, m_vs, m_ws, m_qs, m_rho_i, m_square,
     m_u[buf_idx], m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, yycon2, yycon3, yycon4, yycon5,
     c1, c2, ty2, con43, dssp, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhsy1_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhsy1_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = grid_points[1] - 2;
  gws[0] = grid_points[0] - 2;
  gws[0]*= 5;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsy2_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhsy2_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhsy2_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_u[buf_idx],
     m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     dssp, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhsy2_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhsy2_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));





  if (timeron) timer_stop(t_rhsy);


  if (timeron) timer_start(t_rhsz);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = (size_t)work_num_item;
  gws[1] = (size_t)grid_points[1]-2;
  gws[0] = (size_t)grid_points[0]-2;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsz1_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhsz1_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhsz1_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_us, m_vs, m_ws, m_qs, m_rho_i, m_square,
     m_u[buf_idx], m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1,
     zzcon2, zzcon3, zzcon4, zzcon5, c1, c2,
     tz2, con43, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhsz1_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhsz1_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));


  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = grid_points[1]-2;
  gws[0] = grid_points[0]-2;
  gws[0]*= 5;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhsz2_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhsz2_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhsz2_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_u[buf_idx], m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2], 
     dssp, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhsz2_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhsz2_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));


  if (timeron) timer_stop(t_rhsz);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = work_num_item;
  gws[1] = grid_points[1] - 2;
  gws[0] = grid_points[0] - 2;
  gws[0]*= 5;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  blockSize.x = gws[0]/lws[0]; threadSize.x = lws[0];
  blockSize.y = gws[1]/lws[1]; threadSize.y = lws[1];
  blockSize.z = gws[2]/lws[2]; threadSize.z = lws[2];

  CUCHK(cudaEventRecord(ev_k_rhs2_parallel[work_step][0], cmd_q[KERNEL_Q]));
  cuda_ProfilerStartEventRecord("k_compute_rhs2_parallel",  cmd_q[KERNEL_Q]);
  k_compute_rhs2_parallel<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     dt, work_base, work_num_item, split_flag
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_compute_rhs2_parallel",  cmd_q[KERNEL_Q]);
  CUCHK(cudaEventRecord(ev_k_rhs2_parallel[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  if (timeron) timer_stop(t_rhs);

}

