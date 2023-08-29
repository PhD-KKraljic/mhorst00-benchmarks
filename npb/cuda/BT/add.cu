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

#include "header.h"
#include "timers.h"
#include <stdio.h>

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add (int work_step, 
          int work_base, 
          int work_num_item,
          int buf_idx)
{
  size_t lws[3];
  size_t gws[3];

  if (timeron) timer_start(t_add);

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = grid_points[2] - 2 ;
  gws[1] = work_num_item; 
  gws[0] = grid_points[0] - 2 ;
  gws[0]*= 5;

  gws[2] = RoundWorkSize(gws[2], lws[2]);
  gws[1] = RoundWorkSize(gws[1], lws[1]);
  gws[0] = RoundWorkSize(gws[0], lws[0]);

  dim3 blockSize(gws[0]/lws[0], gws[1]/lws[1], gws[2]/lws[2]);
  dim3 threadSize(lws[0], lws[1], lws[2]);

  CUCHK(cudaEventRecord(loop2_ev_kernel_add[work_step][0], cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_add",  cmd_q[KERNEL_Q]);
  k_add<<< blockSize, threadSize, 0, cmd_q[KERNEL_Q]>>>
    (
     m_u[buf_idx],
     m_rhs[buf_idx],
     grid_points[0], grid_points[1], grid_points[2],
     work_base, work_num_item, split_flag, work_num_item_default_j
    );
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_add",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(loop2_ev_kernel_add[work_step][1], cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  if (timeron) timer_stop(t_add);

}
