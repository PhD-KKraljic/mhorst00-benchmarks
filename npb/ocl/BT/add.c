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
#include <stdio.h>

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add(int work_step, int work_base, int work_num_item, int buf_idx)
{
  cl_int ecode;
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

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_add, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_add, 1, sizeof(cl_mem), &m_rhs[buf_idx]);
  ecode |= clSetKernelArg(k_add, 2, sizeof(int), &grid_points[0]);
  ecode |= clSetKernelArg(k_add, 3, sizeof(int), &grid_points[1]);
  ecode |= clSetKernelArg(k_add, 4, sizeof(int), &grid_points[2]);
  ecode |= clSetKernelArg(k_add, 5, sizeof(int), &work_base);
  ecode |= clSetKernelArg(k_add, 6, sizeof(int), &work_num_item);
  ecode |= clSetKernelArg(k_add, 7, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_add,
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &(loop2_ev_kernel_add[work_step]));
  clu_CheckError(ecode, "clEnqueueNDRangeKernel()");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);

  if (timeron) timer_stop(t_add);
}
