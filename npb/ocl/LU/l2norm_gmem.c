//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB LU code. This OpenCL C  //
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

#include <math.h>
#include <stdio.h>
#include "applu.incl"
#include "timers.h"

cl_kernel k_l2norm_gmem;

void l2norm_init_gmem()
{
  cl_int ecode;

  k_l2norm_gmem = clCreateKernel(p_l2norm_gmem, 
                                 "l2norm_gmem", 
                                 &ecode);
  clu_CheckError(ecode, "clCreateKernel()");
}

void l2norm_release_gmem()
{
  clReleaseKernel(k_l2norm_gmem);
}


void l2norm_body_gmem(int work_step, 
                         int work_max_iter, 
                         int work_base, 
                         int work_num_item,
                         cl_event* ev_wb_ptr, 
                         cl_event* ev_kernel_ptr, 
                         cl_mem* m_sum,
                         int nz0, 
                         int jst, int jend, 
                         int ist, int iend)
{
  cl_int ecode;
  size_t lws, gws;
  int buffer_base = 2;
  int buf_idx = (work_step%2)*buffering_flag;
  int num_wait = (buffering_flag) ? 1 : 0;
  cl_event *wait_ev = (buffering_flag) ? ev_wb_ptr : NULL;

  lws = l2norm_lws;
  gws = work_num_item*(jend-jst)*(iend-ist);
  gws = clu_RoundWorkSize(gws, lws);

  // the size of m_sum is hard coded in lu.c
  ecode  = clSetKernelArg(k_l2norm_gmem, 0, sizeof(cl_mem), &m_rsd[buf_idx]); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 1, sizeof(cl_mem), m_sum); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 2, sizeof(double)*lws*5, NULL); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 3, sizeof(int), &nz0); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 4, sizeof(int), &jst); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 5, sizeof(int), &jend); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 6, sizeof(int), &ist); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 7, sizeof(int), &iend); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 8, sizeof(int), &work_base); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 9, sizeof(int), &work_num_item); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 10, sizeof(int), &split_flag); 
  ecode |= clSetKernelArg(k_l2norm_gmem, 11, sizeof(int), &buffer_base); 
  clu_CheckError(ecode, "clSetKernelArg");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_l2norm_gmem,  
                                 1, NULL, 
                                 &gws, &lws, 
                                 num_wait, wait_ev,
                                 ev_kernel_ptr);
  clu_CheckError(ecode , "clEnqueueNDRangeKernel");

  if (!buffering_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);
}
