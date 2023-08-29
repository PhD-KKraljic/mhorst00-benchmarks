//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB LU code. This CUDA® C  //
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
#include "applu.incl"
extern "C" {
#include "timers.h"
}
//---------------------------------------------------------------------
// to compute the l2-norm of vector v.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------

void l2norm_body_gmem(int step, int iter, 
                      int base, int item,
                      cudaEvent_t* ev_wb_ptr, 
                      cudaEvent_t* ev_kernel_start_ptr,
                      cudaEvent_t* ev_kernel_end_ptr,
                      double* m_sum, int nz0, 
                      int jst, int jend, 
                      int ist, int iend)
{
  size_t lws, gws;
  int buffer_base = 2;
  int buf_idx = (step%2)*buffering_flag;

  lws = l2norm_lws;
  gws = item*(jend-jst)*(iend-ist);
  gws = RoundWorkSize(gws, lws);

  CUCHK(cudaEventRecord(*ev_kernel_start_ptr, cmd_q[KERNEL_Q]));
 
  // the size of m_sum is hard coded in lu.c
  cuda_ProfilerStartEventRecord("k_l2norm_gmem",  cmd_q[KERNEL_Q]);
  k_l2norm_gmem<<< gws / lws, lws, sizeof(double) * lws * 5, cmd_q[KERNEL_Q]>>>
    (m_rsd[buf_idx], m_sum, nz0, jst, jend,
     ist, iend, base, item,
     split_flag, buffer_base);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_l2norm_gmem",  cmd_q[KERNEL_Q]);

  CUCHK(cudaEventRecord(*ev_kernel_end_ptr, cmd_q[KERNEL_Q]));

  if (!buffering_flag)
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
}

