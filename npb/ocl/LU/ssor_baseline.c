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

#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "applu.incl"
#include "timers.h"
#include <math.h>

cl_kernel k_ssor1_baseline, 
          k_ssor2_baseline;

cl_event  *ev_k_ssor1,
          *ev_k_ssor2;

cl_ulong  ev_ns_ssor1_baseline,
          ev_ns_ssor2_baseline;

void ssor1_init_baseline(int iter)
{
  cl_int ecode;
  k_ssor1_baseline = clCreateKernel(p_ssor_baseline, 
                                    "ssor1_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_ssor1 = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void ssor1_release_baseline()
{
  clReleaseKernel(k_ssor1_baseline);

  free(ev_k_ssor1);
}

void ssor1_release_ev_baseline(int iter)
{
  int i;

  for (i = 0; i < iter; i++)
    clReleaseEvent(ev_k_ssor1[i]);
}

void ssor1_baseline(int item, int base, int step, 
                    int buf_idx, cl_event *ev_wb_ptr)
{
  cl_int ecode;
  size_t gws[3], lws[3];
  int num_wait;
  cl_event *wait_ev;

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = min(max_work_item_sizes[0], max_work_group_size);

  gws[2] = (size_t) item;
  gws[1] = (size_t) jend - jst;
  gws[0] = (size_t) iend - ist;
  gws[0] *= 5; 

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_ssor1_baseline, 0, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_ssor1_baseline, 1, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_ssor1_baseline, 2, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_ssor1_baseline, 3, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_ssor1_baseline, 4, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_ssor1_baseline, 5, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_ssor1_baseline, 6, sizeof(double), &dt);
  ecode |= clSetKernelArg(k_ssor1_baseline, 7, sizeof(int), &base);
  ecode |= clSetKernelArg(k_ssor1_baseline, 8, sizeof(int), &item);
  ecode |= clSetKernelArg(k_ssor1_baseline, 9, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg()");

  num_wait = (buffering_flag) ? 1 : 0;
  wait_ev = (buffering_flag) ? ev_wb_ptr : NULL;

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_ssor1_baseline,
                                 3, NULL, 
                                 gws, lws,
                                 num_wait, wait_ev, 
                                 &ev_k_ssor1[step]);
  clu_CheckError(ecode , "clEnqueueNDRangeKerenel()");

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);
}



void ssor2_init_baseline(int iter)
{
  cl_int ecode;
  k_ssor2_baseline = clCreateKernel(p_ssor_baseline, 
                                    "ssor2_baseline", 
                                    &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_ssor2 = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void ssor2_release_baseline()
{
  clReleaseKernel(k_ssor2_baseline);
  free(ev_k_ssor2);
}

void ssor2_release_ev_baseline(int iter)
{
  int i;
  for (i = 0; i < iter; i++)
    clReleaseEvent(ev_k_ssor2[i]);
}

void ssor2_baseline(int item, int base, int step, 
                    int buf_idx, int temp_kst, double tmp2)
{
  cl_int ecode;
  size_t gws[3], lws[3];

  lws[2] = 1;
  lws[1] = 1;
  lws[0] = max_work_group_size;

  gws[2] = (size_t) item;
  gws[1] = (size_t) jend - jst;
  gws[0] = (size_t) iend - ist;
  gws[0] *= 5; 

  gws[2] = clu_RoundWorkSize(gws[2], lws[2]);
  gws[1] = clu_RoundWorkSize(gws[1], lws[1]);
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode  = clSetKernelArg(k_ssor2_baseline, 0, sizeof(cl_mem), &m_u[buf_idx]);
  ecode |= clSetKernelArg(k_ssor2_baseline, 1, sizeof(cl_mem), &m_rsd[buf_idx]);
  ecode |= clSetKernelArg(k_ssor2_baseline, 2, sizeof(double), &tmp2);
  ecode |= clSetKernelArg(k_ssor2_baseline, 3, sizeof(int), &nz);
  ecode |= clSetKernelArg(k_ssor2_baseline, 4, sizeof(int), &jst);
  ecode |= clSetKernelArg(k_ssor2_baseline, 5, sizeof(int), &jend);
  ecode |= clSetKernelArg(k_ssor2_baseline, 6, sizeof(int), &ist);
  ecode |= clSetKernelArg(k_ssor2_baseline, 7, sizeof(int), &iend);
  ecode |= clSetKernelArg(k_ssor2_baseline, 8, sizeof(int), &temp_kst);
  ecode |= clSetKernelArg(k_ssor2_baseline, 9, sizeof(int), &base);
  ecode |= clSetKernelArg(k_ssor2_baseline, 10, sizeof(int), &item);
  ecode |= clSetKernelArg(k_ssor2_baseline, 11, sizeof(int), &split_flag);
  clu_CheckError(ecode, "clSetKernelArg");

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_ssor2_baseline,
                                 3, NULL, 
                                 gws, lws,
                                 0, NULL, 
                                 &ev_k_ssor2[step]);
  clu_CheckError(ecode, "clEnqueueNDRangeKerenel");

  if (!buffering_flag && split_flag)
    clFinish(cmd_q[KERNEL_Q]);
  else
    clFlush(cmd_q[KERNEL_Q]);
}
