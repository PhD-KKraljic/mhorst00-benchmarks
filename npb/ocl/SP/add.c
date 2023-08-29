//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB SP code. This OpenCL C  //
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

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;

  if (timeron) timer_start(t_add);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
        }
      }
    }
  }
  if (timeron) timer_stop(t_add);
}

void add_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_add);

  //---------------------------------------------------------------------
  // add kernel
  //---------------------------------------------------------------------
  int add_base_j = st2;
  int add_offset_j = st - st2;
  int add_gws_j = ed - st + 1;

  err_code  = clSetKernelArg(k_add, 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_add, 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_add, 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_add, 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_add, 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_add, 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_add, 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_add, 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_add, 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_add, 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_add, 10, sizeof(int), &add_base_j);
  err_code |= clSetKernelArg(k_add, 11, sizeof(int), &add_offset_j);
  err_code |= clSetKernelArg(k_add, 12, sizeof(int), &add_gws_j);
  err_code |= clSetKernelArg(k_add, 13, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_add, 14, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_add, 15, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
    
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t add_gws[] = {nz2};
    size_t add_lws[] = {16};
    add_gws[0] = clu_RoundWorkSize(add_gws[0], add_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_add,
                                      1, NULL,
                                      add_gws,
                                      add_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t add_gws[] = {nx2, add_gws_j, nz2};
    size_t add_lws[] = {16, 16, 1};
    add_gws[0] = clu_RoundWorkSize(add_gws[0], add_lws[0]);
    add_gws[1] = clu_RoundWorkSize(add_gws[1], add_lws[1]);
    add_gws[2] = clu_RoundWorkSize(add_gws[2], add_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_add,
                                      3, NULL,
                                      add_gws,
                                      add_lws,
                                      0, NULL, NULL);
  }
  
  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_add);
}
