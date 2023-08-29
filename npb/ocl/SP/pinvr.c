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
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
void pinvr()
{
  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;

  if (timeron) timer_start(t_pinvr);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        r1 = rhs[k][j][i][0];
        r2 = rhs[k][j][i][1];
        r3 = rhs[k][j][i][2];
        r4 = rhs[k][j][i][3];
        r5 = rhs[k][j][i][4];

        t1 = bt * r1;
        t2 = 0.5 * ( r4 + r5 );

        rhs[k][j][i][0] =  bt * ( r4 - r5 );
        rhs[k][j][i][1] = -r3;
        rhs[k][j][i][2] =  r2;
        rhs[k][j][i][3] = -t1 + t2;
        rhs[k][j][i][4] =  t1 + t2;
      }
    }
  }
  if (timeron) timer_stop(t_pinvr);
}

void pinvr_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_pinvr);

  //---------------------------------------------------------------------
  // pinvr kernel
  //---------------------------------------------------------------------
  int pinvr_base_k = st2;
  int pinvr_offset_k = st - st2;
  int pinvr_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_pinvr, 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_pinvr, 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_pinvr, 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_pinvr, 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_pinvr, 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_pinvr, 5, sizeof(int), &pinvr_base_k);
  err_code |= clSetKernelArg(k_pinvr, 6, sizeof(int), &pinvr_offset_k);
  err_code |= clSetKernelArg(k_pinvr, 7, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_pinvr, 8, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_pinvr, 9, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t pinvr_gws[] = {pinvr_gws_k};
    size_t pinvr_lws[] = {1};
    pinvr_gws[0] = clu_RoundWorkSize(pinvr_gws[0], pinvr_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_pinvr,
                                      1, NULL,
                                      pinvr_gws,
                                      pinvr_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t pinvr_gws[] = {nx2, ny2, pinvr_gws_k};
    size_t pinvr_lws[] = {16, 16, 1};
    pinvr_gws[0] = clu_RoundWorkSize(pinvr_gws[0], pinvr_lws[0]);
    pinvr_gws[1] = clu_RoundWorkSize(pinvr_gws[1], pinvr_lws[1]);
    pinvr_gws[2] = clu_RoundWorkSize(pinvr_gws[2], pinvr_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_pinvr,
                                      3, NULL,
                                      pinvr_gws,
                                      pinvr_lws,
                                      0, NULL, NULL);
  }
  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_pinvr);
}
