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
void tzetar()
{
  int i, j, k;
  double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
  double btuz, ac2u, uzik1;

  if (timeron) timer_start(t_tzetar);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        xvel = us[k][j][i];
        yvel = vs[k][j][i];
        zvel = ws[k][j][i];
        ac   = speed[k][j][i];

        ac2u = ac*ac;

        r1 = rhs[k][j][i][0];
        r2 = rhs[k][j][i][1];
        r3 = rhs[k][j][i][2];
        r4 = rhs[k][j][i][3];
        r5 = rhs[k][j][i][4];     

        uzik1 = u[k][j][i][0];
        btuz  = bt * uzik1;

        t1 = btuz/ac * (r4 + r5);
        t2 = r3 + t1;
        t3 = btuz * (r4 - r5);

        rhs[k][j][i][0] = t2;
        rhs[k][j][i][1] = -uzik1*r2 + xvel*t2;
        rhs[k][j][i][2] =  uzik1*r1 + yvel*t2;
        rhs[k][j][i][3] =  zvel*t2  + t3;
        rhs[k][j][i][4] =  uzik1*(-xvel*r2 + yvel*r1) + 
                           qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;
      }
    }
  }
  if (timeron) timer_stop(t_tzetar);
}

void tzetar_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_tzetar);

  //---------------------------------------------------------------------
  // tzetar kernel
  //---------------------------------------------------------------------
  int tzetar_base_j = st2;
  int tzetar_offset_j = st - st2;
  int tzetar_gws_j = ed - st + 1;

  err_code  = clSetKernelArg(k_tzetar, 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_tzetar, 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_tzetar, 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_tzetar, 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_tzetar, 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_tzetar, 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_tzetar, 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_tzetar, 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_tzetar, 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_tzetar, 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_tzetar, 10, sizeof(cl_mem), &buf_us);
  err_code |= clSetKernelArg(k_tzetar, 11, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_tzetar, 12, sizeof(cl_mem), &buf_ws);
  err_code |= clSetKernelArg(k_tzetar, 13, sizeof(cl_mem), &buf_qs);
  err_code |= clSetKernelArg(k_tzetar, 14, sizeof(cl_mem), &buf_speed);
  err_code |= clSetKernelArg(k_tzetar, 15, sizeof(int), &tzetar_base_j);
  err_code |= clSetKernelArg(k_tzetar, 16, sizeof(int), &tzetar_offset_j);
  err_code |= clSetKernelArg(k_tzetar, 17, sizeof(int), &tzetar_gws_j);
  err_code |= clSetKernelArg(k_tzetar, 18, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_tzetar, 19, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_tzetar, 20, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t tzetar_gws[] = {nz2};
    size_t tzetar_lws[] = {16};
    tzetar_gws[0] = clu_RoundWorkSize(tzetar_gws[0], tzetar_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_tzetar,
                                      1, NULL,
                                      tzetar_gws,
                                      tzetar_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t tzetar_gws[] = {nx2, tzetar_gws_j, nz2};
    size_t tzetar_lws[] = {16, 16, 1};
    tzetar_gws[0] = clu_RoundWorkSize(tzetar_gws[0], tzetar_lws[0]);
    tzetar_gws[1] = clu_RoundWorkSize(tzetar_gws[1], tzetar_lws[1]);
    tzetar_gws[2] = clu_RoundWorkSize(tzetar_gws[2], tzetar_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_tzetar,
                                      3, NULL,
                                      tzetar_gws,
                                      tzetar_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_tzetar);
}
