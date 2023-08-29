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
void txinvr()
{
  int i, j, k;
  double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

  if (timeron) timer_start(t_txinvr);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        ru1 = rho_i[k][j][i];
        uu = us[k][j][i];
        vv = vs[k][j][i];
        ww = ws[k][j][i];
        ac = speed[k][j][i];
        ac2inv = ac*ac;

        r1 = rhs[k][j][i][0];
        r2 = rhs[k][j][i][1];
        r3 = rhs[k][j][i][2];
        r4 = rhs[k][j][i][3];
        r5 = rhs[k][j][i][4];

        t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
        t2 = bt * ru1 * ( uu * r1 - r2 );
        t3 = ( bt * ru1 * ac ) * t1;

        rhs[k][j][i][0] = r1 - t1;
        rhs[k][j][i][1] = - ru1 * ( ww*r1 - r4 );
        rhs[k][j][i][2] =   ru1 * ( vv*r1 - r3 );
        rhs[k][j][i][3] = - t2 + t3;
        rhs[k][j][i][4] =   t2 + t3;
      }
    }
  }
  if (timeron) timer_stop(t_txinvr);
}

void txinvr_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_txinvr);

  //---------------------------------------------------------------------
  // txinvr kernel
  //---------------------------------------------------------------------
  int txinvr_base_k = st2;
  int txinvr_offset_k = st - st2;
  int txinvr_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_txinvr, 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_txinvr, 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_txinvr, 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_txinvr, 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_txinvr, 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_txinvr, 5, sizeof(cl_mem), &buf_us);
  err_code |= clSetKernelArg(k_txinvr, 6, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_txinvr, 7, sizeof(cl_mem), &buf_ws);
  err_code |= clSetKernelArg(k_txinvr, 8, sizeof(cl_mem), &buf_qs);
  err_code |= clSetKernelArg(k_txinvr, 9, sizeof(cl_mem), &buf_speed);
  err_code |= clSetKernelArg(k_txinvr, 10, sizeof(cl_mem), &buf_rho_i);
  err_code |= clSetKernelArg(k_txinvr, 11, sizeof(int), &txinvr_base_k);
  err_code |= clSetKernelArg(k_txinvr, 12, sizeof(int), &txinvr_offset_k);
  err_code |= clSetKernelArg(k_txinvr, 13, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_txinvr, 14, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_txinvr, 15, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t txinvr_gws[] = {txinvr_gws_k};
    size_t txinvr_lws[] = {1};
    txinvr_gws[0] = clu_RoundWorkSize(txinvr_gws[0], txinvr_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_txinvr,
                                      1, NULL,
                                      txinvr_gws,
                                      txinvr_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t txinvr_gws[] = {nx2, ny2, txinvr_gws_k};
    size_t txinvr_lws[] = {16, 16, 1};
    txinvr_gws[0] = clu_RoundWorkSize(txinvr_gws[0], txinvr_lws[0]);
    txinvr_gws[1] = clu_RoundWorkSize(txinvr_gws[1], txinvr_lws[1]);
    txinvr_gws[2] = clu_RoundWorkSize(txinvr_gws[2], txinvr_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_txinvr,
                                      3, NULL,
                                      txinvr_gws,
                                      txinvr_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_txinvr);
}
