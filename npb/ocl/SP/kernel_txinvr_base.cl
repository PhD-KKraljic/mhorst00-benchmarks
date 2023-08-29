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

#include "kernel_header.h"

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------
__kernel void txinvr(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_us,
  __global double *g_vs,
  __global double *g_ws,
  __global double *g_qs,
  __global double *g_speed,
  __global double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2)
{
  __global double (*rhs0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rhs0;
  __global double (*rhs1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rhs1;
  __global double (*rhs2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rhs2;
  __global double (*rhs3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rhs3;
  __global double (*rhs4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rhs4;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;

  int i, j, k;
  double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      ru1 = rho_i[k][j][i];
      uu = us[k][j][i];
      vv = vs[k][j][i];
      ww = ws[k][j][i];
      ac = speed[k][j][i];
      ac2inv = ac*ac;

      r1 = rhs0[k][j][i];
      r2 = rhs1[k][j][i];
      r3 = rhs2[k][j][i];
      r4 = rhs3[k][j][i];
      r5 = rhs4[k][j][i];

      t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
      t2 = bt * ru1 * ( uu * r1 - r2 );
      t3 = ( bt * ru1 * ac ) * t1;

      rhs0[k][j][i] = r1 - t1;
      rhs1[k][j][i] = - ru1 * ( ww*r1 - r4 );
      rhs2[k][j][i] =   ru1 * ( vv*r1 - r3 );
      rhs3[k][j][i] = - t2 + t3;
      rhs4[k][j][i] =   t2 + t3;
    }
  }
}
