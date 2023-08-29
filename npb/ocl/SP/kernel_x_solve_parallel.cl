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
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

__kernel void x_solve0(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_us,
  __global double *g_rho_i,
  __global double *g_lhs0,
  __global double *g_lhs1,
  __global double *g_lhs2,
  __global double *g_lhs3,
  __global double *g_lhs4,
  __global double *g_lhsp0,
  __global double *g_lhsp1,
  __global double *g_lhsp2,
  __global double *g_lhsp3,
  __global double *g_lhsp4,
  __global double *g_lhsm0,
  __global double *g_lhsm1,
  __global double *g_lhsm2,
  __global double *g_lhsm3,
  __global double *g_lhsm4,
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
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;

  __global double (*lhs0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs0;
  __global double (*lhs1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs1;
  __global double (*lhs2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs2;
  __global double (*lhs3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs3;
  __global double (*lhs4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs4;
  __global double (*lhsp0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
  __global double (*lhsp1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
  __global double (*lhsp2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
  __global double (*lhsp3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
  __global double (*lhsp4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
  __global double (*lhsm0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
  __global double (*lhsm1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
  __global double (*lhsm2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
  __global double (*lhsm3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
  __global double (*lhsm4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, i1, i2, m;
  double ru1;
  double rhon_im1, rhon_i, rhon_ip1;
  k = offset_k + get_global_id(2);
  j = 1 + get_global_id(1);
  i = get_global_id(0);
  if (base_k + k > nz2 || j > ny2 || i > nx2+1) return;

  if (i == 0 || i == nx2+1) {
    //---------------------------------------------------------------------
    // zap the whole left hand side for starters
    // set all diagonal values to 1. This is overkill, but convenient
    //---------------------------------------------------------------------
    lhs0 [k][j][i] = 0.0;
    lhsp0[k][j][i] = 0.0;
    lhsm0[k][j][i] = 0.0;
    lhs1 [k][j][i] = 0.0;
    lhsp1[k][j][i] = 0.0;
    lhsm1[k][j][i] = 0.0;
    lhs2 [k][j][i] = 0.0;
    lhsp2[k][j][i] = 0.0;
    lhsm2[k][j][i] = 0.0;
    lhs3 [k][j][i] = 0.0;
    lhsp3[k][j][i] = 0.0;
    lhsm3[k][j][i] = 0.0;
    lhs4 [k][j][i] = 0.0;
    lhsp4[k][j][i] = 0.0;
    lhsm4[k][j][i] = 0.0;

    lhs2 [k][j][i] = 1.0;
    lhsp2[k][j][i] = 1.0;
    lhsm2[k][j][i] = 1.0;
  }
  else {
    //---------------------------------------------------------------------
    // Computes the left hand side for the three x-factors
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue
    //---------------------------------------------------------------------
    ru1 = c3c4*rho_i[k][j][i-1];
    rhon_im1 = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));

    ru1 = c3c4*rho_i[k][j][i];
    rhon_i = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));

    ru1 = c3c4*rho_i[k][j][i+1];
    rhon_ip1 = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));

    lhs0[k][j][i] =  0.0;
    lhs1[k][j][i] = -dttx2 * us[k][j][i-1] - dttx1 * rhon_im1;
    lhs2[k][j][i] =  1.0 + c2dttx1 * rhon_i;
    lhs3[k][j][i] =  dttx2 * us[k][j][i+1] - dttx1 * rhon_ip1;
    lhs4[k][j][i] =  0.0;
  }
}

__kernel void x_solve1(
  __global double *g_speed,
  __global double *g_lhs0,
  __global double *g_lhs1,
  __global double *g_lhs2,
  __global double *g_lhs3,
  __global double *g_lhs4,
  __global double *g_lhsp0,
  __global double *g_lhsp1,
  __global double *g_lhsp2,
  __global double *g_lhsp3,
  __global double *g_lhsp4,
  __global double *g_lhsm0,
  __global double *g_lhsm1,
  __global double *g_lhsm2,
  __global double *g_lhsm3,
  __global double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2)
{ 
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;

  __global double (*lhs0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs0;
  __global double (*lhs1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs1;
  __global double (*lhs2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs2;
  __global double (*lhs3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs3;
  __global double (*lhs4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs4;
  __global double (*lhsp0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
  __global double (*lhsp1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
  __global double (*lhsp2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
  __global double (*lhsp3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
  __global double (*lhsp4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
  __global double (*lhsm0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
  __global double (*lhsm1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
  __global double (*lhsm2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
  __global double (*lhsm3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
  __global double (*lhsm4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, i1, i2;
  k = offset_k + get_global_id(2);
  j = 1 + get_global_id(1);
  i = 1 + get_global_id(0);
  if (base_k + k > nz2 || j > ny2 || i > nx2) return;

  //---------------------------------------------------------------------
  // add fourth order dissipation
  //---------------------------------------------------------------------
  if (i == 1) {
    lhs2[k][j][i] = lhs2[k][j][i] + comz5;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
    lhs4[k][j][i] = lhs4[k][j][i] + comz1;
  }
  else if (i == 2) {
    lhs1[k][j][i] = lhs1[k][j][i] - comz4;
    lhs2[k][j][i] = lhs2[k][j][i] + comz6;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
    lhs4[k][j][i] = lhs4[k][j][i] + comz1;
  }
  else if (3 <= i && i <= nx2-2) {
    lhs0[k][j][i] = lhs0[k][j][i] + comz1;
    lhs1[k][j][i] = lhs1[k][j][i] - comz4;
    lhs2[k][j][i] = lhs2[k][j][i] + comz6;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
    lhs4[k][j][i] = lhs4[k][j][i] + comz1;
  }
  else if (i == nx2-1) {
    lhs0[k][j][i] = lhs0[k][j][i] + comz1;
    lhs1[k][j][i] = lhs1[k][j][i] - comz4;
    lhs2[k][j][i] = lhs2[k][j][i] + comz6;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
  }
  else {
    lhs0[k][j][i] = lhs0[k][j][i] + comz1;
    lhs1[k][j][i] = lhs1[k][j][i] - comz4;
    lhs2[k][j][i] = lhs2[k][j][i] + comz5;
  }

  //---------------------------------------------------------------------
  // subsequently, fill the other factors (u+c), (u-c) by adding to
  // the first
  //---------------------------------------------------------------------
  lhsp0[k][j][i] = lhs0[k][j][i];
  lhsp1[k][j][i] = lhs1[k][j][i] - dttx2 * speed[k][j][i-1];
  lhsp2[k][j][i] = lhs2[k][j][i];
  lhsp3[k][j][i] = lhs3[k][j][i] + dttx2 * speed[k][j][i+1];
  lhsp4[k][j][i] = lhs4[k][j][i];
  lhsm0[k][j][i] = lhs0[k][j][i];
  lhsm1[k][j][i] = lhs1[k][j][i] + dttx2 * speed[k][j][i-1];
  lhsm2[k][j][i] = lhs2[k][j][i];
  lhsm3[k][j][i] = lhs3[k][j][i] - dttx2 * speed[k][j][i+1];
  lhsm4[k][j][i] = lhs4[k][j][i];
}

__kernel void x_solve2(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_lhs0,
  __global double *g_lhs1,
  __global double *g_lhs2,
  __global double *g_lhs3,
  __global double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
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
  __global double (*lhs0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs0;
  __global double (*lhs1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs1;
  __global double (*lhs2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs2;
  __global double (*lhs3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs3;
  __global double (*lhs4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs4;

  int i, j, k, i1, i2, m;
  double fac1, fac2;
  k = offset_k + get_global_id(1);
  j = 1 + get_global_id(0);
  if (base_k + k > nz2 || j > ny2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // perform the Thomas algorithm; first, FORWARD ELIMINATION
  //---------------------------------------------------------------------
  for (i = 0; i <= nx2-1; i++) {
    i1 = i + 1;
    i2 = i + 2;
    fac1 = 1.0/lhs2[k][j][i];
    lhs3[k][j][i] = fac1*lhs3[k][j][i];
    lhs4[k][j][i] = fac1*lhs4[k][j][i];

    rhs0[k][j][i] = fac1*rhs0[k][j][i];
    rhs1[k][j][i] = fac1*rhs1[k][j][i];
    rhs2[k][j][i] = fac1*rhs2[k][j][i];

    lhs2[k][j][i1] = lhs2[k][j][i1] - lhs1[k][j][i1]*lhs3[k][j][i];
    lhs3[k][j][i1] = lhs3[k][j][i1] - lhs1[k][j][i1]*lhs4[k][j][i];

    rhs0[k][j][i1] = rhs0[k][j][i1] - lhs1[k][j][i1]*rhs0[k][j][i];
    rhs1[k][j][i1] = rhs1[k][j][i1] - lhs1[k][j][i1]*rhs1[k][j][i];
    rhs2[k][j][i1] = rhs2[k][j][i1] - lhs1[k][j][i1]*rhs2[k][j][i];

    lhs1[k][j][i2] = lhs1[k][j][i2] - lhs0[k][j][i2]*lhs3[k][j][i];
    lhs2[k][j][i2] = lhs2[k][j][i2] - lhs0[k][j][i2]*lhs4[k][j][i];

    rhs0[k][j][i2] = rhs0[k][j][i2] - lhs0[k][j][i2]*rhs0[k][j][i];
    rhs1[k][j][i2] = rhs1[k][j][i2] - lhs0[k][j][i2]*rhs1[k][j][i];
    rhs2[k][j][i2] = rhs2[k][j][i2] - lhs0[k][j][i2]*rhs2[k][j][i];
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different,
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  i  = nx2;
  i1 = nx2 + 1;
  fac1 = 1.0/lhs2[k][j][i];
  lhs3[k][j][i] = fac1*lhs3[k][j][i];
  lhs4[k][j][i] = fac1*lhs4[k][j][i];

  rhs0[k][j][i] = fac1*rhs0[k][j][i];
  rhs1[k][j][i] = fac1*rhs1[k][j][i];
  rhs2[k][j][i] = fac1*rhs2[k][j][i];

  lhs2[k][j][i1] = lhs2[k][j][i1] - lhs1[k][j][i1]*lhs3[k][j][i];
  lhs3[k][j][i1] = lhs3[k][j][i1] - lhs1[k][j][i1]*lhs4[k][j][i];

  rhs0[k][j][i1] = rhs0[k][j][i1] - lhs1[k][j][i1]*rhs0[k][j][i];
  rhs1[k][j][i1] = rhs1[k][j][i1] - lhs1[k][j][i1]*rhs1[k][j][i];
  rhs2[k][j][i1] = rhs2[k][j][i1] - lhs1[k][j][i1]*rhs2[k][j][i];

  //---------------------------------------------------------------------
  // scale the last row immediately
  //---------------------------------------------------------------------
  fac2 = 1.0/lhs2[k][j][i1];
  rhs0[k][j][i1] = fac2*rhs0[k][j][i1];
  rhs1[k][j][i1] = fac2*rhs1[k][j][i1];
  rhs2[k][j][i1] = fac2*rhs2[k][j][i1];
}

__kernel void x_solve3(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_lhsp0,
  __global double *g_lhsp1,
  __global double *g_lhsp2,
  __global double *g_lhsp3,
  __global double *g_lhsp4,
  __global double *g_lhsm0,
  __global double *g_lhsm1,
  __global double *g_lhsm2,
  __global double *g_lhsm3,
  __global double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
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
  __global double (*lhsp0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
  __global double (*lhsp1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
  __global double (*lhsp2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
  __global double (*lhsp3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
  __global double (*lhsp4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
  __global double (*lhsm0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
  __global double (*lhsm1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
  __global double (*lhsm2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
  __global double (*lhsm3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
  __global double (*lhsm4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, i1, i2, m;
  double fac1;
  k = offset_k + get_global_id(1);
  j = 1 + get_global_id(0);
  if (base_k + k > nz2 || j > ny2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors
  //---------------------------------------------------------------------
  for (i = 0; i <= nx2-1; i++) {
    i1 = i + 1;
    i2 = i + 2;

    m = 3;
    fac1 = 1.0/lhsp2[k][j][i];
    lhsp3[k][j][i]    = fac1*lhsp3[k][j][i];
    lhsp4[k][j][i]    = fac1*lhsp4[k][j][i];
    rhs3[k][j][i]  = fac1*rhs3[k][j][i];
    lhsp2[k][j][i1]   = lhsp2[k][j][i1] - lhsp1[k][j][i1]*lhsp3[k][j][i];
    lhsp3[k][j][i1]   = lhsp3[k][j][i1] - lhsp1[k][j][i1]*lhsp4[k][j][i];
    rhs3[k][j][i1] = rhs3[k][j][i1] - lhsp1[k][j][i1]*rhs3[k][j][i];
    lhsp1[k][j][i2]   = lhsp1[k][j][i2] - lhsp0[k][j][i2]*lhsp3[k][j][i];
    lhsp2[k][j][i2]   = lhsp2[k][j][i2] - lhsp0[k][j][i2]*lhsp4[k][j][i];
    rhs3[k][j][i2] = rhs3[k][j][i2] - lhsp0[k][j][i2]*rhs3[k][j][i];

    m = 4;
    fac1 = 1.0/lhsm2[k][j][i];
    lhsm3[k][j][i]    = fac1*lhsm3[k][j][i];
    lhsm4[k][j][i]    = fac1*lhsm4[k][j][i];
    rhs4[k][j][i]  = fac1*rhs4[k][j][i];
    lhsm2[k][j][i1]   = lhsm2[k][j][i1] - lhsm1[k][j][i1]*lhsm3[k][j][i];
    lhsm3[k][j][i1]   = lhsm3[k][j][i1] - lhsm1[k][j][i1]*lhsm4[k][j][i];
    rhs4[k][j][i1] = rhs4[k][j][i1] - lhsm1[k][j][i1]*rhs4[k][j][i];
    lhsm1[k][j][i2]   = lhsm1[k][j][i2] - lhsm0[k][j][i2]*lhsm3[k][j][i];
    lhsm2[k][j][i2]   = lhsm2[k][j][i2] - lhsm0[k][j][i2]*lhsm4[k][j][i];
    rhs4[k][j][i2] = rhs4[k][j][i2] - lhsm0[k][j][i2]*rhs4[k][j][i];
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  i  = nx2;
  i1 = nx2+1;

  m = 3;
  fac1 = 1.0/lhsp2[k][j][i];
  lhsp3[k][j][i]    = fac1*lhsp3[k][j][i];
  lhsp4[k][j][i]    = fac1*lhsp4[k][j][i];
  rhs3[k][j][i]  = fac1*rhs3[k][j][i];
  lhsp2[k][j][i1]   = lhsp2[k][j][i1] - lhsp1[k][j][i1]*lhsp3[k][j][i];
  lhsp3[k][j][i1]   = lhsp3[k][j][i1] - lhsp1[k][j][i1]*lhsp4[k][j][i];
  rhs3[k][j][i1] = rhs3[k][j][i1] - lhsp1[k][j][i1]*rhs3[k][j][i];

  m = 4;
  fac1 = 1.0/lhsm2[k][j][i];
  lhsm3[k][j][i]    = fac1*lhsm3[k][j][i];
  lhsm4[k][j][i]    = fac1*lhsm4[k][j][i];
  rhs4[k][j][i]  = fac1*rhs4[k][j][i];
  lhsm2[k][j][i1]   = lhsm2[k][j][i1] - lhsm1[k][j][i1]*lhsm3[k][j][i];
  lhsm3[k][j][i1]   = lhsm3[k][j][i1] - lhsm1[k][j][i1]*lhsm4[k][j][i];
  rhs4[k][j][i1] = rhs4[k][j][i1] - lhsm1[k][j][i1]*rhs4[k][j][i];

  //---------------------------------------------------------------------
  // Scale the last row immediately
  //---------------------------------------------------------------------
  rhs3[k][j][i1] = rhs3[k][j][i1]/lhsp2[k][j][i1];
  rhs4[k][j][i1] = rhs4[k][j][i1]/lhsm2[k][j][i1];
}

__kernel void x_solve4(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_lhs0,
  __global double *g_lhs1,
  __global double *g_lhs2,
  __global double *g_lhs3,
  __global double *g_lhs4,
  __global double *g_lhsp0,
  __global double *g_lhsp1,
  __global double *g_lhsp2,
  __global double *g_lhsp3,
  __global double *g_lhsp4,
  __global double *g_lhsm0,
  __global double *g_lhsm1,
  __global double *g_lhsm2,
  __global double *g_lhsm3,
  __global double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
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
  __global double (*lhs0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs0;
  __global double (*lhs1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs1;
  __global double (*lhs2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs2;
  __global double (*lhs3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs3;
  __global double (*lhs4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhs4;
  __global double (*lhsp0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
  __global double (*lhsp1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
  __global double (*lhsp2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
  __global double (*lhsp3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
  __global double (*lhsp4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
  __global double (*lhsm0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
  __global double (*lhsm1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
  __global double (*lhsm2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
  __global double (*lhsm3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
  __global double (*lhsm4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, i1, i2, m;
  k = offset_k + get_global_id(1);
  j = 1 + get_global_id(0);
  if (base_k + k > nz2 || j > ny2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION
  //---------------------------------------------------------------------
  i  = nx2;
  i1 = nx2+1;
  rhs0[k][j][i] = rhs0[k][j][i] - lhs3[k][j][i]*rhs0[k][j][i1];
  rhs1[k][j][i] = rhs1[k][j][i] - lhs3[k][j][i]*rhs1[k][j][i1];
  rhs2[k][j][i] = rhs2[k][j][i] - lhs3[k][j][i]*rhs2[k][j][i1];

  rhs3[k][j][i] = rhs3[k][j][i] - lhsp3[k][j][i]*rhs3[k][j][i1];
  rhs4[k][j][i] = rhs4[k][j][i] - lhsm3[k][j][i]*rhs4[k][j][i1];

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (i = nx2-1; i >= 0; i--) {
    i1 = i + 1;
    i2 = i + 2;
    rhs0[k][j][i] = rhs0[k][j][i] -
                      lhs3[k][j][i]*rhs0[k][j][i1] -
                      lhs4[k][j][i]*rhs0[k][j][i2];
    rhs1[k][j][i] = rhs1[k][j][i] -
                      lhs3[k][j][i]*rhs1[k][j][i1] -
                      lhs4[k][j][i]*rhs1[k][j][i2];
    rhs2[k][j][i] = rhs2[k][j][i] -
                      lhs3[k][j][i]*rhs2[k][j][i1] -
                      lhs4[k][j][i]*rhs2[k][j][i2];

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    rhs3[k][j][i] = rhs3[k][j][i] -
                      lhsp3[k][j][i]*rhs3[k][j][i1] -
                      lhsp4[k][j][i]*rhs3[k][j][i2];
    rhs4[k][j][i] = rhs4[k][j][i] -
                      lhsm3[k][j][i]*rhs4[k][j][i1] -
                      lhsm4[k][j][i]*rhs4[k][j][i2];
  }
}

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__kernel void ninvr(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
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

  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;
  k = offset_k + get_global_id(2);
  j = 1 + get_global_id(1);
  i = 1 + get_global_id(0);
  if (base_k + k > nz2 || j > ny2 || i > nx2) return;

  r1 = rhs0[k][j][i];
  r2 = rhs1[k][j][i];
  r3 = rhs2[k][j][i];
  r4 = rhs3[k][j][i];
  r5 = rhs4[k][j][i];

  t1 = bt * r3;
  t2 = 0.5 * ( r4 + r5 );

  rhs0[k][j][i] = -r2;
  rhs1[k][j][i] =  r1;
  rhs2[k][j][i] = bt * ( r4 - r5 );
  rhs3[k][j][i] = -t1 + t2;
  rhs4[k][j][i] =  t1 + t2;
}
