//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB SP code. This CUDA® C  //
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

#include "npbparams.h"
#include "kernel_header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

__global__ void k_x_solve0_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_us,
   double *g_rho_i,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]

   double (*us)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_us;
   double (*rho_i)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_rho_i;

#define lhs0(a, b, c) g_lhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs1(a, b, c) g_lhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs2(a, b, c) g_lhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs3(a, b, c) g_lhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs4(a, b, c) g_lhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp0(a, b, c) g_lhsp0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp1(a, b, c) g_lhsp1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp2(a, b, c) g_lhsp2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp3(a, b, c) g_lhsp3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp4(a, b, c) g_lhsp4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm0(a, b, c) g_lhsm0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm1(a, b, c) g_lhsm1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm2(a, b, c) g_lhsm2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm3(a, b, c) g_lhsm3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm4(a, b, c) g_lhsm4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]

  int i, j, k, i1, i2, m;
  double ru1;
  double rhon_im1, rhon_i, rhon_ip1;
  k = offset_k + blockDim.z * blockIdx.z + threadIdx.z;
  j = 1 + blockDim.y * blockIdx.y + threadIdx.y;
  i = blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || j > ny2 || i > nx2+1) return;

  if (i == 0 || i == nx2+1) {
    //---------------------------------------------------------------------
    // zap the whole left hand side for starters
    // set all diagonal values to 1. This is overkill, but convenient
    //---------------------------------------------------------------------
    lhs0 (i, k, j) = 0.0;
    lhsp0(i, k, j) = 0.0;
    lhsm0(i, k, j) = 0.0;
    lhs1 (i, k, j) = 0.0;
    lhsp1(i, k, j) = 0.0;
    lhsm1(i, k, j) = 0.0;
    lhs2 (i, k, j) = 0.0;
    lhsp2(i, k, j) = 0.0;
    lhsm2(i, k, j) = 0.0;
    lhs3 (i, k, j) = 0.0;
    lhsp3(i, k, j) = 0.0;
    lhsm3(i, k, j) = 0.0;
    lhs4 (i, k, j) = 0.0;
    lhsp4(i, k, j) = 0.0;
    lhsm4(i, k, j) = 0.0;

    lhs2 (i, k, j) = 1.0;
    lhsp2(i, k, j) = 1.0;
    lhsm2(i, k, j) = 1.0;
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

    lhs0(i, k, j) =  0.0;
    lhs1(i, k, j) = -dttx2 * us[k][j][i-1] - dttx1 * rhon_im1;
    lhs2(i, k, j) =  1.0 + c2dttx1 * rhon_i;
    lhs3(i, k, j) =  dttx2 * us[k][j][i+1] - dttx1 * rhon_ip1;
    lhs4(i, k, j) =  0.0;
  }

#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
#undef lhs0
#undef lhs1
#undef lhs2
#undef lhs3
#undef lhs4
#undef lhsp0
#undef lhsp1
#undef lhsp2
#undef lhsp3
#undef lhsp4
#undef lhsm0
#undef lhsm1
#undef lhsm2
#undef lhsm3
#undef lhsm4
}

__global__ void k_x_solve1_opt(
   double *g_speed,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K)
{ 
   double (*speed)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_speed;

#define lhs0(a, b, c) g_lhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs1(a, b, c) g_lhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs2(a, b, c) g_lhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs3(a, b, c) g_lhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs4(a, b, c) g_lhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp0(a, b, c) g_lhsp0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp1(a, b, c) g_lhsp1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp2(a, b, c) g_lhsp2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp3(a, b, c) g_lhsp3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp4(a, b, c) g_lhsp4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm0(a, b, c) g_lhsm0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm1(a, b, c) g_lhsm1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm2(a, b, c) g_lhsm2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm3(a, b, c) g_lhsm3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm4(a, b, c) g_lhsm4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]

  int i, j, k, i1, i2;
  k = offset_k + blockDim.z * blockIdx.z + threadIdx.z;
  j = 1 + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || j > ny2 || i > nx2) return;

  //---------------------------------------------------------------------
  // add fourth order dissipation
  //---------------------------------------------------------------------
  if (i == 1) {
    lhs2(i, k, j) = lhs2(i, k, j) + comz5;
    lhs3(i, k, j) = lhs3(i, k, j) - comz4;
    lhs4(i, k, j) = lhs4(i, k, j) + comz1;
  }
  else if (i == 2) {
    lhs1(i, k, j) = lhs1(i, k, j) - comz4;
    lhs2(i, k, j) = lhs2(i, k, j) + comz6;
    lhs3(i, k, j) = lhs3(i, k, j) - comz4;
    lhs4(i, k, j) = lhs4(i, k, j) + comz1;
  }
  else if (3 <= i && i <= nx2-2) {
    lhs0(i, k, j) = lhs0(i, k, j) + comz1;
    lhs1(i, k, j) = lhs1(i, k, j) - comz4;
    lhs2(i, k, j) = lhs2(i, k, j) + comz6;
    lhs3(i, k, j) = lhs3(i, k, j) - comz4;
    lhs4(i, k, j) = lhs4(i, k, j) + comz1;
  }
  else if (i == nx2-1) {
    lhs0(i, k, j) = lhs0(i, k, j) + comz1;
    lhs1(i, k, j) = lhs1(i, k, j) - comz4;
    lhs2(i, k, j) = lhs2(i, k, j) + comz6;
    lhs3(i, k, j) = lhs3(i, k, j) - comz4;
  }
  else {
    lhs0(i, k, j) = lhs0(i, k, j) + comz1;
    lhs1(i, k, j) = lhs1(i, k, j) - comz4;
    lhs2(i, k, j) = lhs2(i, k, j) + comz5;
  }

  //---------------------------------------------------------------------
  // subsequently, fill the other factors (u+c), (u-c) by adding to
  // the first
  //---------------------------------------------------------------------
  lhsp0(i, k, j) = lhs0(i, k, j);
  lhsp1(i, k, j) = lhs1(i, k, j) - dttx2 * speed[k][j][i-1];
  lhsp2(i, k, j) = lhs2(i, k, j);
  lhsp3(i, k, j) = lhs3(i, k, j) + dttx2 * speed[k][j][i+1];
  lhsp4(i, k, j) = lhs4(i, k, j);
  lhsm0(i, k, j) = lhs0(i, k, j);
  lhsm1(i, k, j) = lhs1(i, k, j) + dttx2 * speed[k][j][i-1];
  lhsm2(i, k, j) = lhs2(i, k, j);
  lhsm3(i, k, j) = lhs3(i, k, j) - dttx2 * speed[k][j][i+1];
  lhsm4(i, k, j) = lhs4(i, k, j);
#undef lhs0
#undef lhs1
#undef lhs2
#undef lhs3
#undef lhs4
#undef lhsp0
#undef lhsp1
#undef lhsp2
#undef lhsp3
#undef lhsp4
#undef lhsm0
#undef lhsm1
#undef lhsm2
#undef lhsm3
#undef lhsm4
}

__global__ void k_x_solve2_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K)
{ 
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs0(a, b, c) g_lhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs1(a, b, c) g_lhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs2(a, b, c) g_lhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs3(a, b, c) g_lhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs4(a, b, c) g_lhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]

  int i, j, k, i1, i2, m;
  double fac1, fac2;
  k = offset_k + blockDim.y * blockIdx.y + threadIdx.y;
  j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
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
    fac1 = 1.0/lhs2(i, k, j);
    lhs3(i, k, j) = fac1*lhs3(i, k, j);
    lhs4(i, k, j) = fac1*lhs4(i, k, j);

    rhs0(i, k, j) = fac1*rhs0(i, k, j);
    rhs1(i, k, j) = fac1*rhs1(i, k, j);
    rhs2(i, k, j) = fac1*rhs2(i, k, j);

    lhs2(i1, k, j) = lhs2(i1, k, j) - lhs1(i1, k, j)*lhs3(i, k, j);
    lhs3(i1, k, j) = lhs3(i1, k, j) - lhs1(i1, k, j)*lhs4(i, k, j);

    rhs0(i1, k, j) = rhs0(i1, k, j) - lhs1(i1, k, j)*rhs0(i, k, j);
    rhs1(i1, k, j) = rhs1(i1, k, j) - lhs1(i1, k, j)*rhs1(i, k, j);
    rhs2(i1, k, j) = rhs2(i1, k, j) - lhs1(i1, k, j)*rhs2(i, k, j);

    lhs1(i2, k, j) = lhs1(i2, k, j) - lhs0(i2, k, j)*lhs3(i, k, j);
    lhs2(i2, k, j) = lhs2(i2, k, j) - lhs0(i2, k, j)*lhs4(i, k, j);

    rhs0(i2, k, j) = rhs0(i2, k, j) - lhs0(i2, k, j)*rhs0(i, k, j);
    rhs1(i2, k, j) = rhs1(i2, k, j) - lhs0(i2, k, j)*rhs1(i, k, j);
    rhs2(i2, k, j) = rhs2(i2, k, j) - lhs0(i2, k, j)*rhs2(i, k, j);
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different,
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  i  = nx2;
  i1 = nx2 + 1;
  fac1 = 1.0/lhs2(i, k, j);
  lhs3(i, k, j) = fac1*lhs3(i, k, j);
  lhs4(i, k, j) = fac1*lhs4(i, k, j);

  rhs0(i, k, j) = fac1*rhs0(i, k, j);
  rhs1(i, k, j) = fac1*rhs1(i, k, j);
  rhs2(i, k, j) = fac1*rhs2(i, k, j);

  lhs2(i1, k, j) = lhs2(i1, k, j) - lhs1(i1, k, j)*lhs3(i, k, j);
  lhs3(i1, k, j) = lhs3(i1, k, j) - lhs1(i1, k, j)*lhs4(i, k, j);

  rhs0(i1, k, j) = rhs0(i1, k, j) - lhs1(i1, k, j)*rhs0(i, k, j);
  rhs1(i1, k, j) = rhs1(i1, k, j) - lhs1(i1, k, j)*rhs1(i, k, j);
  rhs2(i1, k, j) = rhs2(i1, k, j) - lhs1(i1, k, j)*rhs2(i, k, j);

  //---------------------------------------------------------------------
  // scale the last row immediately
  //---------------------------------------------------------------------
  fac2 = 1.0/lhs2(i1, k, j);
  rhs0(i1, k, j) = fac2*rhs0(i1, k, j);
  rhs1(i1, k, j) = fac2*rhs1(i1, k, j);
  rhs2(i1, k, j) = fac2*rhs2(i1, k, j);
#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
#undef lhs0
#undef lhs1
#undef lhs2
#undef lhs3
#undef lhs4

}

__global__ void k_x_solve3_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp0(a, b, c) g_lhsp0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp1(a, b, c) g_lhsp1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp2(a, b, c) g_lhsp2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp3(a, b, c) g_lhsp3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp4(a, b, c) g_lhsp4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm0(a, b, c) g_lhsm0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm1(a, b, c) g_lhsm1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm2(a, b, c) g_lhsm2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm3(a, b, c) g_lhsm3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm4(a, b, c) g_lhsm4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]

  int i, j, k, i1, i2, m;
  double fac1;
  k = offset_k + blockDim.y * blockIdx.y + threadIdx.y;
  j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || j > ny2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors
  //---------------------------------------------------------------------
  for (i = 0; i <= nx2-1; i++) {
    i1 = i + 1;
    i2 = i + 2;

    m = 3;
    fac1 = 1.0/lhsp2(i, k, j);
    lhsp3(i, k, j)    = fac1*lhsp3(i, k, j);
    lhsp4(i, k, j)    = fac1*lhsp4(i, k, j);
    rhs3(i, k, j)  = fac1*rhs3(i, k, j);
    lhsp2(i1, k, j)   = lhsp2(i1, k, j) - lhsp1(i1, k, j)*lhsp3(i, k, j);
    lhsp3(i1, k, j)   = lhsp3(i1, k, j) - lhsp1(i1, k, j)*lhsp4(i, k, j);
    rhs3(i1, k, j) = rhs3(i1, k, j) - lhsp1(i1, k, j)*rhs3(i, k, j);
    lhsp1(i2, k, j)   = lhsp1(i2, k, j) - lhsp0(i2, k, j)*lhsp3(i, k, j);
    lhsp2(i2, k, j)   = lhsp2(i2, k, j) - lhsp0(i2, k, j)*lhsp4(i, k, j);
    rhs3(i2, k, j) = rhs3(i2, k, j) - lhsp0(i2, k, j)*rhs3(i, k, j);

    m = 4;
    fac1 = 1.0/lhsm2(i, k, j);
    lhsm3(i, k, j)    = fac1*lhsm3(i, k, j);
    lhsm4(i, k, j)    = fac1*lhsm4(i, k, j);
    rhs4(i, k, j)  = fac1*rhs4(i, k, j);
    lhsm2(i1, k, j)   = lhsm2(i1, k, j) - lhsm1(i1, k, j)*lhsm3(i, k, j);
    lhsm3(i1, k, j)   = lhsm3(i1, k, j) - lhsm1(i1, k, j)*lhsm4(i, k, j);
    rhs4(i1, k, j) = rhs4(i1, k, j) - lhsm1(i1, k, j)*rhs4(i, k, j);
    lhsm1(i2, k, j)   = lhsm1(i2, k, j) - lhsm0(i2, k, j)*lhsm3(i, k, j);
    lhsm2(i2, k, j)   = lhsm2(i2, k, j) - lhsm0(i2, k, j)*lhsm4(i, k, j);
    rhs4(i2, k, j) = rhs4(i2, k, j) - lhsm0(i2, k, j)*rhs4(i, k, j);
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  i  = nx2;
  i1 = nx2+1;

  m = 3;
  fac1 = 1.0/lhsp2(i, k, j);
  lhsp3(i, k, j)    = fac1*lhsp3(i, k, j);
  lhsp4(i, k, j)    = fac1*lhsp4(i, k, j);
  rhs3(i, k, j)  = fac1*rhs3(i, k, j);
  lhsp2(i1, k, j)   = lhsp2(i1, k, j) - lhsp1(i1, k, j)*lhsp3(i, k, j);
  lhsp3(i1, k, j)   = lhsp3(i1, k, j) - lhsp1(i1, k, j)*lhsp4(i, k, j);
  rhs3(i1, k, j) = rhs3(i1, k, j) - lhsp1(i1, k, j)*rhs3(i, k, j);

  m = 4;
  fac1 = 1.0/lhsm2(i, k, j);
  lhsm3(i, k, j)    = fac1*lhsm3(i, k, j);
  lhsm4(i, k, j)    = fac1*lhsm4(i, k, j);
  rhs4(i, k, j)  = fac1*rhs4(i, k, j);
  lhsm2(i1, k, j)   = lhsm2(i1, k, j) - lhsm1(i1, k, j)*lhsm3(i, k, j);
  lhsm3(i1, k, j)   = lhsm3(i1, k, j) - lhsm1(i1, k, j)*lhsm4(i, k, j);
  rhs4(i1, k, j) = rhs4(i1, k, j) - lhsm1(i1, k, j)*rhs4(i, k, j);

  //---------------------------------------------------------------------
  // Scale the last row immediately
  //---------------------------------------------------------------------
  rhs3(i1, k, j) = rhs3(i1, k, j)/lhsp2(i1, k, j);
  rhs4(i1, k, j) = rhs4(i1, k, j)/lhsm2(i1, k, j);


#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
#undef lhsp0
#undef lhsp1
#undef lhsp2
#undef lhsp3
#undef lhsp4
#undef lhsm0
#undef lhsm1
#undef lhsm2
#undef lhsm3
#undef lhsm4
}

__global__ void k_x_solve4_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_lhs0,
   double *g_lhs1,
   double *g_lhs2,
   double *g_lhs3,
   double *g_lhs4,
   double *g_lhsp0,
   double *g_lhsp1,
   double *g_lhsp2,
   double *g_lhsp3,
   double *g_lhsp4,
   double *g_lhsm0,
   double *g_lhsm1,
   double *g_lhsm2,
   double *g_lhsm3,
   double *g_lhsm4,
  const int base_k, const int offset_k, const int gws_k,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_K)
{ 
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs0(a, b, c) g_lhs0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs1(a, b, c) g_lhs1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs2(a, b, c) g_lhs2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs3(a, b, c) g_lhs3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhs4(a, b, c) g_lhs4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp0(a, b, c) g_lhsp0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp1(a, b, c) g_lhsp1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp2(a, b, c) g_lhsp2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp3(a, b, c) g_lhsp3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsp4(a, b, c) g_lhsp4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm0(a, b, c) g_lhsm0[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm1(a, b, c) g_lhsm1[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm2(a, b, c) g_lhsm2[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm3(a, b, c) g_lhsm3[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]
#define lhsm4(a, b, c) g_lhsm4[((a) * WORK_NUM_ITEM_K + (b)) * (JMAXP+1) + (c)]

  int i, j, k, i1, i2, m;
  k = offset_k + blockDim.y * blockIdx.y + threadIdx.y;
  j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || j > ny2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION
  //---------------------------------------------------------------------
  i  = nx2;
  i1 = nx2+1;
  rhs0(i, k, j) = rhs0(i, k, j) - lhs3(i, k, j)*rhs0(i1, k, j);
  rhs1(i, k, j) = rhs1(i, k, j) - lhs3(i, k, j)*rhs1(i1, k, j);
  rhs2(i, k, j) = rhs2(i, k, j) - lhs3(i, k, j)*rhs2(i1, k, j);

  rhs3(i, k, j) = rhs3(i, k, j) - lhsp3(i, k, j)*rhs3(i1, k, j);
  rhs4(i, k, j) = rhs4(i, k, j) - lhsm3(i, k, j)*rhs4(i1, k, j);

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (i = nx2-1; i >= 0; i--) {
    i1 = i + 1;
    i2 = i + 2;
    rhs0(i, k, j) = rhs0(i, k, j) -
                      lhs3(i, k, j)*rhs0(i1, k, j) -
                      lhs4(i, k, j)*rhs0(i2, k, j);
    rhs1(i, k, j) = rhs1(i, k, j) -
                      lhs3(i, k, j)*rhs1(i1, k, j) -
                      lhs4(i, k, j)*rhs1(i2, k, j);
    rhs2(i, k, j) = rhs2(i, k, j) -
                      lhs3(i, k, j)*rhs2(i1, k, j) -
                      lhs4(i, k, j)*rhs2(i2, k, j);

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    rhs3(i, k, j) = rhs3(i, k, j) -
                      lhsp3(i, k, j)*rhs3(i1, k, j) -
                      lhsp4(i, k, j)*rhs3(i2, k, j);
    rhs4(i, k, j) = rhs4(i, k, j) -
                      lhsm3(i, k, j)*rhs4(i1, k, j) -
                      lhsm4(i, k, j)*rhs4(i2, k, j);
  }
#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
#undef lhs0
#undef lhs1
#undef lhs2
#undef lhs3
#undef lhs4
#undef lhsp0
#undef lhsp1
#undef lhsp2
#undef lhsp3
#undef lhsp4
#undef lhsm0
#undef lhsm1
#undef lhsm2
#undef lhsm3
#undef lhsm4
}

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__global__ void k_ninvr_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2)
{
   double (*rhs0)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_rhs0;
   double (*rhs1)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_rhs1;
   double (*rhs2)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_rhs2;
   double (*rhs3)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_rhs3;
   double (*rhs4)[JMAXP+1][IMAXP+1] = 
    ( double (*)[JMAXP+1][IMAXP+1])g_rhs4;

  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;
  k = offset_k + blockDim.z * blockIdx.z + threadIdx.z;
  j = 1 + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
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
