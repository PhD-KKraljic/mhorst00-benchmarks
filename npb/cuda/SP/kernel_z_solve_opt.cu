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
// compute the reciprocal of density, and the kinetic energy, 
// and the speed of sound. 
//---------------------------------------------------------------------
__global__ void k_z_solve0_opt(
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_square,
   double *g_speed,
   double *g_rho_i,
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define u0(a, b, c) g_u0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u1(a, b, c) g_u1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u2(a, b, c) g_u2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u3(a, b, c) g_u3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u4(a, b, c) g_u4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define us(a, b, c) g_us[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define vs(a, b, c) g_vs[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define ws(a, b, c) g_ws[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define qs(a, b, c) g_qs[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define square(a, b, c) g_square[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define speed(a, b, c) g_speed[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rho_i(a, b, c) g_rho_i[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

  int i, j, k;
  double aux, rho_inv;
  j = offset_j + blockDim.z * blockIdx.z + threadIdx.z;
  k = blockDim.y * blockIdx.y + threadIdx.y;
  i = blockDim.x * blockIdx.x + threadIdx.x;
  if (k > nz2+1 || base_j + j > ny2+1 || i > nx2+1) return;

  rho_inv = 1.0/u0(k, j, i);
  rho_i(k, j, i) = rho_inv;
  us(k, j, i) = u1(k, j, i) * rho_inv;
  vs(k, j, i) = u2(k, j, i) * rho_inv;
  ws(k, j, i) = u3(k, j, i) * rho_inv;
  square(k, j, i) = 0.5* (
      u1(k, j, i)*u1(k, j, i) + 
      u2(k, j, i)*u2(k, j, i) +
      u3(k, j, i)*u3(k, j, i) ) * rho_inv;
  qs(k, j, i) = square(k, j, i) * rho_inv;
  //-------------------------------------------------------------------
  // (don't need speed and ainx until the lhs computation)
  //-------------------------------------------------------------------
  aux = c1c2*rho_inv* (u4(k, j, i) - square(k, j, i));
  speed(k, j, i) = sqrt(aux);

#undef u0
#undef u1
#undef u2
#undef u3
#undef u4
#undef us
#undef vs
#undef ws
#undef qs
#undef square
#undef speed
#undef rho_i
}

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__global__ void k_z_solve1_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_ws,
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
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define ws(a, b, c) g_ws[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rho_i(a, b, c) g_rho_i[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

   double (*lhs0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs0;
   double (*lhs1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs1;
   double (*lhs2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs2;
   double (*lhs3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs3;
   double (*lhs4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs4;
   double (*lhsp0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp0;
   double (*lhsp1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp1;
   double (*lhsp2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp2;
   double (*lhsp3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp3;
   double (*lhsp4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp4;
   double (*lhsm0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm0;
   double (*lhsm1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm1;
   double (*lhsm2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm2;
   double (*lhsm3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm3;
   double (*lhsm4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm4;

  int i, j, k, k1, k2, m;
  double ru1;
  double rhos_km1, rhos_k, rhos_kp1;
  j = offset_j + blockDim.z * blockIdx.z + threadIdx.z;
  k = blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_j + j > ny2 || k > nz2+1 || i > nx2) return;

  if (k == 0 || k == nz2+1) {
    //---------------------------------------------------------------------
    // zap the whole left hand side for starters
    // set all diagonal values to 1. This is overkill, but convenient
    //---------------------------------------------------------------------
    lhs0 [j][k][i] = 0.0;
    lhsp0[j][k][i] = 0.0;
    lhsm0[j][k][i] = 0.0;
    lhs1 [j][k][i] = 0.0;
    lhsp1[j][k][i] = 0.0;
    lhsm1[j][k][i] = 0.0;
    lhs2 [j][k][i] = 0.0;
    lhsp2[j][k][i] = 0.0;
    lhsm2[j][k][i] = 0.0;
    lhs3 [j][k][i] = 0.0;
    lhsp3[j][k][i] = 0.0;
    lhsm3[j][k][i] = 0.0;
    lhs4 [j][k][i] = 0.0;
    lhsp4[j][k][i] = 0.0;
    lhsm4[j][k][i] = 0.0;

    lhs2 [j][k][i] = 1.0;
    lhsp2[j][k][i] = 1.0;
    lhsm2[j][k][i] = 1.0;
  }
  else {
    //---------------------------------------------------------------------
    // Computes the left hand side for the three z-factors
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue
    //---------------------------------------------------------------------
    ru1 = c3c4*rho_i(k-1, j, i);
    rhos_km1 = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));

    ru1 = c3c4*rho_i(k, j, i);
    rhos_k = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));

    ru1 = c3c4*rho_i(k+1, j, i);
    rhos_kp1 = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));

    lhs0[j][k][i] =  0.0;
    lhs1[j][k][i] = -dttz2 * ws(k-1, j, i) - dttz1 * rhos_km1;
    lhs2[j][k][i] =  1.0 + c2dttz1 * rhos_k;
    lhs3[j][k][i] =  dttz2 * ws(k+1, j, i) - dttz1 * rhos_kp1;
    lhs4[j][k][i] =  0.0;
  }
#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef ws
#undef rho_i
}

__global__ void k_z_solve2_opt(
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
  const int base_j, const int offset_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define speed(a, b, c) g_speed[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

   double (*lhs0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs0;
   double (*lhs1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs1;
   double (*lhs2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs2;
   double (*lhs3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs3;
   double (*lhs4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs4;
   double (*lhsp0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp0;
   double (*lhsp1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp1;
   double (*lhsp2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp2;
   double (*lhsp3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp3;
   double (*lhsp4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp4;
   double (*lhsm0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm0;
   double (*lhsm1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm1;
   double (*lhsm2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm2;
   double (*lhsm3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm3;
   double (*lhsm4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm4;

  int i, j, k, k1, k2;
  j = offset_j + blockDim.z * blockIdx.z + threadIdx.z;
  k = 1 + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_j + j > ny2 || k > nz2 || i > nx2) return;

  //---------------------------------------------------------------------
  // add fourth order dissipation
  //---------------------------------------------------------------------
  if (k == 1) {
    lhs2[j][k][i] = lhs2[j][k][i] + comz5;
    lhs3[j][k][i] = lhs3[j][k][i] - comz4;
    lhs4[j][k][i] = lhs4[j][k][i] + comz1;
  }
  else if (k == 2) {
    lhs1[j][k][i] = lhs1[j][k][i] - comz4;
    lhs2[j][k][i] = lhs2[j][k][i] + comz6;
    lhs3[j][k][i] = lhs3[j][k][i] - comz4;
    lhs4[j][k][i] = lhs4[j][k][i] + comz1;
  }
  else if (3 <= k && k <= nz2-2) {
    lhs0[j][k][i] = lhs0[j][k][i] + comz1;
    lhs1[j][k][i] = lhs1[j][k][i] - comz4;
    lhs2[j][k][i] = lhs2[j][k][i] + comz6;
    lhs3[j][k][i] = lhs3[j][k][i] - comz4;
    lhs4[j][k][i] = lhs4[j][k][i] + comz1;
  }
  else if (k == nz2-1) {
    lhs0[j][k][i] = lhs0[j][k][i] + comz1;
    lhs1[j][k][i] = lhs1[j][k][i] - comz4;
    lhs2[j][k][i] = lhs2[j][k][i] + comz6;
    lhs3[j][k][i] = lhs3[j][k][i] - comz4;
  }
  else {
    lhs0[j][k][i] = lhs0[j][k][i] + comz1;
    lhs1[j][k][i] = lhs1[j][k][i] - comz4;
    lhs2[j][k][i] = lhs2[j][k][i] + comz5;
  }

  //---------------------------------------------------------------------
  // subsequently, fill the other factors (u+c), (u-c)
  //---------------------------------------------------------------------
  lhsp0[j][k][i] = lhs0[j][k][i];
  lhsp1[j][k][i] = lhs1[j][k][i] - dttz2 * speed(k-1, j, i);
  lhsp2[j][k][i] = lhs2[j][k][i];
  lhsp3[j][k][i] = lhs3[j][k][i] + dttz2 * speed(k+1, j, i);
  lhsp4[j][k][i] = lhs4[j][k][i];
  lhsm0[j][k][i] = lhs0[j][k][i];
  lhsm1[j][k][i] = lhs1[j][k][i] + dttz2 * speed(k-1, j, i);
  lhsm2[j][k][i] = lhs2[j][k][i];
  lhsm3[j][k][i] = lhs3[j][k][i] - dttz2 * speed(k+1, j, i);
  lhsm4[j][k][i] = lhs4[j][k][i];

#undef speed
}

__global__ void k_z_solve3_opt(
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
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

   double (*lhs0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs0;
   double (*lhs1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs1;
   double (*lhs2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs2;
   double (*lhs3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs3;
   double (*lhs4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs4;

  int i, j, k, k1, k2, m;
  double fac1, fac2;
  j = offset_j + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_j + j > ny2 || i > nx2) return;
  if (j >= offset_j + gws_j) return;

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION
  //---------------------------------------------------------------------
  for (k = 0; k <= nz2-1; k++) {
    k1 = k + 1;
    k2 = k + 2;

    fac1 = 1.0/lhs2[j][k][i];
    lhs3[j][k][i] = fac1*lhs3[j][k][i];
    lhs4[j][k][i] = fac1*lhs4[j][k][i];

    rhs0(k, j, i) = fac1*rhs0(k, j, i);
    rhs1(k, j, i) = fac1*rhs1(k, j, i);
    rhs2(k, j, i) = fac1*rhs2(k, j, i);

    lhs2[j][k1][i] = lhs2[j][k1][i] - lhs1[j][k1][i]*lhs3[j][k][i];
    lhs3[j][k1][i] = lhs3[j][k1][i] - lhs1[j][k1][i]*lhs4[j][k][i];

    rhs0(k1, j, i) = rhs0(k1, j, i) - lhs1[j][k1][i]*rhs0(k, j, i);
    rhs1(k1, j, i) = rhs1(k1, j, i) - lhs1[j][k1][i]*rhs1(k, j, i);
    rhs2(k1, j, i) = rhs2(k1, j, i) - lhs1[j][k1][i]*rhs2(k, j, i);

    lhs1[j][k2][i] = lhs1[j][k2][i] - lhs0[j][k2][i]*lhs3[j][k][i];
    lhs2[j][k2][i] = lhs2[j][k2][i] - lhs0[j][k2][i]*lhs4[j][k][i];

    rhs0(k2, j, i) = rhs0(k2, j, i) - lhs0[j][k2][i]*rhs0(k, j, i);
    rhs1(k2, j, i) = rhs1(k2, j, i) - lhs0[j][k2][i]*rhs1(k, j, i);
    rhs2(k2, j, i) = rhs2(k2, j, i) - lhs0[j][k2][i]*rhs2(k, j, i);
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different,
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  k  = nz2;
  k1 = nz2+1;

  fac1 = 1.0/lhs2[j][k][i];
  lhs3[j][k][i] = fac1*lhs3[j][k][i];
  lhs4[j][k][i] = fac1*lhs4[j][k][i];

  rhs0(k, j, i) = fac1*rhs0(k, j, i);
  rhs1(k, j, i) = fac1*rhs1(k, j, i);
  rhs2(k, j, i) = fac1*rhs2(k, j, i);

  lhs2[j][k1][i] = lhs2[j][k1][i] - lhs1[j][k1][i]*lhs3[j][k][i];
  lhs3[j][k1][i] = lhs3[j][k1][i] - lhs1[j][k1][i]*lhs4[j][k][i];

  rhs0(k1, j, i) = rhs0(k1, j, i) - lhs1[j][k1][i]*rhs0(k, j, i);
  rhs1(k1, j, i) = rhs1(k1, j, i) - lhs1[j][k1][i]*rhs1(k, j, i);
  rhs2(k1, j, i) = rhs2(k1, j, i) - lhs1[j][k1][i]*rhs2(k, j, i);

  //---------------------------------------------------------------------
  // scale the last row immediately
  //---------------------------------------------------------------------
  fac2 = 1.0/lhs2[j][k1][i];
  rhs0(k1, j, i) = fac2*rhs0(k1, j, i);
  rhs1(k1, j, i) = fac2*rhs1(k1, j, i);
  rhs2(k1, j, i) = fac2*rhs2(k1, j, i);

#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
}

__global__ void k_z_solve4_opt(
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
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

   double (*lhsp0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp0;
   double (*lhsp1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp1;
   double (*lhsp2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp2;
   double (*lhsp3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp3;
   double (*lhsp4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp4;
   double (*lhsm0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm0;
   double (*lhsm1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm1;
   double (*lhsm2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm2;
   double (*lhsm3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm3;
   double (*lhsm4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm4;

  int i, j, k, k1, k2, m;
  double fac1;
  j = offset_j + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_j + j > ny2 || i > nx2) return;
  if (j >= offset_j + gws_j) return;

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors
  //---------------------------------------------------------------------
  for (k = 0; k <= nz2-1; k++) {
    k1 = k + 1;
    k2 = k + 2;

    m = 3;
    fac1 = 1.0/lhsp2[j][k][i];
    lhsp3[j][k][i]    = fac1*lhsp3[j][k][i];
    lhsp4[j][k][i]    = fac1*lhsp4[j][k][i];
    rhs3(k, j, i)  = fac1*rhs3(k, j, i);
    lhsp2[j][k1][i]   = lhsp2[j][k1][i] - lhsp1[j][k1][i]*lhsp3[j][k][i];
    lhsp3[j][k1][i]   = lhsp3[j][k1][i] - lhsp1[j][k1][i]*lhsp4[j][k][i];
    rhs3(k1, j, i) = rhs3(k1, j, i) - lhsp1[j][k1][i]*rhs3(k, j, i);
    lhsp1[j][k2][i]   = lhsp1[j][k2][i] - lhsp0[j][k2][i]*lhsp3[j][k][i];
    lhsp2[j][k2][i]   = lhsp2[j][k2][i] - lhsp0[j][k2][i]*lhsp4[j][k][i];
    rhs3(k2, j, i) = rhs3(k2, j, i) - lhsp0[j][k2][i]*rhs3(k, j, i);

    m = 4;
    fac1 = 1.0/lhsm2[j][k][i];
    lhsm3[j][k][i]    = fac1*lhsm3[j][k][i];
    lhsm4[j][k][i]    = fac1*lhsm4[j][k][i];
    rhs4(k, j, i)  = fac1*rhs4(k, j, i);
    lhsm2[j][k1][i]   = lhsm2[j][k1][i] - lhsm1[j][k1][i]*lhsm3[j][k][i];
    lhsm3[j][k1][i]   = lhsm3[j][k1][i] - lhsm1[j][k1][i]*lhsm4[j][k][i];
    rhs4(k1, j, i) = rhs4(k1, j, i) - lhsm1[j][k1][i]*rhs4(k, j, i);
    lhsm1[j][k2][i]   = lhsm1[j][k2][i] - lhsm0[j][k2][i]*lhsm3[j][k][i];
    lhsm2[j][k2][i]   = lhsm2[j][k2][i] - lhsm0[j][k2][i]*lhsm4[j][k][i];
    rhs4(k2, j, i) = rhs4(k2, j, i) - lhsm0[j][k2][i]*rhs4(k, j, i);
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  k  = nz2;
  k1 = nz2+1;

  m = 3;
  fac1 = 1.0/lhsp2[j][k][i];
  lhsp3[j][k][i]    = fac1*lhsp3[j][k][i];
  lhsp4[j][k][i]    = fac1*lhsp4[j][k][i];
  rhs3(k, j, i)  = fac1*rhs3(k, j, i);
  lhsp2[j][k1][i]   = lhsp2[j][k1][i] - lhsp1[j][k1][i]*lhsp3[j][k][i];
  lhsp3[j][k1][i]   = lhsp3[j][k1][i] - lhsp1[j][k1][i]*lhsp4[j][k][i];
  rhs3(k1, j, i) = rhs3(k1, j, i) - lhsp1[j][k1][i]*rhs3(k, j, i);

  m = 4;
  fac1 = 1.0/lhsm2[j][k][i];
  lhsm3[j][k][i]    = fac1*lhsm3[j][k][i];
  lhsm4[j][k][i]    = fac1*lhsm4[j][k][i];
  rhs4(k, j, i)  = fac1*rhs4(k, j, i);
  lhsm2[j][k1][i]   = lhsm2[j][k1][i] - lhsm1[j][k1][i]*lhsm3[j][k][i];
  lhsm3[j][k1][i]   = lhsm3[j][k1][i] - lhsm1[j][k1][i]*lhsm4[j][k][i];
  rhs4(k1, j, i) = rhs4(k1, j, i) - lhsm1[j][k1][i]*rhs4(k, j, i);

  //---------------------------------------------------------------------
  // Scale the last row immediately (some of this is overkill
  // if this is the last cell)
  //---------------------------------------------------------------------
  rhs3(k1, j, i) = rhs3(k1, j, i)/lhsp2[j][k1][i];
  rhs4(k1, j, i) = rhs4(k1, j, i)/lhsm2[j][k1][i];

#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
}

__global__ void k_z_solve5_opt(
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
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

   double (*lhs0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs0;
   double (*lhs1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs1;
   double (*lhs2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs2;
   double (*lhs3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs3;
   double (*lhs4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhs4;
   double (*lhsp0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp0;
   double (*lhsp1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp1;
   double (*lhsp2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp2;
   double (*lhsp3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp3;
   double (*lhsp4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsp4;
   double (*lhsm0)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm0;
   double (*lhsm1)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm1;
   double (*lhsm2)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm2;
   double (*lhsm3)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm3;
   double (*lhsm4)[KMAX][IMAXP+1] = ( double (*)[KMAX][IMAXP+1])g_lhsm4;

  int i, j, k, k1, k2, m;
  j = offset_j + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_j + j > ny2 || i > nx2) return;
  if (j >= offset_j + gws_j) return;

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION
  //---------------------------------------------------------------------
  k  = nz2;
  k1 = nz2+1;
  rhs0(k, j, i) = rhs0(k, j, i) - lhs3[j][k][i]*rhs0(k1, j, i);
  rhs1(k, j, i) = rhs1(k, j, i) - lhs3[j][k][i]*rhs1(k1, j, i);
  rhs2(k, j, i) = rhs2(k, j, i) - lhs3[j][k][i]*rhs2(k1, j, i);

  rhs3(k, j, i) = rhs3(k, j, i) - lhsp3[j][k][i]*rhs3(k1, j, i);
  rhs4(k, j, i) = rhs4(k, j, i) - lhsm3[j][k][i]*rhs4(k1, j, i);

  //---------------------------------------------------------------------
  // Whether or not this is the last processor, we always have
  // to complete the back-substitution
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (k = nz2-1; k >= 0; k--) {
    k1 = k + 1;
    k2 = k + 2;
    rhs0(k, j, i) = rhs0(k, j, i) -
                      lhs3[j][k][i]*rhs0(k1, j, i) -
                      lhs4[j][k][i]*rhs0(k2, j, i);
    rhs1(k, j, i) = rhs1(k, j, i) -
                      lhs3[j][k][i]*rhs1(k1, j, i) -
                      lhs4[j][k][i]*rhs1(k2, j, i);
    rhs2(k, j, i) = rhs2(k, j, i) -
                      lhs3[j][k][i]*rhs2(k1, j, i) -
                      lhs4[j][k][i]*rhs2(k2, j, i);

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    rhs3(k, j, i) = rhs3(k, j, i) -
                      lhsp3[j][k][i]*rhs3(k1, j, i) -
                      lhsp4[j][k][i]*rhs3(k2, j, i);
    rhs4(k, j, i) = rhs4(k, j, i) -
                      lhsm3[j][k][i]*rhs4(k1, j, i) -
                      lhsm4[j][k][i]*rhs4(k2, j, i);
  }
#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
}

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__global__ void k_tzetar_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_u0,
   double *g_u1,
   double *g_u2,
   double *g_u3,
   double *g_u4,
   double *g_us,
   double *g_vs,
   double *g_ws,
   double *g_qs,
   double *g_speed,
  const int base_j, const int offset_j, const int gws_j,
  const int nx2, const int ny2, const int nz2, const int WORK_NUM_ITEM_J)
{
#define rhs0(a, b, c) g_rhs0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs1(a, b, c) g_rhs1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs2(a, b, c) g_rhs2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs3(a, b, c) g_rhs3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define rhs4(a, b, c) g_rhs4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u0(a, b, c) g_u0[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u1(a, b, c) g_u1[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u2(a, b, c) g_u2[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u3(a, b, c) g_u3[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define u4(a, b, c) g_u4[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

#define us(a, b, c) g_us[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define vs(a, b, c) g_vs[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define ws(a, b, c) g_ws[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define qs(a, b, c) g_qs[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]
#define speed(a, b, c) g_speed[((a) * WORK_NUM_ITEM_J + (b)) * (IMAXP+1) + (c)]

  int i, j, k;
  double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
  double btuz, ac2u, uzik1;
  k = 1 + blockDim.z * blockIdx.z + threadIdx.z;
  j = offset_j + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (k > nz2 || base_j + j > ny2 || i > nx2) return;
  if (j >= offset_j + gws_j) return;

  xvel = us(k, j, i);
  yvel = vs(k, j, i);
  zvel = ws(k, j, i);
  ac   = speed(k, j, i);

  ac2u = ac*ac;

  r1 = rhs0(k, j, i);
  r2 = rhs1(k, j, i);
  r3 = rhs2(k, j, i);
  r4 = rhs3(k, j, i);
  r5 = rhs4(k, j, i);     

  uzik1 = u0(k, j, i);
  btuz  = bt * uzik1;

  t1 = btuz/ac * (r4 + r5);
  t2 = r3 + t1;
  t3 = btuz * (r4 - r5);

  rhs0(k, j, i) = t2;
  rhs1(k, j, i) = -uzik1*r2 + xvel*t2;
  rhs2(k, j, i) =  uzik1*r1 + yvel*t2;
  rhs3(k, j, i) =  zvel*t2  + t3;
  rhs4(k, j, i) =  uzik1*(-xvel*r2 + yvel*r1) + 
                     qs(k, j, i)*t2 + c2iv*ac2u*t1 + zvel*t3;

#undef rhs0
#undef rhs1
#undef rhs2
#undef rhs3
#undef rhs4
#undef u0
#undef u1
#undef u2
#undef u3
#undef u4
#undef us
#undef vs
#undef ws
#undef qs
#undef speed
}
