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
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
__global__ void k_y_solve0_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
   double *g_vs,
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
  const int nx2, const int ny2, const int nz2)
{
   double (*rhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs0;
   double (*rhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs1;
   double (*rhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs2;
   double (*rhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs3;
   double (*rhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs4;

   double (*vs)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_vs;
   double (*rho_i)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rho_i;

   double (*lhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs0;
   double (*lhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs1;
   double (*lhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs2;
   double (*lhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs3;
   double (*lhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs4;
   double (*lhsp0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
   double (*lhsp1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
   double (*lhsp2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
   double (*lhsp3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
   double (*lhsp4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
   double (*lhsm0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
   double (*lhsm1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
   double (*lhsm2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
   double (*lhsm3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
   double (*lhsm4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, j1, j2, m;
  double ru1;
  double rhoq_jm1, rhoq_j, rhoq_jp1;
  k = offset_k + blockDim.z * blockIdx.z + threadIdx.z;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || j > ny2+1 || i > nx2) return;

  if (j == 0 || j == ny2+1) {
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
    // Computes the left hand side for the three y-factors
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue
    //---------------------------------------------------------------------
    ru1 = c3c4*rho_i[k][j-1][i];
    rhoq_jm1 = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));

    ru1 = c3c4*rho_i[k][j][i];
    rhoq_j = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));

    ru1 = c3c4*rho_i[k][j+1][i];
    rhoq_jp1 = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));

    lhs0[k][j][i] =  0.0;
    lhs1[k][j][i] = -dtty2 * vs[k][j-1][i] - dtty1 * rhoq_jm1;
    lhs2[k][j][i] =  1.0 + c2dtty1 * rhoq_j;
    lhs3[k][j][i] =  dtty2 * vs[k][j+1][i] - dtty1 * rhoq_jp1;
    lhs4[k][j][i] =  0.0;
  }
}

__global__ void k_y_solve1_opt(
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
  const int nx2, const int ny2, const int nz2)
{
   double (*speed)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_speed;

   double (*lhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs0;
   double (*lhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs1;
   double (*lhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs2;
   double (*lhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs3;
   double (*lhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs4;
   double (*lhsp0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
   double (*lhsp1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
   double (*lhsp2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
   double (*lhsp3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
   double (*lhsp4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
   double (*lhsm0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
   double (*lhsm1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
   double (*lhsm2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
   double (*lhsm3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
   double (*lhsm4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, j1, j2;
  k = offset_k + blockDim.z * blockIdx.z + threadIdx.z;
  j = 1 + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || j > ny2 || i > nx2) return;

  //---------------------------------------------------------------------
  // add fourth order dissipation
  //---------------------------------------------------------------------
  if (j == 1) {
    lhs2[k][j][i] = lhs2[k][j][i] + comz5;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
    lhs4[k][j][i] = lhs4[k][j][i] + comz1;
  }
  else if (j == 2) {
    lhs1[k][j][i] = lhs1[k][j][i] - comz4;
    lhs2[k][j][i] = lhs2[k][j][i] + comz6;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
    lhs4[k][j][i] = lhs4[k][j][i] + comz1;
  }
  else if (3 <= j && j <= ny2-2) {
    lhs0[k][j][i] = lhs0[k][j][i] + comz1;
    lhs1[k][j][i] = lhs1[k][j][i] - comz4;
    lhs2[k][j][i] = lhs2[k][j][i] + comz6;
    lhs3[k][j][i] = lhs3[k][j][i] - comz4;
    lhs4[k][j][i] = lhs4[k][j][i] + comz1;
  }
  else if (j == ny2-1) {
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
  // subsequently, for (the other two factors
  //---------------------------------------------------------------------
  lhsp0[k][j][i] = lhs0[k][j][i];
  lhsp1[k][j][i] = lhs1[k][j][i] - dtty2 * speed[k][j-1][i];
  lhsp2[k][j][i] = lhs2[k][j][i];
  lhsp3[k][j][i] = lhs3[k][j][i] + dtty2 * speed[k][j+1][i];
  lhsp4[k][j][i] = lhs4[k][j][i];
  lhsm0[k][j][i] = lhs0[k][j][i];
  lhsm1[k][j][i] = lhs1[k][j][i] + dtty2 * speed[k][j-1][i];
  lhsm2[k][j][i] = lhs2[k][j][i];
  lhsm3[k][j][i] = lhs3[k][j][i] - dtty2 * speed[k][j+1][i];
  lhsm4[k][j][i] = lhs4[k][j][i];
}

__global__ void k_y_solve2_opt(
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
  const int nx2, const int ny2, const int nz2)
{
   double (*rhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs0;
   double (*rhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs1;
   double (*rhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs2;
   double (*rhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs3;
   double (*rhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs4;
   double (*lhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs0;
   double (*lhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs1;
   double (*lhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs2;
   double (*lhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs3;
   double (*lhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs4;

  int i, j, k, j1, j2, m;
  double fac1, fac2;
  k = offset_k + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || i > nx2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // FORWARD ELIMINATION
  //---------------------------------------------------------------------
  for (j = 0; j <= ny2-1; j++) {
    j1 = j + 1;
    j2 = j + 2;

    fac1 = 1.0/lhs2[k][j][i];
    lhs3[k][j][i] = fac1*lhs3[k][j][i];
    lhs4[k][j][i] = fac1*lhs4[k][j][i];

    rhs0[k][j][i] = fac1*rhs0[k][j][i];
    rhs1[k][j][i] = fac1*rhs1[k][j][i];
    rhs2[k][j][i] = fac1*rhs2[k][j][i];

    lhs2[k][j1][i] = lhs2[k][j1][i] - lhs1[k][j1][i]*lhs3[k][j][i];
    lhs3[k][j1][i] = lhs3[k][j1][i] - lhs1[k][j1][i]*lhs4[k][j][i];

    rhs0[k][j1][i] = rhs0[k][j1][i] - lhs1[k][j1][i]*rhs0[k][j][i];
    rhs1[k][j1][i] = rhs1[k][j1][i] - lhs1[k][j1][i]*rhs1[k][j][i];
    rhs2[k][j1][i] = rhs2[k][j1][i] - lhs1[k][j1][i]*rhs2[k][j][i];

    lhs1[k][j2][i] = lhs1[k][j2][i] - lhs0[k][j2][i]*lhs3[k][j][i];
    lhs2[k][j2][i] = lhs2[k][j2][i] - lhs0[k][j2][i]*lhs4[k][j][i];

    rhs0[k][j2][i] = rhs0[k][j2][i] - lhs0[k][j2][i]*rhs0[k][j][i];
    rhs1[k][j2][i] = rhs1[k][j2][i] - lhs0[k][j2][i]*rhs1[k][j][i];
    rhs2[k][j2][i] = rhs2[k][j2][i] - lhs0[k][j2][i]*rhs2[k][j][i];
  }

  //---------------------------------------------------------------------
  // The last two rows in this grid block are a bit different,
  // since they for (not have two more rows available for the
  // elimination of off-diagonal entries
  //---------------------------------------------------------------------
  j  = ny2;
  j1 = ny2+1;

  fac1 = 1.0/lhs2[k][j][i];
  lhs3[k][j][i] = fac1*lhs3[k][j][i];
  lhs4[k][j][i] = fac1*lhs4[k][j][i];

  rhs0[k][j][i] = fac1*rhs0[k][j][i];
  rhs1[k][j][i] = fac1*rhs1[k][j][i];
  rhs2[k][j][i] = fac1*rhs2[k][j][i];

  lhs2[k][j1][i] = lhs2[k][j1][i] - lhs1[k][j1][i]*lhs3[k][j][i];
  lhs3[k][j1][i] = lhs3[k][j1][i] - lhs1[k][j1][i]*lhs4[k][j][i];

  rhs0[k][j1][i] = rhs0[k][j1][i] - lhs1[k][j1][i]*rhs0[k][j][i];
  rhs1[k][j1][i] = rhs1[k][j1][i] - lhs1[k][j1][i]*rhs1[k][j][i];
  rhs2[k][j1][i] = rhs2[k][j1][i] - lhs1[k][j1][i]*rhs2[k][j][i];

  //---------------------------------------------------------------------
  // scale the last row immediately
  //---------------------------------------------------------------------
  fac2 = 1.0/lhs2[k][j1][i];
  rhs0[k][j1][i] = fac2*rhs0[k][j1][i];
  rhs1[k][j1][i] = fac2*rhs1[k][j1][i];
  rhs2[k][j1][i] = fac2*rhs2[k][j1][i];
}

__global__ void k_y_solve3_opt(
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
  const int nx2, const int ny2, const int nz2)
{
   double (*rhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs0;
   double (*rhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs1;
   double (*rhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs2;
   double (*rhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs3;
   double (*rhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs4;

   double (*lhsp0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
   double (*lhsp1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
   double (*lhsp2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
   double (*lhsp3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
   double (*lhsp4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
   double (*lhsm0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
   double (*lhsm1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
   double (*lhsm2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
   double (*lhsm3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
   double (*lhsm4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, j1, j2, m;
  double fac1;
  k = offset_k + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || i > nx2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // for (the u+c and the u-c factors
  //---------------------------------------------------------------------
  for (j = 0; j <= ny2-1; j++) {
    j1 = j + 1;
    j2 = j + 2;

    m = 3;
    fac1 = 1.0/lhsp2[k][j][i];
    lhsp3[k][j][i]    = fac1*lhsp3[k][j][i];
    lhsp4[k][j][i]    = fac1*lhsp4[k][j][i];
    rhs3[k][j][i]  = fac1*rhs3[k][j][i];
    lhsp2[k][j1][i]   = lhsp2[k][j1][i] - lhsp1[k][j1][i]*lhsp3[k][j][i];
    lhsp3[k][j1][i]   = lhsp3[k][j1][i] - lhsp1[k][j1][i]*lhsp4[k][j][i];
    rhs3[k][j1][i] = rhs3[k][j1][i] - lhsp1[k][j1][i]*rhs3[k][j][i];
    lhsp1[k][j2][i]   = lhsp1[k][j2][i] - lhsp0[k][j2][i]*lhsp3[k][j][i];
    lhsp2[k][j2][i]   = lhsp2[k][j2][i] - lhsp0[k][j2][i]*lhsp4[k][j][i];
    rhs3[k][j2][i] = rhs3[k][j2][i] - lhsp0[k][j2][i]*rhs3[k][j][i];

    m = 4;
    fac1 = 1.0/lhsm2[k][j][i];
    lhsm3[k][j][i]    = fac1*lhsm3[k][j][i];
    lhsm4[k][j][i]    = fac1*lhsm4[k][j][i];
    rhs4[k][j][i]  = fac1*rhs4[k][j][i];
    lhsm2[k][j1][i]   = lhsm2[k][j1][i] - lhsm1[k][j1][i]*lhsm3[k][j][i];
    lhsm3[k][j1][i]   = lhsm3[k][j1][i] - lhsm1[k][j1][i]*lhsm4[k][j][i];
    rhs4[k][j1][i] = rhs4[k][j1][i] - lhsm1[k][j1][i]*rhs4[k][j][i];
    lhsm1[k][j2][i]   = lhsm1[k][j2][i] - lhsm0[k][j2][i]*lhsm3[k][j][i];
    lhsm2[k][j2][i]   = lhsm2[k][j2][i] - lhsm0[k][j2][i]*lhsm4[k][j][i];
    rhs4[k][j2][i] = rhs4[k][j2][i] - lhsm0[k][j2][i]*rhs4[k][j][i];
  }

  //---------------------------------------------------------------------
  // And again the last two rows separately
  //---------------------------------------------------------------------
  j  = ny2;
  j1 = ny2+1;

  m = 3;
  fac1 = 1.0/lhsp2[k][j][i];
  lhsp3[k][j][i]    = fac1*lhsp3[k][j][i];
  lhsp4[k][j][i]    = fac1*lhsp4[k][j][i];
  rhs3[k][j][i]  = fac1*rhs3[k][j][i];
  lhsp2[k][j1][i]   = lhsp2[k][j1][i] - lhsp1[k][j1][i]*lhsp3[k][j][i];
  lhsp3[k][j1][i]   = lhsp3[k][j1][i] - lhsp1[k][j1][i]*lhsp4[k][j][i];
  rhs3[k][j1][i] = rhs3[k][j1][i] - lhsp1[k][j1][i]*rhs3[k][j][i];

  m = 4;
  fac1 = 1.0/lhsm2[k][j][i];
  lhsm3[k][j][i]    = fac1*lhsm3[k][j][i];
  lhsm4[k][j][i]    = fac1*lhsm4[k][j][i];
  rhs4[k][j][i]  = fac1*rhs4[k][j][i];
  lhsm2[k][j1][i]   = lhsm2[k][j1][i] - lhsm1[k][j1][i]*lhsm3[k][j][i];
  lhsm3[k][j1][i]   = lhsm3[k][j1][i] - lhsm1[k][j1][i]*lhsm4[k][j][i];
  rhs4[k][j1][i] = rhs4[k][j1][i] - lhsm1[k][j1][i]*rhs4[k][j][i];

  //---------------------------------------------------------------------
  // Scale the last row immediately
  //---------------------------------------------------------------------
  rhs3[k][j1][i]   = rhs3[k][j1][i]/lhsp2[k][j1][i];
  rhs4[k][j1][i]   = rhs4[k][j1][i]/lhsm2[k][j1][i];
}

__global__ void k_y_solve4_opt(
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
  const int nx2, const int ny2, const int nz2)
{
   double (*rhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs0;
   double (*rhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs1;
   double (*rhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs2;
   double (*rhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs3;
   double (*rhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs4;

   double (*lhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs0;
   double (*lhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs1;
   double (*lhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs2;
   double (*lhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs3;
   double (*lhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhs4;
   double (*lhsp0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp0;
   double (*lhsp1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp1;
   double (*lhsp2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp2;
   double (*lhsp3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp3;
   double (*lhsp4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsp4;
   double (*lhsm0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm0;
   double (*lhsm1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm1;
   double (*lhsm2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm2;
   double (*lhsm3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm3;
   double (*lhsm4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_lhsm4;

  int i, j, k, j1, j2, m;
  k = offset_k + blockDim.y * blockIdx.y + threadIdx.y;
  i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (base_k + k > nz2 || i > nx2) return;
  if (k >= offset_k + gws_k) return;

  //---------------------------------------------------------------------
  // BACKSUBSTITUTION
  //---------------------------------------------------------------------
  j  = ny2;
  j1 = ny2+1;
  rhs0[k][j][i] = rhs0[k][j][i] - lhs3[k][j][i]*rhs0[k][j1][i];
  rhs1[k][j][i] = rhs1[k][j][i] - lhs3[k][j][i]*rhs1[k][j1][i];
  rhs2[k][j][i] = rhs2[k][j][i] - lhs3[k][j][i]*rhs2[k][j1][i];

  rhs3[k][j][i] = rhs3[k][j][i] - lhsp3[k][j][i]*rhs3[k][j1][i];
  rhs4[k][j][i] = rhs4[k][j][i] - lhsm3[k][j][i]*rhs4[k][j1][i];

  //---------------------------------------------------------------------
  // The first three factors
  //---------------------------------------------------------------------
  for (j = ny2-1; j >= 0; j--) {
    j1 = j + 1;
    j2 = j + 2;

    rhs0[k][j][i] = rhs0[k][j][i] -
                      lhs3[k][j][i]*rhs0[k][j1][i] -
                      lhs4[k][j][i]*rhs0[k][j2][i];
    rhs1[k][j][i] = rhs1[k][j][i] -
                      lhs3[k][j][i]*rhs1[k][j1][i] -
                      lhs4[k][j][i]*rhs1[k][j2][i];
    rhs2[k][j][i] = rhs2[k][j][i] -
                      lhs3[k][j][i]*rhs2[k][j1][i] -
                      lhs4[k][j][i]*rhs2[k][j2][i];

    //-------------------------------------------------------------------
    // And the remaining two
    //-------------------------------------------------------------------
    rhs3[k][j][i] = rhs3[k][j][i] -
                      lhsp3[k][j][i]*rhs3[k][j1][i] -
                      lhsp4[k][j][i]*rhs3[k][j2][i];
    rhs4[k][j][i] = rhs4[k][j][i] -
                      lhsm3[k][j][i]*rhs4[k][j1][i] -
                      lhsm4[k][j][i]*rhs4[k][j2][i];
  }
}


//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
__global__ void k_pinvr_opt(
   double *g_rhs0,
   double *g_rhs1,
   double *g_rhs2,
   double *g_rhs3,
   double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2)
{
   double (*rhs0)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs0;
   double (*rhs1)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs1;
   double (*rhs2)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs2;
   double (*rhs3)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs3;
   double (*rhs4)[JMAXP+1][IMAXP+1] = ( double (*)[JMAXP+1][IMAXP+1])g_rhs4;

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

  t1 = bt * r1;
  t2 = 0.5 * ( r4 + r5 );

  rhs0[k][j][i] =  bt * ( r4 - r5 );
  rhs1[k][j][i] = -r3;
  rhs2[k][j][i] =  r2;
  rhs3[k][j][i] = -t1 + t2;
  rhs4[k][j][i] =  t1 + t2;
}
