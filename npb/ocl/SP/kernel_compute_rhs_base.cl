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
// compute the reciprocal of density, and the kinetic energy, 
// and the speed of sound. 
//---------------------------------------------------------------------
__kernel void compute_rhs0(
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
  __global double *g_us,
  __global double *g_vs,
  __global double *g_ws,
  __global double *g_qs,
  __global double *g_square,
  __global double *g_speed,
  __global double *g_rho_i,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2)
{
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*speed)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_speed;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;

  int i, j, k;
  double aux, rho_inv;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2+1) return;

  for (j = 0; j <= ny2+1; j++) {
    for (i = 0; i <= nx2+1; i++) {
      rho_inv = 1.0/u0[k][j][i];
      rho_i[k][j][i] = rho_inv;
      us[k][j][i] = u1[k][j][i] * rho_inv;
      vs[k][j][i] = u2[k][j][i] * rho_inv;
      ws[k][j][i] = u3[k][j][i] * rho_inv;
      square[k][j][i] = 0.5* (
          u1[k][j][i]*u1[k][j][i] + 
          u2[k][j][i]*u2[k][j][i] +
          u3[k][j][i]*u3[k][j][i] ) * rho_inv;
      qs[k][j][i] = square[k][j][i] * rho_inv;
      //-------------------------------------------------------------------
      // (don't need speed and ainx until the lhs computation)
      //-------------------------------------------------------------------
      aux = c1c2*rho_inv* (u4[k][j][i] - square[k][j][i]);
      speed[k][j][i] = sqrt(aux);
    }
  }
}

//---------------------------------------------------------------------
// copy the exact forcing term to the right hand side;  because 
// this forcing term is known, we can store it on the whole grid
// including the boundary                   
//---------------------------------------------------------------------
__kernel void compute_rhs1(
  __global double *g_forcing0,
  __global double *g_forcing1,
  __global double *g_forcing2,
  __global double *g_forcing3,
  __global double *g_forcing4,
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  const int base_k, const int offset_k,
  const int nx2, const int ny2, const int nz2)
{
  __global double (*forcing0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_forcing0;
  __global double (*forcing1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_forcing1;
  __global double (*forcing2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_forcing2;
  __global double (*forcing3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_forcing3;
  __global double (*forcing4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_forcing4;
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

  int i, j, k, m;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2+1) return;

  for (j = 0; j <= ny2+1; j++) {
    for (i = 0; i <= nx2+1; i++) {
      rhs0[k][j][i] = forcing0[k][j][i];
      rhs1[k][j][i] = forcing1[k][j][i];
      rhs2[k][j][i] = forcing2[k][j][i];
      rhs3[k][j][i] = forcing3[k][j][i];
      rhs4[k][j][i] = forcing4[k][j][i];
    }
  }
}

//---------------------------------------------------------------------
// compute xi-direction fluxes 
//---------------------------------------------------------------------
__kernel void compute_rhs2(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
  __global double *g_us,
  __global double *g_vs,
  __global double *g_ws,
  __global double *g_qs,
  __global double *g_square,
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
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;

  int i, j, k, m;
  double uijk, up1, um1;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      uijk = us[k][j][i];
      up1  = us[k][j][i+1];
      um1  = us[k][j][i-1];

      rhs0[k][j][i] = rhs0[k][j][i] + dx1tx1 * 
        (u0[k][j][i+1] - 2.0*u0[k][j][i] + u0[k][j][i-1]) -
        tx2 * (u1[k][j][i+1] - u1[k][j][i-1]);

      rhs1[k][j][i] = rhs1[k][j][i] + dx2tx1 * 
        (u1[k][j][i+1] - 2.0*u1[k][j][i] + u1[k][j][i-1]) +
        xxcon2*con43 * (up1 - 2.0*uijk + um1) -
        tx2 * (u1[k][j][i+1]*up1 - u1[k][j][i-1]*um1 +
              (u4[k][j][i+1] - square[k][j][i+1] -
               u4[k][j][i-1] + square[k][j][i-1]) * c2);

      rhs2[k][j][i] = rhs2[k][j][i] + dx3tx1 * 
        (u2[k][j][i+1] - 2.0*u2[k][j][i] + u2[k][j][i-1]) +
        xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] + vs[k][j][i-1]) -
        tx2 * (u2[k][j][i+1]*up1 - u2[k][j][i-1]*um1);

      rhs3[k][j][i] = rhs3[k][j][i] + dx4tx1 * 
        (u3[k][j][i+1] - 2.0*u3[k][j][i] + u3[k][j][i-1]) +
        xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] + ws[k][j][i-1]) -
        tx2 * (u3[k][j][i+1]*up1 - u3[k][j][i-1]*um1);

      rhs4[k][j][i] = rhs4[k][j][i] + dx5tx1 * 
        (u4[k][j][i+1] - 2.0*u4[k][j][i] + u4[k][j][i-1]) +
        xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] + qs[k][j][i-1]) +
        xxcon4 * (up1*up1 -       2.0*uijk*uijk + um1*um1) +
        xxcon5 * (u4[k][j][i+1]*rho_i[k][j][i+1] - 
              2.0*u4[k][j][i]*rho_i[k][j][i] +
                  u4[k][j][i-1]*rho_i[k][j][i-1]) -
        tx2 * ( (c1*u4[k][j][i+1] - c2*square[k][j][i+1])*up1 -
                (c1*u4[k][j][i-1] - c2*square[k][j][i-1])*um1 );
    }
  }
}

__kernel void compute_rhs3(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
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
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  int i, j, k, m;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      //---------------------------------------------------------------------
      // add fourth order xi-direction dissipation               
      //---------------------------------------------------------------------
      if (i == 1) {
        rhs0[k][j][i] = rhs0[k][j][i]- dssp * 
          (5.0*u0[k][j][i] - 4.0*u0[k][j][i+1] + u0[k][j][i+2]);
        rhs1[k][j][i] = rhs1[k][j][i]- dssp * 
          (5.0*u1[k][j][i] - 4.0*u1[k][j][i+1] + u1[k][j][i+2]);
        rhs2[k][j][i] = rhs2[k][j][i]- dssp * 
          (5.0*u2[k][j][i] - 4.0*u2[k][j][i+1] + u2[k][j][i+2]);
        rhs3[k][j][i] = rhs3[k][j][i]- dssp * 
          (5.0*u3[k][j][i] - 4.0*u3[k][j][i+1] + u3[k][j][i+2]);
        rhs4[k][j][i] = rhs4[k][j][i]- dssp * 
          (5.0*u4[k][j][i] - 4.0*u4[k][j][i+1] + u4[k][j][i+2]);
      }
      else if (i == 2) {
          rhs0[k][j][i] = rhs0[k][j][i] - dssp * 
            (-4.0*u0[k][j][i-1] + 6.0*u0[k][j][i] -
              4.0*u0[k][j][i+1] + u0[k][j][i+2]);
          rhs1[k][j][i] = rhs1[k][j][i] - dssp * 
            (-4.0*u1[k][j][i-1] + 6.0*u1[k][j][i] -
              4.0*u1[k][j][i+1] + u1[k][j][i+2]);
          rhs2[k][j][i] = rhs2[k][j][i] - dssp * 
            (-4.0*u2[k][j][i-1] + 6.0*u2[k][j][i] -
              4.0*u2[k][j][i+1] + u2[k][j][i+2]);
          rhs3[k][j][i] = rhs3[k][j][i] - dssp * 
            (-4.0*u3[k][j][i-1] + 6.0*u3[k][j][i] -
              4.0*u3[k][j][i+1] + u3[k][j][i+2]);
          rhs4[k][j][i] = rhs4[k][j][i] - dssp * 
            (-4.0*u4[k][j][i-1] + 6.0*u4[k][j][i] -
              4.0*u4[k][j][i+1] + u4[k][j][i+2]);
      }
      else if (3 <= i && i <= nx2-2) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp * 
          ( u0[k][j][i-2] - 4.0*u0[k][j][i-1] + 
          6.0*u0[k][j][i] - 4.0*u0[k][j][i+1] + 
            u0[k][j][i+2] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp * 
          ( u1[k][j][i-2] - 4.0*u1[k][j][i-1] + 
          6.0*u1[k][j][i] - 4.0*u1[k][j][i+1] + 
            u1[k][j][i+2] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp * 
          ( u2[k][j][i-2] - 4.0*u2[k][j][i-1] + 
          6.0*u2[k][j][i] - 4.0*u2[k][j][i+1] + 
            u2[k][j][i+2] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp * 
          ( u3[k][j][i-2] - 4.0*u3[k][j][i-1] + 
          6.0*u3[k][j][i] - 4.0*u3[k][j][i+1] + 
            u3[k][j][i+2] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp * 
          ( u4[k][j][i-2] - 4.0*u4[k][j][i-1] + 
          6.0*u4[k][j][i] - 4.0*u4[k][j][i+1] + 
            u4[k][j][i+2] );
      }
      else if (i == nx2-1) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp *
          ( u0[k][j][i-2] - 4.0*u0[k][j][i-1] + 
          6.0*u0[k][j][i] - 4.0*u0[k][j][i+1] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp *
          ( u1[k][j][i-2] - 4.0*u1[k][j][i-1] + 
          6.0*u1[k][j][i] - 4.0*u1[k][j][i+1] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp *
          ( u2[k][j][i-2] - 4.0*u2[k][j][i-1] + 
          6.0*u2[k][j][i] - 4.0*u2[k][j][i+1] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp *
          ( u3[k][j][i-2] - 4.0*u3[k][j][i-1] + 
          6.0*u3[k][j][i] - 4.0*u3[k][j][i+1] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp *
          ( u4[k][j][i-2] - 4.0*u4[k][j][i-1] + 
          6.0*u4[k][j][i] - 4.0*u4[k][j][i+1] );
      }
      else {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp *
          ( u0[k][j][i-2] - 4.0*u0[k][j][i-1] + 5.0*u0[k][j][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp *
          ( u1[k][j][i-2] - 4.0*u1[k][j][i-1] + 5.0*u1[k][j][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp *
          ( u2[k][j][i-2] - 4.0*u2[k][j][i-1] + 5.0*u2[k][j][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp *
          ( u3[k][j][i-2] - 4.0*u3[k][j][i-1] + 5.0*u3[k][j][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp *
          ( u4[k][j][i-2] - 4.0*u4[k][j][i-1] + 5.0*u4[k][j][i] );
      }
    }
  }
}

//---------------------------------------------------------------------
// compute eta-direction fluxes 
//---------------------------------------------------------------------
__kernel void compute_rhs4(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
  __global double *g_us,
  __global double *g_vs,
  __global double *g_ws,
  __global double *g_qs,
  __global double *g_square,
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
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;

  int i, j, k, m;
  double vijk, vp1, vm1;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      vijk = vs[k][j][i];
      vp1  = vs[k][j+1][i];
      vm1  = vs[k][j-1][i];

      rhs0[k][j][i] = rhs0[k][j][i] + dy1ty1 * 
        (u0[k][j+1][i] - 2.0*u0[k][j][i] + u0[k][j-1][i]) -
        ty2 * (u2[k][j+1][i] - u2[k][j-1][i]);

      rhs1[k][j][i] = rhs1[k][j][i] + dy2ty1 * 
        (u1[k][j+1][i] - 2.0*u1[k][j][i] + u1[k][j-1][i]) +
        yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
        ty2 * (u1[k][j+1][i]*vp1 - u1[k][j-1][i]*vm1);

      rhs2[k][j][i] = rhs2[k][j][i] + dy3ty1 * 
        (u2[k][j+1][i] - 2.0*u2[k][j][i] + u2[k][j-1][i]) +
        yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
        ty2 * (u2[k][j+1][i]*vp1 - u2[k][j-1][i]*vm1 +
              (u4[k][j+1][i] - square[k][j+1][i] - 
               u4[k][j-1][i] + square[k][j-1][i]) * c2);

      rhs3[k][j][i] = rhs3[k][j][i] + dy4ty1 * 
        (u3[k][j+1][i] - 2.0*u3[k][j][i] + u3[k][j-1][i]) +
        yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
        ty2 * (u3[k][j+1][i]*vp1 - u3[k][j-1][i]*vm1);

      rhs4[k][j][i] = rhs4[k][j][i] + dy5ty1 * 
        (u4[k][j+1][i] - 2.0*u4[k][j][i] + u4[k][j-1][i]) +
        yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
        yycon4 * (vp1*vp1       - 2.0*vijk*vijk + vm1*vm1) +
        yycon5 * (u4[k][j+1][i]*rho_i[k][j+1][i] - 
                2.0*u4[k][j][i]*rho_i[k][j][i] +
                  u4[k][j-1][i]*rho_i[k][j-1][i]) -
        ty2 * ((c1*u4[k][j+1][i] - c2*square[k][j+1][i]) * vp1 -
               (c1*u4[k][j-1][i] - c2*square[k][j-1][i]) * vm1);
    }
  }
}

__kernel void compute_rhs5(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
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
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  int i, j, k, m;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      //---------------------------------------------------------------------
      // add fourth order eta-direction dissipation         
      //---------------------------------------------------------------------
      if (j == 1) {
        rhs0[k][j][i] = rhs0[k][j][i]- dssp * 
          ( 5.0*u0[k][j][i] - 4.0*u0[k][j+1][i] + u0[k][j+2][i]);
        rhs1[k][j][i] = rhs1[k][j][i]- dssp * 
          ( 5.0*u1[k][j][i] - 4.0*u1[k][j+1][i] + u1[k][j+2][i]);
        rhs2[k][j][i] = rhs2[k][j][i]- dssp * 
          ( 5.0*u2[k][j][i] - 4.0*u2[k][j+1][i] + u2[k][j+2][i]);
        rhs3[k][j][i] = rhs3[k][j][i]- dssp * 
          ( 5.0*u3[k][j][i] - 4.0*u3[k][j+1][i] + u3[k][j+2][i]);
        rhs4[k][j][i] = rhs4[k][j][i]- dssp * 
          ( 5.0*u4[k][j][i] - 4.0*u4[k][j+1][i] + u4[k][j+2][i]);
      }
      else if (j == 2) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp * 
          (-4.0*u0[k][j-1][i] + 6.0*u0[k][j][i] -
            4.0*u0[k][j+1][i] + u0[k][j+2][i]);
        rhs1[k][j][i] = rhs1[k][j][i] - dssp * 
          (-4.0*u1[k][j-1][i] + 6.0*u1[k][j][i] -
            4.0*u1[k][j+1][i] + u1[k][j+2][i]);
        rhs2[k][j][i] = rhs2[k][j][i] - dssp * 
          (-4.0*u2[k][j-1][i] + 6.0*u2[k][j][i] -
            4.0*u2[k][j+1][i] + u2[k][j+2][i]);
        rhs3[k][j][i] = rhs3[k][j][i] - dssp * 
          (-4.0*u3[k][j-1][i] + 6.0*u3[k][j][i] -
            4.0*u3[k][j+1][i] + u3[k][j+2][i]);
        rhs4[k][j][i] = rhs4[k][j][i] - dssp * 
          (-4.0*u4[k][j-1][i] + 6.0*u4[k][j][i] -
            4.0*u4[k][j+1][i] + u4[k][j+2][i]);
      }
      else if (3 <= j && j <= ny2-2) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp * 
          ( u0[k][j-2][i] - 4.0*u0[k][j-1][i] + 
          6.0*u0[k][j][i] - 4.0*u0[k][j+1][i] + 
            u0[k][j+2][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp * 
          ( u1[k][j-2][i] - 4.0*u1[k][j-1][i] + 
          6.0*u1[k][j][i] - 4.0*u1[k][j+1][i] + 
            u1[k][j+2][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp * 
          ( u2[k][j-2][i] - 4.0*u2[k][j-1][i] + 
          6.0*u2[k][j][i] - 4.0*u2[k][j+1][i] + 
            u2[k][j+2][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp * 
          ( u3[k][j-2][i] - 4.0*u3[k][j-1][i] + 
          6.0*u3[k][j][i] - 4.0*u3[k][j+1][i] + 
            u3[k][j+2][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp * 
          ( u4[k][j-2][i] - 4.0*u4[k][j-1][i] + 
          6.0*u4[k][j][i] - 4.0*u4[k][j+1][i] + 
            u4[k][j+2][i] );
      }
      else if (j == ny2-1) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp *
          ( u0[k][j-2][i] - 4.0*u0[k][j-1][i] + 
          6.0*u0[k][j][i] - 4.0*u0[k][j+1][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp *
          ( u1[k][j-2][i] - 4.0*u1[k][j-1][i] + 
          6.0*u1[k][j][i] - 4.0*u1[k][j+1][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp *
          ( u2[k][j-2][i] - 4.0*u2[k][j-1][i] + 
          6.0*u2[k][j][i] - 4.0*u2[k][j+1][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp *
          ( u3[k][j-2][i] - 4.0*u3[k][j-1][i] + 
          6.0*u3[k][j][i] - 4.0*u3[k][j+1][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp *
          ( u4[k][j-2][i] - 4.0*u4[k][j-1][i] + 
          6.0*u4[k][j][i] - 4.0*u4[k][j+1][i] );
      }
      else {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp *
          ( u0[k][j-2][i] - 4.0*u0[k][j-1][i] + 5.0*u0[k][j][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp *
          ( u1[k][j-2][i] - 4.0*u1[k][j-1][i] + 5.0*u1[k][j][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp *
          ( u2[k][j-2][i] - 4.0*u2[k][j-1][i] + 5.0*u2[k][j][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp *
          ( u3[k][j-2][i] - 4.0*u3[k][j-1][i] + 5.0*u3[k][j][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp *
          ( u4[k][j-2][i] - 4.0*u4[k][j-1][i] + 5.0*u4[k][j][i] );
      }
    }
  }
}

//---------------------------------------------------------------------
// compute zeta-direction fluxes 
//---------------------------------------------------------------------
__kernel void compute_rhs6(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
  __global double *g_us,
  __global double *g_vs,
  __global double *g_ws,
  __global double *g_qs,
  __global double *g_square,
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
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  __global double (*us)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_us;
  __global double (*vs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_vs;
  __global double (*ws)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_ws;
  __global double (*qs)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_qs;
  __global double (*square)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_square;
  __global double (*rho_i)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_rho_i;

  int i, j, k, m;
  double wijk, wp1, wm1;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      wijk = ws[k][j][i];
      wp1  = ws[k+1][j][i];
      wm1  = ws[k-1][j][i];

      rhs0[k][j][i] = rhs0[k][j][i] + dz1tz1 * 
        (u0[k+1][j][i] - 2.0*u0[k][j][i] + u0[k-1][j][i]) -
        tz2 * (u3[k+1][j][i] - u3[k-1][j][i]);

      rhs1[k][j][i] = rhs1[k][j][i] + dz2tz1 * 
        (u1[k+1][j][i] - 2.0*u1[k][j][i] + u1[k-1][j][i]) +
        zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + us[k-1][j][i]) -
        tz2 * (u1[k+1][j][i]*wp1 - u1[k-1][j][i]*wm1);

      rhs2[k][j][i] = rhs2[k][j][i] + dz3tz1 * 
        (u2[k+1][j][i] - 2.0*u2[k][j][i] + u2[k-1][j][i]) +
        zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + vs[k-1][j][i]) -
        tz2 * (u2[k+1][j][i]*wp1 - u2[k-1][j][i]*wm1);

      rhs3[k][j][i] = rhs3[k][j][i] + dz4tz1 * 
        (u3[k+1][j][i] - 2.0*u3[k][j][i] + u3[k-1][j][i]) +
        zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
        tz2 * (u3[k+1][j][i]*wp1 - u3[k-1][j][i]*wm1 +
              (u4[k+1][j][i] - square[k+1][j][i] - 
               u4[k-1][j][i] + square[k-1][j][i]) * c2);

      rhs4[k][j][i] = rhs4[k][j][i] + dz5tz1 * 
        (u4[k+1][j][i] - 2.0*u4[k][j][i] + u4[k-1][j][i]) +
        zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + qs[k-1][j][i]) +
        zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
        zzcon5 * (u4[k+1][j][i]*rho_i[k+1][j][i] - 
                2.0*u4[k][j][i]*rho_i[k][j][i] +
                  u4[k-1][j][i]*rho_i[k-1][j][i]) -
        tz2 * ((c1*u4[k+1][j][i] - c2*square[k+1][j][i])*wp1 -
               (c1*u4[k-1][j][i] - c2*square[k-1][j][i])*wm1);
    }
  }
}

__kernel void compute_rhs7(
  __global double *g_rhs0,
  __global double *g_rhs1,
  __global double *g_rhs2,
  __global double *g_rhs3,
  __global double *g_rhs4,
  __global double *g_u0,
  __global double *g_u1,
  __global double *g_u2,
  __global double *g_u3,
  __global double *g_u4,
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
  __global double (*u0)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u0;
  __global double (*u1)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u1;
  __global double (*u2)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u2;
  __global double (*u3)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u3;
  __global double (*u4)[JMAXP+1][IMAXP+1] = 
    (__global double (*)[JMAXP+1][IMAXP+1])g_u4;

  int i, j, k, m;
  double wijk, wp1, wm1;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      //---------------------------------------------------------------------
      // add fourth order zeta-direction dissipation                
      //---------------------------------------------------------------------
      if (base_k + k == 1) {
        rhs0[k][j][i] = rhs0[k][j][i]- dssp * 
          (5.0*u0[k][j][i] - 4.0*u0[k+1][j][i] + u0[k+2][j][i]);
        rhs1[k][j][i] = rhs1[k][j][i]- dssp * 
          (5.0*u1[k][j][i] - 4.0*u1[k+1][j][i] + u1[k+2][j][i]);
        rhs2[k][j][i] = rhs2[k][j][i]- dssp * 
          (5.0*u2[k][j][i] - 4.0*u2[k+1][j][i] + u2[k+2][j][i]);
        rhs3[k][j][i] = rhs3[k][j][i]- dssp * 
          (5.0*u3[k][j][i] - 4.0*u3[k+1][j][i] + u3[k+2][j][i]);
        rhs4[k][j][i] = rhs4[k][j][i]- dssp * 
          (5.0*u4[k][j][i] - 4.0*u4[k+1][j][i] + u4[k+2][j][i]);
      }
      else if (base_k + k == 2) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp * 
          (-4.0*u0[k-1][j][i] + 6.0*u0[k][j][i] -
            4.0*u0[k+1][j][i] + u0[k+2][j][i]);
        rhs1[k][j][i] = rhs1[k][j][i] - dssp * 
          (-4.0*u1[k-1][j][i] + 6.0*u1[k][j][i] -
            4.0*u1[k+1][j][i] + u1[k+2][j][i]);
        rhs2[k][j][i] = rhs2[k][j][i] - dssp * 
          (-4.0*u2[k-1][j][i] + 6.0*u2[k][j][i] -
            4.0*u2[k+1][j][i] + u2[k+2][j][i]);
        rhs3[k][j][i] = rhs3[k][j][i] - dssp * 
          (-4.0*u3[k-1][j][i] + 6.0*u3[k][j][i] -
            4.0*u3[k+1][j][i] + u3[k+2][j][i]);
        rhs4[k][j][i] = rhs4[k][j][i] - dssp * 
          (-4.0*u4[k-1][j][i] + 6.0*u4[k][j][i] -
            4.0*u4[k+1][j][i] + u4[k+2][j][i]);
      }
      else if (3 <= base_k + k && base_k + k <= nz2-2) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp * 
          ( u0[k-2][j][i] - 4.0*u0[k-1][j][i] + 
          6.0*u0[k][j][i] - 4.0*u0[k+1][j][i] + 
            u0[k+2][j][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp * 
          ( u1[k-2][j][i] - 4.0*u1[k-1][j][i] + 
          6.0*u1[k][j][i] - 4.0*u1[k+1][j][i] + 
            u1[k+2][j][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp * 
          ( u2[k-2][j][i] - 4.0*u2[k-1][j][i] + 
          6.0*u2[k][j][i] - 4.0*u2[k+1][j][i] + 
            u2[k+2][j][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp * 
          ( u3[k-2][j][i] - 4.0*u3[k-1][j][i] + 
          6.0*u3[k][j][i] - 4.0*u3[k+1][j][i] + 
            u3[k+2][j][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp * 
          ( u4[k-2][j][i] - 4.0*u4[k-1][j][i] + 
          6.0*u4[k][j][i] - 4.0*u4[k+1][j][i] + 
            u4[k+2][j][i] );
      }
      else if (base_k + k == nz2-1) {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp *
          ( u0[k-2][j][i] - 4.0*u0[k-1][j][i] + 
          6.0*u0[k][j][i] - 4.0*u0[k+1][j][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp *
          ( u1[k-2][j][i] - 4.0*u1[k-1][j][i] + 
          6.0*u1[k][j][i] - 4.0*u1[k+1][j][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp *
          ( u2[k-2][j][i] - 4.0*u2[k-1][j][i] + 
          6.0*u2[k][j][i] - 4.0*u2[k+1][j][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp *
          ( u3[k-2][j][i] - 4.0*u3[k-1][j][i] + 
          6.0*u3[k][j][i] - 4.0*u3[k+1][j][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp *
          ( u4[k-2][j][i] - 4.0*u4[k-1][j][i] + 
          6.0*u4[k][j][i] - 4.0*u4[k+1][j][i] );
      }
      else {
        rhs0[k][j][i] = rhs0[k][j][i] - dssp *
          ( u0[k-2][j][i] - 4.0*u0[k-1][j][i] + 5.0*u0[k][j][i] );
        rhs1[k][j][i] = rhs1[k][j][i] - dssp *
          ( u1[k-2][j][i] - 4.0*u1[k-1][j][i] + 5.0*u1[k][j][i] );
        rhs2[k][j][i] = rhs2[k][j][i] - dssp *
          ( u2[k-2][j][i] - 4.0*u2[k-1][j][i] + 5.0*u2[k][j][i] );
        rhs3[k][j][i] = rhs3[k][j][i] - dssp *
          ( u3[k-2][j][i] - 4.0*u3[k-1][j][i] + 5.0*u3[k][j][i] );
        rhs4[k][j][i] = rhs4[k][j][i] - dssp *
          ( u4[k-2][j][i] - 4.0*u4[k-1][j][i] + 5.0*u4[k][j][i] );
      }
    }
  }
}

__kernel void compute_rhs8(
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

  int i, j, k, m;
  k = offset_k + get_global_id(0);
  if (base_k + k > nz2) return;

  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      rhs0[k][j][i] = rhs0[k][j][i] * dt;
      rhs1[k][j][i] = rhs1[k][j][i] * dt;
      rhs2[k][j][i] = rhs2[k][j][i] * dt;
      rhs3[k][j][i] = rhs3[k][j][i] * dt;
      rhs4[k][j][i] = rhs4[k][j][i] * dt;
    }
  }
}
