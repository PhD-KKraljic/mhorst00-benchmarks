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

#include "header.h"
#include <stdio.h>

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
void z_solve()
{
  int i, j, k, k1, k2, m;
  double ru1, fac1, fac2;

  if (timeron) timer_start(t_zsolve);
  for (j = 1; j <= ny2; j++) {
    lhsinitj(nz2+1, nx2);

    //---------------------------------------------------------------------
    // Computes the left hand side for the three z-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue                          
    //---------------------------------------------------------------------
    for (i = 1; i <= nx2; i++) {
      for (k = 0; k <= nz2+1; k++) {
        ru1 = c3c4*rho_i[k][j][i];
        cv[k] = ws[k][j][i];
        rhos[k] = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
      }

      for (k = 1; k <= nz2; k++) {
        lhs[k][i][0] =  0.0;
        lhs[k][i][1] = -dttz2 * cv[k-1] - dttz1 * rhos[k-1];
        lhs[k][i][2] =  1.0 + c2dttz1 * rhos[k];
        lhs[k][i][3] =  dttz2 * cv[k+1] - dttz1 * rhos[k+1];
        lhs[k][i][4] =  0.0;
      }
    }

    //---------------------------------------------------------------------
    // add fourth order dissipation                                  
    //---------------------------------------------------------------------
    for (i = 1; i <= nx2; i++) {
      k = 1;
      lhs[k][i][2] = lhs[k][i][2] + comz5;
      lhs[k][i][3] = lhs[k][i][3] - comz4;
      lhs[k][i][4] = lhs[k][i][4] + comz1;

      k = 2;
      lhs[k][i][1] = lhs[k][i][1] - comz4;
      lhs[k][i][2] = lhs[k][i][2] + comz6;
      lhs[k][i][3] = lhs[k][i][3] - comz4;
      lhs[k][i][4] = lhs[k][i][4] + comz1;
    }

    for (k = 3; k <= nz2-2; k++) {
      for (i = 1; i <= nx2; i++) {
        lhs[k][i][0] = lhs[k][i][0] + comz1;
        lhs[k][i][1] = lhs[k][i][1] - comz4;
        lhs[k][i][2] = lhs[k][i][2] + comz6;
        lhs[k][i][3] = lhs[k][i][3] - comz4;
        lhs[k][i][4] = lhs[k][i][4] + comz1;
      }
    }

    for (i = 1; i <= nx2; i++) {
      k = nz2-1;
      lhs[k][i][0] = lhs[k][i][0] + comz1;
      lhs[k][i][1] = lhs[k][i][1] - comz4;
      lhs[k][i][2] = lhs[k][i][2] + comz6;
      lhs[k][i][3] = lhs[k][i][3] - comz4;

      k = nz2;
      lhs[k][i][0] = lhs[k][i][0] + comz1;
      lhs[k][i][1] = lhs[k][i][1] - comz4;
      lhs[k][i][2] = lhs[k][i][2] + comz5;
    }

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) 
    //---------------------------------------------------------------------
    for (k = 1; k <= nz2; k++) {
      for (i = 1; i <= nx2; i++) {
        lhsp[k][i][0] = lhs[k][i][0];
        lhsp[k][i][1] = lhs[k][i][1] - dttz2 * speed[k-1][j][i];
        lhsp[k][i][2] = lhs[k][i][2];
        lhsp[k][i][3] = lhs[k][i][3] + dttz2 * speed[k+1][j][i];
        lhsp[k][i][4] = lhs[k][i][4];
        lhsm[k][i][0] = lhs[k][i][0];
        lhsm[k][i][1] = lhs[k][i][1] + dttz2 * speed[k-1][j][i];
        lhsm[k][i][2] = lhs[k][i][2];
        lhsm[k][i][3] = lhs[k][i][3] - dttz2 * speed[k+1][j][i];
        lhsm[k][i][4] = lhs[k][i][4];
      }
    }


    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
    for (k = 0; k <= grid_points[2]-3; k++) {
      k1 = k + 1;
      k2 = k + 2;
      for (i = 1; i <= nx2; i++) {
        fac1 = 1.0/lhs[k][i][2];
        lhs[k][i][3] = fac1*lhs[k][i][3];
        lhs[k][i][4] = fac1*lhs[k][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
        }
        lhs[k1][i][2] = lhs[k1][i][2] - lhs[k1][i][1]*lhs[k][i][3];
        lhs[k1][i][3] = lhs[k1][i][3] - lhs[k1][i][1]*lhs[k][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhs[k1][i][1]*rhs[k][j][i][m];
        }
        lhs[k2][i][1] = lhs[k2][i][1] - lhs[k2][i][0]*lhs[k][i][3];
        lhs[k2][i][2] = lhs[k2][i][2] - lhs[k2][i][0]*lhs[k][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k2][j][i][m] = rhs[k2][j][i][m] - lhs[k2][i][0]*rhs[k][j][i][m];
        }
      }
    }

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    k  = grid_points[2]-2;
    k1 = grid_points[2]-1;
    for (i = 1; i <= nx2; i++) {
      fac1 = 1.0/lhs[k][i][2];
      lhs[k][i][3] = fac1*lhs[k][i][3];
      lhs[k][i][4] = fac1*lhs[k][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      lhs[k1][i][2] = lhs[k1][i][2] - lhs[k1][i][1]*lhs[k][i][3];
      lhs[k1][i][3] = lhs[k1][i][3] - lhs[k1][i][1]*lhs[k][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhs[k1][i][1]*rhs[k][j][i][m];
      }

      //---------------------------------------------------------------------
      // scale the last row immediately
      //---------------------------------------------------------------------
      fac2 = 1.0/lhs[k1][i][2];
      for (m = 0; m < 3; m++) {
        rhs[k1][j][i][m] = fac2*rhs[k1][j][i][m];
      }
    }

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors               
    //---------------------------------------------------------------------
    for (k = 0; k <= grid_points[2]-3; k++) {
      k1 = k + 1;
      k2 = k + 2;
      for (i = 1; i <= nx2; i++) {
        m = 3;
        fac1 = 1.0/lhsp[k][i][2];
        lhsp[k][i][3]    = fac1*lhsp[k][i][3];
        lhsp[k][i][4]    = fac1*lhsp[k][i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        lhsp[k1][i][2]   = lhsp[k1][i][2] - lhsp[k1][i][1]*lhsp[k][i][3];
        lhsp[k1][i][3]   = lhsp[k1][i][3] - lhsp[k1][i][1]*lhsp[k][i][4];
        rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsp[k1][i][1]*rhs[k][j][i][m];
        lhsp[k2][i][1]   = lhsp[k2][i][1] - lhsp[k2][i][0]*lhsp[k][i][3];
        lhsp[k2][i][2]   = lhsp[k2][i][2] - lhsp[k2][i][0]*lhsp[k][i][4];
        rhs[k2][j][i][m] = rhs[k2][j][i][m] - lhsp[k2][i][0]*rhs[k][j][i][m];

        m = 4;
        fac1 = 1.0/lhsm[k][i][2];
        lhsm[k][i][3]    = fac1*lhsm[k][i][3];
        lhsm[k][i][4]    = fac1*lhsm[k][i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        lhsm[k1][i][2]   = lhsm[k1][i][2] - lhsm[k1][i][1]*lhsm[k][i][3];
        lhsm[k1][i][3]   = lhsm[k1][i][3] - lhsm[k1][i][1]*lhsm[k][i][4];
        rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsm[k1][i][1]*rhs[k][j][i][m];
        lhsm[k2][i][1]   = lhsm[k2][i][1] - lhsm[k2][i][0]*lhsm[k][i][3];
        lhsm[k2][i][2]   = lhsm[k2][i][2] - lhsm[k2][i][0]*lhsm[k][i][4];
        rhs[k2][j][i][m] = rhs[k2][j][i][m] - lhsm[k2][i][0]*rhs[k][j][i][m];
      }
    }

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    k  = grid_points[2]-2;
    k1 = grid_points[2]-1;
    for (i = 1; i <= nx2; i++) {
      m = 3;
      fac1 = 1.0/lhsp[k][i][2];
      lhsp[k][i][3]    = fac1*lhsp[k][i][3];
      lhsp[k][i][4]    = fac1*lhsp[k][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsp[k1][i][2]   = lhsp[k1][i][2] - lhsp[k1][i][1]*lhsp[k][i][3];
      lhsp[k1][i][3]   = lhsp[k1][i][3] - lhsp[k1][i][1]*lhsp[k][i][4];
      rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsp[k1][i][1]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/lhsm[k][i][2];
      lhsm[k][i][3]    = fac1*lhsm[k][i][3];
      lhsm[k][i][4]    = fac1*lhsm[k][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsm[k1][i][2]   = lhsm[k1][i][2] - lhsm[k1][i][1]*lhsm[k][i][3];
      lhsm[k1][i][3]   = lhsm[k1][i][3] - lhsm[k1][i][1]*lhsm[k][i][4];
      rhs[k1][j][i][m] = rhs[k1][j][i][m] - lhsm[k1][i][1]*rhs[k][j][i][m];

      //---------------------------------------------------------------------
      // Scale the last row immediately (some of this is overkill
      // if this is the last cell)
      //---------------------------------------------------------------------
      rhs[k1][j][i][3] = rhs[k1][j][i][3]/lhsp[k1][i][2];
      rhs[k1][j][i][4] = rhs[k1][j][i][4]/lhsm[k1][i][2];
    }


    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    k  = grid_points[2]-2;
    k1 = grid_points[2]-1;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[k][i][3]*rhs[k1][j][i][m];
      }

      rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[k][i][3]*rhs[k1][j][i][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[k][i][3]*rhs[k1][j][i][4];
    }

    //---------------------------------------------------------------------
    // Whether or not this is the last processor, we always have
    // to complete the back-substitution 
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
    for (k = grid_points[2]-3; k >= 0; k--) {
      k1 = k + 1;
      k2 = k + 2;
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - 
                            lhs[k][i][3]*rhs[k1][j][i][m] -
                            lhs[k][i][4]*rhs[k2][j][i][m];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[k][j][i][3] = rhs[k][j][i][3] - 
                          lhsp[k][i][3]*rhs[k1][j][i][3] -
                          lhsp[k][i][4]*rhs[k2][j][i][3];
        rhs[k][j][i][4] = rhs[k][j][i][4] - 
                          lhsm[k][i][3]*rhs[k1][j][i][4] -
                          lhsm[k][i][4]*rhs[k2][j][i][4];
      }
    }
  }
  if (timeron) timer_stop(t_zsolve);

  tzetar();
}

void z_solve_gpu(int t, int st, int ed, int st2, int ed2)
{
    dim3 blockSize, threadSize;
    if (timeron) timer_start(t_zsolve);

    if (NUM_PARTITIONS > 1) {
        //---------------------------------------------------------------------
        // z_solve0 kernel
        //---------------------------------------------------------------------
        int z_solve0_base_j = st2;
        int z_solve0_offset_j = st - st2;
        int z_solve0_gws_j = ed - st + 1;

        if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
          int z_solve0_gws[] = {z_solve0_gws_j};
          int z_solve0_lws[] = {1};
          z_solve0_gws[0] = RoundWorkSize(z_solve0_gws[0], z_solve0_lws[0]);
          blockSize.x = z_solve0_gws[0]/z_solve0_lws[0];
          blockSize.y = 1;
          blockSize.z = 1;
          threadSize.x = z_solve0_lws[0];
          threadSize.y = 1;
          threadSize.z = 1;

          cuda_ProfilerStartEventRecord("k_z_solve0_base", cmd_queue[0]);
          k_z_solve0_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
              (
               buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, buf_us,
               buf_vs, buf_ws, buf_qs, buf_square, buf_speed, buf_rho_i,
               z_solve0_base_j, z_solve0_offset_j, nx2, ny2, nz2, WORK_NUM_ITEM_J
              );
          cuda_ProfilerEndEventRecord("k_z_solve0_base", cmd_queue[0]);
        }
        else {
          int z_solve0_gws[] = {nx2+2, nz2+2, z_solve0_gws_j};
          int z_solve0_lws[] = {16, 16, 1};
          z_solve0_gws[0] = RoundWorkSize(z_solve0_gws[0], z_solve0_lws[0]);
          z_solve0_gws[1] = RoundWorkSize(z_solve0_gws[1], z_solve0_lws[1]);
          z_solve0_gws[2] = RoundWorkSize(z_solve0_gws[2], z_solve0_lws[2]);
          blockSize.x = z_solve0_gws[0]/z_solve0_lws[0];
          blockSize.y = z_solve0_gws[1]/z_solve0_lws[1];
          blockSize.z = z_solve0_gws[2]/z_solve0_lws[2];
          threadSize.x = z_solve0_lws[0];
          threadSize.y = z_solve0_lws[1];
          threadSize.z = z_solve0_lws[2];

          cuda_ProfilerStartEventRecord("k_z_solve0_opt", cmd_queue[0]);
          k_z_solve0_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
              (
               buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, buf_us,
               buf_vs, buf_ws, buf_qs, buf_square, buf_speed, buf_rho_i,
               z_solve0_base_j, z_solve0_offset_j, nx2, ny2, nz2, WORK_NUM_ITEM_J
              );
          cuda_ProfilerEndEventRecord("k_z_solve0_opt", cmd_queue[0]);
        }

        CUCHK(cudaGetLastError());
        //---------------------------------------------------------------------
    }

    //---------------------------------------------------------------------
    // z_solve1 kernel
    //---------------------------------------------------------------------
    int z_solve1_base_j = st2;
    int z_solve1_offset_j = st - st2;
    int z_solve1_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
      int z_solve1_gws[] = {z_solve1_gws_j};
      int z_solve1_lws[] = {1};
      z_solve1_gws[0] = RoundWorkSize(z_solve1_gws[0], z_solve1_lws[0]);
      blockSize.x = z_solve1_gws[0]/z_solve1_lws[0];
      blockSize.y = 1;
      blockSize.z = 1;
      threadSize.x = z_solve1_lws[0];
      threadSize.y = 1;
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve1_base", cmd_queue[0]);
      k_z_solve1_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, buf_ws,
           buf_rho_i, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve1_base_j, z_solve1_offset_j, nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve1_base", cmd_queue[0]);
    }
    else {
      int z_solve1_gws[] = {nx2, nz2+2, z_solve1_gws_j};
      int z_solve1_lws[] = {16, 16, 1};
      z_solve1_gws[0] = RoundWorkSize(z_solve1_gws[0], z_solve1_lws[0]);
      z_solve1_gws[1] = RoundWorkSize(z_solve1_gws[1], z_solve1_lws[1]);
      z_solve1_gws[2] = RoundWorkSize(z_solve1_gws[2], z_solve1_lws[2]);
      blockSize.x = z_solve1_gws[0]/z_solve1_lws[0];
      blockSize.y = z_solve1_gws[1]/z_solve1_lws[1];
      blockSize.z = z_solve1_gws[2]/z_solve1_lws[2];
      threadSize.x = z_solve1_lws[0];
      threadSize.y = z_solve1_lws[1];
      threadSize.z = z_solve1_lws[2];

      cuda_ProfilerStartEventRecord("k_z_solve1_opt", cmd_queue[0]);
      k_z_solve1_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, buf_ws,
           buf_rho_i, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve1_base_j, z_solve1_offset_j, nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve1_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // z_solve2 kernel
    //---------------------------------------------------------------------
    int z_solve2_base_j = st2;
    int z_solve2_offset_j = st - st2;
    int z_solve2_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
      int z_solve2_gws[] = {z_solve2_gws_j};
      int z_solve2_lws[] = {1};
      z_solve2_gws[0] = RoundWorkSize(z_solve2_gws[0], z_solve2_lws[0]);
      blockSize.x = z_solve2_gws[0]/z_solve2_lws[0];
      blockSize.y = 1;
      blockSize.z = 1;
      threadSize.x = z_solve2_lws[0];
      threadSize.y = 1;
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve2_base", cmd_queue[0]);
      k_z_solve2_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_speed, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve2_base_j, z_solve2_offset_j, nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve2_base", cmd_queue[0]);
    }
    else {
      int z_solve2_gws[] = {nx2, nz2, z_solve2_gws_j};
      int z_solve2_lws[] = {16, 16, 1};
      z_solve2_gws[0] = RoundWorkSize(z_solve2_gws[0], z_solve2_lws[0]);
      z_solve2_gws[1] = RoundWorkSize(z_solve2_gws[1], z_solve2_lws[1]);
      z_solve2_gws[2] = RoundWorkSize(z_solve2_gws[2], z_solve2_lws[2]);
      blockSize.x = z_solve2_gws[0]/z_solve2_lws[0];
      blockSize.y = z_solve2_gws[1]/z_solve2_lws[1];
      blockSize.z = z_solve2_gws[2]/z_solve2_lws[2];
      threadSize.x = z_solve2_lws[0];
      threadSize.y = z_solve2_lws[1];
      threadSize.z = z_solve2_lws[2];

      cuda_ProfilerStartEventRecord("k_z_solve2_opt", cmd_queue[0]);
      k_z_solve2_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_speed, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve2_base_j, z_solve2_offset_j, nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve2_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // z_solve3 kernel
    //---------------------------------------------------------------------
    int z_solve3_base_j = st2;
    int z_solve3_offset_j = st - st2;
    int z_solve3_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
      int z_solve3_gws[] = {z_solve3_gws_j};
      int z_solve3_lws[] = {16};
      z_solve3_gws[0] = RoundWorkSize(z_solve3_gws[0], z_solve3_lws[0]);
      blockSize.x = z_solve3_gws[0]/z_solve3_lws[0];
      blockSize.y = 1;
      blockSize.z = 1;
      threadSize.x = z_solve3_lws[0];
      threadSize.y = 1;
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve3_base", cmd_queue[0]);
      k_z_solve3_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
           buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           z_solve3_base_j, z_solve3_offset_j, z_solve3_gws_j,
           nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve3_base", cmd_queue[0]);
    }
    else {
      int z_solve3_gws[] = {nx2, z_solve3_gws_j};
      int z_solve3_lws[] = {16, 16};
      z_solve3_gws[0] = RoundWorkSize(z_solve3_gws[0], z_solve3_lws[0]);
      z_solve3_gws[1] = RoundWorkSize(z_solve3_gws[1], z_solve3_lws[1]);
      blockSize.x = z_solve3_gws[0]/z_solve3_lws[0];
      blockSize.y = z_solve3_gws[1]/z_solve3_lws[1];
      blockSize.z = 1;
      threadSize.x = z_solve3_lws[0];
      threadSize.y = z_solve3_lws[1];
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve3_opt", cmd_queue[0]);
      k_z_solve3_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
           buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           z_solve3_base_j, z_solve3_offset_j, z_solve3_gws_j,
           nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve3_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // z_solve4 kernel
    //---------------------------------------------------------------------
    int z_solve4_base_j = st2;
    int z_solve4_offset_j = st - st2;
    int z_solve4_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
      int z_solve4_gws[] = {z_solve4_gws_j};
      int z_solve4_lws[] = {16};
      z_solve4_gws[0] = RoundWorkSize(z_solve4_gws[0], z_solve4_lws[0]);
      blockSize.x = z_solve4_gws[0]/z_solve4_lws[0];
      blockSize.y = 1;
      blockSize.z = 1;
      threadSize.x = z_solve4_lws[0];
      threadSize.y = 1;
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve4_base", cmd_queue[0]);
      k_z_solve4_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve4_base_j, z_solve4_offset_j, z_solve4_gws_j,
           nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve4_base", cmd_queue[0]);
    }
    else {
      int z_solve4_gws[] = {nx2, z_solve4_gws_j};
      int z_solve4_lws[] = {16, 16};
      z_solve4_gws[0] = RoundWorkSize(z_solve4_gws[0], z_solve4_lws[0]);
      z_solve4_gws[1] = RoundWorkSize(z_solve4_gws[1], z_solve4_lws[1]);
      blockSize.x = z_solve4_gws[0]/z_solve4_lws[0];
      blockSize.y = z_solve4_gws[1]/z_solve4_lws[1];
      blockSize.z = 1;
      threadSize.x = z_solve4_lws[0];
      threadSize.y = z_solve4_lws[1];
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve4_opt", cmd_queue[0]);
      k_z_solve4_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve4_base_j, z_solve4_offset_j, z_solve4_gws_j,
           nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve4_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // z_solve5 kernel
    //---------------------------------------------------------------------
    int z_solve5_base_j = st2;
    int z_solve5_offset_j = st - st2;
    int z_solve5_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
      int z_solve5_gws[] = {z_solve5_gws_j};
      int z_solve5_lws[] = {16};
      z_solve5_gws[0] = RoundWorkSize(z_solve5_gws[0], z_solve5_lws[0]);
      blockSize.x = z_solve5_gws[0]/z_solve5_lws[0];
      blockSize.y = 1;
      blockSize.z = 1;
      threadSize.x = z_solve5_lws[0];
      threadSize.y = 1;
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve5_base", cmd_queue[0]);
      k_z_solve5_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
           buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve5_base_j, z_solve5_offset_j, z_solve5_gws_j,
           nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve5_base", cmd_queue[0]);
    }
    else {
      int z_solve5_gws[] = {nx2, z_solve5_gws_j};
      int z_solve5_lws[] = {16, 16};
      z_solve5_gws[0] = RoundWorkSize(z_solve5_gws[0], z_solve5_lws[0]);
      z_solve5_gws[1] = RoundWorkSize(z_solve5_gws[1], z_solve5_lws[1]);
      blockSize.x = z_solve5_gws[0]/z_solve5_lws[0];
      blockSize.y = z_solve5_gws[1]/z_solve5_lws[1];
      blockSize.z = 1;
      threadSize.x = z_solve5_lws[0];
      threadSize.y = z_solve5_lws[1];
      threadSize.z = 1;

      cuda_ProfilerStartEventRecord("k_z_solve5_opt", cmd_queue[0]);
      k_z_solve5_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
          (
           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
           buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
           z_solve5_base_j, z_solve5_offset_j, z_solve5_gws_j,
           nx2, ny2, nz2, WORK_NUM_ITEM_J
          );
      cuda_ProfilerEndEventRecord("k_z_solve5_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    if (timeron) timer_stop(t_zsolve);

    tzetar_gpu(t, st, ed, st2, ed2);
}
