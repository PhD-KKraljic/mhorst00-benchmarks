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
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
void y_solve()
{
  int i, j, k, j1, j2, m;
  double ru1, fac1, fac2;

  if (timeron) timer_start(t_ysolve);
  for (k = 1; k <= grid_points[2]-2; k++) {
    lhsinitj(ny2+1, nx2);

    //---------------------------------------------------------------------
    // Computes the left hand side for the three y-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue         
    //---------------------------------------------------------------------
    for (i = 1; i <= grid_points[0]-2; i++) {
      for (j = 0; j <= grid_points[1]-1; j++) {
        ru1 = c3c4*rho_i[k][j][i];
        cv[j] = vs[k][j][i];
        rhoq[j] = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
      }

      for (j = 1; j <= grid_points[1]-2; j++) {
        lhs[j][i][0] =  0.0;
        lhs[j][i][1] = -dtty2 * cv[j-1] - dtty1 * rhoq[j-1];
        lhs[j][i][2] =  1.0 + c2dtty1 * rhoq[j];
        lhs[j][i][3] =  dtty2 * cv[j+1] - dtty1 * rhoq[j+1];
        lhs[j][i][4] =  0.0;
      }
    }

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
    for (i = 1; i <= grid_points[0]-2; i++) {
      j = 1;
      lhs[j][i][2] = lhs[j][i][2] + comz5;
      lhs[j][i][3] = lhs[j][i][3] - comz4;
      lhs[j][i][4] = lhs[j][i][4] + comz1;

      lhs[j+1][i][1] = lhs[j+1][i][1] - comz4;
      lhs[j+1][i][2] = lhs[j+1][i][2] + comz6;
      lhs[j+1][i][3] = lhs[j+1][i][3] - comz4;
      lhs[j+1][i][4] = lhs[j+1][i][4] + comz1;
    }

    for (j = 3; j <= grid_points[1]-4; j++) {
      for (i = 1; i <= grid_points[0]-2; i++) {
        lhs[j][i][0] = lhs[j][i][0] + comz1;
        lhs[j][i][1] = lhs[j][i][1] - comz4;
        lhs[j][i][2] = lhs[j][i][2] + comz6;
        lhs[j][i][3] = lhs[j][i][3] - comz4;
        lhs[j][i][4] = lhs[j][i][4] + comz1;
      }
    }

    for (i = 1; i <= grid_points[0]-2; i++) {
      j = grid_points[1]-3;
      lhs[j][i][0] = lhs[j][i][0] + comz1;
      lhs[j][i][1] = lhs[j][i][1] - comz4;
      lhs[j][i][2] = lhs[j][i][2] + comz6;
      lhs[j][i][3] = lhs[j][i][3] - comz4;

      lhs[j+1][i][0] = lhs[j+1][i][0] + comz1;
      lhs[j+1][i][1] = lhs[j+1][i][1] - comz4;
      lhs[j+1][i][2] = lhs[j+1][i][2] + comz5;
    }

    //---------------------------------------------------------------------
    // subsequently, for (the other two factors                    
    //---------------------------------------------------------------------
    for (j = 1; j <= grid_points[1]-2; j++) {
      for (i = 1; i <= grid_points[0]-2; i++) {
        lhsp[j][i][0] = lhs[j][i][0];
        lhsp[j][i][1] = lhs[j][i][1] - dtty2 * speed[k][j-1][i];
        lhsp[j][i][2] = lhs[j][i][2];
        lhsp[j][i][3] = lhs[j][i][3] + dtty2 * speed[k][j+1][i];
        lhsp[j][i][4] = lhs[j][i][4];
        lhsm[j][i][0] = lhs[j][i][0];
        lhsm[j][i][1] = lhs[j][i][1] + dtty2 * speed[k][j-1][i];
        lhsm[j][i][2] = lhs[j][i][2];
        lhsm[j][i][3] = lhs[j][i][3] - dtty2 * speed[k][j+1][i];
        lhsm[j][i][4] = lhs[j][i][4];
      }
    }

    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
    for (j = 0; j <= grid_points[1]-3; j++) {
      j1 = j + 1;
      j2 = j + 2;
      for (i = 1; i <= grid_points[0]-2; i++) {
        fac1 = 1.0/lhs[j][i][2];
        lhs[j][i][3] = fac1*lhs[j][i][3];
        lhs[j][i][4] = fac1*lhs[j][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
        }
        lhs[j1][i][2] = lhs[j1][i][2] - lhs[j1][i][1]*lhs[j][i][3];
        lhs[j1][i][3] = lhs[j1][i][3] - lhs[j1][i][1]*lhs[j][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhs[j1][i][1]*rhs[k][j][i][m];
        }
        lhs[j2][i][1] = lhs[j2][i][1] - lhs[j2][i][0]*lhs[j][i][3];
        lhs[j2][i][2] = lhs[j2][i][2] - lhs[j2][i][0]*lhs[j][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j2][i][m] = rhs[k][j2][i][m] - lhs[j2][i][0]*rhs[k][j][i][m];
        }
      }
    }

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    j  = grid_points[1]-2;
    j1 = grid_points[1]-1;
    for (i = 1; i <= grid_points[0]-2; i++) {
      fac1 = 1.0/lhs[j][i][2];
      lhs[j][i][3] = fac1*lhs[j][i][3];
      lhs[j][i][4] = fac1*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      lhs[j1][i][2] = lhs[j1][i][2] - lhs[j1][i][1]*lhs[j][i][3];
      lhs[j1][i][3] = lhs[j1][i][3] - lhs[j1][i][1]*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhs[j1][i][1]*rhs[k][j][i][m];
      }
      //---------------------------------------------------------------------
      // scale the last row immediately 
      //---------------------------------------------------------------------
      fac2 = 1.0/lhs[j1][i][2];
      for (m = 0; m < 3; m++) {
        rhs[k][j1][i][m] = fac2*rhs[k][j1][i][m];
      }
    }

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
    for (j = 0; j <= grid_points[1]-3; j++) {
      j1 = j + 1;
      j2 = j + 2;
      for (i = 1; i <= grid_points[0]-2; i++) {
        m = 3;
        fac1 = 1.0/lhsp[j][i][2];
        lhsp[j][i][3]    = fac1*lhsp[j][i][3];
        lhsp[j][i][4]    = fac1*lhsp[j][i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        lhsp[j1][i][2]   = lhsp[j1][i][2] - lhsp[j1][i][1]*lhsp[j][i][3];
        lhsp[j1][i][3]   = lhsp[j1][i][3] - lhsp[j1][i][1]*lhsp[j][i][4];
        rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsp[j1][i][1]*rhs[k][j][i][m];
        lhsp[j2][i][1]   = lhsp[j2][i][1] - lhsp[j2][i][0]*lhsp[j][i][3];
        lhsp[j2][i][2]   = lhsp[j2][i][2] - lhsp[j2][i][0]*lhsp[j][i][4];
        rhs[k][j2][i][m] = rhs[k][j2][i][m] - lhsp[j2][i][0]*rhs[k][j][i][m];

        m = 4;
        fac1 = 1.0/lhsm[j][i][2];
        lhsm[j][i][3]    = fac1*lhsm[j][i][3];
        lhsm[j][i][4]    = fac1*lhsm[j][i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        lhsm[j1][i][2]   = lhsm[j1][i][2] - lhsm[j1][i][1]*lhsm[j][i][3];
        lhsm[j1][i][3]   = lhsm[j1][i][3] - lhsm[j1][i][1]*lhsm[j][i][4];
        rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsm[j1][i][1]*rhs[k][j][i][m];
        lhsm[j2][i][1]   = lhsm[j2][i][1] - lhsm[j2][i][0]*lhsm[j][i][3];
        lhsm[j2][i][2]   = lhsm[j2][i][2] - lhsm[j2][i][0]*lhsm[j][i][4];
        rhs[k][j2][i][m] = rhs[k][j2][i][m] - lhsm[j2][i][0]*rhs[k][j][i][m];
      }
    }

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    j  = grid_points[1]-2;
    j1 = grid_points[1]-1;
    for (i = 1; i <= grid_points[0]-2; i++) {
      m = 3;
      fac1 = 1.0/lhsp[j][i][2];
      lhsp[j][i][3]    = fac1*lhsp[j][i][3];
      lhsp[j][i][4]    = fac1*lhsp[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsp[j1][i][2]   = lhsp[j1][i][2] - lhsp[j1][i][1]*lhsp[j][i][3];
      lhsp[j1][i][3]   = lhsp[j1][i][3] - lhsp[j1][i][1]*lhsp[j][i][4];
      rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsp[j1][i][1]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/lhsm[j][i][2];
      lhsm[j][i][3]    = fac1*lhsm[j][i][3];
      lhsm[j][i][4]    = fac1*lhsm[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsm[j1][i][2]   = lhsm[j1][i][2] - lhsm[j1][i][1]*lhsm[j][i][3];
      lhsm[j1][i][3]   = lhsm[j1][i][3] - lhsm[j1][i][1]*lhsm[j][i][4];
      rhs[k][j1][i][m] = rhs[k][j1][i][m] - lhsm[j1][i][1]*rhs[k][j][i][m];

      //---------------------------------------------------------------------
      // Scale the last row immediately 
      //---------------------------------------------------------------------
      rhs[k][j1][i][3]   = rhs[k][j1][i][3]/lhsp[j1][i][2];
      rhs[k][j1][i][4]   = rhs[k][j1][i][4]/lhsm[j1][i][2];
    }


    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    j  = grid_points[1]-2;
    j1 = grid_points[1]-1;
    for (i = 1; i <= grid_points[0]-2; i++) {
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][i][3]*rhs[k][j1][i][m];
      }

      rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[j][i][3]*rhs[k][j1][i][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[j][i][3]*rhs[k][j1][i][4];
    }

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
    for (j = grid_points[1]-3; j >= 0; j--) {
      j1 = j + 1;
      j2 = j + 2;
      for (i = 1; i <= grid_points[0]-2; i++) {
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - 
                            lhs[j][i][3]*rhs[k][j1][i][m] -
                            lhs[j][i][4]*rhs[k][j2][i][m];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[k][j][i][3] = rhs[k][j][i][3] - 
                          lhsp[j][i][3]*rhs[k][j1][i][3] -
                          lhsp[j][i][4]*rhs[k][j2][i][3];
        rhs[k][j][i][4] = rhs[k][j][i][4] - 
                          lhsm[j][i][3]*rhs[k][j1][i][4] -
                          lhsm[j][i][4]*rhs[k][j2][i][4];
      }
    }
  }
  if (timeron) timer_stop(t_ysolve);

  pinvr();
}

void y_solve_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_ysolve);

  //---------------------------------------------------------------------
  // y_solve0 kernel
  //---------------------------------------------------------------------
  int y_solve0_base_k = st2;
  int y_solve0_offset_k = st - st2;
  int y_solve0_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_y_solve[0], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_y_solve[0], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_y_solve[0], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_y_solve[0], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_y_solve[0], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_y_solve[0], 5, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_y_solve[0], 6, sizeof(cl_mem), &buf_rho_i);
  err_code |= clSetKernelArg(k_y_solve[0], 7, sizeof(cl_mem), &buf_lhs0);
  err_code |= clSetKernelArg(k_y_solve[0], 8, sizeof(cl_mem), &buf_lhs1);
  err_code |= clSetKernelArg(k_y_solve[0], 9, sizeof(cl_mem), &buf_lhs2);
  err_code |= clSetKernelArg(k_y_solve[0], 10, sizeof(cl_mem), &buf_lhs3);
  err_code |= clSetKernelArg(k_y_solve[0], 11, sizeof(cl_mem), &buf_lhs4);
  err_code |= clSetKernelArg(k_y_solve[0], 12, sizeof(cl_mem), &buf_lhsp0);
  err_code |= clSetKernelArg(k_y_solve[0], 13, sizeof(cl_mem), &buf_lhsp1);
  err_code |= clSetKernelArg(k_y_solve[0], 14, sizeof(cl_mem), &buf_lhsp2);
  err_code |= clSetKernelArg(k_y_solve[0], 15, sizeof(cl_mem), &buf_lhsp3);
  err_code |= clSetKernelArg(k_y_solve[0], 16, sizeof(cl_mem), &buf_lhsp4);
  err_code |= clSetKernelArg(k_y_solve[0], 17, sizeof(cl_mem), &buf_lhsm0);
  err_code |= clSetKernelArg(k_y_solve[0], 18, sizeof(cl_mem), &buf_lhsm1);
  err_code |= clSetKernelArg(k_y_solve[0], 19, sizeof(cl_mem), &buf_lhsm2);
  err_code |= clSetKernelArg(k_y_solve[0], 20, sizeof(cl_mem), &buf_lhsm3);
  err_code |= clSetKernelArg(k_y_solve[0], 21, sizeof(cl_mem), &buf_lhsm4);
  err_code |= clSetKernelArg(k_y_solve[0], 22, sizeof(int), &y_solve0_base_k);
  err_code |= clSetKernelArg(k_y_solve[0], 23, sizeof(int), &y_solve0_offset_k);
  err_code |= clSetKernelArg(k_y_solve[0], 24, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_y_solve[0], 25, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_y_solve[0], 26, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t y_solve0_gws[] = {y_solve0_gws_k};
    size_t y_solve0_lws[] = {1};
    y_solve0_gws[0] = clu_RoundWorkSize(y_solve0_gws[0], y_solve0_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[0],
                                      1, NULL,
                                      y_solve0_gws,
                                      y_solve0_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t y_solve0_gws[] = {nx2, ny2+2, y_solve0_gws_k};
    size_t y_solve0_lws[] = {16, 16, 1};
    y_solve0_gws[0] = clu_RoundWorkSize(y_solve0_gws[0], y_solve0_lws[0]);
    y_solve0_gws[1] = clu_RoundWorkSize(y_solve0_gws[1], y_solve0_lws[1]);
    y_solve0_gws[2] = clu_RoundWorkSize(y_solve0_gws[2], y_solve0_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[0],
                                      3, NULL,
                                      y_solve0_gws,
                                      y_solve0_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // y_solve1 kernel
  //---------------------------------------------------------------------
  int y_solve1_base_k = st2;
  int y_solve1_offset_k = st - st2;
  int y_solve1_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_y_solve[1], 0, sizeof(cl_mem), &buf_speed);
  err_code |= clSetKernelArg(k_y_solve[1], 1, sizeof(cl_mem), &buf_lhs0);
  err_code |= clSetKernelArg(k_y_solve[1], 2, sizeof(cl_mem), &buf_lhs1);
  err_code |= clSetKernelArg(k_y_solve[1], 3, sizeof(cl_mem), &buf_lhs2);
  err_code |= clSetKernelArg(k_y_solve[1], 4, sizeof(cl_mem), &buf_lhs3);
  err_code |= clSetKernelArg(k_y_solve[1], 5, sizeof(cl_mem), &buf_lhs4);
  err_code |= clSetKernelArg(k_y_solve[1], 6, sizeof(cl_mem), &buf_lhsp0);
  err_code |= clSetKernelArg(k_y_solve[1], 7, sizeof(cl_mem), &buf_lhsp1);
  err_code |= clSetKernelArg(k_y_solve[1], 8, sizeof(cl_mem), &buf_lhsp2);
  err_code |= clSetKernelArg(k_y_solve[1], 9, sizeof(cl_mem), &buf_lhsp3);
  err_code |= clSetKernelArg(k_y_solve[1], 10, sizeof(cl_mem), &buf_lhsp4);
  err_code |= clSetKernelArg(k_y_solve[1], 11, sizeof(cl_mem), &buf_lhsm0);
  err_code |= clSetKernelArg(k_y_solve[1], 12, sizeof(cl_mem), &buf_lhsm1);
  err_code |= clSetKernelArg(k_y_solve[1], 13, sizeof(cl_mem), &buf_lhsm2);
  err_code |= clSetKernelArg(k_y_solve[1], 14, sizeof(cl_mem), &buf_lhsm3);
  err_code |= clSetKernelArg(k_y_solve[1], 15, sizeof(cl_mem), &buf_lhsm4);
  err_code |= clSetKernelArg(k_y_solve[1], 16, sizeof(int), &y_solve1_base_k);
  err_code |= clSetKernelArg(k_y_solve[1], 17, sizeof(int), &y_solve1_offset_k);
  err_code |= clSetKernelArg(k_y_solve[1], 18, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_y_solve[1], 19, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_y_solve[1], 20, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t y_solve1_gws[] = {y_solve1_gws_k};
    size_t y_solve1_lws[] = {1};
    y_solve1_gws[0] = clu_RoundWorkSize(y_solve1_gws[0], y_solve1_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[1],
                                      1, NULL,
                                      y_solve1_gws,
                                      y_solve1_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t y_solve1_gws[] = {nx2, ny2, y_solve1_gws_k};
    size_t y_solve1_lws[] = {16, 16, 1};
    y_solve1_gws[0] = clu_RoundWorkSize(y_solve1_gws[0], y_solve1_lws[0]);
    y_solve1_gws[1] = clu_RoundWorkSize(y_solve1_gws[1], y_solve1_lws[1]);
    y_solve1_gws[2] = clu_RoundWorkSize(y_solve1_gws[2], y_solve1_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[1],
                                      3, NULL,
                                      y_solve1_gws,
                                      y_solve1_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // y_solve2 kernel
  //---------------------------------------------------------------------
  int y_solve2_base_k = st2;
  int y_solve2_offset_k = st - st2;
  int y_solve2_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_y_solve[2], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_y_solve[2], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_y_solve[2], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_y_solve[2], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_y_solve[2], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_y_solve[2], 5, sizeof(cl_mem), &buf_lhs0);
  err_code |= clSetKernelArg(k_y_solve[2], 6, sizeof(cl_mem), &buf_lhs1);
  err_code |= clSetKernelArg(k_y_solve[2], 7, sizeof(cl_mem), &buf_lhs2);
  err_code |= clSetKernelArg(k_y_solve[2], 8, sizeof(cl_mem), &buf_lhs3);
  err_code |= clSetKernelArg(k_y_solve[2], 9, sizeof(cl_mem), &buf_lhs4);
  err_code |= clSetKernelArg(k_y_solve[2], 10, sizeof(int), &y_solve2_base_k);
  err_code |= clSetKernelArg(k_y_solve[2], 11, sizeof(int), &y_solve2_offset_k);
  err_code |= clSetKernelArg(k_y_solve[2], 12, sizeof(int), &y_solve2_gws_k);
  err_code |= clSetKernelArg(k_y_solve[2], 13, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_y_solve[2], 14, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_y_solve[2], 15, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t y_solve2_gws[] = {y_solve2_gws_k};
    size_t y_solve2_lws[] = {16};
    y_solve2_gws[0] = clu_RoundWorkSize(y_solve2_gws[0], y_solve2_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[2],
                                      1, NULL,
                                      y_solve2_gws,
                                      y_solve2_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t y_solve2_gws[] = {nx2, y_solve2_gws_k};
    size_t y_solve2_lws[] = {16, 16};
    y_solve2_gws[0] = clu_RoundWorkSize(y_solve2_gws[0], y_solve2_lws[0]);
    y_solve2_gws[1] = clu_RoundWorkSize(y_solve2_gws[1], y_solve2_lws[1]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[2],
                                      2, NULL,
                                      y_solve2_gws,
                                      y_solve2_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // y_solve3 kernel
  //---------------------------------------------------------------------
  int y_solve3_base_k = st2;
  int y_solve3_offset_k = st - st2;
  int y_solve3_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_y_solve[3], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_y_solve[3], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_y_solve[3], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_y_solve[3], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_y_solve[3], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_y_solve[3], 5, sizeof(cl_mem), &buf_lhsp0);
  err_code |= clSetKernelArg(k_y_solve[3], 6, sizeof(cl_mem), &buf_lhsp1);
  err_code |= clSetKernelArg(k_y_solve[3], 7, sizeof(cl_mem), &buf_lhsp2);
  err_code |= clSetKernelArg(k_y_solve[3], 8, sizeof(cl_mem), &buf_lhsp3);
  err_code |= clSetKernelArg(k_y_solve[3], 9, sizeof(cl_mem), &buf_lhsp4);
  err_code |= clSetKernelArg(k_y_solve[3], 10, sizeof(cl_mem), &buf_lhsm0);
  err_code |= clSetKernelArg(k_y_solve[3], 11, sizeof(cl_mem), &buf_lhsm1);
  err_code |= clSetKernelArg(k_y_solve[3], 12, sizeof(cl_mem), &buf_lhsm2);
  err_code |= clSetKernelArg(k_y_solve[3], 13, sizeof(cl_mem), &buf_lhsm3);
  err_code |= clSetKernelArg(k_y_solve[3], 14, sizeof(cl_mem), &buf_lhsm4);
  err_code |= clSetKernelArg(k_y_solve[3], 15, sizeof(int), &y_solve3_base_k);
  err_code |= clSetKernelArg(k_y_solve[3], 16, sizeof(int), &y_solve3_offset_k);
  err_code |= clSetKernelArg(k_y_solve[3], 17, sizeof(int), &y_solve3_gws_k);
  err_code |= clSetKernelArg(k_y_solve[3], 18, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_y_solve[3], 19, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_y_solve[3], 20, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t y_solve3_gws[] = {y_solve3_gws_k};
    size_t y_solve3_lws[] = {16};
    y_solve3_gws[0] = clu_RoundWorkSize(y_solve3_gws[0], y_solve3_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[3],
                                      1, NULL,
                                      y_solve3_gws,
                                      y_solve3_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t y_solve3_gws[] = {nx2, y_solve3_gws_k};
    size_t y_solve3_lws[] = {16, 16};
    y_solve3_gws[0] = clu_RoundWorkSize(y_solve3_gws[0], y_solve3_lws[0]);
    y_solve3_gws[1] = clu_RoundWorkSize(y_solve3_gws[1], y_solve3_lws[1]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[3],
                                      2, NULL,
                                      y_solve3_gws,
                                      y_solve3_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // y_solve4 kernel
  //---------------------------------------------------------------------
  int y_solve4_base_k = st2;
  int y_solve4_offset_k = st - st2;
  int y_solve4_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_y_solve[4], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_y_solve[4], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_y_solve[4], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_y_solve[4], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_y_solve[4], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_y_solve[4], 5, sizeof(cl_mem), &buf_lhs0);
  err_code |= clSetKernelArg(k_y_solve[4], 6, sizeof(cl_mem), &buf_lhs1);
  err_code |= clSetKernelArg(k_y_solve[4], 7, sizeof(cl_mem), &buf_lhs2);
  err_code |= clSetKernelArg(k_y_solve[4], 8, sizeof(cl_mem), &buf_lhs3);
  err_code |= clSetKernelArg(k_y_solve[4], 9, sizeof(cl_mem), &buf_lhs4);
  err_code |= clSetKernelArg(k_y_solve[4], 10, sizeof(cl_mem), &buf_lhsp0);
  err_code |= clSetKernelArg(k_y_solve[4], 11, sizeof(cl_mem), &buf_lhsp1);
  err_code |= clSetKernelArg(k_y_solve[4], 12, sizeof(cl_mem), &buf_lhsp2);
  err_code |= clSetKernelArg(k_y_solve[4], 13, sizeof(cl_mem), &buf_lhsp3);
  err_code |= clSetKernelArg(k_y_solve[4], 14, sizeof(cl_mem), &buf_lhsp4);
  err_code |= clSetKernelArg(k_y_solve[4], 15, sizeof(cl_mem), &buf_lhsm0);
  err_code |= clSetKernelArg(k_y_solve[4], 16, sizeof(cl_mem), &buf_lhsm1);
  err_code |= clSetKernelArg(k_y_solve[4], 17, sizeof(cl_mem), &buf_lhsm2);
  err_code |= clSetKernelArg(k_y_solve[4], 18, sizeof(cl_mem), &buf_lhsm3);
  err_code |= clSetKernelArg(k_y_solve[4], 19, sizeof(cl_mem), &buf_lhsm4);
  err_code |= clSetKernelArg(k_y_solve[4], 20, sizeof(int), &y_solve4_base_k);
  err_code |= clSetKernelArg(k_y_solve[4], 21, sizeof(int), &y_solve4_offset_k);
  err_code |= clSetKernelArg(k_y_solve[4], 22, sizeof(int), &y_solve4_gws_k);
  err_code |= clSetKernelArg(k_y_solve[4], 23, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_y_solve[4], 24, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_y_solve[4], 25, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t y_solve4_gws[] = {y_solve4_gws_k};
    size_t y_solve4_lws[] = {16};
    y_solve4_gws[0] = clu_RoundWorkSize(y_solve4_gws[0], y_solve4_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[4],
                                      1, NULL,
                                      y_solve4_gws,
                                      y_solve4_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t y_solve4_gws[] = {nx2, y_solve4_gws_k};
    size_t y_solve4_lws[] = {16, 16};
    y_solve4_gws[0] = clu_RoundWorkSize(y_solve4_gws[0], y_solve4_lws[0]);
    y_solve4_gws[1] = clu_RoundWorkSize(y_solve4_gws[1], y_solve4_lws[1]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_y_solve[4],
                                      2, NULL,
                                      y_solve4_gws,
                                      y_solve4_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_ysolve);

  pinvr_gpu(t, st, ed, st2, ed2);
}
