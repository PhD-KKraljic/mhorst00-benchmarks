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
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
void x_solve()
{
  int i, j, k, i1, i2, m;
  double ru1, fac1, fac2;

  if (timeron) timer_start(t_xsolve);
  for (k = 1; k <= nz2; k++) {
    lhsinit(nx2+1, ny2);

    //---------------------------------------------------------------------
    // Computes the left hand side for the three x-factors  
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue                   
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        ru1 = c3c4*rho_i[k][j][i];
        cv[i] = us[k][j][i];
        rhon[i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
      }

      for (i = 1; i <= nx2; i++) {
        lhs[j][i][0] =  0.0;
        lhs[j][i][1] = -dttx2 * cv[i-1] - dttx1 * rhon[i-1];
        lhs[j][i][2] =  1.0 + c2dttx1 * rhon[i];
        lhs[j][i][3] =  dttx2 * cv[i+1] - dttx1 * rhon[i+1];
        lhs[j][i][4] =  0.0;
      }
    }

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      i = 1;
      lhs[j][i][2] = lhs[j][i][2] + comz5;
      lhs[j][i][3] = lhs[j][i][3] - comz4;
      lhs[j][i][4] = lhs[j][i][4] + comz1;

      lhs[j][i+1][1] = lhs[j][i+1][1] - comz4;
      lhs[j][i+1][2] = lhs[j][i+1][2] + comz6;
      lhs[j][i+1][3] = lhs[j][i+1][3] - comz4;
      lhs[j][i+1][4] = lhs[j][i+1][4] + comz1;
    }

    for (j = 1; j <= ny2; j++) {
      for (i = 3; i <= grid_points[0]-4; i++) {
        lhs[j][i][0] = lhs[j][i][0] + comz1;
        lhs[j][i][1] = lhs[j][i][1] - comz4;
        lhs[j][i][2] = lhs[j][i][2] + comz6;
        lhs[j][i][3] = lhs[j][i][3] - comz4;
        lhs[j][i][4] = lhs[j][i][4] + comz1;
      }
    }

    for (j = 1; j <= ny2; j++) {
      i = grid_points[0]-3;
      lhs[j][i][0] = lhs[j][i][0] + comz1;
      lhs[j][i][1] = lhs[j][i][1] - comz4;
      lhs[j][i][2] = lhs[j][i][2] + comz6;
      lhs[j][i][3] = lhs[j][i][3] - comz4;

      lhs[j][i+1][0] = lhs[j][i+1][0] + comz1;
      lhs[j][i+1][1] = lhs[j][i+1][1] - comz4;
      lhs[j][i+1][2] = lhs[j][i+1][2] + comz5;
    }

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) by adding to 
    // the first  
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        lhsp[j][i][0] = lhs[j][i][0];
        lhsp[j][i][1] = lhs[j][i][1] - dttx2 * speed[k][j][i-1];
        lhsp[j][i][2] = lhs[j][i][2];
        lhsp[j][i][3] = lhs[j][i][3] + dttx2 * speed[k][j][i+1];
        lhsp[j][i][4] = lhs[j][i][4];
        lhsm[j][i][0] = lhs[j][i][0];
        lhsm[j][i][1] = lhs[j][i][1] + dttx2 * speed[k][j][i-1];
        lhsm[j][i][2] = lhs[j][i][2];
        lhsm[j][i][3] = lhs[j][i][3] - dttx2 * speed[k][j][i+1];
        lhsm[j][i][4] = lhs[j][i][4];
      }
    }

    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // perform the Thomas algorithm; first, FORWARD ELIMINATION     
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      for (i = 0; i <= grid_points[0]-3; i++) {
        i1 = i + 1;
        i2 = i + 2;
        fac1 = 1.0/lhs[j][i][2];
        lhs[j][i][3] = fac1*lhs[j][i][3];
        lhs[j][i][4] = fac1*lhs[j][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
        }
        lhs[j][i1][2] = lhs[j][i1][2] - lhs[j][i1][1]*lhs[j][i][3];
        lhs[j][i1][3] = lhs[j][i1][3] - lhs[j][i1][1]*lhs[j][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs[j][i1][1]*rhs[k][j][i][m];
        }
        lhs[j][i2][1] = lhs[j][i2][1] - lhs[j][i2][0]*lhs[j][i][3];
        lhs[j][i2][2] = lhs[j][i2][2] - lhs[j][i2][0]*lhs[j][i][4];
        for (m = 0; m < 3; m++) {
          rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs[j][i2][0]*rhs[k][j][i][m];
        }
      }
    }

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      i  = grid_points[0]-2;
      i1 = grid_points[0]-1;
      fac1 = 1.0/lhs[j][i][2];
      lhs[j][i][3] = fac1*lhs[j][i][3];
      lhs[j][i][4] = fac1*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = fac1*rhs[k][j][i][m];
      }
      lhs[j][i1][2] = lhs[j][i1][2] - lhs[j][i1][1]*lhs[j][i][3];
      lhs[j][i1][3] = lhs[j][i1][3] - lhs[j][i1][1]*lhs[j][i][4];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs[j][i1][1]*rhs[k][j][i][m];
      }

      //---------------------------------------------------------------------
      // scale the last row immediately 
      //---------------------------------------------------------------------
      fac2 = 1.0/lhs[j][i1][2];
      for (m = 0; m < 3; m++) {
        rhs[k][j][i1][m] = fac2*rhs[k][j][i1][m];
      }
    }

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      for (i = 0; i <= grid_points[0]-3; i++) {
        i1 = i + 1;
        i2 = i + 2;

        m = 3;
        fac1 = 1.0/lhsp[j][i][2];
        lhsp[j][i][3]    = fac1*lhsp[j][i][3];
        lhsp[j][i][4]    = fac1*lhsp[j][i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        lhsp[j][i1][2]   = lhsp[j][i1][2] - lhsp[j][i1][1]*lhsp[j][i][3];
        lhsp[j][i1][3]   = lhsp[j][i1][3] - lhsp[j][i1][1]*lhsp[j][i][4];
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp[j][i1][1]*rhs[k][j][i][m];
        lhsp[j][i2][1]   = lhsp[j][i2][1] - lhsp[j][i2][0]*lhsp[j][i][3];
        lhsp[j][i2][2]   = lhsp[j][i2][2] - lhsp[j][i2][0]*lhsp[j][i][4];
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp[j][i2][0]*rhs[k][j][i][m];

        m = 4;
        fac1 = 1.0/lhsm[j][i][2];
        lhsm[j][i][3]    = fac1*lhsm[j][i][3];
        lhsm[j][i][4]    = fac1*lhsm[j][i][4];
        rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
        lhsm[j][i1][2]   = lhsm[j][i1][2] - lhsm[j][i1][1]*lhsm[j][i][3];
        lhsm[j][i1][3]   = lhsm[j][i1][3] - lhsm[j][i1][1]*lhsm[j][i][4];
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm[j][i1][1]*rhs[k][j][i][m];
        lhsm[j][i2][1]   = lhsm[j][i2][1] - lhsm[j][i2][0]*lhsm[j][i][3];
        lhsm[j][i2][2]   = lhsm[j][i2][2] - lhsm[j][i2][0]*lhsm[j][i][4];
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm[j][i2][0]*rhs[k][j][i][m];
      }
    }

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      i  = grid_points[0]-2;
      i1 = grid_points[0]-1;

      m = 3;
      fac1 = 1.0/lhsp[j][i][2];
      lhsp[j][i][3]    = fac1*lhsp[j][i][3];
      lhsp[j][i][4]    = fac1*lhsp[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsp[j][i1][2]   = lhsp[j][i1][2] - lhsp[j][i1][1]*lhsp[j][i][3];
      lhsp[j][i1][3]   = lhsp[j][i1][3] - lhsp[j][i1][1]*lhsp[j][i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp[j][i1][1]*rhs[k][j][i][m];

      m = 4;
      fac1 = 1.0/lhsm[j][i][2];
      lhsm[j][i][3]    = fac1*lhsm[j][i][3];
      lhsm[j][i][4]    = fac1*lhsm[j][i][4];
      rhs[k][j][i][m]  = fac1*rhs[k][j][i][m];
      lhsm[j][i1][2]   = lhsm[j][i1][2] - lhsm[j][i1][1]*lhsm[j][i][3];
      lhsm[j][i1][3]   = lhsm[j][i1][3] - lhsm[j][i1][1]*lhsm[j][i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm[j][i1][1]*rhs[k][j][i][m];

      //---------------------------------------------------------------------
      // Scale the last row immediately
      //---------------------------------------------------------------------
      rhs[k][j][i1][3] = rhs[k][j][i1][3]/lhsp[j][i1][2];
      rhs[k][j][i1][4] = rhs[k][j][i1][4]/lhsm[j][i1][2];
    }

    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      i  = grid_points[0]-2;
      i1 = grid_points[0]-1;
      for (m = 0; m < 3; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][i][3]*rhs[k][j][i1][m];
      }

      rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[j][i][3]*rhs[k][j][i1][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[j][i][3]*rhs[k][j][i1][4];
    }

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      for (i = grid_points[0]-3; i >= 0; i--) {
        i1 = i + 1;
        i2 = i + 2;
        for (m = 0; m < 3; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - 
                            lhs[j][i][3]*rhs[k][j][i1][m] -
                            lhs[j][i][4]*rhs[k][j][i2][m];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[k][j][i][3] = rhs[k][j][i][3] - 
                          lhsp[j][i][3]*rhs[k][j][i1][3] -
                          lhsp[j][i][4]*rhs[k][j][i2][3];
        rhs[k][j][i][4] = rhs[k][j][i][4] - 
                          lhsm[j][i][3]*rhs[k][j][i1][4] -
                          lhsm[j][i][4]*rhs[k][j][i2][4];
      }
    }
  }
  if (timeron) timer_stop(t_xsolve);

  //---------------------------------------------------------------------
  // Do the block-diagonal inversion          
  //---------------------------------------------------------------------
  ninvr();
}

void x_solve_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_xsolve);

  //---------------------------------------------------------------------
  // transpose kernel
  //---------------------------------------------------------------------
  const int P0 = WORK_NUM_ITEM_K * (JMAXP+1), Q0 = IMAXP+1;
  int gws0[] = {Q0, P0}, lws0[] = {16, 16};
  gws0[0] = RoundWorkSize(gws0[0], lws0[0]);
  gws0[1] = RoundWorkSize(gws0[1], lws0[1]);
  dim3 blockSize(gws0[0]/lws0[0], gws0[1]/lws0[1]);
  dim3 threadSize(lws0[0], lws0[1]);

  if (opt_level_t == 3) {
    // rhs0
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs0, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs0, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));

    // rhs1
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs1, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs1, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    
    // rhs2
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs2, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs2, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));

    // rhs3
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs3, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs3, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));

    // rhs4
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs4, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs4, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
  }
  else if (opt_level_t == 5) {
    // rhs0
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs0, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs0, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));

    // rhs1
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs1, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs1, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    
    // rhs2
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs2, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs2, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));

    // rhs3
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs3, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs3, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));

    // rhs4
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_rhs4, buf_temp, P0, Q0 );
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);
    CUCHK(cudaMemcpyAsync(buf_rhs4, buf_temp, P0 * Q0 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
  }
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // x_solve0 kernel
  //---------------------------------------------------------------------
  int x_solve0_base_k = st2;
  int x_solve0_offset_k = st - st2;
  int x_solve0_gws_k = ed - st + 1;

  if (opt_level_t == 0 || opt_level_t == 2) {
    int x_solve0_gws[] = {x_solve0_gws_k};
    int x_solve0_lws[] = {1};
    x_solve0_gws[0] = RoundWorkSize(x_solve0_gws[0], x_solve0_lws[0]);

    blockSize.x = x_solve0_gws[0]/x_solve0_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve0_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve0_base", cmd_queue[0]);
    k_x_solve0_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, buf_us,
         buf_rho_i, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve0_base_k, x_solve0_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve0_base", cmd_queue[0]);
  }
  else if (opt_level_t == 1) {
    int x_solve0_gws[] = {nx2+2, ny2, x_solve0_gws_k};
    int x_solve0_lws[] = {16, 16, 1};
    x_solve0_gws[0] = RoundWorkSize(x_solve0_gws[0], x_solve0_lws[0]);
    x_solve0_gws[1] = RoundWorkSize(x_solve0_gws[1], x_solve0_lws[1]);
    x_solve0_gws[2] = RoundWorkSize(x_solve0_gws[2], x_solve0_lws[2]);

    blockSize.x = x_solve0_gws[0]/x_solve0_lws[0];
    blockSize.y = x_solve0_gws[1]/x_solve0_lws[1];
    blockSize.z = x_solve0_gws[2]/x_solve0_lws[2];
    threadSize.x = x_solve0_lws[0];
    threadSize.y = x_solve0_lws[1];
    threadSize.z = x_solve0_lws[2];

    cuda_ProfilerStartEventRecord("k_x_solve0_parallel", cmd_queue[0]);
    k_x_solve0_parallel<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, buf_us,
         buf_rho_i, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve0_base_k, x_solve0_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve0_parallel", cmd_queue[0]);
  }
  else if (opt_level_t == 3) {
    int x_solve0_gws[] = {x_solve0_gws_k};
    int x_solve0_lws[] = {1};
    x_solve0_gws[0] = RoundWorkSize(x_solve0_gws[0], x_solve0_lws[0]);

    blockSize.x = x_solve0_gws[0]/x_solve0_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve0_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve0_layout", cmd_queue[0]);
    k_x_solve0_layout<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, buf_us,
         buf_rho_i, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve0_base_k, x_solve0_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve0_layout", cmd_queue[0]);
  }
  else {
    int x_solve0_gws[] = {nx2+2, ny2, x_solve0_gws_k};
    int x_solve0_lws[] = {16, 16, 1};
    x_solve0_gws[0] = RoundWorkSize(x_solve0_gws[0], x_solve0_lws[0]);
    x_solve0_gws[1] = RoundWorkSize(x_solve0_gws[1], x_solve0_lws[1]);
    x_solve0_gws[2] = RoundWorkSize(x_solve0_gws[2], x_solve0_lws[2]);

    blockSize.x = x_solve0_gws[0]/x_solve0_lws[0];
    blockSize.y = x_solve0_gws[1]/x_solve0_lws[1];
    blockSize.z = x_solve0_gws[2]/x_solve0_lws[2];
    threadSize.x = x_solve0_lws[0];
    threadSize.y = x_solve0_lws[1];
    threadSize.z = x_solve0_lws[2];

    cuda_ProfilerStartEventRecord("k_x_solve0_opt", cmd_queue[0]);
    k_x_solve0_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, buf_us,
         buf_rho_i, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve0_base_k, x_solve0_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve0_opt", cmd_queue[0]);
  }

  CUCHK(cudaGetLastError());
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // x_solve1 kernel
  //---------------------------------------------------------------------
  int x_solve1_base_k = st2;
  int x_solve1_offset_k = st - st2;
  int x_solve1_gws_k = ed - st + 1;

  if (opt_level_t == 0 || opt_level_t == 2) {
    int x_solve1_gws[] = {x_solve1_gws_k};
    int x_solve1_lws[] = {1};
    x_solve1_gws[0] = RoundWorkSize(x_solve1_gws[0], x_solve1_lws[0]);
    blockSize.x = x_solve1_gws[0]/x_solve1_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve1_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve1_base", cmd_queue[0]);
    k_x_solve1_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_speed, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve1_base_k, x_solve1_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve1_base", cmd_queue[0]);
  }
  else if (opt_level_t == 1) {
    int x_solve1_gws[] = {nx2, ny2, x_solve1_gws_k};
    int x_solve1_lws[] = {16, 16, 1};
    x_solve1_gws[0] = RoundWorkSize(x_solve1_gws[0], x_solve1_lws[0]);
    x_solve1_gws[1] = RoundWorkSize(x_solve1_gws[1], x_solve1_lws[1]);
    x_solve1_gws[2] = RoundWorkSize(x_solve1_gws[2], x_solve1_lws[2]);
    blockSize.x = x_solve1_gws[0]/x_solve1_lws[0];
    blockSize.y = x_solve1_gws[1]/x_solve1_lws[1];
    blockSize.z = x_solve1_gws[2]/x_solve1_lws[2];
    threadSize.x = x_solve1_lws[0];
    threadSize.y = x_solve1_lws[1];
    threadSize.z = x_solve1_lws[2];

    cuda_ProfilerStartEventRecord("k_x_solve1_parallel", cmd_queue[0]);
    k_x_solve1_parallel<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_speed, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve1_base_k, x_solve1_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve1_parallel", cmd_queue[0]);
  }
  else if (opt_level_t == 3) {
    int x_solve1_gws[] = {x_solve1_gws_k};
    int x_solve1_lws[] = {1};
    x_solve1_gws[0] = RoundWorkSize(x_solve1_gws[0], x_solve1_lws[0]);
    blockSize.x = x_solve1_gws[0]/x_solve1_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve1_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve1_layout", cmd_queue[0]);
    k_x_solve1_layout<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_speed, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve1_base_k, x_solve1_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve1_layout", cmd_queue[0]);
  }
  else {
    int x_solve1_gws[] = {nx2, ny2, x_solve1_gws_k};
    int x_solve1_lws[] = {16, 16, 1};
    x_solve1_gws[0] = RoundWorkSize(x_solve1_gws[0], x_solve1_lws[0]);
    x_solve1_gws[1] = RoundWorkSize(x_solve1_gws[1], x_solve1_lws[1]);
    x_solve1_gws[2] = RoundWorkSize(x_solve1_gws[2], x_solve1_lws[2]);
    blockSize.x = x_solve1_gws[0]/x_solve1_lws[0];
    blockSize.y = x_solve1_gws[1]/x_solve1_lws[1];
    blockSize.z = x_solve1_gws[2]/x_solve1_lws[2];
    threadSize.x = x_solve1_lws[0];
    threadSize.y = x_solve1_lws[1];
    threadSize.z = x_solve1_lws[2];

    cuda_ProfilerStartEventRecord("k_x_solve1_opt", cmd_queue[0]);
    k_x_solve1_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_speed, buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve1_base_k, x_solve1_offset_k, nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve1_opt", cmd_queue[0]);
  }
  
  CUCHK(cudaGetLastError());
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // x_solve2 kernel
  //---------------------------------------------------------------------
  int x_solve2_base_k = st2;
  int x_solve2_offset_k = st - st2;
  int x_solve2_gws_k = ed - st + 1;

  if (opt_level_t == 0 || opt_level_t == 2) {
    int x_solve2_gws[] = {x_solve2_gws_k};
    int x_solve2_lws[] = {16};
    x_solve2_gws[0] = RoundWorkSize(x_solve2_gws[0], x_solve2_lws[0]);
    blockSize.x = x_solve2_gws[0]/x_solve2_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve2_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve2_base", cmd_queue[0]);
    k_x_solve2_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         x_solve2_base_k, x_solve2_offset_k, x_solve2_gws_k, 
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve2_base", cmd_queue[0]);
  }
  else if (opt_level_t == 1) {
    int x_solve2_gws[] = {ny2, x_solve2_gws_k};
    int x_solve2_lws[] = {16, 16};
    x_solve2_gws[0] = RoundWorkSize(x_solve2_gws[0], x_solve2_lws[0]);
    x_solve2_gws[1] = RoundWorkSize(x_solve2_gws[1], x_solve2_lws[1]);
    blockSize.x = x_solve2_gws[0]/x_solve2_lws[0];
    blockSize.y = x_solve2_gws[1]/x_solve2_lws[1];
    blockSize.z = 1;
    threadSize.x = x_solve2_lws[0];
    threadSize.y = x_solve2_lws[1];
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve2_parallel", cmd_queue[0]);
    k_x_solve2_parallel<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         x_solve2_base_k, x_solve2_offset_k, x_solve2_gws_k, 
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve2_parallel", cmd_queue[0]);
  }
  else if (opt_level_t == 3) {
    int x_solve2_gws[] = {x_solve2_gws_k};
    int x_solve2_lws[] = {16};
    x_solve2_gws[0] = RoundWorkSize(x_solve2_gws[0], x_solve2_lws[0]);
    blockSize.x = x_solve2_gws[0]/x_solve2_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve2_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve2_layout", cmd_queue[0]);
    k_x_solve2_layout<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         x_solve2_base_k, x_solve2_offset_k, x_solve2_gws_k, 
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve2_layout", cmd_queue[0]);
  }
  else {
    int x_solve2_gws[] = {ny2, x_solve2_gws_k};
    int x_solve2_lws[] = {16, 16};
    x_solve2_gws[0] = RoundWorkSize(x_solve2_gws[0], x_solve2_lws[0]);
    x_solve2_gws[1] = RoundWorkSize(x_solve2_gws[1], x_solve2_lws[1]);
    blockSize.x = x_solve2_gws[0]/x_solve2_lws[0];
    blockSize.y = x_solve2_gws[1]/x_solve2_lws[1];
    blockSize.z = 1;
    threadSize.x = x_solve2_lws[0];
    threadSize.y = x_solve2_lws[1];
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve2_opt", cmd_queue[0]);
    k_x_solve2_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         x_solve2_base_k, x_solve2_offset_k, x_solve2_gws_k, 
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve2_opt", cmd_queue[0]);
  }

  CUCHK(cudaGetLastError());
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // x_solve3 kernel
  //---------------------------------------------------------------------
  int x_solve3_base_k = st2;
  int x_solve3_offset_k = st - st2;
  int x_solve3_gws_k = ed - st + 1;

  if (opt_level_t == 0 || opt_level_t == 2) {
    int x_solve3_gws[] = {x_solve3_gws_k};
    int x_solve3_lws[] = {16};
    x_solve3_gws[0] = RoundWorkSize(x_solve3_gws[0], x_solve3_lws[0]);
    blockSize.x = x_solve3_gws[0]/x_solve3_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve3_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve3_base", cmd_queue[0]);
    k_x_solve3_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve3_base_k, x_solve3_offset_k, x_solve3_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve3_base", cmd_queue[0]);
  }
  else if (opt_level_t == 1) {
    int x_solve3_gws[] = {ny2, x_solve3_gws_k};
    int x_solve3_lws[] = {16, 16};
    x_solve3_gws[0] = RoundWorkSize(x_solve3_gws[0], x_solve3_lws[0]);
    x_solve3_gws[1] = RoundWorkSize(x_solve3_gws[1], x_solve3_lws[1]);
    blockSize.x = x_solve3_gws[0]/x_solve3_lws[0];
    blockSize.y = x_solve3_gws[1]/x_solve3_lws[1];
    blockSize.z = 1;
    threadSize.x = x_solve3_lws[0];
    threadSize.y = x_solve3_lws[1];
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve3_parallel", cmd_queue[0]);
    k_x_solve3_parallel<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve3_base_k, x_solve3_offset_k, x_solve3_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve3_parallel", cmd_queue[0]);
  }
  else if (opt_level_t == 3) {
    int x_solve3_gws[] = {x_solve3_gws_k};
    int x_solve3_lws[] = {16};
    x_solve3_gws[0] = RoundWorkSize(x_solve3_gws[0], x_solve3_lws[0]);
    blockSize.x = x_solve3_gws[0]/x_solve3_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve3_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve3_layout", cmd_queue[0]);
    k_x_solve3_layout<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve3_base_k, x_solve3_offset_k, x_solve3_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve3_layout", cmd_queue[0]);
  }
  else {
    int x_solve3_gws[] = {ny2, x_solve3_gws_k};
    int x_solve3_lws[] = {16, 16};
    x_solve3_gws[0] = RoundWorkSize(x_solve3_gws[0], x_solve3_lws[0]);
    x_solve3_gws[1] = RoundWorkSize(x_solve3_gws[1], x_solve3_lws[1]);
    blockSize.x = x_solve3_gws[0]/x_solve3_lws[0];
    blockSize.y = x_solve3_gws[1]/x_solve3_lws[1];
    blockSize.z = 1;
    threadSize.x = x_solve3_lws[0];
    threadSize.y = x_solve3_lws[1];
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve3_opt", cmd_queue[0]);
    k_x_solve3_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve3_base_k, x_solve3_offset_k, x_solve3_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve3_opt", cmd_queue[0]);
  }

  CUCHK(cudaGetLastError());
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // x_solve4 kernel
  //---------------------------------------------------------------------
  int x_solve4_base_k = st2;
  int x_solve4_offset_k = st - st2;
  int x_solve4_gws_k = ed - st + 1;

  if (opt_level_t == 0 || opt_level_t == 2) {
    int x_solve4_gws[] = {x_solve4_gws_k};
    int x_solve4_lws[] = {16};
    x_solve4_gws[0] = RoundWorkSize(x_solve4_gws[0], x_solve4_lws[0]);
    blockSize.x = x_solve4_gws[0]/x_solve4_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve4_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve4_base", cmd_queue[0]);
    k_x_solve4_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve4_base_k, x_solve4_offset_k, x_solve4_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve4_base", cmd_queue[0]);
  }
  else if (opt_level_t == 1) {
    int x_solve4_gws[] = {ny2, x_solve4_gws_k};
    int x_solve4_lws[] = {16, 16};
    x_solve4_gws[0] = RoundWorkSize(x_solve4_gws[0], x_solve4_lws[0]);
    x_solve4_gws[1] = RoundWorkSize(x_solve4_gws[1], x_solve4_lws[1]);
    blockSize.x = x_solve4_gws[0]/x_solve4_lws[0];
    blockSize.y = x_solve4_gws[1]/x_solve4_lws[1];
    blockSize.z = 1;
    threadSize.x = x_solve4_lws[0];
    threadSize.y = x_solve4_lws[1];
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve4_parallel", cmd_queue[0]);
    k_x_solve4_parallel<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve4_base_k, x_solve4_offset_k, x_solve4_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve4_parallel", cmd_queue[0]);
  }
  else if (opt_level_t == 3) {
    int x_solve4_gws[] = {x_solve4_gws_k};
    int x_solve4_lws[] = {16};
    x_solve4_gws[0] = RoundWorkSize(x_solve4_gws[0], x_solve4_lws[0]);
    blockSize.x = x_solve4_gws[0]/x_solve4_lws[0];
    blockSize.y = 1;
    blockSize.z = 1;
    threadSize.x = x_solve4_lws[0];
    threadSize.y = 1;
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve4_layout", cmd_queue[0]);
    k_x_solve4_layout<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve4_base_k, x_solve4_offset_k, x_solve4_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve4_layout", cmd_queue[0]);
  }
  else {
    int x_solve4_gws[] = {ny2, x_solve4_gws_k};
    int x_solve4_lws[] = {16, 16};
    x_solve4_gws[0] = RoundWorkSize(x_solve4_gws[0], x_solve4_lws[0]);
    x_solve4_gws[1] = RoundWorkSize(x_solve4_gws[1], x_solve4_lws[1]);
    blockSize.x = x_solve4_gws[0]/x_solve4_lws[0];
    blockSize.y = x_solve4_gws[1]/x_solve4_lws[1];
    blockSize.z = 1;
    threadSize.x = x_solve4_lws[0];
    threadSize.y = x_solve4_lws[1];
    threadSize.z = 1;

    cuda_ProfilerStartEventRecord("k_x_solve4_opt", cmd_queue[0]);
    k_x_solve4_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
         buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4,
         buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4,
         buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4,
         x_solve4_base_k, x_solve4_offset_k, x_solve4_gws_k,
         nx2, ny2, nz2, WORK_NUM_ITEM_K
        );
    cuda_ProfilerEndEventRecord("k_x_solve4_opt", cmd_queue[0]);
  }

  CUCHK(cudaGetLastError());
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // transpose kernel
  //---------------------------------------------------------------------
  const int P1 = IMAXP+1, Q1 = WORK_NUM_ITEM_K * (JMAXP+1);
  int gws1[] = {Q1, P1};
  int lws1[] = {16, 16};
  gws1[0] = RoundWorkSize(gws1[0], lws1[0]);
  gws1[1] = RoundWorkSize(gws1[1], lws1[1]);

  blockSize.x = gws1[0]/lws1[0];
  blockSize.y = gws1[1]/lws1[1];
  blockSize.z = 1;
  threadSize.x = lws1[0];
  threadSize.y = lws1[1];
  threadSize.z = 1;

  if (opt_level_t == 3) {
    // rhs0
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs0, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs0, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);

    // rhs1
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs1, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs1, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);

    // rhs2
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs2, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs2, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);

    // rhs3
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs3, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs3, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);

    // rhs4
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs4, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_base", cmd_queue[0]);
    k_transpose_base<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs4, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_base", cmd_queue[0]);
  }
  else if (opt_level_t == 5) {
    // rhs0
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs0, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs0, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);

    // rhs1
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs1, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs1, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);

    // rhs2
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs2, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs2, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);

    // rhs3
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs3, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs3, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);

    // rhs4
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs4, P1 * Q1 * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue[0]));
    cuda_ProfilerStartEventRecord("k_transpose_opt", cmd_queue[0]);
    k_transpose_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>> ( buf_temp, buf_rhs4, P1, Q1);
    CUCHK(cudaGetLastError());
    cuda_ProfilerEndEventRecord("k_transpose_opt", cmd_queue[0]);
  }
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_xsolve);

  //---------------------------------------------------------------------
  // Do the block-diagonal inversion          
  //---------------------------------------------------------------------
  ninvr_gpu(t, st, ed, st2, ed2);
}
