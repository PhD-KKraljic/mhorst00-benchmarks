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

#include <math.h>
#include "header.h"

void compute_rhs()
{
  int i, j, k, m;
  double aux, rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;

  if (timeron) timer_start(t_rhs);
  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound. 
  //---------------------------------------------------------------------
  for (k = 0; k <= grid_points[2]-1; k++) {
    for (j = 0; j <= grid_points[1]-1; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        rho_inv = 1.0/u[k][j][i][0];
        rho_i[k][j][i] = rho_inv;
        us[k][j][i] = u[k][j][i][1] * rho_inv;
        vs[k][j][i] = u[k][j][i][2] * rho_inv;
        ws[k][j][i] = u[k][j][i][3] * rho_inv;
        square[k][j][i] = 0.5* (
            u[k][j][i][1]*u[k][j][i][1] + 
            u[k][j][i][2]*u[k][j][i][2] +
            u[k][j][i][3]*u[k][j][i][3] ) * rho_inv;
        qs[k][j][i] = square[k][j][i] * rho_inv;
        //-------------------------------------------------------------------
        // (don't need speed and ainx until the lhs computation)
        //-------------------------------------------------------------------
        aux = c1c2*rho_inv* (u[k][j][i][4] - square[k][j][i]);
        speed[k][j][i] = sqrt(aux);
      }
    }
  }

  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------
  for (k = 0; k <= grid_points[2]-1; k++) {
    for (j = 0; j <= grid_points[1]-1; j++) {
      for (i = 0; i <= grid_points[0]-1; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][j][i][m] = forcing[k][j][i][m];
        }
      }
    }
  }

  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsx);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[k][j][i][0] = rhs[k][j][i][0] + dx1tx1 * 
          (u[k][j][i+1][0] - 2.0*u[k][j][i][0] + u[k][j][i-1][0]) -
          tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]);

        rhs[k][j][i][1] = rhs[k][j][i][1] + dx2tx1 * 
          (u[k][j][i+1][1] - 2.0*u[k][j][i][1] + u[k][j][i-1][1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[k][j][i+1][1]*up1 - u[k][j][i-1][1]*um1 +
                (u[k][j][i+1][4] - square[k][j][i+1] -
                 u[k][j][i-1][4] + square[k][j][i-1]) * c2);

        rhs[k][j][i][2] = rhs[k][j][i][2] + dx3tx1 * 
          (u[k][j][i+1][2] - 2.0*u[k][j][i][2] + u[k][j][i-1][2]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] + vs[k][j][i-1]) -
          tx2 * (u[k][j][i+1][2]*up1 - u[k][j][i-1][2]*um1);

        rhs[k][j][i][3] = rhs[k][j][i][3] + dx4tx1 * 
          (u[k][j][i+1][3] - 2.0*u[k][j][i][3] + u[k][j][i-1][3]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] + ws[k][j][i-1]) -
          tx2 * (u[k][j][i+1][3]*up1 - u[k][j][i-1][3]*um1);

        rhs[k][j][i][4] = rhs[k][j][i][4] + dx5tx1 * 
          (u[k][j][i+1][4] - 2.0*u[k][j][i][4] + u[k][j][i-1][4]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] + qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + um1*um1) +
          xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] - 
                2.0*u[k][j][i][4]*rho_i[k][j][i] +
                    u[k][j][i-1][4]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[k][j][i+1][4] - c2*square[k][j][i+1])*up1 -
                  (c1*u[k][j][i-1][4] - c2*square[k][j][i-1])*um1 );
      }
    }

    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
    for (j = 1; j <= ny2; j++) {
      i = 1;
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
          (5.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] + u[k][j][i+2][m]);
      }

      i = 2;
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
          (-4.0*u[k][j][i-1][m] + 6.0*u[k][j][i][m] -
            4.0*u[k][j][i+1][m] + u[k][j][i+2][m]);
      }
    }

    for (j = 1; j <= ny2; j++) {
      for (i = 3; i <= nx2-2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
            ( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
            6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] + 
              u[k][j][i+2][m] );
        }
      }
    }

    for (j = 1; j <= ny2; j++) {
      i = nx2-1;
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
          6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] );
      }

      i = nx2;
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 5.0*u[k][j][i][m] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsx);

  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsy);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];

        rhs[k][j][i][0] = rhs[k][j][i][0] + dy1ty1 * 
          (u[k][j+1][i][0] - 2.0*u[k][j][i][0] + u[k][j-1][i][0]) -
          ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]);

        rhs[k][j][i][1] = rhs[k][j][i][1] + dy2ty1 * 
          (u[k][j+1][i][1] - 2.0*u[k][j][i][1] + u[k][j-1][i][1]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
          ty2 * (u[k][j+1][i][1]*vp1 - u[k][j-1][i][1]*vm1);

        rhs[k][j][i][2] = rhs[k][j][i][2] + dy3ty1 * 
          (u[k][j+1][i][2] - 2.0*u[k][j][i][2] + u[k][j-1][i][2]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[k][j+1][i][2]*vp1 - u[k][j-1][i][2]*vm1 +
                (u[k][j+1][i][4] - square[k][j+1][i] - 
                 u[k][j-1][i][4] + square[k][j-1][i]) * c2);

        rhs[k][j][i][3] = rhs[k][j][i][3] + dy4ty1 * 
          (u[k][j+1][i][3] - 2.0*u[k][j][i][3] + u[k][j-1][i][3]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
          ty2 * (u[k][j+1][i][3]*vp1 - u[k][j-1][i][3]*vm1);

        rhs[k][j][i][4] = rhs[k][j][i][4] + dy5ty1 * 
          (u[k][j+1][i][4] - 2.0*u[k][j][i][4] + u[k][j-1][i][4]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + vm1*vm1) +
          yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] - 
                  2.0*u[k][j][i][4]*rho_i[k][j][i] +
                    u[k][j-1][i][4]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[k][j+1][i][4] - c2*square[k][j+1][i]) * vp1 -
                 (c1*u[k][j-1][i][4] - c2*square[k][j-1][i]) * vm1);
      }
    }

    //---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
    j = 1;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
          ( 5.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + u[k][j+2][i][m]);
      }
    }

    j = 2;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
          (-4.0*u[k][j-1][i][m] + 6.0*u[k][j][i][m] -
            4.0*u[k][j+1][i][m] + u[k][j+2][i][m]);
      }
    }

    for (j = 3; j <= ny2-2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
            ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
            6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + 
              u[k][j+2][i][m] );
        }
      }
    }

    j = ny2-1;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
          6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] );
      }
    }

    j = ny2;
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 5.0*u[k][j][i][m] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsy);

  //---------------------------------------------------------------------
  // compute zeta-direction fluxes 
  //---------------------------------------------------------------------
  if (timeron) timer_start(t_rhsz);
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[k][j][i][0] = rhs[k][j][i][0] + dz1tz1 * 
          (u[k+1][j][i][0] - 2.0*u[k][j][i][0] + u[k-1][j][i][0]) -
          tz2 * (u[k+1][j][i][3] - u[k-1][j][i][3]);

        rhs[k][j][i][1] = rhs[k][j][i][1] + dz2tz1 * 
          (u[k+1][j][i][1] - 2.0*u[k][j][i][1] + u[k-1][j][i][1]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + us[k-1][j][i]) -
          tz2 * (u[k+1][j][i][1]*wp1 - u[k-1][j][i][1]*wm1);

        rhs[k][j][i][2] = rhs[k][j][i][2] + dz3tz1 * 
          (u[k+1][j][i][2] - 2.0*u[k][j][i][2] + u[k-1][j][i][2]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + vs[k-1][j][i]) -
          tz2 * (u[k+1][j][i][2]*wp1 - u[k-1][j][i][2]*wm1);

        rhs[k][j][i][3] = rhs[k][j][i][3] + dz4tz1 * 
          (u[k+1][j][i][3] - 2.0*u[k][j][i][3] + u[k-1][j][i][3]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[k+1][j][i][3]*wp1 - u[k-1][j][i][3]*wm1 +
                (u[k+1][j][i][4] - square[k+1][j][i] - 
                 u[k-1][j][i][4] + square[k-1][j][i]) * c2);

        rhs[k][j][i][4] = rhs[k][j][i][4] + dz5tz1 * 
          (u[k+1][j][i][4] - 2.0*u[k][j][i][4] + u[k-1][j][i][4]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
          zzcon5 * (u[k+1][j][i][4]*rho_i[k+1][j][i] - 
                  2.0*u[k][j][i][4]*rho_i[k][j][i] +
                    u[k-1][j][i][4]*rho_i[k-1][j][i]) -
          tz2 * ((c1*u[k+1][j][i][4] - c2*square[k+1][j][i])*wp1 -
                 (c1*u[k-1][j][i][4] - c2*square[k-1][j][i])*wm1);
      }
    }
  }

  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  k = 1;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
          (5.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] + u[k+2][j][i][m]);
      }
    }
  }

  k = 2;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
          (-4.0*u[k-1][j][i][m] + 6.0*u[k][j][i][m] -
            4.0*u[k+1][j][i][m] + u[k+2][j][i][m]);
      }
    }
  }

  for (k = 3; k <= nz2-2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
            ( u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
            6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] + 
              u[k+2][j][i][m] );
        }
      }
    }
  }

  k = nz2-1;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
          6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] );
      }
    }
  }

  k = nz2;
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 5.0*u[k][j][i][m] );
      }
    }
  }
  if (timeron) timer_stop(t_rhsz);

  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
        }
      }
    }
  }
  if (timeron) timer_stop(t_rhs);
}

void compute_rhs_gpu(int t, int st, int ed, int st2, int ed2)
{
  if (timeron) timer_start(t_rhs);

  //---------------------------------------------------------------------
  // compute_rhs0 kernel
  //---------------------------------------------------------------------
  int compute_rhs0_base_k = st2;
  int compute_rhs0_offset_k = 0;
  int compute_rhs0_gws_k = ed2 - st2 + 1;

  err_code  = clSetKernelArg(k_compute_rhs[0], 0, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[0], 1, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[0], 2, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[0], 3, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[0], 4, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[0], 5, sizeof(cl_mem), &buf_us);
  err_code |= clSetKernelArg(k_compute_rhs[0], 6, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_compute_rhs[0], 7, sizeof(cl_mem), &buf_ws);
  err_code |= clSetKernelArg(k_compute_rhs[0], 8, sizeof(cl_mem), &buf_qs);
  err_code |= clSetKernelArg(k_compute_rhs[0], 9, sizeof(cl_mem), &buf_square);
  err_code |= clSetKernelArg(k_compute_rhs[0], 10, sizeof(cl_mem), &buf_speed);
  err_code |= clSetKernelArg(k_compute_rhs[0], 11, sizeof(cl_mem), &buf_rho_i);
  err_code |= clSetKernelArg(k_compute_rhs[0], 12, sizeof(int), &compute_rhs0_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[0], 13, sizeof(int), &compute_rhs0_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[0], 14, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[0], 15, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[0], 16, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs0_gws[] = {compute_rhs0_gws_k};
    size_t compute_rhs0_lws[] = {1};
    compute_rhs0_gws[0] = clu_RoundWorkSize(compute_rhs0_gws[0], compute_rhs0_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[0],
                                      1, NULL,
                                      compute_rhs0_gws,
                                      compute_rhs0_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs0_gws[] = {nx2+2, ny2+2, compute_rhs0_gws_k};
    size_t compute_rhs0_lws[] = {16, 16, 1};
    compute_rhs0_gws[0] = clu_RoundWorkSize(compute_rhs0_gws[0], compute_rhs0_lws[0]);
    compute_rhs0_gws[1] = clu_RoundWorkSize(compute_rhs0_gws[1], compute_rhs0_lws[1]);
    compute_rhs0_gws[2] = clu_RoundWorkSize(compute_rhs0_gws[2], compute_rhs0_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[0],
                                      3, NULL,
                                      compute_rhs0_gws,
                                      compute_rhs0_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // compute_rhs1 kernel
  //---------------------------------------------------------------------
  int compute_rhs1_base_k = st2;
  int compute_rhs1_offset_k = 0;
  int compute_rhs1_gws_k = ed2 - st2 + 1;

  err_code  = clSetKernelArg(k_compute_rhs[1], 0, sizeof(cl_mem), &buf_forcing0);
  err_code |= clSetKernelArg(k_compute_rhs[1], 1, sizeof(cl_mem), &buf_forcing1);
  err_code |= clSetKernelArg(k_compute_rhs[1], 2, sizeof(cl_mem), &buf_forcing2);
  err_code |= clSetKernelArg(k_compute_rhs[1], 3, sizeof(cl_mem), &buf_forcing3);
  err_code |= clSetKernelArg(k_compute_rhs[1], 4, sizeof(cl_mem), &buf_forcing4);
  err_code |= clSetKernelArg(k_compute_rhs[1], 5, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[1], 6, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[1], 7, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[1], 8, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[1], 9, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[1], 10, sizeof(int), &compute_rhs1_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[1], 11, sizeof(int), &compute_rhs1_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[1], 12, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[1], 13, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[1], 14, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs1_gws[] = {compute_rhs1_gws_k};
    size_t compute_rhs1_lws[] = {1};
    compute_rhs1_gws[0] = clu_RoundWorkSize(compute_rhs1_gws[0], compute_rhs1_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[1],
                                      1, NULL,
                                      compute_rhs1_gws,
                                      compute_rhs1_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs1_gws[] = {nx2+2, ny2+2, compute_rhs1_gws_k};
    size_t compute_rhs1_lws[] = {16, 16, 1};
    compute_rhs1_gws[0] = clu_RoundWorkSize(compute_rhs1_gws[0], compute_rhs1_lws[0]);
    compute_rhs1_gws[1] = clu_RoundWorkSize(compute_rhs1_gws[1], compute_rhs1_lws[1]);
    compute_rhs1_gws[2] = clu_RoundWorkSize(compute_rhs1_gws[2], compute_rhs1_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[1],
                                      3, NULL,
                                      compute_rhs1_gws,
                                      compute_rhs1_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_start(t_rhsx);

  //---------------------------------------------------------------------
  // compute_rhs2 kernel
  //---------------------------------------------------------------------
  int compute_rhs2_base_k = st2;
  int compute_rhs2_offset_k = st - st2;
  int compute_rhs2_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[2], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[2], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[2], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[2], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[2], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[2], 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[2], 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[2], 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[2], 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[2], 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[2], 10, sizeof(cl_mem), &buf_us);
  err_code |= clSetKernelArg(k_compute_rhs[2], 11, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_compute_rhs[2], 12, sizeof(cl_mem), &buf_ws);
  err_code |= clSetKernelArg(k_compute_rhs[2], 13, sizeof(cl_mem), &buf_qs);
  err_code |= clSetKernelArg(k_compute_rhs[2], 14, sizeof(cl_mem), &buf_square);
  err_code |= clSetKernelArg(k_compute_rhs[2], 15, sizeof(cl_mem), &buf_rho_i);
  err_code |= clSetKernelArg(k_compute_rhs[2], 16, sizeof(int), &compute_rhs2_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[2], 17, sizeof(int), &compute_rhs2_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[2], 18, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[2], 19, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[2], 20, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs2_gws[] = {compute_rhs2_gws_k};
    size_t compute_rhs2_lws[] = {1};
    compute_rhs2_gws[0] = clu_RoundWorkSize(compute_rhs2_gws[0], compute_rhs2_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[2],
                                      1, NULL,
                                      compute_rhs2_gws,
                                      compute_rhs2_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs2_gws[] = {nx2, ny2, compute_rhs2_gws_k};
    size_t compute_rhs2_lws[] = {16, 16, 1};
    compute_rhs2_gws[0] = clu_RoundWorkSize(compute_rhs2_gws[0], compute_rhs2_lws[0]);
    compute_rhs2_gws[1] = clu_RoundWorkSize(compute_rhs2_gws[1], compute_rhs2_lws[1]);
    compute_rhs2_gws[2] = clu_RoundWorkSize(compute_rhs2_gws[2], compute_rhs2_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[2],
                                      3, NULL,
                                      compute_rhs2_gws,
                                      compute_rhs2_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------
  
  //---------------------------------------------------------------------
  // compute_rhs3 kernel
  //---------------------------------------------------------------------
  int compute_rhs3_base_k = st2;
  int compute_rhs3_offset_k = st - st2;
  int compute_rhs3_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[3], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[3], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[3], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[3], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[3], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[3], 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[3], 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[3], 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[3], 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[3], 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[3], 10, sizeof(int), &compute_rhs3_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[3], 11, sizeof(int), &compute_rhs3_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[3], 12, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[3], 13, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[3], 14, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs3_gws[] = {compute_rhs3_gws_k};
    size_t compute_rhs3_lws[] = {1};
    compute_rhs3_gws[0] = clu_RoundWorkSize(compute_rhs3_gws[0], compute_rhs3_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[3],
                                      1, NULL,
                                      compute_rhs3_gws,
                                      compute_rhs3_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs3_gws[] = {nx2, ny2, compute_rhs3_gws_k};
    size_t compute_rhs3_lws[] = {16, 16, 1};
    compute_rhs3_gws[0] = clu_RoundWorkSize(compute_rhs3_gws[0], compute_rhs3_lws[0]);
    compute_rhs3_gws[1] = clu_RoundWorkSize(compute_rhs3_gws[1], compute_rhs3_lws[1]);
    compute_rhs3_gws[2] = clu_RoundWorkSize(compute_rhs3_gws[2], compute_rhs3_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[3],
                                      3, NULL,
                                      compute_rhs3_gws,
                                      compute_rhs3_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_rhsx);

  if (timeron) timer_start(t_rhsy);

  //---------------------------------------------------------------------
  // compute_rhs4 kernel
  //---------------------------------------------------------------------
  int compute_rhs4_base_k = st2;
  int compute_rhs4_offset_k = st - st2;
  int compute_rhs4_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[4], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[4], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[4], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[4], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[4], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[4], 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[4], 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[4], 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[4], 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[4], 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[4], 10, sizeof(cl_mem), &buf_us);
  err_code |= clSetKernelArg(k_compute_rhs[4], 11, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_compute_rhs[4], 12, sizeof(cl_mem), &buf_ws);
  err_code |= clSetKernelArg(k_compute_rhs[4], 13, sizeof(cl_mem), &buf_qs);
  err_code |= clSetKernelArg(k_compute_rhs[4], 14, sizeof(cl_mem), &buf_square);
  err_code |= clSetKernelArg(k_compute_rhs[4], 15, sizeof(cl_mem), &buf_rho_i);
  err_code |= clSetKernelArg(k_compute_rhs[4], 16, sizeof(int), &compute_rhs4_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[4], 17, sizeof(int), &compute_rhs4_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[4], 18, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[4], 19, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[4], 20, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs4_gws[] = {compute_rhs4_gws_k};
    size_t compute_rhs4_lws[] = {1};
    compute_rhs4_gws[0] = clu_RoundWorkSize(compute_rhs4_gws[0], compute_rhs4_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[4],
                                      1, NULL,
                                      compute_rhs4_gws,
                                      compute_rhs4_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs4_gws[] = {nx2, ny2, compute_rhs4_gws_k};
    size_t compute_rhs4_lws[] = {16, 16, 1};
    compute_rhs4_gws[0] = clu_RoundWorkSize(compute_rhs4_gws[0], compute_rhs4_lws[0]);
    compute_rhs4_gws[1] = clu_RoundWorkSize(compute_rhs4_gws[1], compute_rhs4_lws[1]);
    compute_rhs4_gws[2] = clu_RoundWorkSize(compute_rhs4_gws[2], compute_rhs4_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[4],
                                      3, NULL,
                                      compute_rhs4_gws,
                                      compute_rhs4_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // compute_rhs5 kernel
  //---------------------------------------------------------------------
  int compute_rhs5_base_k = st2;
  int compute_rhs5_offset_k = st - st2;
  int compute_rhs5_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[5], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[5], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[5], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[5], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[5], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[5], 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[5], 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[5], 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[5], 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[5], 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[5], 10, sizeof(int), &compute_rhs5_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[5], 11, sizeof(int), &compute_rhs5_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[5], 12, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[5], 13, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[5], 14, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs5_gws[] = {compute_rhs5_gws_k};
    size_t compute_rhs5_lws[] = {1};
    compute_rhs5_gws[0] = clu_RoundWorkSize(compute_rhs5_gws[0], compute_rhs5_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[5],
                                      1, NULL,
                                      compute_rhs5_gws,
                                      compute_rhs5_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs5_gws[] = {nx2, ny2, compute_rhs5_gws_k};
    size_t compute_rhs5_lws[] = {16, 16, 1};
    compute_rhs5_gws[0] = clu_RoundWorkSize(compute_rhs5_gws[0], compute_rhs5_lws[0]);
    compute_rhs5_gws[1] = clu_RoundWorkSize(compute_rhs5_gws[1], compute_rhs5_lws[1]);
    compute_rhs5_gws[2] = clu_RoundWorkSize(compute_rhs5_gws[2], compute_rhs5_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[5],
                                      3, NULL,
                                      compute_rhs5_gws,
                                      compute_rhs5_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_rhsy);

  if (timeron) timer_start(t_rhsz);

  //---------------------------------------------------------------------
  // compute_rhs6 kernel
  //---------------------------------------------------------------------
  int compute_rhs6_base_k = st2;
  int compute_rhs6_offset_k = st - st2;
  int compute_rhs6_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[6], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[6], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[6], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[6], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[6], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[6], 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[6], 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[6], 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[6], 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[6], 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[6], 10, sizeof(cl_mem), &buf_us);
  err_code |= clSetKernelArg(k_compute_rhs[6], 11, sizeof(cl_mem), &buf_vs);
  err_code |= clSetKernelArg(k_compute_rhs[6], 12, sizeof(cl_mem), &buf_ws);
  err_code |= clSetKernelArg(k_compute_rhs[6], 13, sizeof(cl_mem), &buf_qs);
  err_code |= clSetKernelArg(k_compute_rhs[6], 14, sizeof(cl_mem), &buf_square);
  err_code |= clSetKernelArg(k_compute_rhs[6], 15, sizeof(cl_mem), &buf_rho_i);
  err_code |= clSetKernelArg(k_compute_rhs[6], 16, sizeof(int), &compute_rhs6_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[6], 17, sizeof(int), &compute_rhs6_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[6], 18, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[6], 19, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[6], 20, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs6_gws[] = {compute_rhs6_gws_k};
    size_t compute_rhs6_lws[] = {1};
    compute_rhs6_gws[0] = clu_RoundWorkSize(compute_rhs6_gws[0], compute_rhs6_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[6],
                                      1, NULL,
                                      compute_rhs6_gws,
                                      compute_rhs6_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs6_gws[] = {nx2, ny2, compute_rhs6_gws_k};
    size_t compute_rhs6_lws[] = {16, 16, 1};
    compute_rhs6_gws[0] = clu_RoundWorkSize(compute_rhs6_gws[0], compute_rhs6_lws[0]);
    compute_rhs6_gws[1] = clu_RoundWorkSize(compute_rhs6_gws[1], compute_rhs6_lws[1]);
    compute_rhs6_gws[2] = clu_RoundWorkSize(compute_rhs6_gws[2], compute_rhs6_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[6],
                                      3, NULL,
                                      compute_rhs6_gws,
                                      compute_rhs6_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------
 
  //---------------------------------------------------------------------
  // compute_rhs7 kernel
  //---------------------------------------------------------------------
  int compute_rhs7_base_k = st2;
  int compute_rhs7_offset_k = st - st2;
  int compute_rhs7_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[7], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[7], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[7], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[7], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[7], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[7], 5, sizeof(cl_mem), &buf_u0);
  err_code |= clSetKernelArg(k_compute_rhs[7], 6, sizeof(cl_mem), &buf_u1);
  err_code |= clSetKernelArg(k_compute_rhs[7], 7, sizeof(cl_mem), &buf_u2);
  err_code |= clSetKernelArg(k_compute_rhs[7], 8, sizeof(cl_mem), &buf_u3);
  err_code |= clSetKernelArg(k_compute_rhs[7], 9, sizeof(cl_mem), &buf_u4);
  err_code |= clSetKernelArg(k_compute_rhs[7], 10, sizeof(int), &compute_rhs7_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[7], 11, sizeof(int), &compute_rhs7_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[7], 12, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[7], 13, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[7], 14, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs7_gws[] = {compute_rhs7_gws_k};
    size_t compute_rhs7_lws[] = {1};
    compute_rhs7_gws[0] = clu_RoundWorkSize(compute_rhs7_gws[0], compute_rhs7_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[7],
                                      1, NULL,
                                      compute_rhs7_gws,
                                      compute_rhs7_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs7_gws[] = {nx2, ny2, compute_rhs7_gws_k};
    size_t compute_rhs7_lws[] = {16, 16, 1};
    compute_rhs7_gws[0] = clu_RoundWorkSize(compute_rhs7_gws[0], compute_rhs7_lws[0]);
    compute_rhs7_gws[1] = clu_RoundWorkSize(compute_rhs7_gws[1], compute_rhs7_lws[1]);
    compute_rhs7_gws[2] = clu_RoundWorkSize(compute_rhs7_gws[2], compute_rhs7_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[7],
                                      3, NULL,
                                      compute_rhs7_gws,
                                      compute_rhs7_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_rhsz);

  //---------------------------------------------------------------------
  // compute_rhs8 kernel
  //---------------------------------------------------------------------
  int compute_rhs8_base_k = st2;
  int compute_rhs8_offset_k = st - st2;
  int compute_rhs8_gws_k = ed - st + 1;

  err_code  = clSetKernelArg(k_compute_rhs[8], 0, sizeof(cl_mem), &buf_rhs0);
  err_code |= clSetKernelArg(k_compute_rhs[8], 1, sizeof(cl_mem), &buf_rhs1);
  err_code |= clSetKernelArg(k_compute_rhs[8], 2, sizeof(cl_mem), &buf_rhs2);
  err_code |= clSetKernelArg(k_compute_rhs[8], 3, sizeof(cl_mem), &buf_rhs3);
  err_code |= clSetKernelArg(k_compute_rhs[8], 4, sizeof(cl_mem), &buf_rhs4);
  err_code |= clSetKernelArg(k_compute_rhs[8], 5, sizeof(int), &compute_rhs8_base_k);
  err_code |= clSetKernelArg(k_compute_rhs[8], 6, sizeof(int), &compute_rhs8_offset_k);
  err_code |= clSetKernelArg(k_compute_rhs[8], 7, sizeof(int), &nx2);
  err_code |= clSetKernelArg(k_compute_rhs[8], 8, sizeof(int), &ny2);
  err_code |= clSetKernelArg(k_compute_rhs[8], 9, sizeof(int), &nz2);
  clu_CheckError(err_code, "clSetKernelArg()");
  
  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    size_t compute_rhs8_gws[] = {compute_rhs8_gws_k};
    size_t compute_rhs8_lws[] = {1};
    compute_rhs8_gws[0] = clu_RoundWorkSize(compute_rhs8_gws[0], compute_rhs8_lws[0]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[8],
                                      1, NULL,
                                      compute_rhs8_gws,
                                      compute_rhs8_lws,
                                      0, NULL, NULL);
  }
  else {
    size_t compute_rhs8_gws[] = {nx2, ny2, compute_rhs8_gws_k};
    size_t compute_rhs8_lws[] = {16, 16, 1};
    compute_rhs8_gws[0] = clu_RoundWorkSize(compute_rhs8_gws[0], compute_rhs8_lws[0]);
    compute_rhs8_gws[1] = clu_RoundWorkSize(compute_rhs8_gws[1], compute_rhs8_lws[1]);
    compute_rhs8_gws[2] = clu_RoundWorkSize(compute_rhs8_gws[2], compute_rhs8_lws[2]);

    err_code = clEnqueueNDRangeKernel(cmd_queue[0],
                                      k_compute_rhs[8],
                                      3, NULL,
                                      compute_rhs8_gws,
                                      compute_rhs8_lws,
                                      0, NULL, NULL);
  }

  clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  //---------------------------------------------------------------------

  if (timeron) timer_stop(t_rhs);
}
