//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB BT code. This OpenCL C  //
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

//---------------------------------------------------------------------
// compute the reciprocal of density, and the kinetic energy, 
// and the speed of sound.
//---------------------------------------------------------------------

__kernel void rhs_datagen_parallel(__global double *m_rho_i, 
                                   __global double *m_us, 
                                   __global double *m_vs, 
                                   __global double *m_ws, 
                                   __global double *m_qs, 
                                   __global double *m_square,
                                   __global double *m_u,
                                   int gp0, int gp1, int gp2,
                                   int copy_buffer_base, 
                                   int copy_num_item, 
                                   int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);
  if (k >= copy_num_item || j > gp1-1 || i > gp0-1) 
    return;

  k += copy_buffer_base;
  
  // pointer casting  
  __global double (*rho_i)[JMAXP+1][IMAXP+1]
    = (__global double (*)[JMAXP+1][IMAXP+1])m_rho_i;

  __global double (*us)[JMAXP+1][IMAXP+1]
    = (__global double(*)[JMAXP+1][IMAXP+1])m_us;

  __global double (*vs)[JMAXP+1][IMAXP+1]
    = (__global double(*)[JMAXP+1][IMAXP+1])m_vs;

  __global double (*ws)[JMAXP+1][IMAXP+1]
    = (__global double(*)[JMAXP+1][IMAXP+1])m_ws;

  __global double (*qs)[JMAXP+1][IMAXP+1]
    = (__global double(*)[JMAXP+1][IMAXP+1])m_qs;

  __global double (*square)[JMAXP+1][IMAXP+1]
    = (__global double(*)[JMAXP+1][IMAXP+1])m_square;

  __global double (*u)[JMAXP+1][IMAXP+1][5]
    = (__global double(*)[JMAXP+1][IMAXP+1][5])m_u;

  double rho_inv;
  double t_u[4];
  int m;

  for (m = 0; m < 4; m++) {
    t_u[m] = u[k][j][i][m];
  }

  rho_inv = 1.0/t_u[0];
  rho_i[k][j][i] = rho_inv;
  us[k][j][i] = t_u[1] * rho_inv;
  vs[k][j][i] = t_u[2] * rho_inv;
  ws[k][j][i] = t_u[3] * rho_inv;
  square[k][j][i] = 0.5* (
      t_u[1]*t_u[1] + 
      t_u[2]*t_u[2] +
      t_u[3]*t_u[3] ) * rho_inv;
  qs[k][j][i] = square[k][j][i] * rho_inv;
}

//---------------------------------------------------------------------
// copy the exact forcing term to the right hand side;  because 
// this forcing term is known, we can store it on the whole grid
// including the boundary                   
// Input(write buffer) - m_forcing
// Output(read buffer) - m_rhs
//---------------------------------------------------------------------
__kernel void rhs1_parallel(__global double *m_rhs, 
                            __global double *m_forcing,
                            int gp0, int gp1, int gp2,
                            int work_base, 
                            int work_num_item, 
                            int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int t_i = get_global_id(0);
  int i = t_i / 5;
  int m = t_i % 5;

  if (k + work_base > gp2-1 || k >= work_num_item || j > gp1-1 || i > gp0-1) return;

  if (split_flag) k += 2;
  else k += work_base;

  __global double (*rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double(*)[JMAXP+1][IMAXP+1][5])m_rhs;
  __global double (*forcing)[JMAXP+1][IMAXP+1][5]
    = (__global double(*)[JMAXP+1][IMAXP+1][5])m_forcing;

        rhs[k][j][i][m] = forcing[k][j][i][m];
}

//---------------------------------------------------------------------
// compute xi-direction fluxes 
// Input(write buffer) - us, vs, ws, qs, square, rhs, rho_i, u
// Output(read buffer) - rhs
//---------------------------------------------------------------------
__kernel void rhsx1_parallel(__global double *m_us, 
                             __global double *m_vs,
                             __global double *m_ws, 
                             __global double *m_qs,
                             __global double *m_rho_i, 
                             __global double *m_square,
                             __global double *m_u, 
                             __global double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dx1tx1, double dx2tx1, 
                             double dx3tx1, double dx4tx1, 
                             double dx5tx1,
                             double xxcon2, double xxcon3, 
                             double xxcon4, double xxcon5,
                             double c1, double c2, 
                             double tx2, double con43, 
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  // caution : 1 <= k <= gp-2
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int i = get_global_id(0)+1;
  
  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;

  double uijk, up1, um1;

  // pointer casting 
  
  __global double (* us)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_us;
  __global double (* vs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_vs;  
  __global double (* ws)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_ws;  
  __global double (* qs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_qs;  
  __global double (* rho_i)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_rho_i; 
  __global double (* square)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_square; 
  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_u; 
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_rhs; 



  uijk = us[k][j][i];
  up1  = us[k][j][i+1];
  um1  = us[k][j][i-1];

  rhs[k][j][i][0] = rhs[k][j][i][0] + dx1tx1 * 
    (u[k][j][i+1][0] - 2.0*u[k][j][i][0] + 
     u[k][j][i-1][0]) -
    tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]);

  rhs[k][j][i][1] = rhs[k][j][i][1] + dx2tx1 * 
    (u[k][j][i+1][1] - 2.0*u[k][j][i][1] + 
     u[k][j][i-1][1]) +
    xxcon2*con43 * (up1 - 2.0*uijk + um1) -
    tx2 * (u[k][j][i+1][1]*up1 - 
        u[k][j][i-1][1]*um1 +
        (u[k][j][i+1][4]- square[k][j][i+1]-
         u[k][j][i-1][4]+ square[k][j][i-1])*
        c2);

  rhs[k][j][i][2] = rhs[k][j][i][2] + dx3tx1 * 
    (u[k][j][i+1][2] - 2.0*u[k][j][i][2] +
     u[k][j][i-1][2]) +
    xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
        vs[k][j][i-1]) -
    tx2 * (u[k][j][i+1][2]*up1 - 
        u[k][j][i-1][2]*um1);

  rhs[k][j][i][3] = rhs[k][j][i][3] + dx4tx1 * 
    (u[k][j][i+1][3] - 2.0*u[k][j][i][3] +
     u[k][j][i-1][3]) +
    xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
        ws[k][j][i-1]) -
    tx2 * (u[k][j][i+1][3]*up1 - 
        u[k][j][i-1][3]*um1);

  rhs[k][j][i][4] = rhs[k][j][i][4] + dx5tx1 * 
    (u[k][j][i+1][4] - 2.0*u[k][j][i][4] +
     u[k][j][i-1][4]) +
    xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
        qs[k][j][i-1]) +
    xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
        um1*um1) +
    xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] - 
        2.0*u[k][j][i][4]*rho_i[k][j][i] +
        u[k][j][i-1][4]*rho_i[k][j][i-1]) -
    tx2 * ( (c1*u[k][j][i+1][4] - 
          c2*square[k][j][i+1])*up1 -
        (c1*u[k][j][i-1][4] - 
         c2*square[k][j][i-1])*um1 );

}

__kernel void rhsx2_parallel(__global double *m_u, 
                             __global double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int t_i = get_global_id(0);
  int i = t_i / 5 + 1;
  int m = t_i % 5; 
  
  if (k+work_base < 1 || k+work_base > gp2 - 2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;


  // pointer casting 
  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_u; 
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_rhs; 

  if (i == 1) {
      rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
        ( 5.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] +
          u[k][j][i+2][m]);
  }
  else if (i == 2) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
        (-4.0*u[k][j][i-1][m] + 6.0*u[k][j][i][m] -
         4.0*u[k][j][i+1][m] + u[k][j][i+2][m]);
  }
  else if (i == gp0-3) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
        ( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
          6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] );
  }
  else if (i == gp0-2) {
      rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
        ( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] +
          5.0*u[k][j][i][m] );
  }
  else {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
          (  u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
             6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] + 
             u[k][j][i+2][m] );
  }
}

//---------------------------------------------------------------------
// compute eta-direction fluxes 
//---------------------------------------------------------------------
__kernel void rhsy1_parallel(__global double *m_us, 
                             __global double *m_vs,
                             __global double *m_ws, 
                             __global double *m_qs,
                             __global double *m_rho_i, 
                             __global double *m_square,
                             __global double *m_u, 
                             __global double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dy1ty1, double dy2ty1, 
                             double dy3ty1, double dy4ty1, 
                             double dy5ty1,
                             double yycon2, double yycon3, 
                             double yycon4, double yycon5,
                             double c1, double c2, 
                             double ty2, double con43, 
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int i = get_global_id(0)+1;

  int l_j = get_local_id(1)+1;
  int l_size = get_local_size(1);

  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;
  double vijk, vp1, vm1;


  // pointer casting 
  __global double (* us)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_us;
  __global double (* vs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_vs;  
  __global double (* ws)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_ws;  
  __global double (* qs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_qs;  
  __global double (* rho_i)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_rho_i; 
  __global double (* square)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_square; 
  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_u; 
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_rhs; 

  vijk = vs[k][j][i];
  vp1  = vs[k][j+1][i];
  vm1  = vs[k][j-1][i];
  rhs[k][j][i][0] = rhs[k][j][i][0] + dy1ty1 * 
    (u[k][j+1][i][0] - 2.0*u[k][j][i][0] + 
     u[k][j-1][i][0]) -
    ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]);
  rhs[k][j][i][1] = rhs[k][j][i][1] + dy2ty1 * 
    (u[k][j+1][i][1] - 2.0*u[k][j][i][1] + 
     u[k][j-1][i][1]) +
    yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + 
        us[k][j-1][i]) -
    ty2 * (u[k][j+1][i][1]*vp1 - 
        u[k][j-1][i][1]*vm1);
  rhs[k][j][i][2] = rhs[k][j][i][2] + dy3ty1 * 
    (u[k][j+1][i][2] - 2.0*u[k][j][i][2] + 
     u[k][j-1][i][2]) +
    yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
    ty2 * (u[k][j+1][i][2]*vp1 - 
        u[k][j-1][i][2]*vm1 +
        (u[k][j+1][i][4] - square[k][j+1][i] - 
         u[k][j-1][i][4] + square[k][j-1][i])
        *c2);
  rhs[k][j][i][3] = rhs[k][j][i][3] + dy4ty1 * 
    (u[k][j+1][i][3] - 2.0*u[k][j][i][3] + 
     u[k][j-1][i][3]) +
    yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + 
        ws[k][j-1][i]) -
    ty2 * (u[k][j+1][i][3]*vp1 - 
        u[k][j-1][i][3]*vm1);
  rhs[k][j][i][4] = rhs[k][j][i][4] + dy5ty1 * 
    (u[k][j+1][i][4] - 2.0*u[k][j][i][4] + 
     u[k][j-1][i][4]) +
    yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + 
        qs[k][j-1][i]) +
    yycon4 * (vp1*vp1       - 2.0*vijk*vijk + 
        vm1*vm1) +
    yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] - 
        2.0*u[k][j][i][4]*rho_i[k][j][i] +
        u[k][j-1][i][4]*rho_i[k][j-1][i]) -
    ty2 * ((c1*u[k][j+1][i][4] - 
          c2*square[k][j+1][i]) * vp1 -
        (c1*u[k][j-1][i][4] - 
         c2*square[k][j-1][i]) * vm1);


}

__kernel void rhsy2_parallel(__global double *m_u, 
                             __global double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int t_i = get_global_id(0);
  int i = t_i / 5 + 1;
  int m = t_i % 5;

  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;



  // pointer casting 
  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_u; 
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_rhs; 

  if (j == 1) {
    rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
      ( 5.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] +
        u[k][j+2][i][m]);
  }
  else if (j == 2) {
    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
      (-4.0*u[k][j-1][i][m] + 6.0*u[k][j][i][m] -
       4.0*u[k][j+1][i][m] + u[k][j+2][i][m]);
  }
  else if (j == gp1-3) {
    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
      ( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
        6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] );
  }
  else if (j == gp1-2) {
    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
      ( u[k][j-2][i][m] - 4.*u[k][j-1][i][m] +
        5.*u[k][j][i][m] );
  }
  else {
    rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
      (  u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
         6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + 
         u[k][j+2][i][m] );
  }

}

__kernel void rhsz1_parallel(__global double *m_us, 
                             __global double *m_vs,
                             __global double *m_ws, 
                             __global double *m_qs,
                             __global double *m_rho_i, 
                             __global double *m_square,
                             __global double *m_u, 
                             __global double *m_rhs,
                             int gp0, int gp1, int gp2,
                             double dz1tz1, double dz2tz1, 
                             double dz3tz1, double dz4tz1, 
                             double dz5tz1,
                             double zzcon2, double zzcon3, 
                             double zzcon4, double zzcon5,
                             double c1, double c2, 
                             double tz2, double con43,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int i = get_global_id(0)+1;

  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;
  double wijk, wp1, wm1;

  __global double (* us)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_us;
  __global double (* vs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_vs;  
  __global double (* ws)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_ws;  
  __global double (* qs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_qs;  
  __global double (* rho_i)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_rho_i; 
  __global double (* square)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_square; 
  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_u; 
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_rhs;


  wijk = ws[k][j][i];
  wp1  = ws[k+1][j][i];
  wm1  = ws[k-1][j][i];

  rhs[k][j][i][0] = rhs[k][j][i][0] + dz1tz1 * 
    (u[k+1][j][i][0] - 2.0*u[k][j][i][0] + 
     u[k-1][j][i][0]) -
    tz2 * (u[k+1][j][i][3] - u[k-1][j][i][3]);
  rhs[k][j][i][1] = rhs[k][j][i][1] + dz2tz1 * 
    (u[k+1][j][i][1] - 2.0*u[k][j][i][1] + 
     u[k-1][j][i][1]) +
    zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + 
        us[k-1][j][i]) -
    tz2 * (u[k+1][j][i][1]*wp1 - 
        u[k-1][j][i][1]*wm1);
  rhs[k][j][i][2] = rhs[k][j][i][2] + dz3tz1 * 
    (u[k+1][j][i][2] - 2.0*u[k][j][i][2] + 
     u[k-1][j][i][2]) +
    zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + 
        vs[k-1][j][i]) -
    tz2 * (u[k+1][j][i][2]*wp1 - 
        u[k-1][j][i][2]*wm1);
  rhs[k][j][i][3] = rhs[k][j][i][3] + dz4tz1 * 
    (u[k+1][j][i][3] - 2.0*u[k][j][i][3] + 
     u[k-1][j][i][3]) +
    zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
    tz2 * (u[k+1][j][i][3]*wp1 - 
        u[k-1][j][i][3]*wm1 +
        (u[k+1][j][i][4] - square[k+1][j][i] - 
         u[k-1][j][i][4] + square[k-1][j][i])
        *c2);
  rhs[k][j][i][4] = rhs[k][j][i][4] + dz5tz1 * 
    (u[k+1][j][i][4] - 2.0*u[k][j][i][4] + 
     u[k-1][j][i][4]) +
    zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + 
        qs[k-1][j][i]) +
    zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + 
        wm1*wm1) +
    zzcon5 * (u[k+1][j][i][4]*rho_i[k+1][j][i] - 
        2.0*u[k][j][i][4]*rho_i[k][j][i] +
        u[k-1][j][i][4]*rho_i[k-1][j][i]) -
    tz2 * ( (c1*u[k+1][j][i][4] - 
          c2*square[k+1][j][i])*wp1 -
        (c1*u[k-1][j][i][4] - 
         c2*square[k-1][j][i])*wm1);

}

__kernel void rhsz2_parallel(__global double *m_u, 
                             __global double *m_rhs,
                             int gp0, int gp1, int gp2, 
                             double dssp,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int t_i = get_global_id(0);
  int i = t_i / 5 + 1;
  int m = t_i % 5;
  int original_k;
  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  original_k = k + work_base;

  if (split_flag) k += 2;
  else k += work_base;

  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_u; 
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5]) m_rhs;

  if (original_k == 1) {
        rhs[k][j][i][m] = rhs[k][j][i][m]- dssp * 
          ( 5.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] +
            u[k+2][j][i][m]);
  }
  else if (original_k == 2) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
          (-4.0*u[k-1][j][i][m] + 6.0*u[k][j][i][m] -
           4.0*u[k+1][j][i][m] + u[k+2][j][i][m]);
  }
  else if (original_k == gp2-3) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
            6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] );
  }
  else if (original_k == gp2-2) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - dssp *
          ( u[k-2][j][i][m] - 4.*u[k-1][j][i][m] +
            5.*u[k][j][i][m] );
  }
  else {
          rhs[k][j][i][m] = rhs[k][j][i][m] - dssp * 
            (  u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
               6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] + 
               u[k+2][j][i][m] );
  }
}

__kernel void rhs2_parallel(__global double *m_rhs,
                            int gp0, int gp1, int gp2, 
                            double dt,
                            int work_base, 
                            int work_num_item, 
                            int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+1;
  int t_i = get_global_id(0);
  int i = t_i / 5 + 1;
  int m = t_i % 5;

  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;


  __global double ( * rhs ) [JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5])m_rhs;

  rhs[k][j][i][m] = rhs[k][j][i][m] * dt;
}
