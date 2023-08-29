//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB LU code. This CUDA® C  //
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

#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "applu.incl"
extern "C" {
#include "timers.h"
}


__device__
static void flux1_to_rsd(double rsd[5], 
                         double flux1m1[5], 
                         double flux1p1[5], 
                         double t2)
{
  int m;
  for(m = 0; m < 5; m++){
    rsd[m] = rsd[m] - t2 * (flux1p1[m] - flux1m1[m]);
  }
}

__device__
static void flux2_to_rsd(double rsd[5], 
                         double p_um1[5], 
                         double p_u[5], 
                         double p_up1[5], 
                         double flux2[5], 
                         double flux2p1[5], 
                         double d[5], 
                         double t1, 
                         double t3)
{

  rsd[0] = rsd[0] + d[0] * t1 * ( p_um1[0] - 2.0*p_u[0] + p_up1[0] );

  rsd[1] = rsd[1] + t3 * C3 * C4 * ( flux2p1[1] - flux2[1] )
    + d[1] * t1 * ( p_um1[1] - 2.0*p_u[1] + p_up1[1] );

  rsd[2] = rsd[2] + t3 * C3 * C4 * ( flux2p1[2] - flux2[2] )
    + d[2] * t1 * ( p_um1[2] - 2.0*p_u[2] + p_up1[2] );

  rsd[3] = rsd[3] + t3 * C3 * C4 * ( flux2p1[3] - flux2[3] )
    + d[3] * t1 * ( p_um1[3] - 2.0*p_u[3] + p_up1[3] );

  rsd[4] = rsd[4] + t3 * C3 * C4 * ( flux2p1[4] - flux2[4] )
    + d[4] * t1 * ( p_um1[4] - 2.0*p_u[4] + p_up1[4] );
}



__global__
void k_rhs1_gmem(double *m_rsd, 
                 double * m_frct, 
                 int nx, int ny, int nz,
                 int work_base, 
                 int work_num_item, 
                 int split_flag)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int t_i = blockDim.x * blockIdx.x + threadIdx.x;

  int i = t_i / 5;
  int m = t_i % 5;
  if(k+work_base >= nz || k >= work_num_item || j >= ny || i >= nx) return;
  
  if(split_flag) k += 2;
  else k += work_base;




  double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;
  double (* frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_frct;

  rsd[k][j][i][m] = - frct[k][j][i][m];

}



__global__
void k_rhs1_datagen_gmem(double *m_u, 
                         double *m_rho_i,
                         double *m_qs, 
                         int nx, int ny, int nz,
                         int u_copy_buffer_base, 
                         int u_copy_num_item)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(k >= u_copy_num_item || j >= ny || i >= nx) return;

  k += u_copy_buffer_base;

  double tmp;

  double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i;
  double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_qs;

  tmp = 1.0 / u[k][j][i][0];
  rho_i[k][j][i] = tmp;
  qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
      + u[k][j][i][2] * u[k][j][i][2]
      + u[k][j][i][3] * u[k][j][i][3] ) * tmp;
}


//---------------------------------------------------------------------
// xi-direction flux differences
//---------------------------------------------------------------------

__device__
static void compute_flux1_x(double *flux1, 
                            double *p_u, 
                            double q, 
                            double u21)
{
  flux1[0] = p_u[1];
  flux1[1] = p_u[1] * u21 + C2 * ( p_u[4] - q );
  flux1[2] = p_u[2] * u21;
  flux1[3] = p_u[3] * u21;
  flux1[4] = ( C1 * p_u[4] - C2 * q ) * u21;

}

__device__
static void compute_flux2_x(double *flux2, 
                            double *p_um1, 
                            double *p_u, 
                            double tmpm1, 
                            double tmp, 
                            double tx3)
{
  double u21i, u31i, u41i, u51i, u21im1, u31im1, u41im1, u51im1;

  u21i = tmp * p_u[1];
  u31i = tmp * p_u[2];
  u41i = tmp * p_u[3];
  u51i = tmp * p_u[4];

  u21im1 = tmpm1 * p_um1[1];
  u31im1 = tmpm1 * p_um1[2];
  u41im1 = tmpm1 * p_um1[3];
  u51im1 = tmpm1 * p_um1[4];

  flux2[0] = 0.0;

  flux2[1] = (4.0/3.0) * tx3 * (u21i-u21im1);
  flux2[2] = tx3 * ( u31i - u31im1 );
  flux2[3] = tx3 * ( u41i - u41im1 );
  flux2[4] = 0.50 * ( 1.0 - C1*C5 )
    * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
        - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
    + (1.0/6.0)
    * tx3 * ( u21i*u21i - u21im1*u21im1 )
    + C1 * C5 * tx3 * ( u51i - u51im1 );


}


__launch_bounds__(min(min(ISIZ2-2, MAX_THREAD_BLOCK_SIZE), MAX_THREAD_DIM_0))
__global__
void k_rhsx_gmem(double *m_u, 
                 double *m_rho_i,
                 double *m_qs, 
                 double *m_rsd,
                 int jst, int jend, 
                 int ist, int iend,
                 double tx1, double tx2, 
                 double tx3, 
                 double dx1, double dx2, 
                 double dx3, double dx4, 
                 double dx5,
                 double dssp, 
                 int nx, int nz,
                 int work_base, 
                 int work_num_item, 
                 int split_flag)
{
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x + jst;

  double flux1[3][5], flux2[2][5];
  double p_u[5][5], p_rsd[5];
  double dx[5] = {dx1, dx2, dx3, dx4, dx5};
  int i, m;

  if(k+work_base < 1 || k+work_base >= nz - 1 || k >= work_num_item || j >= jend) 
    return;

  double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i ;
  double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_qs ;
  double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_rsd ;

  if(split_flag) k += 2;
  else k += work_base;

  // get flux1[0][m] i == 0
  for(m = 0; m < 5; m++) p_u[1][m] = u[k][j][0][m];
  compute_flux1_x(flux1[0], p_u[1], qs[k][j][0], p_u[1][1]*rho_i[k][j][0]);

  // get flux1[1][m] i == 1
  for(m = 0; m < 5; m++) p_u[2][m] = u[k][j][1][m];
  compute_flux1_x(flux1[1], p_u[2], qs[k][j][1], p_u[2][1]*rho_i[k][j][1]);

  // get flux1[1][m] i == 2
  for(m = 0; m < 5; m++) p_u[3][m] = u[k][j][2][m];
  compute_flux1_x(flux1[2], p_u[3], qs[k][j][2], p_u[3][1]*rho_i[k][j][2]);

  // get flux2[0][m] i == 1 
  compute_flux2_x(flux2[0], p_u[1], p_u[2], rho_i[k][j][0], rho_i[k][j][1], tx3);

  // get flux2[0][m] i == 2
  compute_flux2_x(flux2[1], p_u[2], p_u[3], rho_i[k][j][1], rho_i[k][j][2], tx3);

  // load rsd 
  for(m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][1][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], tx2);

  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dx, tx1, tx3);

  for(m = 0; m < 5; m++){
    p_u[4][m] = u[k][j][3][m];

    // fourth-order dissipation
    rsd[k][j][1][m] = p_rsd[m]
      - dssp * ( + 5.0 * p_u[2][m]
          - 4.0 * p_u[3][m]
          +       p_u[4][m] );

  }



  // move p_u, flux1 and flux2
  for(m = 0; m < 5; m++){
    p_u[0][m] = p_u[1][m];
    p_u[1][m] = p_u[2][m];
    p_u[2][m] = p_u[3][m];
    p_u[3][m] = p_u[4][m];
    p_u[4][m] = u[k][j][4][m];

    flux1[0][m] = flux1[1][m];
    flux1[1][m] = flux1[2][m];
    flux2[0][m] = flux2[1][m];
  }





  // get flux1[2][m] i == 3
  compute_flux1_x(flux1[2], p_u[3], qs[k][j][3], p_u[3][1]*rho_i[k][j][3]);
  // get flux2[1][m] i == 3
  compute_flux2_x(flux2[1], p_u[2], p_u[3], rho_i[k][j][2], rho_i[k][j][3], tx3);

  // load rsd
  for(m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][2][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], tx2);
  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dx, tx1, tx3);

  for(m = 0; m < 5; m++){
    rsd[k][j][2][m] = p_rsd[m]
      - dssp * ( - 4.0 * p_u[1][m]
          + 6.0 * p_u[2][m]
          - 4.0 * p_u[3][m]
          +       p_u[4][m] );
  }

  for (i = 3; i < nx - 3; i++) {

    // move p_u, flux1 and flux2
    for(m = 0; m < 5; m++){
      p_u[0][m] = p_u[1][m];
      p_u[1][m] = p_u[2][m];
      p_u[2][m] = p_u[3][m];
      p_u[3][m] = p_u[4][m];
      p_u[4][m] = u[k][j][i+2][m];

      flux1[0][m] = flux1[1][m];
      flux1[1][m] = flux1[2][m];
      flux2[0][m] = flux2[1][m];
    }

    // get flux1[2][m]
    compute_flux1_x(flux1[2], p_u[3], qs[k][j][i+1], p_u[3][1]*rho_i[k][j][i+1]);
    // get flux2[1][m]
    compute_flux2_x(flux2[1], p_u[2], p_u[3], rho_i[k][j][i], rho_i[k][j][i+1], tx3);

    // load rsd
    for(m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][i][m];

    // flux1 to rsd
    flux1_to_rsd(p_rsd, flux1[0], flux1[2], tx2);
    // flux2 to rsd
    flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dx, tx1, tx3);

    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] = p_rsd[m]
        - dssp * (         p_u[0][m]
            - 4.0 * p_u[1][m]
            + 6.0 * p_u[2][m]
            - 4.0 * p_u[3][m]
            +       p_u[4][m] );
    }
  }

  // move p_u, flux1 and flux2
  for(m = 0; m < 5; m++){
    p_u[0][m] = p_u[1][m];
    p_u[1][m] = p_u[2][m];
    p_u[2][m] = p_u[3][m];
    p_u[3][m] = p_u[4][m];
    p_u[4][m] = u[k][j][i+2][m];

    flux1[0][m] = flux1[1][m];
    flux1[1][m] = flux1[2][m];
    flux2[0][m] = flux2[1][m];
  }


  // get flux1[2][m]
  compute_flux1_x(flux1[2], p_u[3], qs[k][j][i+1], p_u[3][1]*rho_i[k][j][i+1]);
  // get flux2[1][m]
  compute_flux2_x(flux2[1], p_u[2], p_u[3], rho_i[k][j][i], rho_i[k][j][i+1], tx3);

  // load rsd
  for(m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][i][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], tx2);
  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dx, tx1, tx3);

  for (m = 0; m < 5; m++) {
    rsd[k][j][i][m] = p_rsd[m]
      - dssp * (         p_u[0][m]
          - 4.0 * p_u[1][m]
          + 6.0 * p_u[2][m]
          - 4.0 * p_u[3][m] );
  }

  i++;

  // move p_u, flux1 and flux2
  for(m = 0; m < 5; m++){
    p_u[0][m] = p_u[1][m];
    p_u[1][m] = p_u[2][m];
    p_u[2][m] = p_u[3][m];
    p_u[3][m] = p_u[4][m];

    flux1[0][m] = flux1[1][m];
    flux1[1][m] = flux1[2][m];
    flux2[0][m] = flux2[1][m];
  }

  // get flux1[2][m]
  compute_flux1_x(flux1[2], p_u[3], qs[k][j][i+1], p_u[3][1]*rho_i[k][j][i+1]);
  // get flux2[1][m]
  compute_flux2_x(flux2[1], p_u[2], p_u[3], rho_i[k][j][i], rho_i[k][j][i+1], tx3);

  // load rsd
  for(m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][i][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], tx2);
  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dx, tx1, tx3);

  for (m = 0; m < 5; m++) {
    rsd[k][j][i][m] = p_rsd[m]
      - dssp * (         p_u[0][m]
          - 4.0 * p_u[1][m]
          + 5.0 * p_u[2][m] );
  }
}

//---------------------------------------------------------------------
// eta-direction flux differences
//---------------------------------------------------------------------

__device__
static void compute_flux1_y(double flux1[5], 
                            double p_u[5], 
                            double q, 
                            double u31)
{
  flux1[0] = p_u[2];
  flux1[1] = p_u[1] * u31;
  flux1[2] = p_u[2] * u31 + C2 * (p_u[4]-q);
  flux1[3] = p_u[3] * u31;
  flux1[4] = ( C1 * p_u[4] - C2 * q ) * u31;

}

__device__
static void compute_flux2_y(double flux2[5], 
                            double p_um1[5], 
                            double p_u[5], 
                            double tmpm1, 
                            double tmp, 
                            double ty3)
{
  double u21j, u31j, u41j, u51j, u21jm1, u31jm1, u41jm1, u51jm1;

  u21j = tmp * p_u[1];
  u31j = tmp * p_u[2];
  u41j = tmp * p_u[3];
  u51j = tmp * p_u[4];

  u21jm1 = tmpm1 * p_um1[1];
  u31jm1 = tmpm1 * p_um1[2];
  u41jm1 = tmpm1 * p_um1[3];
  u51jm1 = tmpm1 * p_um1[4];

  flux2[0] = 0.0;
  flux2[1] = ty3 * ( u21j - u21jm1 );
  flux2[2] = (4.0/3.0) * ty3 * (u31j-u31jm1);
  flux2[3] = ty3 * ( u41j - u41jm1 );
  flux2[4] = 0.50 * ( 1.0 - C1*C5 )
    * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
        - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
    + (1.0/6.0)
    * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
    + C1 * C5 * ty3 * ( u51j - u51jm1 );

}

__launch_bounds__(min(min(ISIZ1-2, MAX_THREAD_DIM_0), MAX_THREAD_BLOCK_SIZE))
__global__
void k_rhsy_gmem(double *m_u, 
                 double *m_rho_i,
                 double *m_qs, 
                 double *m_rsd,
                 int ist, int iend, 
                 double ty1, double ty2, 
                 double ty3,
                 double dy1, double dy2, 
                 double dy3, double dy4, 
                 double dy5,
                 double dssp, 
                 int ny, int nz, 
                 int work_base, 
                 int work_num_item,
                 int split_flag)
{
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x + ist;


  if(k+work_base < 1 || k+work_base >= nz - 1 || k >= work_num_item || i >= iend) 
    return;

  double flux1[3][5], flux2[2][5];
  double p_u[5][5], p_rsd[5];
  double dy[5] = {dy1, dy2, dy3, dy4, dy5};
  int j, m;

  double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i ;
  double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_qs ;
  double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_rsd ;
  
  if (split_flag) k += 2;
  else k += work_base;
  

  // get flux1[0][m] j == 0
  for (m = 0; m < 5; m++) p_u[1][m] = u[k][0][i][m];
  compute_flux1_y(flux1[0], p_u[1], qs[k][0][i], p_u[1][2] * rho_i[k][0][i]);

  // get flux1[1][m] j == 1
  for (m = 0; m < 5; m++) p_u[2][m] = u[k][1][i][m];
  compute_flux1_y(flux1[1], p_u[2], qs[k][1][i], p_u[2][2] * rho_i[k][1][i]);

  // get flux1[2][m] j == 2
  for (m = 0; m < 5; m++) p_u[3][m] = u[k][2][i][m];
  compute_flux1_y(flux1[2], p_u[3], qs[k][2][i], p_u[3][2] * rho_i[k][2][i]);



  //get flux2[0][m] j == 1  
  compute_flux2_y(flux2[0], p_u[1], p_u[2], rho_i[k][0][i], rho_i[k][1][i], ty3);

  //get flux2[1][m] j == 2    
  compute_flux2_y(flux2[1], p_u[2], p_u[3], rho_i[k][1][i], rho_i[k][2][i], ty3);

  // load rsd
  for (m = 0; m < 5; m++) p_rsd[m] = rsd[k][1][i][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], ty2);

  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dy, ty1, ty3);

  for (m = 0; m < 5; m++) {
    p_u[4][m] = u[k][3][i][m];

    // fourth-order dissipation
    p_rsd[m] = p_rsd[m]
      - dssp * ( + 5.0 * p_u[2][m]
          - 4.0 * p_u[3][m]
          +       p_u[4][m] );
  }

  // store rsd
  for (m = 0; m < 5; m++) rsd[k][1][i][m] = p_rsd[m];

  // move p_u, flux1 and flux2
  for (m = 0; m < 5; m++) {
    p_u[0][m] = p_u[1][m];
    p_u[1][m] = p_u[2][m];
    p_u[2][m] = p_u[3][m];
    p_u[3][m] = p_u[4][m];

    flux1[0][m] = flux1[1][m];
    flux1[1][m] = flux1[2][m];
    flux2[0][m] = flux2[1][m];
  }

  
  for (m = 0; m < 5; m++)
    p_u[4][m] = u[k][4][i][m];

  // get flux1[2][m] j == 3
  compute_flux1_y(flux1[2], p_u[3], qs[k][3][i], p_u[3][2] * rho_i[k][3][i]);

  //get flux2[1][m] j == 3  
  compute_flux2_y(flux2[1], p_u[2], p_u[3], rho_i[k][2][i], rho_i[k][3][i], ty3);

  // load rsd
  for (m = 0; m < 5; m++) p_rsd[m] = rsd[k][2][i][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], ty2);

  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dy, ty1, ty3);


  // fourth-order dissipation
  for (m = 0; m < 5; m++) { 
    p_rsd[m] = p_rsd[m]
      - dssp * ( - 4.0 * p_u[1][m]
          + 6.0 * p_u[2][m]
          - 4.0 * p_u[3][m]
          +       p_u[4][m] );
  }

  // store rsd
  for (m = 0; m < 5; m++) rsd[k][2][i][m] = p_rsd[m];

  for (j = 3; j < ny - 3; j++) {  
    for (m = 0; m < 5; m++) {
      p_u[0][m] = p_u[1][m];
      p_u[1][m] = p_u[2][m];
      p_u[2][m] = p_u[3][m];
      p_u[3][m] = p_u[4][m];
      p_u[4][m] = u[k][j+2][i][m];

      flux1[0][m] = flux1[1][m];
      flux1[1][m] = flux1[2][m];

      flux2[0][m] = flux2[1][m];
    }


    // get flux1[2][m] 
    compute_flux1_y(flux1[2], p_u[3], qs[k][j+1][i], p_u[3][2] * rho_i[k][j+1][i]);
    
    //get flux2[1][m]
    compute_flux2_y(flux2[1], p_u[2], p_u[3], rho_i[k][j][i], rho_i[k][j+1][i], ty3);

    // load rsd
    for (m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][i][m];

    // flux1 to rsd
    flux1_to_rsd(p_rsd, flux1[0], flux1[2], ty2);

    // flux2 to rsd
    flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dy, ty1, ty3);

    for (m = 0; m < 5; m++) {   
      p_rsd[m] = p_rsd[m]
        - dssp * ( p_u[0][m] - 4.0 * p_u[1][m] + 6.0 * p_u[2][m] - 4.0 * p_u[3][m] + p_u[4][m] );
    }



    // store rsd
    for (m = 0; m < 5; m++) rsd[k][j][i][m] = p_rsd[m];
  }






  for (m = 0; m < 5; m++) {
    p_u[0][m] = p_u[1][m];
    p_u[1][m] = p_u[2][m];
    p_u[2][m] = p_u[3][m];
    p_u[3][m] = p_u[4][m];
    p_u[4][m] = u[k][j+2][i][m];

    flux1[0][m] = flux1[1][m];
    flux1[1][m] = flux1[2][m];
    flux2[0][m] = flux2[1][m];
  }

  // get flux1[2][m] 
  compute_flux1_y(flux1[2], p_u[3], qs[k][j+1][i], p_u[3][2] * rho_i[k][j+1][i]);

  //get flux2[1][m]
  compute_flux2_y(flux2[1], p_u[2], p_u[3], rho_i[k][j][i], rho_i[k][j+1][i], ty3);

  // load rsd
  for (m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][i][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], ty2);

  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dy, ty1, ty3);

  for (m = 0; m < 5; m++) { 
    p_rsd[m] = p_rsd[m]
      - dssp * (         p_u[0][m]
          - 4.0 * p_u[1][m]
          + 6.0 * p_u[2][m]
          - 4.0 * p_u[3][m] );

  }

  // store rsd
  for (m = 0; m < 5; m++) rsd[k][j][i][m] = p_rsd[m];



  j++;


  for (m = 0; m < 5; m++) {
    p_u[0][m] = p_u[1][m];
    p_u[1][m] = p_u[2][m];
    p_u[2][m] = p_u[3][m];
    p_u[3][m] = p_u[4][m];

    flux1[0][m] = flux1[1][m];
    flux1[1][m] = flux1[2][m];
    flux2[0][m] = flux2[1][m];
  }




  // get flux1[2][m] 
  compute_flux1_y(flux1[2], p_u[3], qs[k][j+1][i], p_u[3][2] * rho_i[k][j+1][i]);


  //get flux2[1][m]
  compute_flux2_y(flux2[1], p_u[2], p_u[3], rho_i[k][j][i], rho_i[k][j+1][i], ty3);

  // load rsd
  for (m = 0; m < 5; m++) p_rsd[m] = rsd[k][j][i][m];

  // flux1 to rsd
  flux1_to_rsd(p_rsd, flux1[0], flux1[2], ty2);

  // flux2 to rsd
  flux2_to_rsd(p_rsd, p_u[1], p_u[2], p_u[3], flux2[0], flux2[1], dy, ty1, ty3);

  for (m = 0; m < 5; m++) { 
    p_rsd[m] = p_rsd[m]
      - dssp * (         p_u[0][m]
          - 4.0 * p_u[1][m]
          + 5.0 * p_u[2][m] );
  }

  // store rsd
  for (m = 0; m < 5; m++) rsd[k][j][i][m] = p_rsd[m];
}



//---------------------------------------------------------------------
// zeta-direction flux differences
//---------------------------------------------------------------------

__device__
static void compute_flux1_z(double *flux1, 
                            double *p_utmp, 
                            double q)
{
  double u41 = p_utmp[3] * p_utmp[5];

  flux1[0] = p_utmp[3];
  flux1[1] = p_utmp[1] * u41;
  flux1[2] = p_utmp[2] * u41;
  flux1[3] = p_utmp[3] * u41 + C2 * (p_utmp[4]-q);
  flux1[4] = ( C1 * p_utmp[4] - C2 * q ) * u41;

}


__device__
static void compute_flux2_z(double *flux2, 
                            double *p_utmpm1, 
                            double *p_utmp, 
                            double tz3)
{
  double tmp, u21k, u31k, u41k, u51k, u21km1, u31km1, u41km1, u51km1;

  tmp = p_utmp[5];

  u21k = tmp * p_utmp[1];
  u31k = tmp * p_utmp[2];
  u41k = tmp * p_utmp[3];
  u51k = tmp * p_utmp[4];

  tmp = p_utmpm1[5];

  u21km1 = tmp * p_utmpm1[1];
  u31km1 = tmp * p_utmpm1[2];
  u41km1 = tmp * p_utmpm1[3];
  u51km1 = tmp * p_utmpm1[4];


  flux2[1] = tz3 * ( u21k - u21km1 );
  flux2[2] = tz3 * ( u31k - u31km1 );
  flux2[3] = (4.0/3.0) * tz3 * (u41k-u41km1);
  flux2[4] = 0.50 * ( 1.0 - C1*C5 )
    * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
        - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
    + (1.0/6.0)
    * tz3 * ( u41k*u41k - u41km1*u41km1 )
    + C1 * C5 * tz3 * ( u51k - u51km1 );

}

// WARNING : work_base  = 0 (not split)
__launch_bounds__(min(min(ISIZ1-2, MAX_THREAD_DIM_0), MAX_THREAD_BLOCK_SIZE))
__global__
void k_rhsz_gmem(double *m_u, 
                 double *m_rho_i,
                 double *m_qs, 
                 double * m_rsd,
                 int jst, int jend, 
                 int ist, int iend,
                 double tz1, double tz2, 
                 double tz3,
                 double dz1, double dz2, 
                 double dz3, double dz4, 
                 double dz5,
                 double dssp, int nz, 
                 int work_base, 
                 int work_num_item, 
                 int split_flag)
{
  int j = blockDim.y * blockIdx.y + threadIdx.y + jst;
  int i = blockDim.x * blockIdx.x + threadIdx.x + ist;

  int kst, kend;

  if (j >= jend || i >= iend) return;

  if(split_flag) {

    if(work_base < 1) kst = 1;
    else kst = 0;

    if(work_base+work_num_item >= nz-1) kend = nz-1 - work_base;
    else kend = work_num_item;

    // front padding align
    kst += 2;
    kend += 2;
  }
  else {
    kst = 1;
    kend = nz-1;
  }


  double flux1[3][5], flux2[2][5];
  double p_utmp[5][6], p_rsd[5];
  double dz[5] = { dz1, dz2, dz3, dz4, dz5 };
  int k, m;
  int orig_kst, orig_k;

  
  double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u ;
  double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i ;
  double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_qs ;
  double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_rsd ;


  if (split_flag) orig_kst = kst - 2 + work_base;
  else orig_kst = kst;

  if (orig_kst >= 2) {
    for(m = 0; m < 5; m++) p_utmp[1][m] = u[kst-2][j][i][m];
    p_utmp[1][5] = rho_i[kst-2][j][i];
  }
  else {
    for(m = 0; m < 5; m++) p_utmp[1][m] = 0.0;
    p_utmp[1][5] = 0.0;
  }



  // get flux1[0][m] k == 0
  for (m = 0; m < 5; m++) p_utmp[2][m] = u[kst-1][j][i][m];
  p_utmp[2][5] = rho_i[kst-1][j][i];
  compute_flux1_z(flux1[1], p_utmp[2], qs[kst-1][j][i]);

  // get flux1[1][m] k == 1
  for (m = 0; m < 5; m++) p_utmp[3][m] = u[kst][j][i][m];
  p_utmp[3][5] = rho_i[kst][j][i];
  compute_flux1_z(flux1[2], p_utmp[3], qs[kst][j][i]);

  // get flux2[0][m] k == 1
  compute_flux2_z(flux2[1], p_utmp[2], p_utmp[3], tz3);

  for (m = 0; m < 5; m++) p_utmp[4][m] = u[kst+1][j][i][m];
  p_utmp[4][5] = rho_i[kst+1][j][i];




  for (k = kst; k < kend; k++) {

    if (split_flag) orig_k = k - 2+ work_base;
    else orig_k = k;

    // move p_utmp, flux1, and flux2
    for (m = 0; m < 5; m++) {
      p_utmp[0][m] = p_utmp[1][m];
      p_utmp[1][m] = p_utmp[2][m];
      p_utmp[2][m] = p_utmp[3][m];
      p_utmp[3][m] = p_utmp[4][m];

      flux1[0][m] = flux1[1][m];
      flux1[1][m] = flux1[2][m];

      flux2[0][m] = flux2[1][m];
    }
    for (m = 0; m < 4; m++) {
      p_utmp[m][5] = p_utmp[m+1][5]; 
    }

    if (orig_k < nz-2) {
      for(m = 0; m < 5; m++){
        p_utmp[4][m] = u[k+2][j][i][m];
      }
      p_utmp[4][5] = rho_i[k+2][j][i];
    }



    // get flux1[2][m]
    compute_flux1_z(flux1[2], p_utmp[3], qs[k+1][j][i]);
    // get flux2[1][m]
    compute_flux2_z(flux2[1], p_utmp[2], p_utmp[3], tz3);

    for(m = 0; m < 5; m++)p_rsd[m] = rsd[k][j][i][m];

    //flux1 to rsd
    flux1_to_rsd(p_rsd, flux1[0], flux1[2], tz2);

    //flux2 to rsd
    flux2_to_rsd(p_rsd, p_utmp[1], p_utmp[2], p_utmp[3], flux2[0], flux2[1], dz, tz1, tz3);


    // fourth-order dissipation
    if (orig_k == 1) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = p_rsd[m] 
          - dssp * ( + 5.0 * p_utmp[2][m]
              - 4.0 * p_utmp[3][m]
              +       p_utmp[4][m] );
      }
    }
    else if (orig_k == 2) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = p_rsd[m]
          - dssp * ( - 4.0 * p_utmp[1][m]
              + 6.0 * p_utmp[2][m]
              - 4.0 * p_utmp[3][m]
              +       p_utmp[4][m] );
      }
    }
    else if (orig_k == nz-3) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = p_rsd[m]
          - dssp * (  p_utmp[0][m]
              - 4.0 * p_utmp[1][m]
              + 6.0 * p_utmp[2][m]
              - 4.0 * p_utmp[3][m]);
      }
    }
    else if (orig_k == nz-2) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = p_rsd[m]
          - dssp * (  p_utmp[0][m]
              - 4.0 * p_utmp[1][m]
              + 5.0 * p_utmp[2][m]);
      }
    }
    else {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = p_rsd[m]
          - dssp * (  p_utmp[0][m]
              - 4.0 * p_utmp[1][m]
              + 6.0 * p_utmp[2][m]
              - 4.0 * p_utmp[3][m]
              +       p_utmp[4][m] );
      }
    }
  }
}
