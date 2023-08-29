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

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "applu.incl"
#include "kernel_constants.h"

extern "C" {
#include "timers.h"
}


__global__
void k_jbl_datagen_baseline(double *m_u, 
                            double *m_qs,
                            double *m_rho_i,
                            int kend, int jend, int iend, 
                            int work_base, 
                            int work_num_item)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (k+work_base > kend || k > work_num_item || j >= jend || i >= iend) 
    return;

  double tmp;

  double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*) [ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;
  double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (double (*) [ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;


  tmp = 1.0 / u[k][j][i][0];
  rho_i[k][j][i] = tmp;
  qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
      + u[k][j][i][2] * u[k][j][i][2]
      + u[k][j][i][3] * u[k][j][i][3] ) * tmp;

}

__global__
void k_jbl_datacopy_baseline(double *m_rsd, 
                             double *m_rsd_next,
                             double *m_u, 
                             double *m_u_next,
                             int jst, int jend, 
                             int ist, int iend, 
                             int work_num_item)
{

  double (* rsd) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;  
  double (* rsd_next) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd_next;

  double (* u) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;  
  double (* u_next) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u_next;

  int j = blockDim.y * blockIdx.y + threadIdx.y + jst;
  int t_i = blockDim.x * blockIdx.x + threadIdx.x;

  int k = work_num_item;
  int i = t_i / 5 + ist;
  int m = t_i % 5;

  if (j >= jend || i >= iend) 
    return;

  rsd_next[0][j][i][m] = rsd[k][j][i][m];
  u_next[0][j][i][m] = u[k][j][i][m];
}

__global__
void k_jacld_baseline(double *m_rsd,
                      double *m_u,
                      double *m_qs,
                      double *m_rho_i,
                      double *m_a,
                      double *m_b,
                      double *m_c,
                      double *m_d,
                      int nz, int ny, int nx,
                      int jst, int jend, 
                      int ist, int iend, 
                      int temp_kst, int temp_kend)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z + temp_kst;
  int j = blockDim.y * blockIdx.y + threadIdx.y + jst;
  int i = blockDim.x * blockIdx.x + threadIdx.x + ist;

  double r43, c1345, c34,
         tmp1, tmp2, tmp3;

  if (k >= temp_kend || j >= jend || i >= iend)
    return;

  double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

  double (*a)[ISIZ1/2*2+1][5][5];
  double (*b)[ISIZ1/2*2+1][5][5];
  double (*c)[ISIZ1/2*2+1][5][5];
  double (*d)[ISIZ1/2*2+1][5][5];

  u = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  qs = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;
  rho_i = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;

  a = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_a)[k];
  b = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_b)[k];
  c = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_c)[k];
  d = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_d)[k];


  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  //---------------------------------------------------------------------
  // form the block daigonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  d[j][i][0][0] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
  d[j][i][1][0] =  0.0;
  d[j][i][2][0] =  0.0;
  d[j][i][3][0] =  0.0;
  d[j][i][4][0] =  0.0;

  d[j][i][0][1] = -dt * 2.0
    * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * u[k][j][i][1];
  d[j][i][1][1] =  1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
    + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
  d[j][i][2][1] = 0.0;
  d[j][i][3][1] = 0.0;
  d[j][i][4][1] = 0.0;

  d[j][i][0][2] = -dt * 2.0 
    * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * u[k][j][i][2];
  d[j][i][1][2] = 0.0;
  d[j][i][2][2] = 1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
    + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
  d[j][i][3][2] = 0.0;
  d[j][i][4][2] = 0.0;

  d[j][i][0][3] = -dt * 2.0
    * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * u[k][j][i][3];
  d[j][i][1][3] = 0.0;
  d[j][i][2][3] = 0.0;
  d[j][i][3][3] = 1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
    + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
  d[j][i][4][3] = 0.0;

  d[j][i][0][4] = -dt * 2.0
    * ( ( ( tx1 * ( r43*c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][1]*u[k][j][i][1] )
          + ( tx1 * ( c34 - c1345 )
            + ty1 * ( r43*c34 - c1345 )
            + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][2]*u[k][j][i][2] )
          + ( tx1 * ( c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( r43*c34 - c1345 ) ) * (u[k][j][i][3]*u[k][j][i][3])
        ) * tmp3
        + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u[k][j][i][4] );

  d[j][i][1][4] = dt * 2.0 * tmp2 * u[k][j][i][1]
    * ( tx1 * ( r43*c34 - c1345 )
        + ty1 * (     c34 - c1345 )
        + tz1 * (     c34 - c1345 ) );
  d[j][i][2][4] = dt * 2.0 * tmp2 * u[k][j][i][2]
    * ( tx1 * ( c34 - c1345 )
        + ty1 * ( r43*c34 -c1345 )
        + tz1 * ( c34 - c1345 ) );
  d[j][i][3][4] = dt * 2.0 * tmp2 * u[k][j][i][3]
    * ( tx1 * ( c34 - c1345 )
        + ty1 * ( c34 - c1345 )
        + tz1 * ( r43*c34 - c1345 ) );
  d[j][i][4][4] = 1.0
    + dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
    + dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

  //---------------------------------------------------------------------
  // form the first block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k-1][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  a[j][i][0][0] = - dt * tz1 * dz1;
  a[j][i][1][0] =   0.0;
  a[j][i][2][0] =   0.0;
  a[j][i][3][0] = - dt * tz2;
  a[j][i][4][0] =   0.0;

  a[j][i][0][1] = - dt * tz2
    * ( - ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k-1][j][i][1] );
  a[j][i][1][1] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * c34 * tmp1
    - dt * tz1 * dz2;
  a[j][i][2][1] = 0.0;
  a[j][i][3][1] = - dt * tz2 * ( u[k-1][j][i][1] * tmp1 );
  a[j][i][4][1] = 0.0;

  a[j][i][0][2] = - dt * tz2
    * ( - ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k-1][j][i][2] );
  a[j][i][1][2] = 0.0;
  a[j][i][2][2] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * ( c34 * tmp1 )
    - dt * tz1 * dz3;
  a[j][i][3][2] = - dt * tz2 * ( u[k-1][j][i][2] * tmp1 );
  a[j][i][4][2] = 0.0;

  a[j][i][0][3] = - dt * tz2
    * ( - ( u[k-1][j][i][3] * tmp1 ) * ( u[k-1][j][i][3] * tmp1 )
        + C2 * qs[k-1][j][i] * tmp1 )
    - dt * tz1 * ( - r43 * c34 * tmp2 * u[k-1][j][i][3] );
  a[j][i][1][3] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][1] * tmp1 ) );
  a[j][i][2][3] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][2] * tmp1 ) );
  a[j][i][3][3] = - dt * tz2 * ( 2.0 - C2 )
    * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * ( r43 * c34 * tmp1 )
    - dt * tz1 * dz4;
  a[j][i][4][3] = - dt * tz2 * C2;

  a[j][i][0][4] = - dt * tz2
    * ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * u[k-1][j][i][4] )
        * u[k-1][j][i][3] * tmp2 )
    - dt * tz1
    * ( - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][1]*u[k-1][j][i][1])
        - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][2]*u[k-1][j][i][2])
        - ( r43*c34 - c1345 )* tmp3 * (u[k-1][j][i][3]*u[k-1][j][i][3])
        - c1345 * tmp2 * u[k-1][j][i][4] );
  a[j][i][1][4] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k-1][j][i][1];
  a[j][i][2][4] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k-1][j][i][2];
  a[j][i][3][4] = - dt * tz2
    * ( C1 * ( u[k-1][j][i][4] * tmp1 )
        - C2 * ( qs[k-1][j][i] * tmp1
          + u[k-1][j][i][3]*u[k-1][j][i][3] * tmp2 ) )
    - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[k-1][j][i][3];
  a[j][i][4][4] = - dt * tz2
    * ( C1 * ( u[k-1][j][i][3] * tmp1 ) )
    - dt * tz1 * c1345 * tmp1
    - dt * tz1 * dz5;

  //---------------------------------------------------------------------
  // form the second block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j-1][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  b[j][i][0][0] = - dt * ty1 * dy1;
  b[j][i][1][0] =   0.0;
  b[j][i][2][0] = - dt * ty2;
  b[j][i][3][0] =   0.0;
  b[j][i][4][0] =   0.0;

  b[j][i][0][1] = - dt * ty2
    * ( - ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j-1][i][1] );
  b[j][i][1][1] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy2;
  b[j][i][2][1] = - dt * ty2 * ( u[k][j-1][i][1] * tmp1 );
  b[j][i][3][1] = 0.0;
  b[j][i][4][1] = 0.0;

  b[j][i][0][2] = - dt * ty2
    * ( - ( u[k][j-1][i][2] * tmp1 ) * ( u[k][j-1][i][2] * tmp1 )
        + C2 * ( qs[k][j-1][i] * tmp1 ) )
    - dt * ty1 * ( - r43 * c34 * tmp2 * u[k][j-1][i][2] );
  b[j][i][1][2] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][1] * tmp1 ) );
  b[j][i][2][2] = - dt * ty2 * ( (2.0 - C2) * (u[k][j-1][i][2] * tmp1) )
    - dt * ty1 * ( r43 * c34 * tmp1 )
    - dt * ty1 * dy3;
  b[j][i][3][2] = - dt * ty2 * ( - C2 * ( u[k][j-1][i][3] * tmp1 ) );
  b[j][i][4][2] = - dt * ty2 * C2;

  b[j][i][0][3] = - dt * ty2
    * ( - ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j-1][i][3] );
  b[j][i][1][3] = 0.0;
  b[j][i][2][3] = - dt * ty2 * ( u[k][j-1][i][3] * tmp1 );
  b[j][i][3][3] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy4;
  b[j][i][4][3] = 0.0;

  b[j][i][0][4] = - dt * ty2
    * ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * u[k][j-1][i][4] )
        * ( u[k][j-1][i][2] * tmp2 ) )
    - dt * ty1
    * ( - (     c34 - c1345 )*tmp3*(u[k][j-1][i][1]*u[k][j-1][i][1])
        - ( r43*c34 - c1345 )*tmp3*(u[k][j-1][i][2]*u[k][j-1][i][2])
        - (     c34 - c1345 )*tmp3*(u[k][j-1][i][3]*u[k][j-1][i][3])
        - c1345*tmp2*u[k][j-1][i][4] );
  b[j][i][1][4] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j-1][i][1];
  b[j][i][2][4] = - dt * ty2
    * ( C1 * ( u[k][j-1][i][4] * tmp1 )
        - C2 * ( qs[k][j-1][i] * tmp1
          + u[k][j-1][i][2]*u[k][j-1][i][2] * tmp2 ) )
    - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u[k][j-1][i][2];
  b[j][i][3][4] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j-1][i][3];
  b[j][i][4][4] = - dt * ty2
    * ( C1 * ( u[k][j-1][i][2] * tmp1 ) )
    - dt * ty1 * c1345 * tmp1
    - dt * ty1 * dy5;

  //---------------------------------------------------------------------
  // form the third block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i-1];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  c[j][i][0][0] = - dt * tx1 * dx1;
  c[j][i][1][0] = - dt * tx2;
  c[j][i][2][0] =   0.0;
  c[j][i][3][0] =   0.0;
  c[j][i][4][0] =   0.0;

  c[j][i][0][1] = - dt * tx2
    * ( - ( u[k][j][i-1][1] * tmp1 ) * ( u[k][j][i-1][1] * tmp1 )
        + C2 * qs[k][j][i-1] * tmp1 )
    - dt * tx1 * ( - r43 * c34 * tmp2 * u[k][j][i-1][1] );
  c[j][i][1][1] = - dt * tx2
    * ( ( 2.0 - C2 ) * ( u[k][j][i-1][1] * tmp1 ) )
    - dt * tx1 * ( r43 * c34 * tmp1 )
    - dt * tx1 * dx2;
  c[j][i][2][1] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][2] * tmp1 ) );
  c[j][i][3][1] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][3] * tmp1 ) );
  c[j][i][4][1] = - dt * tx2 * C2;

  c[j][i][0][2] = - dt * tx2
    * ( - ( u[k][j][i-1][1] * u[k][j][i-1][2] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i-1][2] );
  c[j][i][1][2] = - dt * tx2 * ( u[k][j][i-1][2] * tmp1 );
  c[j][i][2][2] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx3;
  c[j][i][3][2] = 0.0;
  c[j][i][4][2] = 0.0;

  c[j][i][0][3] = - dt * tx2
    * ( - ( u[k][j][i-1][1]*u[k][j][i-1][3] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i-1][3] );
  c[j][i][1][3] = - dt * tx2 * ( u[k][j][i-1][3] * tmp1 );
  c[j][i][2][3] = 0.0;
  c[j][i][3][3] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
  c[j][i][4][3] = 0.0;

  c[j][i][0][4] = - dt * tx2
    * ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * u[k][j][i-1][4] )
        * u[k][j][i-1][1] * tmp2 )
    - dt * tx1
    * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[k][j][i-1][1]*u[k][j][i-1][1] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][2]*u[k][j][i-1][2] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][3]*u[k][j][i-1][3] )
        - c1345 * tmp2 * u[k][j][i-1][4] );
  c[j][i][1][4] = - dt * tx2
    * ( C1 * ( u[k][j][i-1][4] * tmp1 )
        - C2 * ( u[k][j][i-1][1]*u[k][j][i-1][1] * tmp2
          + qs[k][j][i-1] * tmp1 ) )
    - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u[k][j][i-1][1];
  c[j][i][2][4] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][2]*u[k][j][i-1][1] ) * tmp2 )
    - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[k][j][i-1][2];
  c[j][i][3][4] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][3]*u[k][j][i-1][1] ) * tmp2 )
    - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[k][j][i-1][3];
  c[j][i][4][4] = - dt * tx2
    * ( C1 * ( u[k][j][i-1][1] * tmp1 ) )
    - dt * tx1 * c1345 * tmp1
    - dt * tx1 * dx5;



}

__global__
void k_blts_KL_baseline(double *m_rsd,
                        double *m_u,
                        double *m_qs,
                        double *m_rho_i,
                        double *m_a,
                        double *m_b,
                        double *m_c,
                        double *m_d,
                        int nz, int ny, int nx,
                        int wf_sum, 
                        int wf_base_k, 
                        int wf_base_j,
                        int jst, int jend, 
                        int ist, int iend, 
                        int temp_kst, int temp_kend)
{
  int k, j, i, m;
  double tmp, tmp1;
  double tmat[5][5], tv[5];

  double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];

  double (*a)[5];
  double (*b)[5];
  double (*c)[5];
  double (*d)[5];

  int gid1 = blockDim.y * blockIdx.y + threadIdx.y;
  int gid0 = blockDim.x * blockIdx.x + threadIdx.x;

  k = gid1 + temp_kst + wf_base_k;
  j = gid0 + jst + wf_base_j;
  i = wf_sum - gid1 - gid0 - wf_base_k - wf_base_j + ist;

  if (k >= temp_kend || j >= jend || i < ist || i >= iend) return;

  rsd = (double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;

  a = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_a)[k][j][i];
  b = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_b)[k][j][i];
  c = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_c)[k][j][i];
  d = ((double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_d)[k][j][i];


  for (m = 0; m < 5; m++) {
    tv[m] =  rsd[k][j][i][m]
      - omega * (  a[0][m] * rsd[k-1][j][i][0]
          + a[1][m] * rsd[k-1][j][i][1]
          + a[2][m] * rsd[k-1][j][i][2]
          + a[3][m] * rsd[k-1][j][i][3]
          + a[4][m] * rsd[k-1][j][i][4] );
  }

  for (m = 0; m < 5; m++) {
    tv[m] =  tv[m]
      - omega * ( b[0][m] * rsd[k][j-1][i][0]
          + c[0][m] * rsd[k][j][i-1][0]
          + b[1][m] * rsd[k][j-1][i][1]
          + c[1][m] * rsd[k][j][i-1][1]
          + b[2][m] * rsd[k][j-1][i][2]
          + c[2][m] * rsd[k][j][i-1][2]
          + b[3][m] * rsd[k][j-1][i][3]
          + c[3][m] * rsd[k][j][i-1][3]
          + b[4][m] * rsd[k][j-1][i][4]
          + c[4][m] * rsd[k][j][i-1][4] );
  }

  //---------------------------------------------------------------------
  // diagonal block inversion
  // 
  // forward elimination
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    tmat[m][0] = d[0][m];
    tmat[m][1] = d[1][m];
    tmat[m][2] = d[2][m];
    tmat[m][3] = d[3][m];
    tmat[m][4] = d[4][m];
  }

  tmp1 = 1.0 / tmat[0][0];
  tmp = tmp1 * tmat[1][0];
  tmat[1][1] =  tmat[1][1] - tmp * tmat[0][1];
  tmat[1][2] =  tmat[1][2] - tmp * tmat[0][2];
  tmat[1][3] =  tmat[1][3] - tmp * tmat[0][3];
  tmat[1][4] =  tmat[1][4] - tmp * tmat[0][4];
  tv[1] = tv[1] - tv[0] * tmp;

  tmp = tmp1 * tmat[2][0];
  tmat[2][1] =  tmat[2][1] - tmp * tmat[0][1];
  tmat[2][2] =  tmat[2][2] - tmp * tmat[0][2];
  tmat[2][3] =  tmat[2][3] - tmp * tmat[0][3];
  tmat[2][4] =  tmat[2][4] - tmp * tmat[0][4];
  tv[2] = tv[2] - tv[0] * tmp;

  tmp = tmp1 * tmat[3][0];
  tmat[3][1] =  tmat[3][1] - tmp * tmat[0][1];
  tmat[3][2] =  tmat[3][2] - tmp * tmat[0][2];
  tmat[3][3] =  tmat[3][3] - tmp * tmat[0][3];
  tmat[3][4] =  tmat[3][4] - tmp * tmat[0][4];
  tv[3] = tv[3] - tv[0] * tmp;

  tmp = tmp1 * tmat[4][0];
  tmat[4][1] =  tmat[4][1] - tmp * tmat[0][1];
  tmat[4][2] =  tmat[4][2] - tmp * tmat[0][2];
  tmat[4][3] =  tmat[4][3] - tmp * tmat[0][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[0][4];
  tv[4] = tv[4] - tv[0] * tmp;

  tmp1 = 1.0 / tmat[1][1];
  tmp = tmp1 * tmat[2][1];
  tmat[2][2] =  tmat[2][2] - tmp * tmat[1][2];
  tmat[2][3] =  tmat[2][3] - tmp * tmat[1][3];
  tmat[2][4] =  tmat[2][4] - tmp * tmat[1][4];
  tv[2] = tv[2] - tv[1] * tmp;

  tmp = tmp1 * tmat[3][1];
  tmat[3][2] =  tmat[3][2] - tmp * tmat[1][2];
  tmat[3][3] =  tmat[3][3] - tmp * tmat[1][3];
  tmat[3][4] =  tmat[3][4] - tmp * tmat[1][4];
  tv[3] = tv[3] - tv[1] * tmp;

  tmp = tmp1 * tmat[4][1];
  tmat[4][2] =  tmat[4][2] - tmp * tmat[1][2];
  tmat[4][3] =  tmat[4][3] - tmp * tmat[1][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[1][4];
  tv[4] = tv[4] - tv[1] * tmp;

  tmp1 = 1.0 / tmat[2][2];
  tmp = tmp1 * tmat[3][2];
  tmat[3][3] =  tmat[3][3] - tmp * tmat[2][3];
  tmat[3][4] =  tmat[3][4] - tmp * tmat[2][4];
  tv[3] = tv[3] - tv[2] * tmp;

  tmp = tmp1 * tmat[4][2];
  tmat[4][3] =  tmat[4][3] - tmp * tmat[2][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[2][4];
  tv[4] = tv[4] - tv[2] * tmp;

  tmp1 = 1.0 / tmat[3][3];
  tmp = tmp1 * tmat[4][3];
  tmat[4][4] =  tmat[4][4] - tmp * tmat[3][4];
  tv[4] = tv[4] - tv[3] * tmp;

  //---------------------------------------------------------------------
  // back substitution
  //---------------------------------------------------------------------
  rsd[k][j][i][4] = tv[4] / tmat[4][4];

  tv[3] = tv[3] 
    - tmat[3][4] * rsd[k][j][i][4];
  rsd[k][j][i][3] = tv[3] / tmat[3][3];

  tv[2] = tv[2]
    - tmat[2][3] * rsd[k][j][i][3]
    - tmat[2][4] * rsd[k][j][i][4];
  rsd[k][j][i][2] = tv[2] / tmat[2][2];

  tv[1] = tv[1]
    - tmat[1][2] * rsd[k][j][i][2]
    - tmat[1][3] * rsd[k][j][i][3]
    - tmat[1][4] * rsd[k][j][i][4];
  rsd[k][j][i][1] = tv[1] / tmat[1][1];

  tv[0] = tv[0]
    - tmat[0][1] * rsd[k][j][i][1]
    - tmat[0][2] * rsd[k][j][i][2]
    - tmat[0][3] * rsd[k][j][i][3]
    - tmat[0][4] * rsd[k][j][i][4];
  rsd[k][j][i][0] = tv[0] / tmat[0][0];
}

