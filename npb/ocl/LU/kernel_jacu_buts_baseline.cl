//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB LU code. This OpenCL C  //
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

#include "kernel_constants.h"

__kernel void jacu_buts_datagen_baseline(__global double *m_u, 
                                         __global double *m_qs,
                                         __global double *m_rho_i, 
                                         int jst, int jend, 
                                         int ist, int iend,
                                         int temp_kst, 
                                         int temp_kend, 
                                         int work_num_item)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+jst;
  int i = get_global_id(0)+ist;

  if (k > work_num_item || j > jend || i > iend) return;

  k += temp_kst;

  double tmp;

  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  __global double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*) [ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*) [ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;


  tmp = 1.0 / u[k][j][i][0];
  rho_i[k][j][i] = tmp;
  qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
      + u[k][j][i][2] * u[k][j][i][2]
      + u[k][j][i][3] * u[k][j][i][3] ) * tmp;


}

__kernel void jacu_baseline(__global double *m_rsd,
                            __global double *m_u,
                            __global double *m_qs,
                            __global double *m_rho_i,
                            __global double *m_au,
                            __global double *m_bu,
                            __global double *m_cu,
                            __global double *m_du,
                            int nz, int ny, int nx,
                            int jst, int jend, 
                            int ist, int iend, 
                            int temp_kst, int temp_kend)
{

  int k = get_global_id(2) + temp_kst;
  int j = get_global_id(1) + jst;
  int i = get_global_id(0) + ist;

  double r43, c1345, c34,
         tmp, tmp1, tmp2, tmp3;

  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

  __global double (*au)[ISIZ1/2*2+1][5][5];
  __global double (*bu)[ISIZ1/2*2+1][5][5];
  __global double (*cu)[ISIZ1/2*2+1][5][5];
  __global double (*du)[ISIZ1/2*2+1][5][5];

  if (k >= temp_kend || j >= jend || i >= iend)
    return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;
  u = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  qs = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;

  au = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_au)[k];
  bu = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_bu)[k];
  cu = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_cu)[k];
  du = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_du)[k];

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  //---------------------------------------------------------------------
  // form the block daigonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  du[j][i][0][0] = 1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
  du[j][i][1][0] = 0.0;
  du[j][i][2][0] = 0.0;
  du[j][i][3][0] = 0.0;
  du[j][i][4][0] = 0.0;

  du[j][i][0][1] =  dt * 2.0
    * ( - tx1 * r43 - ty1 - tz1 )
    * ( c34 * tmp2 * u[k][j][i][1] );
  du[j][i][1][1] =  1.0
    + dt * 2.0 * c34 * tmp1 
    * (  tx1 * r43 + ty1 + tz1 )
    + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
  du[j][i][2][1] = 0.0;
  du[j][i][3][1] = 0.0;
  du[j][i][4][1] = 0.0;

  du[j][i][0][2] = dt * 2.0
    * ( - tx1 - ty1 * r43 - tz1 )
    * ( c34 * tmp2 * u[k][j][i][2] );
  du[j][i][1][2] = 0.0;
  du[j][i][2][2] = 1.0
    + dt * 2.0 * c34 * tmp1
    * (  tx1 + ty1 * r43 + tz1 )
    + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
  du[j][i][3][2] = 0.0;
  du[j][i][4][2] = 0.0;

  du[j][i][0][3] = dt * 2.0
    * ( - tx1 - ty1 - tz1 * r43 )
    * ( c34 * tmp2 * u[k][j][i][3] );
  du[j][i][1][3] = 0.0;
  du[j][i][2][3] = 0.0;
  du[j][i][3][3] = 1.0
    + dt * 2.0 * c34 * tmp1
    * (  tx1 + ty1 + tz1 * r43 )
    + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
  du[j][i][4][3] = 0.0;

  du[j][i][0][4] = -dt * 2.0
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

  du[j][i][1][4] = dt * 2.0
    * ( tx1 * ( r43*c34 - c1345 )
        + ty1 * (     c34 - c1345 )
        + tz1 * (     c34 - c1345 ) ) * tmp2 * u[k][j][i][1];
  du[j][i][2][4] = dt * 2.0
    * ( tx1 * ( c34 - c1345 )
        + ty1 * ( r43*c34 -c1345 )
        + tz1 * ( c34 - c1345 ) ) * tmp2 * u[k][j][i][2];
  du[j][i][3][4] = dt * 2.0
    * ( tx1 * ( c34 - c1345 )
        + ty1 * ( c34 - c1345 )
        + tz1 * ( r43*c34 - c1345 ) ) * tmp2 * u[k][j][i][3];
  du[j][i][4][4] = 1.0
    + dt * 2.0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1
    + dt * 2.0 * ( tx1 * dx5 + ty1 * dy5 + tz1 * dz5 );

  //---------------------------------------------------------------------
  // form the first block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i+1];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  au[j][i][0][0] = - dt * tx1 * dx1;
  au[j][i][1][0] =   dt * tx2;
  au[j][i][2][0] =   0.0;
  au[j][i][3][0] =   0.0;
  au[j][i][4][0] =   0.0;

  au[j][i][0][1] =  dt * tx2
    * ( - ( u[k][j][i+1][1] * tmp1 ) * ( u[k][j][i+1][1] * tmp1 )
        + C2 * qs[k][j][i+1] * tmp1 )
    - dt * tx1 * ( - r43 * c34 * tmp2 * u[k][j][i+1][1] );
  au[j][i][1][1] =  dt * tx2
    * ( ( 2.0 - C2 ) * ( u[k][j][i+1][1] * tmp1 ) )
    - dt * tx1 * ( r43 * c34 * tmp1 )
    - dt * tx1 * dx2;
  au[j][i][2][1] =  dt * tx2
    * ( - C2 * ( u[k][j][i+1][2] * tmp1 ) );
  au[j][i][3][1] =  dt * tx2
    * ( - C2 * ( u[k][j][i+1][3] * tmp1 ) );
  au[j][i][4][1] =  dt * tx2 * C2 ;

  au[j][i][0][2] =  dt * tx2
    * ( - ( u[k][j][i+1][1] * u[k][j][i+1][2] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i+1][2] );
  au[j][i][1][2] =  dt * tx2 * ( u[k][j][i+1][2] * tmp1 );
  au[j][i][2][2] =  dt * tx2 * ( u[k][j][i+1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx3;
  au[j][i][3][2] = 0.0;
  au[j][i][4][2] = 0.0;

  au[j][i][0][3] = dt * tx2
    * ( - ( u[k][j][i+1][1]*u[k][j][i+1][3] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i+1][3] );
  au[j][i][1][3] = dt * tx2 * ( u[k][j][i+1][3] * tmp1 );
  au[j][i][2][3] = 0.0;
  au[j][i][3][3] = dt * tx2 * ( u[k][j][i+1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx4;
  au[j][i][4][3] = 0.0;

  au[j][i][0][4] = dt * tx2
    * ( ( C2 * 2.0 * qs[k][j][i+1]
          - C1 * u[k][j][i+1][4] )
        * ( u[k][j][i+1][1] * tmp2 ) )
    - dt * tx1
    * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[k][j][i+1][1]*u[k][j][i+1][1] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i+1][2]*u[k][j][i+1][2] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i+1][3]*u[k][j][i+1][3] )
        - c1345 * tmp2 * u[k][j][i+1][4] );
  au[j][i][1][4] = dt * tx2
    * ( C1 * ( u[k][j][i+1][4] * tmp1 )
        - C2
        * ( u[k][j][i+1][1]*u[k][j][i+1][1] * tmp2
          + qs[k][j][i+1] * tmp1 ) )
    - dt * tx1
    * ( r43*c34 - c1345 ) * tmp2 * u[k][j][i+1][1];
  au[j][i][2][4] = dt * tx2
    * ( - C2 * ( u[k][j][i+1][2]*u[k][j][i+1][1] ) * tmp2 )
    - dt * tx1
    * (  c34 - c1345 ) * tmp2 * u[k][j][i+1][2];
  au[j][i][3][4] = dt * tx2
    * ( - C2 * ( u[k][j][i+1][3]*u[k][j][i+1][1] ) * tmp2 )
    - dt * tx1
    * (  c34 - c1345 ) * tmp2 * u[k][j][i+1][3];
  au[j][i][4][4] = dt * tx2
    * ( C1 * ( u[k][j][i+1][1] * tmp1 ) )
    - dt * tx1 * c1345 * tmp1
    - dt * tx1 * dx5;

  //---------------------------------------------------------------------
  // form the second block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j+1][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  bu[j][i][0][0] = - dt * ty1 * dy1;
  bu[j][i][1][0] =   0.0;
  bu[j][i][2][0] =  dt * ty2;
  bu[j][i][3][0] =   0.0;
  bu[j][i][4][0] =   0.0;

  bu[j][i][0][1] =  dt * ty2
    * ( - ( u[k][j+1][i][1]*u[k][j+1][i][2] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j+1][i][1] );
  bu[j][i][1][1] =  dt * ty2 * ( u[k][j+1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy2;
  bu[j][i][2][1] =  dt * ty2 * ( u[k][j+1][i][1] * tmp1 );
  bu[j][i][3][1] = 0.0;
  bu[j][i][4][1] = 0.0;

  bu[j][i][0][2] =  dt * ty2
    * ( - ( u[k][j+1][i][2] * tmp1 ) * ( u[k][j+1][i][2] * tmp1 )
        + C2 * ( qs[k][j+1][i] * tmp1 ) )
    - dt * ty1 * ( - r43 * c34 * tmp2 * u[k][j+1][i][2] );
  bu[j][i][1][2] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][1] * tmp1 ) );
  bu[j][i][2][2] =  dt * ty2 * ( ( 2.0 - C2 )
      * ( u[k][j+1][i][2] * tmp1 ) )
    - dt * ty1 * ( r43 * c34 * tmp1 )
    - dt * ty1 * dy3;
  bu[j][i][3][2] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][3] * tmp1 ) );
  bu[j][i][4][2] =  dt * ty2 * C2;

  bu[j][i][0][3] =  dt * ty2
    * ( - ( u[k][j+1][i][2]*u[k][j+1][i][3] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j+1][i][3] );
  bu[j][i][1][3] = 0.0;
  bu[j][i][2][3] =  dt * ty2 * ( u[k][j+1][i][3] * tmp1 );
  bu[j][i][3][3] =  dt * ty2 * ( u[k][j+1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy4;
  bu[j][i][4][3] = 0.0;

  bu[j][i][0][4] =  dt * ty2
    * ( ( C2 * 2.0 * qs[k][j+1][i]
          - C1 * u[k][j+1][i][4] )
        * ( u[k][j+1][i][2] * tmp2 ) )
    - dt * ty1
    * ( - (     c34 - c1345 )*tmp3*(u[k][j+1][i][1]*u[k][j+1][i][1])
        - ( r43*c34 - c1345 )*tmp3*(u[k][j+1][i][2]*u[k][j+1][i][2])
        - (     c34 - c1345 )*tmp3*(u[k][j+1][i][3]*u[k][j+1][i][3])
        - c1345*tmp2*u[k][j+1][i][4] );
  bu[j][i][1][4] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][1]*u[k][j+1][i][2] ) * tmp2 )
    - dt * ty1
    * ( c34 - c1345 ) * tmp2 * u[k][j+1][i][1];
  bu[j][i][2][4] =  dt * ty2
    * ( C1 * ( u[k][j+1][i][4] * tmp1 )
        - C2 
        * ( qs[k][j+1][i] * tmp1
          + u[k][j+1][i][2]*u[k][j+1][i][2] * tmp2 ) )
    - dt * ty1
    * ( r43*c34 - c1345 ) * tmp2 * u[k][j+1][i][2];
  bu[j][i][3][4] =  dt * ty2
    * ( - C2 * ( u[k][j+1][i][2]*u[k][j+1][i][3] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j+1][i][3];
  bu[j][i][4][4] =  dt * ty2
    * ( C1 * ( u[k][j+1][i][2] * tmp1 ) )
    - dt * ty1 * c1345 * tmp1
    - dt * ty1 * dy5;

  //---------------------------------------------------------------------
  // form the third block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k+1][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  cu[j][i][0][0] = - dt * tz1 * dz1;
  cu[j][i][1][0] =   0.0;
  cu[j][i][2][0] =   0.0;
  cu[j][i][3][0] = dt * tz2;
  cu[j][i][4][0] =   0.0;

  cu[j][i][0][1] = dt * tz2
    * ( - ( u[k+1][j][i][1]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k+1][j][i][1] );
  cu[j][i][1][1] = dt * tz2 * ( u[k+1][j][i][3] * tmp1 )
    - dt * tz1 * c34 * tmp1
    - dt * tz1 * dz2;
  cu[j][i][2][1] = 0.0;
  cu[j][i][3][1] = dt * tz2 * ( u[k+1][j][i][1] * tmp1 );
  cu[j][i][4][1] = 0.0;

  cu[j][i][0][2] = dt * tz2
    * ( - ( u[k+1][j][i][2]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k+1][j][i][2] );
  cu[j][i][1][2] = 0.0;
  cu[j][i][2][2] = dt * tz2 * ( u[k+1][j][i][3] * tmp1 )
    - dt * tz1 * ( c34 * tmp1 )
    - dt * tz1 * dz3;
  cu[j][i][3][2] = dt * tz2 * ( u[k+1][j][i][2] * tmp1 );
  cu[j][i][4][2] = 0.0;

  cu[j][i][0][3] = dt * tz2
    * ( - ( u[k+1][j][i][3] * tmp1 ) * ( u[k+1][j][i][3] * tmp1 )
        + C2 * ( qs[k+1][j][i] * tmp1 ) )
    - dt * tz1 * ( - r43 * c34 * tmp2 * u[k+1][j][i][3] );
  cu[j][i][1][3] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][1] * tmp1 ) );
  cu[j][i][2][3] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][2] * tmp1 ) );
  cu[j][i][3][3] = dt * tz2 * ( 2.0 - C2 )
    * ( u[k+1][j][i][3] * tmp1 )
    - dt * tz1 * ( r43 * c34 * tmp1 )
    - dt * tz1 * dz4;
  cu[j][i][4][3] = dt * tz2 * C2;

  cu[j][i][0][4] = dt * tz2
    * ( ( C2 * 2.0 * qs[k+1][j][i]
          - C1 * u[k+1][j][i][4] )
        * ( u[k+1][j][i][3] * tmp2 ) )
    - dt * tz1
    * ( - ( c34 - c1345 ) * tmp3 * (u[k+1][j][i][1]*u[k+1][j][i][1])
        - ( c34 - c1345 ) * tmp3 * (u[k+1][j][i][2]*u[k+1][j][i][2])
        - ( r43*c34 - c1345 )* tmp3 * (u[k+1][j][i][3]*u[k+1][j][i][3])
        - c1345 * tmp2 * u[k+1][j][i][4] );
  cu[j][i][1][4] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][1]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k+1][j][i][1];
  cu[j][i][2][4] = dt * tz2
    * ( - C2 * ( u[k+1][j][i][2]*u[k+1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k+1][j][i][2];
  cu[j][i][3][4] = dt * tz2
    * ( C1 * ( u[k+1][j][i][4] * tmp1 )
        - C2
        * ( qs[k+1][j][i] * tmp1
          + u[k+1][j][i][3]*u[k+1][j][i][3] * tmp2 ) )
    - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[k+1][j][i][3];
  cu[j][i][4][4] = dt * tz2
    * ( C1 * ( u[k+1][j][i][3] * tmp1 ) )
    - dt * tz1 * c1345 * tmp1
    - dt * tz1 * dz5;





}

__kernel void buts_KL_baseline(__global double *m_rsd,
                               __global double *m_u,
                               __global double *m_qs,
                               __global double *m_rho_i,
                               __global double *m_au,
                               __global double *m_bu,
                               __global double *m_cu,
                               __global double *m_du,
                               int nz, int ny, int nx,
                               int wf_sum, int wf_base_k, int wf_base_j,
                               int jst, int jend, 
                               int ist, int iend, 
                               int temp_kst, int temp_kend)
{
  int k, j, i, m;
  double r43, c1345, c34;
  double tmp, tmp1, tmp2, tmp3;
  double tmat[5][5], tv[5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

  __global double (*au)[5];
  __global double (*bu)[5];
  __global double (*cu)[5];
  __global double (*du)[5];

  k = get_global_id(1) + temp_kst + wf_base_k;
  j = get_global_id(0) + jst + wf_base_j;
  i = wf_sum - get_global_id(1) - get_global_id(0) - wf_base_k - wf_base_j + ist;
  if (k >= temp_kend || j >= jend || i < ist || i >= iend) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;
  u = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  qs = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;

  au = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_au)[k][j][i];
  bu = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_bu)[k][j][i];
  cu = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_cu)[k][j][i];
  du = ((__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5][5])m_du)[k][j][i];

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  for (m = 0; m < 5; m++) {
    tv[m] = 
      omega * (  cu[0][m] * rsd[k+1][j][i][0]
          + cu[1][m] * rsd[k+1][j][i][1]
          + cu[2][m] * rsd[k+1][j][i][2]
          + cu[3][m] * rsd[k+1][j][i][3]
          + cu[4][m] * rsd[k+1][j][i][4] );
  }
  for (m = 0; m < 5; m++) {
    tv[m] = tv[m]
      + omega * ( bu[0][m] * rsd[k][j+1][i][0]
          + au[0][m] * rsd[k][j][i+1][0]
          + bu[1][m] * rsd[k][j+1][i][1]
          + au[1][m] * rsd[k][j][i+1][1]
          + bu[2][m] * rsd[k][j+1][i][2]
          + au[2][m] * rsd[k][j][i+1][2]
          + bu[3][m] * rsd[k][j+1][i][3]
          + au[3][m] * rsd[k][j][i+1][3]
          + bu[4][m] * rsd[k][j+1][i][4]
          + au[4][m] * rsd[k][j][i+1][4] );
  }

  //---------------------------------------------------------------------
  // diagonal block inversion
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    tmat[m][0] = du[0][m];
    tmat[m][1] = du[1][m];
    tmat[m][2] = du[2][m];
    tmat[m][3] = du[3][m];
    tmat[m][4] = du[4][m];
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
  tv[4] = tv[4] / tmat[4][4];

  tv[3] = tv[3] - tmat[3][4] * tv[4];
  tv[3] = tv[3] / tmat[3][3];

  tv[2] = tv[2]
    - tmat[2][3] * tv[3]
    - tmat[2][4] * tv[4];
  tv[2] = tv[2] / tmat[2][2];

  tv[1] = tv[1]
    - tmat[1][2] * tv[2]
    - tmat[1][3] * tv[3]
    - tmat[1][4] * tv[4];
  tv[1] = tv[1] / tmat[1][1];

  tv[0] = tv[0]
    - tmat[0][1] * tv[1]
    - tmat[0][2] * tv[2]
    - tmat[0][3] * tv[3]
    - tmat[0][4] * tv[4];
  tv[0] = tv[0] / tmat[0][0];

  rsd[k][j][i][0] = rsd[k][j][i][0] - tv[0];
  rsd[k][j][i][1] = rsd[k][j][i][1] - tv[1];
  rsd[k][j][i][2] = rsd[k][j][i][2] - tv[2];
  rsd[k][j][i][3] = rsd[k][j][i][3] - tv[3];
  rsd[k][j][i][4] = rsd[k][j][i][4] - tv[4];
}
