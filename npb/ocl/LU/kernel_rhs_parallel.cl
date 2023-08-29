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

__kernel void rhs1_parallel(__global double * m_rsd, 
                            __global double * m_frct, 
                            int nx, int ny, int nz,
                            int work_base, 
                            int work_num_item, 
                            int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int t_i = get_global_id(0);
  int i = t_i / 5;
  int m = t_i % 5;
  if(k+work_base >= nz || k >= work_num_item || j >= ny || i >= nx) return;
  
  if(split_flag) k += 2;
  else k += work_base;

  __global double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;
  __global double (* frct)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_frct;

  rsd[k][j][i][m] = - frct[k][j][i][m];

}



__kernel void rhs1_datagen_parallel(__global double *m_u, 
                                    __global double *m_rho_i,
                                    __global double *m_qs, 
                                    int nx, int ny, int nz,
                                    int u_copy_buffer_base, 
                                    int u_copy_num_item)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);
  
  if(k >= u_copy_num_item || j >= ny || i >= nx) return;

  k += u_copy_buffer_base;

  double tmp;

  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i;
  __global double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_qs;

  tmp = 1.0 / u[k][j][i][0];
  rho_i[k][j][i] = tmp;
  qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
      + u[k][j][i][2] * u[k][j][i][2]
      + u[k][j][i][3] * u[k][j][i][3] ) * tmp;
}



//---------------------------------------------------------------------
// xi-direction flux differences
//---------------------------------------------------------------------
__kernel void rhsx1_parallel(__global double *m_flux,
                             __global double *m_u,
                             __global double *m_rho_i, 
                             __global double *m_qs,
                             __global double *m_rsd,
                             int jst, int jend, 
                             int ist, int iend,
                             double tx2, int nz,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{

  int k = get_global_id(2);
  int j = get_global_id(1)+jst;
  int i = get_global_id(0)+ist;
  int l_i = get_local_id(0)+1;
  int l_isize = get_local_size(0);
  int wg_id = get_group_id(0);
  int dummy = 0;
  int m;

  if(k+work_base < 1 || k+work_base >= nz - 1 || k >= work_num_item || j >= jend || i >= iend) dummy = 1;


  double u21, q;

  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i ;
  __global double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_qs ;
  __global double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_rsd ;
   __global double (* tmp_flux)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_flux ;

  if (split_flag) k += 2;
  else k += work_base;

  __global double (* flux)[5] = &(tmp_flux[k][j][wg_id*l_isize]);


  if (!dummy) {
    flux[l_i][0] = u[k][j][i][1];
    u21 = u[k][j][i][1] * rho_i[k][j][i];

    q = qs[k][j][i];

    flux[l_i][1] = u[k][j][i][1] * u21 + C2 * ( u[k][j][i][4] - q );
    flux[l_i][2] = u[k][j][i][2] * u21;
    flux[l_i][3] = u[k][j][i][3] * u21;
    flux[l_i][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u21;

    if(l_i == 1){

      flux[l_i-1][0] = u[k][j][i-1][1];
      u21 = u[k][j][i-1][1] * rho_i[k][j][i-1];

      q = qs[k][j][i-1];

      flux[l_i-1][1] = u[k][j][i-1][1] * u21 + C2 * ( u[k][j][i-1][4] - q );
      flux[l_i-1][2] = u[k][j][i-1][2] * u21;
      flux[l_i-1][3] = u[k][j][i-1][3] * u21;
      flux[l_i-1][4] = ( C1 * u[k][j][i-1][4] - C2 * q ) * u21;

    }

    if(l_i == l_isize || i == iend -1){

      flux[l_i+1][0] = u[k][j][i+1][1];
      u21 = u[k][j][i+1][1] * rho_i[k][j][i+1];

      q = qs[k][j][i+1];

      flux[l_i+1][1] = u[k][j][i+1][1] * u21 + C2 * ( u[k][j][i+1][4] - q );
      flux[l_i+1][2] = u[k][j][i+1][2] * u21;
      flux[l_i+1][3] = u[k][j][i+1][3] * u21;
      flux[l_i+1][4] = ( C1 * u[k][j][i+1][4] - C2 * q ) * u21;

    }
  } 

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (!dummy) {
    for (m = 0; m < 5; m++) {
      rsd[k][j][i][m] =  rsd[k][j][i][m]
        - tx2 * ( flux[l_i+1][m] - flux[l_i-1][m] );
    }
  }
}




__kernel void rhsx2_parallel(__global double *m_flux,
                             __global double *m_u, 
                             __global double *m_rsd,
                             __global double *m_rho_i,
                             int jst, int jend, 
                             int ist, int iend, 
                             double tx1, double tx2, double tx3, 
                             double dx1, double dx2, 
                             double dx3, double dx4, 
                             double dx5,
                             int nx, int nz,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1)+jst;
  int i = get_global_id(0)+ist;
  int l_i = get_local_id(0)+1;
  int l_isize = get_local_size(0);
  int wg_id = get_group_id(0);
  int dummy = 0;
  int m;
  if(k+work_base < 1 || k+work_base >= nz - 1 || k >= work_num_item || j >= jend || i >= iend ) dummy = 1;

  double u21i, u31i, u41i, u51i, u21im1, u31im1, u41im1, u51im1, tmp;

  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1] )m_rho_i ;
  __global double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_rsd ;
   __global double (* tmp_flux)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_flux ;

  if (split_flag) k += 2;
  else k += work_base;

  __global double (* flux)[5] = &(tmp_flux[k][j][wg_id*l_isize]);
  __global double (* l_u)[5] = &(u[k][j][wg_id*l_isize]);

  if (!dummy) {
    tmp = rho_i[k][j][i];

    u21i = tmp * l_u[l_i][1];
    u31i = tmp * l_u[l_i][2];
    u41i = tmp * l_u[l_i][3];
    u51i = tmp * l_u[l_i][4];

    tmp = rho_i[k][j][i-1];

    u21im1 = tmp * l_u[l_i-1][1];
    u31im1 = tmp * l_u[l_i-1][2];
    u41im1 = tmp * l_u[l_i-1][3];
    u51im1 = tmp * l_u[l_i-1][4];

    flux[l_i][1] = (4.0/3.0) * tx3 * (u21i-u21im1);
    flux[l_i][2] = tx3 * ( u31i - u31im1 );
    flux[l_i][3] = tx3 * ( u41i - u41im1 );
    flux[l_i][4] = 0.50 * ( 1.0 - C1*C5 )
      * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
          - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
      + (1.0/6.0)
      * tx3 * ( u21i*u21i - u21im1*u21im1 )
      + C1 * C5 * tx3 * ( u51i - u51im1 );

    if(l_i == l_isize || i == iend - 1){
      l_i++;
      i++;

      tmp = rho_i[k][j][i];

      u21i = tmp * l_u[l_i][1];
      u31i = tmp * l_u[l_i][2];
      u41i = tmp * l_u[l_i][3];
      u51i = tmp * l_u[l_i][4];

      tmp = rho_i[k][j][i-1];

      u21im1 = tmp * l_u[l_i-1][1];
      u31im1 = tmp * l_u[l_i-1][2];
      u41im1 = tmp * l_u[l_i-1][3];
      u51im1 = tmp * l_u[l_i-1][4];

      flux[l_i][1] = (4.0/3.0) * tx3 * (u21i-u21im1);
      flux[l_i][2] = tx3 * ( u31i - u31im1 );
      flux[l_i][3] = tx3 * ( u41i - u41im1 );
      flux[l_i][4] = 0.50 * ( 1.0 - C1*C5 )
        * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
            - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
        + (1.0/6.0)
        * tx3 * ( u21i*u21i - u21im1*u21im1 )
        + C1 * C5 * tx3 * ( u51i - u51im1 );

      l_i--;
      i--;
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);


  if (!dummy) {

    rsd[k][j][i][0] = rsd[k][j][i][0]
      + dx1 * tx1 * (        l_u[l_i-1][0]
          - 2.0 * l_u[l_i][0]
          +       l_u[l_i+1][0] );
    rsd[k][j][i][1] = rsd[k][j][i][1]
      + tx3 * C3 * C4 * ( flux[l_i+1][1] - flux[l_i][1] )
      + dx2 * tx1 * (        l_u[l_i-1][1]
          - 2.0 * l_u[l_i][1]
          +       l_u[l_i+1][1] );
    rsd[k][j][i][2] = rsd[k][j][i][2]
      + tx3 * C3 * C4 * ( flux[l_i+1][2] - flux[l_i][2] )
      + dx3 * tx1 * (        l_u[l_i-1][2]
          - 2.0 * l_u[l_i][2]
          +       l_u[l_i+1][2] );
    rsd[k][j][i][3] = rsd[k][j][i][3]
      + tx3 * C3 * C4 * ( flux[l_i+1][3] - flux[l_i][3] )
      + dx4 * tx1 * (        l_u[l_i-1][3]
          - 2.0 * l_u[l_i][3]
          +       l_u[l_i+1][3] );
    rsd[k][j][i][4] = rsd[k][j][i][4]
      + tx3 * C3 * C4 * ( flux[l_i+1][4] - flux[l_i][4] )
      + dx5 * tx1 * (        l_u[l_i-1][4]
          - 2.0 * l_u[l_i][4]
          +       l_u[l_i+1][4] );

  }
}

//---------------------------------------------------------------------
// Fourth-order dissipation
//---------------------------------------------------------------------
__kernel void rhsx3_parallel(__global double *m_u,
                             __global double *m_rsd,
                             int jst, int jend,
                             double dssp, int nx, int nz,
                             int work_base, 
                             int work_num_item, 
                             int split_flag)
{

  int k = get_global_id(2);
  int j = get_global_id(1)+jst;
  int t_i = get_global_id(0);
  int i = t_i / 5 + 1;
  int m = t_i % 5;
  int l_i = get_local_id(0) / 5 + 2;
  int l_isize = get_local_size(0) / 5;
  int dummy = 0;

  if(k+work_base < 1 || k+work_base >= nz - 1 || k >= work_num_item || j >= jend || i >= nx-1) return;

  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_u;
  __global double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5] )m_rsd ;

  if(split_flag) k += 2;
  else k += work_base;

  if (i == 1) {
    rsd[k][j][1][m] = rsd[k][j][1][m]
      - dssp * ( + 5.0 * u[k][j][1][m]
          - 4.0 * u[k][j][2][m]
          +       u[k][j][3][m] );
  }
  else if (i == 2) {
    rsd[k][j][2][m] = rsd[k][j][2][m]
      - dssp * ( - 4.0 * u[k][j][1][m]
          + 6.0 * u[k][j][2][m]
          - 4.0 * u[k][j][3][m]
          +       u[k][j][4][m] );
  }
  else if (i == nx-3) {
    rsd[k][j][nx-3][m] = rsd[k][j][nx-3][m]
      - dssp * (         u[k][j][nx-5][m]
          - 4.0 * u[k][j][nx-4][m]
          + 6.0 * u[k][j][nx-3][m]
          - 4.0 * u[k][j][nx-2][m] );
  }
  else if (i == nx-2) {
    rsd[k][j][nx-2][m] = rsd[k][j][nx-2][m]
      - dssp * (         u[k][j][nx-4][m]
          - 4.0 * u[k][j][nx-3][m]
          + 5.0 * u[k][j][nx-2][m] );

  }
  else {
    rsd[k][j][i][m] = rsd[k][j][i][m]
      - dssp * (         u[k][j][i-2][m]
          - 4.0 * u[k][j][i-1][m]
          + 6.0 * u[k][j][i][m]
          - 4.0 * u[k][j][i+1][m]
          +       u[k][j][i+2][m] );
  }
}

__kernel void rhsy_parallel(__global double *m_flux,
                            __global double *m_u,
                            __global double *m_rho_i,
                            __global double *m_qs,
                            __global double *m_rsd,
                            double ty1, double ty2,
                            double ty3,
                            double dy1, double dy2,
                            double dy3, double dy4,
                            double dy5,
                            double dssp,
                            int nz, int ny,
                            int jst, int jend,
                            int ist, int iend,
                            int work_base,
                            int work_num_item,
                            int split_flag)
{

  int k = get_global_id(0);
  int j, i, m;
  double u31, q,
         u21j, u31j,
         u41j, u51j,
         u21jm1, u31jm1,
         u41jm1, u51jm1,
         tmp;

  if (k+work_base < 1 || k+work_base >= nz-1) 
    return;

  if (split_flag) k += 2;
  else k += work_base;

   __global double (* tmp_flux)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_flux ;
  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_u;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1]) m_rho_i ;
  __global double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1]) m_qs ;
  __global double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_rsd ;

  __global double (* flux)[5];

  for (i = ist; i < iend; i++) {
    flux = tmp_flux[k][i];

    for (j = 0; j < ny; j++) {
      flux[j][0] = u[k][j][i][2];
      u31 = u[k][j][i][2] * rho_i[k][j][i];

      q = qs[k][j][i];

      flux[j][1] = u[k][j][i][1] * u31;
      flux[j][2] = u[k][j][i][2] * u31 + C2 * (u[k][j][i][4]-q);
      flux[j][3] = u[k][j][i][3] * u31;
      flux[j][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u31;
    }

    for (j = jst; j < jend; j++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] =  rsd[k][j][i][m]
          - ty2 * ( flux[j+1][m] - flux[j-1][m] );
      }
    }

    for (j = jst; j < ny; j++) {
      tmp = rho_i[k][j][i];

      u21j = tmp * u[k][j][i][1];
      u31j = tmp * u[k][j][i][2];
      u41j = tmp * u[k][j][i][3];
      u51j = tmp * u[k][j][i][4];

      tmp = rho_i[k][j-1][i];
      u21jm1 = tmp * u[k][j-1][i][1];
      u31jm1 = tmp * u[k][j-1][i][2];
      u41jm1 = tmp * u[k][j-1][i][3];
      u51jm1 = tmp * u[k][j-1][i][4];

      flux[j][1] = ty3 * ( u21j - u21jm1 );
      flux[j][2] = (4.0/3.0) * ty3 * (u31j-u31jm1);
      flux[j][3] = ty3 * ( u41j - u41jm1 );
      flux[j][4] = 0.50 * ( 1.0 - C1*C5 )
        * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
            - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
        + (1.0/6.0)
        * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
        + C1 * C5 * ty3 * ( u51j - u51jm1 );
    }

    for (j = jst; j < jend; j++) {
      rsd[k][j][i][0] = rsd[k][j][i][0]
        + dy1 * ty1 * (         u[k][j-1][i][0]
            - 2.0 * u[k][j][i][0]
            +       u[k][j+1][i][0] );

      rsd[k][j][i][1] = rsd[k][j][i][1]
        + ty3 * C3 * C4 * ( flux[j+1][1] - flux[j][1] )
        + dy2 * ty1 * (         u[k][j-1][i][1]
            - 2.0 * u[k][j][i][1]
            +       u[k][j+1][i][1] );

      rsd[k][j][i][2] = rsd[k][j][i][2]
        + ty3 * C3 * C4 * ( flux[j+1][2] - flux[j][2] )
        + dy3 * ty1 * (         u[k][j-1][i][2]
            - 2.0 * u[k][j][i][2]
            +       u[k][j+1][i][2] );

      rsd[k][j][i][3] = rsd[k][j][i][3]
        + ty3 * C3 * C4 * ( flux[j+1][3] - flux[j][3] )
        + dy4 * ty1 * (         u[k][j-1][i][3]
            - 2.0 * u[k][j][i][3]
            +       u[k][j+1][i][3] );

      rsd[k][j][i][4] = rsd[k][j][i][4]
        + ty3 * C3 * C4 * ( flux[j+1][4] - flux[j][4] )
        + dy5 * ty1 * (         u[k][j-1][i][4]
            - 2.0 * u[k][j][i][4]
            +       u[k][j+1][i][4] );
    }
  }

  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------
  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      rsd[k][1][i][m] = rsd[k][1][i][m]
        - dssp * ( + 5.0 * u[k][1][i][m]
            - 4.0 * u[k][2][i][m]
            +       u[k][3][i][m] );
      rsd[k][2][i][m] = rsd[k][2][i][m]
        - dssp * ( - 4.0 * u[k][1][i][m]
            + 6.0 * u[k][2][i][m]
            - 4.0 * u[k][3][i][m]
            +       u[k][4][i][m] );
    }
  }

  for (j = 3; j < ny - 3; j++) {
    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rsd[k][j][i][m]
          - dssp * (         u[k][j-2][i][m]
              - 4.0 * u[k][j-1][i][m]
              + 6.0 * u[k][j][i][m]
              - 4.0 * u[k][j+1][i][m]
              +       u[k][j+2][i][m] );
      }
    }
  }

  for (i = ist; i < iend; i++) {
    for (m = 0; m < 5; m++) {
      rsd[k][ny-3][i][m] = rsd[k][ny-3][i][m]
        - dssp * (         u[k][ny-5][i][m]
            - 4.0 * u[k][ny-4][i][m]
            + 6.0 * u[k][ny-3][i][m]
            - 4.0 * u[k][ny-2][i][m] );
      rsd[k][ny-2][i][m] = rsd[k][ny-2][i][m]
        - dssp * (         u[k][ny-4][i][m]
            - 4.0 * u[k][ny-3][i][m]
            + 5.0 * u[k][ny-2][i][m] );
    }
  }
}


// Not implemented for split case
__kernel void rhsz_parallel(__global double *m_flux,
                            __global double *m_utmp,
                            __global double *m_rtmp,
                            __global double *m_u,
                            __global double *m_rho_i,
                            __global double *m_qs,
                            __global double *m_rsd,
                            double tz1, double tz2,
                            double tz3,
                            double dz1, double dz2,
                            double dz3, double dz4,
                            double dz5,
                            double dssp,
                            int nz,
                            int jst, int jend,
                            int ist, int iend,
                            int work_base,
                            int work_num_item,
                            int split_flag)
{
  int j = get_global_id(1) + jst;
  int i = get_global_id(0) + ist;
  int k, m;
  int kst, kend;
  int orig_kst, orig_k;

  double u41, q, 
         u21k, u31k,
         u41k, u51k,
         u21km1, u31km1,
         u41km1, u51km1,
         tmp;

   __global double (* tmp_flux)[ISIZ2/2*2+1][WORK_NUM_ITEM_DEFAULT][5]
    = (__global double (*)[ISIZ2/2*2+1][WORK_NUM_ITEM_DEFAULT][5]) m_flux;
   __global double (* tmp_utmp)[ISIZ2/2*2+1][WORK_NUM_ITEM_DEFAULT][6]
    = (__global double (*)[ISIZ2/2*2+1][WORK_NUM_ITEM_DEFAULT][6]) m_utmp;
   __global double (* tmp_rtmp)[ISIZ2/2*2+1][WORK_NUM_ITEM_DEFAULT][5]
    = (__global double (*)[ISIZ2/2*2+1][WORK_NUM_ITEM_DEFAULT][5]) m_rtmp;

  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_u;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1]) m_rho_i ;
  __global double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1]) m_qs ;
  __global double (* rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]) m_rsd ;

  __global double (* flux)[5] = tmp_flux[j][i];
  __global double (* utmp)[6] = tmp_utmp[j][i];
  __global double (* rtmp)[5] = tmp_rtmp[j][i];

  if (j >= jend || i >= iend) 
    return;

  if (split_flag) {

    kst = (work_base < 1) ? 1 : 0;
    kend = min(nz-1 - work_base, work_num_item);

    // front padding align
    kst += 2;
    kend += 2;
  }
  else {
    kst = 1;
    kend = nz-1;
  }

  if (split_flag) orig_kst = kst - 2 + work_base;
  else orig_kst = kst;

  for (k = kst - 2; k < kend + 2; k++) {
    orig_k = (split_flag) ? k-2 + work_base : k;
    if (orig_k < 0 || orig_k > nz-1)
      continue; 
    utmp[k][0] = u[k][j][i][0];
    utmp[k][1] = u[k][j][i][1];
    utmp[k][2] = u[k][j][i][2];
    utmp[k][3] = u[k][j][i][3];
    utmp[k][4] = u[k][j][i][4];
    utmp[k][5] = rho_i[k][j][i];
  }

  for (k = kst - 1; k < kend + 1; k++) {
    flux[k][0] = utmp[k][3];
    u41 = utmp[k][3] * utmp[k][5];

    q = qs[k][j][i];

    flux[k][1] = utmp[k][1] * u41;
    flux[k][2] = utmp[k][2] * u41;
    flux[k][3] = utmp[k][3] * u41 + C2 * (utmp[k][4]-q);
    flux[k][4] = ( C1 * utmp[k][4] - C2 * q ) * u41;
  }

  for (k = kst; k < kend; k++) {
    for (m = 0; m < 5; m++) {
      rtmp[k][m] =  rsd[k][j][i][m]
        - tz2 * ( flux[k+1][m] - flux[k-1][m] );
    }
  }

  for (k = kst; k < kend + 1; k++) {
    tmp = utmp[k][5];

    u21k = tmp * utmp[k][1];
    u31k = tmp * utmp[k][2];
    u41k = tmp * utmp[k][3];
    u51k = tmp * utmp[k][4];

    tmp = utmp[k-1][5];

    u21km1 = tmp * utmp[k-1][1];
    u31km1 = tmp * utmp[k-1][2];
    u41km1 = tmp * utmp[k-1][3];
    u51km1 = tmp * utmp[k-1][4];

    flux[k][1] = tz3 * ( u21k - u21km1 );
    flux[k][2] = tz3 * ( u31k - u31km1 );
    flux[k][3] = (4.0/3.0) * tz3 * (u41k-u41km1);
    flux[k][4] = 0.50 * ( 1.0 - C1*C5 )
      * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
          - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
      + (1.0/6.0)
      * tz3 * ( u41k*u41k - u41km1*u41km1 )
      + C1 * C5 * tz3 * ( u51k - u51km1 );

  }

  for (k = kst; k < kend; k++) {
    rtmp[k][0] = rtmp[k][0]
      + dz1 * tz1 * (         utmp[k-1][0]
          - 2.0 * utmp[k][0]
          +       utmp[k+1][0] );
    rtmp[k][1] = rtmp[k][1]
      + tz3 * C3 * C4 * ( flux[k+1][1] - flux[k][1] )
      + dz2 * tz1 * (         utmp[k-1][1]
          - 2.0 * utmp[k][1]
          +       utmp[k+1][1] );
    rtmp[k][2] = rtmp[k][2]
      + tz3 * C3 * C4 * ( flux[k+1][2] - flux[k][2] )
      + dz3 * tz1 * (         utmp[k-1][2]
          - 2.0 * utmp[k][2]
          +       utmp[k+1][2] );
    rtmp[k][3] = rtmp[k][3]
      + tz3 * C3 * C4 * ( flux[k+1][3] - flux[k][3] )
      + dz4 * tz1 * (         utmp[k-1][3]
          - 2.0 * utmp[k][3]
          +       utmp[k+1][3] );
    rtmp[k][4] = rtmp[k][4]
      + tz3 * C3 * C4 * ( flux[k+1][4] - flux[k][4] )
      + dz5 * tz1 * (         utmp[k-1][4]
          - 2.0 * utmp[k][4]
          +       utmp[k+1][4] );
  }
  //---------------------------------------------------------------------
  // fourth-order dissipation
  //---------------------------------------------------------------------
  for (k = kst; k < kend; k++) {
    orig_k = (split_flag) ? k-2 + work_base : k;

    if (orig_k == 1) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rtmp[k][m]
          - dssp * ( + 5.0 * utmp[k][m]
              - 4.0 * utmp[k+1][m]
              +       utmp[k+2][m] );
      }
    }
    else if (orig_k == 2) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rtmp[k][m]
          - dssp * ( - 4.0 * utmp[k-1][m]
              + 6.0 * utmp[k][m]
              - 4.0 * utmp[k+1][m]
              +       utmp[k+2][m] );
      }
    }
    else if (orig_k == nz-3) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rtmp[k][m]
          - dssp * (      utmp[k-2][m]
              - 4.0 * utmp[k-1][m]
              + 6.0 * utmp[k][m]
              - 4.0 * utmp[k+1][m] );

      }
    }
    else if (orig_k == nz-2) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rtmp[k][m]
          - dssp * (      utmp[k-2][m]
              - 4.0 * utmp[k-1][m]
              + 5.0 * utmp[k][m] );
      }
    }
    else {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] = rtmp[k][m]
          - dssp * (         utmp[k-2][m]
              - 4.0 * utmp[k-1][m]
              + 6.0 * utmp[k][m]
              - 4.0 * utmp[k+1][m]
              +       utmp[k+2][m] );
      }
    }
  }
}

