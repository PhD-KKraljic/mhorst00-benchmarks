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

__kernel void jacld_blts_datagen_fullopt(__global double *m_u, 
                                         __global double *m_qs,
                                         __global double *m_rho_i,
                                         int kend, int jend, int iend, 
                                         int work_base, 
                                         int work_num_item)
{

  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);

  if (k+work_base > kend || k > work_num_item || j >= jend || i >= iend) return;

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

__kernel void jacld_blts_datacopy_fullopt(__global double *m_rsd, 
                                          __global double *m_rsd_next,
                                          __global double *m_u, 
                                          __global double *m_u_next,
                                          int jst, int jend, 
                                          int ist, int iend, 
                                          int work_num_item)
{

  __global double (* rsd) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;  
  __global double (* rsd_next) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd_next;

  __global double (* u) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;  
  __global double (* u_next) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u_next;

  int k = work_num_item;
  int j = get_global_id(1) + jst;
  int t_i = get_global_id(0);
  int i = t_i / 5 + ist;
  int m = t_i % 5;

  if(j >= jend || i >= iend) return;

  rsd_next[0][j][i][m] = rsd[k][j][i][m];
  u_next[0][j][i][m] = u[k][j][i][m];


}



//---------------------------------------------------------------------
// compute the lower triangular part of the jacobian matrix
//---------------------------------------------------------------------
  __kernel __attribute__((reqd_work_group_size(JACLD_BLTS_LWS, 1, 1)))
void jacld_blts_BR_fullopt(__global double *m_v, 
                           __global double *m_rho_i,
                           __global double *m_u, 
                           __global double *m_qs,
                           int kst, int kend, 
                           int jst, int jend, 
                           int ist, int iend,
                           int wg00_head_k, 
                           int wg00_head_j, 
                           int wg00_head_i, 
                           int wg00_block_k, 
                           int wg00_block_j, 
                           int wg00_block_i, 
                           int num_block_k, 
                           int num_block_j, 
                           int num_block_i, 
                           int block_size,
                           int block_size_k)
{
  double abcd[5][5];
  double tv[5];

  int wg_id = get_group_id(0);
  int l_j, l_i;
  int l_isize = block_size;


  int wg_head_k, wg_head_j, wg_head_i;

  int k, j, i, m, n;
  int cur_diag_blocks;
  int step;
  int iter;
  int dummy = 0;
  double r43;
  double c1345;
  double c34;
  double tmp, tmp1, tmp2, tmp3;

  __global double (* v) [ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_v;
  __global double (* rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*) [ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;
  __global double (* u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  __global double (* qs)[ISIZ2/2*2+1][ISIZ1/2*2+1]
    = (__global double (*) [ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;


  double temp_v[5];
  double temp_u[5];

  int g_wg_id;
  int head;
  int wg_block_i, wg_block_j;
  int quotient, rsd, diag;
  int tmp_min, tmp_max;
  int h_b_diag, diag_tmp, b_id;
  int id, h_id, h_diag;

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  wg_head_k = wg00_head_k;
  wg_head_j = wg00_head_j;
  wg_head_i = wg00_head_i;

  id = get_local_id(0);

  // get current work_groups_head
  while(1){
    cur_diag_blocks = 
      (wg00_block_j+1 < num_block_i - wg00_block_i) ? (wg00_block_j+1) : (num_block_i - wg00_block_i);

    if(cur_diag_blocks > wg_id){
      wg_head_j -= wg_id*block_size;
      wg_head_i += wg_id*block_size;
      break;
    }
    else{
      if(wg00_block_j < num_block_j){
        wg00_block_j++;
        wg_head_j += block_size;

        if(wg00_block_j >= num_block_j){
          wg00_block_j--;
          wg00_block_i++;
          wg_head_j -= block_size;
          wg_head_i += block_size;

        }

      }
      else{
        wg00_block_i++;
        wg_head_i += block_size;
      }
      wg_head_k -= block_size_k;
      wg_id -= cur_diag_blocks;
    }
  }
  wg_id = get_group_id(0);

  // remapping work item order
  if (id < block_size*(block_size+1)/2) {
    diag = floor((sqrt((float)(1+8*id))-1)/2);
    head = diag*(diag + 1) / 2;
    l_j = diag - (id - head);
    l_i = id - head;
  }
  else {
    diag = 2*block_size-1 - floor((sqrt((float)(1+8*(block_size*block_size-1 - id)))-1)/2) - 1;

    diag_tmp = 2*block_size-2 - diag;
    head = block_size*block_size - 1 - diag_tmp*(diag_tmp + 1) / 2;

    l_i = (block_size - 1) - (head - id);
    l_j = (block_size - 1) - diag_tmp + head - id;
  }

  k = wg_head_k - (l_i + l_j);
  j = wg_head_j + l_j;
  i = wg_head_i + l_i;

  iter = min(kend - wg_head_k, block_size_k);
  iter += min(jend - wg_head_j, block_size);
  iter += min(iend - wg_head_i, block_size);
  iter -= 2;

  for (step = 0; step < iter; step++) {

    dummy = 0;

    // remapping item order
    if (step >= block_size_k) {
      // get id of wg0's item
      h_diag = step - block_size_k + 1;

      if (h_diag <= block_size) {
        h_id = h_diag*(h_diag+1)/2; 
      }
      else {
        diag_tmp = 2*block_size-1 - h_diag;
        h_id = block_size*block_size - diag_tmp*(diag_tmp+1)/2;
      }

      id = get_local_id(0) + h_id;

      if (id < block_size*(block_size+1)/2) {
        diag = floor((sqrt((float)(1+8*id))-1)/2);
        head = diag*(diag + 1) / 2;
        l_j = diag - (id - head);
        l_i = id - head;
      }
      else {
        diag = 2*block_size-1 - floor((sqrt((float)(1+8*(block_size*block_size-1 - id)))-1)/2) - 1;

        diag_tmp = 2*block_size-2 - diag;
        head = block_size*block_size - 1 - diag_tmp*(diag_tmp + 1) / 2;

        l_i = (block_size - 1) - (head - id);
        l_j = (block_size - 1) - diag_tmp + head - id;
      }

      k = wg_head_k + block_size_k-1 - ((l_i+l_j) - h_diag);;
      j = wg_head_j + l_j;
      i = wg_head_i + l_i;
    }
    else 
      k = wg_head_k - (l_i+l_j) + step;

    if (k >= wg_head_k + block_size_k || k < wg_head_k || k >= kend || j >= jend || i >= iend || id >= block_size*block_size) 
      dummy = 1;


    if (!dummy) {

      //#####################################################
      //#####################################################
      //          PART 1 - 1
      //#####################################################
      //#####################################################

      for (m = 1; m < 5; m++) temp_u[m] = u[k-1][j][i][m];

      tmp1 = rho_i[k-1][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      abcd[0][0] = - dt * tz1 * dz1;
      abcd[1][0] =   0.0;
      abcd[2][0] =   0.0;
      abcd[3][0] = - dt * tz2;
      abcd[4][0] =   0.0;

      abcd[0][1] = - dt * tz2
        * ( - ( temp_u[1]*temp_u[3] ) * tmp2 )
        - dt * tz1 * ( - c34 * tmp2 * temp_u[1] );
      abcd[1][1] = - dt * tz2 * ( temp_u[3] * tmp1 )
        - dt * tz1 * c34 * tmp1
        - dt * tz1 * dz2;
      abcd[2][1] = 0.0;
      abcd[3][1] = - dt * tz2 * ( temp_u[1] * tmp1 );
      abcd[4][1] = 0.0;

      abcd[0][2] = - dt * tz2
        * ( - ( temp_u[2]*temp_u[3] ) * tmp2 )
        - dt * tz1 * ( - c34 * tmp2 * temp_u[2] );
      abcd[1][2] = 0.0;
      abcd[2][2] = - dt * tz2 * ( temp_u[3] * tmp1 )
        - dt * tz1 * ( c34 * tmp1 )
        - dt * tz1 * dz3;
      abcd[3][2] = - dt * tz2 * ( temp_u[2] * tmp1 );
      abcd[4][2] = 0.0;

      abcd[0][3] = - dt * tz2
        * ( - ( temp_u[3] * tmp1 ) * ( temp_u[3] * tmp1 )
            + C2 * qs[k-1][j][i] * tmp1 )
        - dt * tz1 * ( - r43 * c34 * tmp2 * temp_u[3] );
      abcd[1][3] = - dt * tz2
        * ( - C2 * ( temp_u[1] * tmp1 ) );
      abcd[2][3] = - dt * tz2
        * ( - C2 * ( temp_u[2] * tmp1 ) );
      abcd[3][3] = - dt * tz2 * ( 2.0 - C2 )
        * ( temp_u[3] * tmp1 )
        - dt * tz1 * ( r43 * c34 * tmp1 )
        - dt * tz1 * dz4;
      abcd[4][3] = - dt * tz2 * C2;

      abcd[0][4] = - dt * tz2
        * ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * temp_u[4] )
            * temp_u[3] * tmp2 )
        - dt * tz1
        * ( - ( c34 - c1345 ) * tmp3 * (temp_u[1]*temp_u[1])
            - ( c34 - c1345 ) * tmp3 * (temp_u[2]*temp_u[2])
            - ( r43*c34 - c1345 )* tmp3 * (temp_u[3]*temp_u[3])
            - c1345 * tmp2 * temp_u[4] );
      abcd[1][4] = - dt * tz2
        * ( - C2 * ( temp_u[1]*temp_u[3] ) * tmp2 )
        - dt * tz1 * ( c34 - c1345 ) * tmp2 * temp_u[1];
      abcd[2][4] = - dt * tz2
        * ( - C2 * ( temp_u[2]*temp_u[3] ) * tmp2 )
        - dt * tz1 * ( c34 - c1345 ) * tmp2 * temp_u[2];
      abcd[3][4] = - dt * tz2
        * ( C1 * ( temp_u[4] * tmp1 )
            - C2 * ( qs[k-1][j][i] * tmp1
              + temp_u[3]*temp_u[3] * tmp2 ) )
        - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * temp_u[3];
      abcd[4][4] = - dt * tz2
        * ( C1 * ( temp_u[3] * tmp1 ) )
        - dt * tz1 * c1345 * tmp1
        - dt * tz1 * dz5;


      for (m = 0; m < 5; m++) {
        temp_v[m] = v[k][j][i][m]
          - omega * (  abcd[0][m] * v[k-1][j][i][0]
              + abcd[1][m] * v[k-1][j][i][1]
              + abcd[2][m] * v[k-1][j][i][2]
              + abcd[3][m] * v[k-1][j][i][3]
              + abcd[4][m] * v[k-1][j][i][4] );
      }

      //#####################################################
      //#####################################################
      //          PART 1 - 2
      //#####################################################
      //#####################################################

      for (m = 1; m < 5; m++) temp_u[m] = u[k][j][i-1][m];

      tmp1 = rho_i[k][j][i-1];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      abcd[0][0] = - dt * tx1 * dx1;
      abcd[1][0] = - dt * tx2;
      abcd[2][0] =   0.0;
      abcd[3][0] =   0.0;
      abcd[4][0] =   0.0;

      abcd[0][1] = - dt * tx2
        * ( - ( temp_u[1] * tmp1 ) * ( temp_u[1] * tmp1 )
            + C2 * qs[k][j][i-1] * tmp1 )
        - dt * tx1 * ( - r43 * c34 * tmp2 * temp_u[1] );
      abcd[1][1] = - dt * tx2
        * ( ( 2.0 - C2 ) * ( temp_u[1] * tmp1 ) )
        - dt * tx1 * ( r43 * c34 * tmp1 )
        - dt * tx1 * dx2;
      abcd[2][1] = - dt * tx2
        * ( - C2 * ( temp_u[2] * tmp1 ) );
      abcd[3][1] = - dt * tx2
        * ( - C2 * ( temp_u[3] * tmp1 ) );
      abcd[4][1] = - dt * tx2 * C2;

      abcd[0][2] = - dt * tx2
        * ( - ( temp_u[1] * temp_u[2] ) * tmp2 )
        - dt * tx1 * ( - c34 * tmp2 * temp_u[2] );
      abcd[1][2] = - dt * tx2 * ( temp_u[2] * tmp1 );
      abcd[2][2] = - dt * tx2 * ( temp_u[1] * tmp1 )
        - dt * tx1 * ( c34 * tmp1 )
        - dt * tx1 * dx3;
      abcd[3][2] = 0.0;
      abcd[4][2] = 0.0;

      abcd[0][3] = - dt * tx2
        * ( - ( temp_u[1]*temp_u[3] ) * tmp2 )
        - dt * tx1 * ( - c34 * tmp2 * temp_u[3] );
      abcd[1][3] = - dt * tx2 * ( temp_u[3] * tmp1 );
      abcd[2][3] = 0.0;
      abcd[3][3] = - dt * tx2 * ( temp_u[1] * tmp1 )
        - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
      abcd[4][3] = 0.0;

      abcd[0][4] = - dt * tx2
        * ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * temp_u[4] )
            * temp_u[1] * tmp2 )
        - dt * tx1
        * ( - ( r43*c34 - c1345 ) * tmp3 * ( temp_u[1]*temp_u[1] )
            - (     c34 - c1345 ) * tmp3 * ( temp_u[2]*temp_u[2] )
            - (     c34 - c1345 ) * tmp3 * ( temp_u[3]*temp_u[3] )
            - c1345 * tmp2 * temp_u[4] );
      abcd[1][4] = - dt * tx2
        * ( C1 * ( temp_u[4] * tmp1 )
            - C2 * ( temp_u[1]*temp_u[1] * tmp2
              + qs[k][j][i-1] * tmp1 ) )
        - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * temp_u[1];
      abcd[2][4] = - dt * tx2
        * ( - C2 * ( temp_u[2]*temp_u[1] ) * tmp2 )
        - dt * tx1 * (  c34 - c1345 ) * tmp2 * temp_u[2];
      abcd[3][4] = - dt * tx2
        * ( - C2 * ( temp_u[3]*temp_u[1] ) * tmp2 )
        - dt * tx1 * (  c34 - c1345 ) * tmp2 * temp_u[3];
      abcd[4][4] = - dt * tx2
        * ( C1 * ( temp_u[1] * tmp1 ) )
        - dt * tx1 * c1345 * tmp1
        - dt * tx1 * dx5;


      for (m = 0; m < 5; m++) {
        tv[m] = temp_v[m]
          - omega * (  abcd[0][m] * v[k][j][i-1][0]
              + abcd[1][m] * v[k][j][i-1][1]
              + abcd[2][m] * v[k][j][i-1][2]
              + abcd[3][m] * v[k][j][i-1][3]
              + abcd[4][m] * v[k][j][i-1][4] );
      }

      //#####################################################
      //#####################################################
      //          PART 1 - 3
      //#####################################################
      //#####################################################


      for (m = 1; m < 5; m++) temp_u[m] = u[k][j-1][i][m];


      tmp1 = rho_i[k][j-1][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      abcd[0][0] = - dt * ty1 * dy1;
      abcd[1][0] =   0.0;
      abcd[2][0] = - dt * ty2;
      abcd[3][0] =   0.0;
      abcd[4][0] =   0.0;

      abcd[0][1] = - dt * ty2
        * ( - ( temp_u[1]*temp_u[2] ) * tmp2 )
        - dt * ty1 * ( - c34 * tmp2 * temp_u[1] );
      abcd[1][1] = - dt * ty2 * ( temp_u[2] * tmp1 )
        - dt * ty1 * ( c34 * tmp1 )
        - dt * ty1 * dy2;
      abcd[2][1] = - dt * ty2 * ( temp_u[1] * tmp1 );
      abcd[3][1] = 0.0;
      abcd[4][1] = 0.0;

      abcd[0][2] = - dt * ty2
        * ( - ( temp_u[2] * tmp1 ) * ( temp_u[2] * tmp1 )
            + C2 * ( qs[k][j-1][i] * tmp1 ) )
        - dt * ty1 * ( - r43 * c34 * tmp2 * temp_u[2] );
      abcd[1][2] = - dt * ty2
        * ( - C2 * ( temp_u[1] * tmp1 ) );
      abcd[2][2] = - dt * ty2 * ( (2.0 - C2) * (temp_u[2] * tmp1) )
        - dt * ty1 * ( r43 * c34 * tmp1 )
        - dt * ty1 * dy3;
      abcd[3][2] = - dt * ty2 * ( - C2 * ( temp_u[3] * tmp1 ) );
      abcd[4][2] = - dt * ty2 * C2;

      abcd[0][3] = - dt * ty2
        * ( - ( temp_u[2]*temp_u[3] ) * tmp2 )
        - dt * ty1 * ( - c34 * tmp2 * temp_u[3] );
      abcd[1][3] = 0.0;
      abcd[2][3] = - dt * ty2 * ( temp_u[3] * tmp1 );
      abcd[3][3] = - dt * ty2 * ( temp_u[2] * tmp1 )
        - dt * ty1 * ( c34 * tmp1 )
        - dt * ty1 * dy4;
      abcd[4][3] = 0.0;

      abcd[0][4] = - dt * ty2
        * ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * temp_u[4] )
            * ( temp_u[2] * tmp2 ) )
        - dt * ty1
        * ( - (     c34 - c1345 )*tmp3*(temp_u[1]*temp_u[1])
            - ( r43*c34 - c1345 )*tmp3*(temp_u[2]*temp_u[2])
            - (     c34 - c1345 )*tmp3*(temp_u[3]*temp_u[3])
            - c1345*tmp2*temp_u[4] );
      abcd[1][4] = - dt * ty2
        * ( - C2 * ( temp_u[1]*temp_u[2] ) * tmp2 )
        - dt * ty1 * ( c34 - c1345 ) * tmp2 * temp_u[1];
      abcd[2][4] = - dt * ty2
        * ( C1 * ( temp_u[4] * tmp1 )
            - C2 * ( qs[k][j-1][i] * tmp1
              + temp_u[2]*temp_u[2] * tmp2 ) )
        - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * temp_u[2];
      abcd[3][4] = - dt * ty2
        * ( - C2 * ( temp_u[2]*temp_u[3] ) * tmp2 )
        - dt * ty1 * ( c34 - c1345 ) * tmp2 * temp_u[3];
      abcd[4][4] = - dt * ty2
        * ( C1 * ( temp_u[2] * tmp1 ) )
        - dt * ty1 * c1345 * tmp1
        - dt * ty1 * dy5;



      for (m = 0; m < 5; m++) {
        tv[m] = tv[m]
          - omega * (  abcd[0][m] * v[k][j-1][i][0]
              + abcd[1][m] * v[k][j-1][i][1]
              + abcd[2][m] * v[k][j-1][i][2]
              + abcd[3][m] * v[k][j-1][i][3]
              + abcd[4][m] * v[k][j-1][i][4] );
      }

      //#####################################################
      //#####################################################
      //          PART 1 - 4
      //#####################################################
      //#####################################################

      for (m = 1; m < 5; m++) temp_u[m] = u[k][j][i][m];

      tmp1 = rho_i[k][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      abcd[0][0] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
      abcd[1][0] =  0.0;
      abcd[2][0] =  0.0;
      abcd[3][0] =  0.0;
      abcd[4][0] =  0.0;

      abcd[0][1] = -dt * 2.0
        * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * temp_u[1];
      abcd[1][1] =  1.0
        + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
        + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
      abcd[2][1] = 0.0;
      abcd[3][1] = 0.0;
      abcd[4][1] = 0.0;

      abcd[0][2] = -dt * 2.0 
        * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * temp_u[2];
      abcd[1][2] = 0.0;
      abcd[2][2] = 1.0
        + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
        + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
      abcd[3][2] = 0.0;
      abcd[4][2] = 0.0;

      abcd[0][3] = -dt * 2.0
        * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * temp_u[3];
      abcd[1][3] = 0.0;
      abcd[2][3] = 0.0;
      abcd[3][3] = 1.0
        + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
        + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
      abcd[4][3] = 0.0;

      abcd[0][4] = -dt * 2.0
        * ( ( ( tx1 * ( r43*c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( temp_u[1]*temp_u[1] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( r43*c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( temp_u[2]*temp_u[2] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( r43*c34 - c1345 ) ) * (temp_u[3]*temp_u[3])
            ) * tmp3
            + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * temp_u[4] );

      abcd[1][4] = dt * 2.0 * tmp2 * temp_u[1]
        * ( tx1 * ( r43*c34 - c1345 )
            + ty1 * (     c34 - c1345 )
            + tz1 * (     c34 - c1345 ) );
      abcd[2][4] = dt * 2.0 * tmp2 * temp_u[2]
        * ( tx1 * ( c34 - c1345 )
            + ty1 * ( r43*c34 -c1345 )
            + tz1 * ( c34 - c1345 ) );
      abcd[3][4] = dt * 2.0 * tmp2 * temp_u[3]
        * ( tx1 * ( c34 - c1345 )
            + ty1 * ( c34 - c1345 )
            + tz1 * ( r43*c34 - c1345 ) );
      abcd[4][4] = 1.0
        + dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
        + dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

      //---------------------------------------------------------------------
      // diagonal block inversion
      // 
      // forward elimination
      //---------------------------------------------------------------------

      for (m = 0; m < 5; m++) {
        for (n = 0; n <= m; n++) {
          tmp = abcd[m][n];
          abcd[m][n] = abcd[n][m];
          abcd[n][m] = tmp;
        }
      }


      tmp1 = 1.0 / abcd[0][0];
      tmp = tmp1 * abcd[1][0];
      abcd[1][1] =  abcd[1][1] - tmp * abcd[0][1];
      abcd[1][2] =  abcd[1][2] - tmp * abcd[0][2];
      abcd[1][3] =  abcd[1][3] - tmp * abcd[0][3];
      abcd[1][4] =  abcd[1][4] - tmp * abcd[0][4];
      tv[1] = tv[1] - tv[0] * tmp;


      tmp = tmp1 * abcd[2][0];
      abcd[2][1] =  abcd[2][1] - tmp * abcd[0][1];
      abcd[2][2] =  abcd[2][2] - tmp * abcd[0][2];
      abcd[2][3] =  abcd[2][3] - tmp * abcd[0][3];
      abcd[2][4] =  abcd[2][4] - tmp * abcd[0][4];
      tv[2] = tv[2] - tv[0] * tmp;


      tmp = tmp1 * abcd[3][0];
      abcd[3][1] =  abcd[3][1] - tmp * abcd[0][1];
      abcd[3][2] =  abcd[3][2] - tmp * abcd[0][2];
      abcd[3][3] =  abcd[3][3] - tmp * abcd[0][3];
      abcd[3][4] =  abcd[3][4] - tmp * abcd[0][4];
      tv[3] = tv[3] - tv[0] * tmp;

      tmp = tmp1 * abcd[4][0];
      abcd[4][1] =  abcd[4][1] - tmp * abcd[0][1];
      abcd[4][2] =  abcd[4][2] - tmp * abcd[0][2];
      abcd[4][3] =  abcd[4][3] - tmp * abcd[0][3];
      abcd[4][4] =  abcd[4][4] - tmp * abcd[0][4];
      tv[4] = tv[4] - tv[0] * tmp;

      tmp1 = 1.0 / abcd[1][1];
      tmp = tmp1 * abcd[2][1];
      abcd[2][2] =  abcd[2][2] - tmp * abcd[1][2];
      abcd[2][3] =  abcd[2][3] - tmp * abcd[1][3];
      abcd[2][4] =  abcd[2][4] - tmp * abcd[1][4];
      tv[2] = tv[2] - tv[1] * tmp;

      tmp = tmp1 * abcd[3][1];
      abcd[3][2] =  abcd[3][2] - tmp * abcd[1][2];
      abcd[3][3] =  abcd[3][3] - tmp * abcd[1][3];
      abcd[3][4] =  abcd[3][4] - tmp * abcd[1][4];
      tv[3] = tv[3] - tv[1] * tmp;

      tmp = tmp1 * abcd[4][1];
      abcd[4][2] =  abcd[4][2] - tmp * abcd[1][2];
      abcd[4][3] =  abcd[4][3] - tmp * abcd[1][3];
      abcd[4][4] =  abcd[4][4] - tmp * abcd[1][4];
      tv[4] = tv[4] - tv[1] * tmp;

      tmp1 = 1.0 / abcd[2][2];
      tmp = tmp1 * abcd[3][2];
      abcd[3][3] =  abcd[3][3] - tmp * abcd[2][3];
      abcd[3][4] =  abcd[3][4] - tmp * abcd[2][4];
      tv[3] = tv[3] - tv[2] * tmp;

      tmp = tmp1 * abcd[4][2];
      abcd[4][3] =  abcd[4][3] - tmp * abcd[2][3];
      abcd[4][4] =  abcd[4][4] - tmp * abcd[2][4];
      tv[4] = tv[4] - tv[2] * tmp;

      tmp1 = 1.0 / abcd[3][3];
      tmp = tmp1 * abcd[4][3];
      abcd[4][4] =  abcd[4][4] - tmp * abcd[3][4];
      tv[4] = tv[4] - tv[3] * tmp;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------

      temp_v[4] = tv[4] / abcd[4][4];

      tv[3] = tv[3] 
        - abcd[3][4] * temp_v[4];
      temp_v[3] = tv[3] / abcd[3][3];

      tv[2] = tv[2]
        - abcd[2][3] * temp_v[3]
        - abcd[2][4] * temp_v[4];
      temp_v[2] = tv[2] / abcd[2][2];

      tv[1] = tv[1]
        - abcd[1][2] * temp_v[2]
        - abcd[1][3] * temp_v[3]
        - abcd[1][4] * temp_v[4];
      temp_v[1] = tv[1] / abcd[1][1];

      tv[0] = tv[0]
        - abcd[0][1] * temp_v[1]
        - abcd[0][2] * temp_v[2]
        - abcd[0][3] * temp_v[3]
        - abcd[0][4] * temp_v[4];
      temp_v[0] = tv[0] / abcd[0][0];
    }

    if (!dummy)
    {
      for (m = 0; m < 5; m++) {
        v[k][j][i][m] = temp_v[m];
      } 
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

}

__kernel void jacld_blts_KL_fullopt(__global double *m_rsd,
                                    __global double *m_u,
                                    __global double *m_qs,
                                    __global double *m_rho_i,
                                    int nz, int ny, int nx,
                                    int wf_sum, 
                                    int wf_base_k, 
                                    int wf_base_j,
                                    int jst, int jend, 
                                    int ist, int iend, 
                                    int temp_kst, int temp_kend)
{
  int k, j, i, m;
  double a[5][5], b[5][5], c[5][5], d[5][5];
  double r43, c1345, c34;
  double tmp, tmp1, tmp2, tmp3;
  double tmat[5][5], tv[5];
  __global double (*rsd)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*u)[ISIZ2/2*2+1][ISIZ1/2*2+1][5];
  __global double (*qs)[ISIZ2/2*2+1][ISIZ1/2*2+1];
  __global double (*rho_i)[ISIZ2/2*2+1][ISIZ1/2*2+1];

  k = get_global_id(1) + temp_kst + wf_base_k;
  j = get_global_id(0) + jst + wf_base_j;
  i = wf_sum - get_global_id(1) - get_global_id(0) - wf_base_k - wf_base_j + ist;
  if (k >= temp_kend || j >= jend || i < ist || i >= iend) return;

  rsd = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_rsd;
  u = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_u;
  qs = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_qs;
  rho_i = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1])m_rho_i;

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  //---------------------------------------------------------------------
  // form the block daigonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  d[0][0] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
  d[1][0] =  0.0;
  d[2][0] =  0.0;
  d[3][0] =  0.0;
  d[4][0] =  0.0;

  d[0][1] = -dt * 2.0
    * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * u[k][j][i][1];
  d[1][1] =  1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
    + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
  d[2][1] = 0.0;
  d[3][1] = 0.0;
  d[4][1] = 0.0;

  d[0][2] = -dt * 2.0 
    * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * u[k][j][i][2];
  d[1][2] = 0.0;
  d[2][2] = 1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
    + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
  d[3][2] = 0.0;
  d[4][2] = 0.0;

  d[0][3] = -dt * 2.0
    * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * u[k][j][i][3];
  d[1][3] = 0.0;
  d[2][3] = 0.0;
  d[3][3] = 1.0
    + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
    + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
  d[4][3] = 0.0;

  d[0][4] = -dt * 2.0
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

  d[1][4] = dt * 2.0 * tmp2 * u[k][j][i][1]
    * ( tx1 * ( r43*c34 - c1345 )
        + ty1 * (     c34 - c1345 )
        + tz1 * (     c34 - c1345 ) );
  d[2][4] = dt * 2.0 * tmp2 * u[k][j][i][2]
    * ( tx1 * ( c34 - c1345 )
        + ty1 * ( r43*c34 -c1345 )
        + tz1 * ( c34 - c1345 ) );
  d[3][4] = dt * 2.0 * tmp2 * u[k][j][i][3]
    * ( tx1 * ( c34 - c1345 )
        + ty1 * ( c34 - c1345 )
        + tz1 * ( r43*c34 - c1345 ) );
  d[4][4] = 1.0
    + dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
    + dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

  //---------------------------------------------------------------------
  // form the first block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k-1][j][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  a[0][0] = - dt * tz1 * dz1;
  a[1][0] =   0.0;
  a[2][0] =   0.0;
  a[3][0] = - dt * tz2;
  a[4][0] =   0.0;

  a[0][1] = - dt * tz2
    * ( - ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k-1][j][i][1] );
  a[1][1] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * c34 * tmp1
    - dt * tz1 * dz2;
  a[2][1] = 0.0;
  a[3][1] = - dt * tz2 * ( u[k-1][j][i][1] * tmp1 );
  a[4][1] = 0.0;

  a[0][2] = - dt * tz2
    * ( - ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( - c34 * tmp2 * u[k-1][j][i][2] );
  a[1][2] = 0.0;
  a[2][2] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * ( c34 * tmp1 )
    - dt * tz1 * dz3;
  a[3][2] = - dt * tz2 * ( u[k-1][j][i][2] * tmp1 );
  a[4][2] = 0.0;

  a[0][3] = - dt * tz2
    * ( - ( u[k-1][j][i][3] * tmp1 ) * ( u[k-1][j][i][3] * tmp1 )
        + C2 * qs[k-1][j][i] * tmp1 )
    - dt * tz1 * ( - r43 * c34 * tmp2 * u[k-1][j][i][3] );
  a[1][3] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][1] * tmp1 ) );
  a[2][3] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][2] * tmp1 ) );
  a[3][3] = - dt * tz2 * ( 2.0 - C2 )
    * ( u[k-1][j][i][3] * tmp1 )
    - dt * tz1 * ( r43 * c34 * tmp1 )
    - dt * tz1 * dz4;
  a[4][3] = - dt * tz2 * C2;

  a[0][4] = - dt * tz2
    * ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * u[k-1][j][i][4] )
        * u[k-1][j][i][3] * tmp2 )
    - dt * tz1
    * ( - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][1]*u[k-1][j][i][1])
        - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][2]*u[k-1][j][i][2])
        - ( r43*c34 - c1345 )* tmp3 * (u[k-1][j][i][3]*u[k-1][j][i][3])
        - c1345 * tmp2 * u[k-1][j][i][4] );
  a[1][4] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k-1][j][i][1];
  a[2][4] = - dt * tz2
    * ( - C2 * ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2 )
    - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[k-1][j][i][2];
  a[3][4] = - dt * tz2
    * ( C1 * ( u[k-1][j][i][4] * tmp1 )
        - C2 * ( qs[k-1][j][i] * tmp1
          + u[k-1][j][i][3]*u[k-1][j][i][3] * tmp2 ) )
    - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[k-1][j][i][3];
  a[4][4] = - dt * tz2
    * ( C1 * ( u[k-1][j][i][3] * tmp1 ) )
    - dt * tz1 * c1345 * tmp1
    - dt * tz1 * dz5;

  //---------------------------------------------------------------------
  // form the second block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j-1][i];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  b[0][0] = - dt * ty1 * dy1;
  b[1][0] =   0.0;
  b[2][0] = - dt * ty2;
  b[3][0] =   0.0;
  b[4][0] =   0.0;

  b[0][1] = - dt * ty2
    * ( - ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j-1][i][1] );
  b[1][1] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy2;
  b[2][1] = - dt * ty2 * ( u[k][j-1][i][1] * tmp1 );
  b[3][1] = 0.0;
  b[4][1] = 0.0;

  b[0][2] = - dt * ty2
    * ( - ( u[k][j-1][i][2] * tmp1 ) * ( u[k][j-1][i][2] * tmp1 )
        + C2 * ( qs[k][j-1][i] * tmp1 ) )
    - dt * ty1 * ( - r43 * c34 * tmp2 * u[k][j-1][i][2] );
  b[1][2] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][1] * tmp1 ) );
  b[2][2] = - dt * ty2 * ( (2.0 - C2) * (u[k][j-1][i][2] * tmp1) )
    - dt * ty1 * ( r43 * c34 * tmp1 )
    - dt * ty1 * dy3;
  b[3][2] = - dt * ty2 * ( - C2 * ( u[k][j-1][i][3] * tmp1 ) );
  b[4][2] = - dt * ty2 * C2;

  b[0][3] = - dt * ty2
    * ( - ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2 )
    - dt * ty1 * ( - c34 * tmp2 * u[k][j-1][i][3] );
  b[1][3] = 0.0;
  b[2][3] = - dt * ty2 * ( u[k][j-1][i][3] * tmp1 );
  b[3][3] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
    - dt * ty1 * ( c34 * tmp1 )
    - dt * ty1 * dy4;
  b[4][3] = 0.0;

  b[0][4] = - dt * ty2
    * ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * u[k][j-1][i][4] )
        * ( u[k][j-1][i][2] * tmp2 ) )
    - dt * ty1
    * ( - (     c34 - c1345 )*tmp3*(u[k][j-1][i][1]*u[k][j-1][i][1])
        - ( r43*c34 - c1345 )*tmp3*(u[k][j-1][i][2]*u[k][j-1][i][2])
        - (     c34 - c1345 )*tmp3*(u[k][j-1][i][3]*u[k][j-1][i][3])
        - c1345*tmp2*u[k][j-1][i][4] );
  b[1][4] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j-1][i][1];
  b[2][4] = - dt * ty2
    * ( C1 * ( u[k][j-1][i][4] * tmp1 )
        - C2 * ( qs[k][j-1][i] * tmp1
          + u[k][j-1][i][2]*u[k][j-1][i][2] * tmp2 ) )
    - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u[k][j-1][i][2];
  b[3][4] = - dt * ty2
    * ( - C2 * ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2 )
    - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[k][j-1][i][3];
  b[4][4] = - dt * ty2
    * ( C1 * ( u[k][j-1][i][2] * tmp1 ) )
    - dt * ty1 * c1345 * tmp1
    - dt * ty1 * dy5;

  //---------------------------------------------------------------------
  // form the third block sub-diagonal
  //---------------------------------------------------------------------
  tmp1 = rho_i[k][j][i-1];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  c[0][0] = - dt * tx1 * dx1;
  c[1][0] = - dt * tx2;
  c[2][0] =   0.0;
  c[3][0] =   0.0;
  c[4][0] =   0.0;

  c[0][1] = - dt * tx2
    * ( - ( u[k][j][i-1][1] * tmp1 ) * ( u[k][j][i-1][1] * tmp1 )
        + C2 * qs[k][j][i-1] * tmp1 )
    - dt * tx1 * ( - r43 * c34 * tmp2 * u[k][j][i-1][1] );
  c[1][1] = - dt * tx2
    * ( ( 2.0 - C2 ) * ( u[k][j][i-1][1] * tmp1 ) )
    - dt * tx1 * ( r43 * c34 * tmp1 )
    - dt * tx1 * dx2;
  c[2][1] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][2] * tmp1 ) );
  c[3][1] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][3] * tmp1 ) );
  c[4][1] = - dt * tx2 * C2;

  c[0][2] = - dt * tx2
    * ( - ( u[k][j][i-1][1] * u[k][j][i-1][2] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i-1][2] );
  c[1][2] = - dt * tx2 * ( u[k][j][i-1][2] * tmp1 );
  c[2][2] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 )
    - dt * tx1 * dx3;
  c[3][2] = 0.0;
  c[4][2] = 0.0;

  c[0][3] = - dt * tx2
    * ( - ( u[k][j][i-1][1]*u[k][j][i-1][3] ) * tmp2 )
    - dt * tx1 * ( - c34 * tmp2 * u[k][j][i-1][3] );
  c[1][3] = - dt * tx2 * ( u[k][j][i-1][3] * tmp1 );
  c[2][3] = 0.0;
  c[3][3] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
    - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
  c[4][3] = 0.0;

  c[0][4] = - dt * tx2
    * ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * u[k][j][i-1][4] )
        * u[k][j][i-1][1] * tmp2 )
    - dt * tx1
    * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[k][j][i-1][1]*u[k][j][i-1][1] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][2]*u[k][j][i-1][2] )
        - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][3]*u[k][j][i-1][3] )
        - c1345 * tmp2 * u[k][j][i-1][4] );
  c[1][4] = - dt * tx2
    * ( C1 * ( u[k][j][i-1][4] * tmp1 )
        - C2 * ( u[k][j][i-1][1]*u[k][j][i-1][1] * tmp2
          + qs[k][j][i-1] * tmp1 ) )
    - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u[k][j][i-1][1];
  c[2][4] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][2]*u[k][j][i-1][1] ) * tmp2 )
    - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[k][j][i-1][2];
  c[3][4] = - dt * tx2
    * ( - C2 * ( u[k][j][i-1][3]*u[k][j][i-1][1] ) * tmp2 )
    - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[k][j][i-1][3];
  c[4][4] = - dt * tx2
    * ( C1 * ( u[k][j][i-1][1] * tmp1 ) )
    - dt * tx1 * c1345 * tmp1
    - dt * tx1 * dx5;

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
