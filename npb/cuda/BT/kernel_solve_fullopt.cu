//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB BT code. This CUDA® C  //
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
// This function computes the left hand side in the xi-direction
//---------------------------------------------------------------------
__global__ 
void k_x_solve1_fullopt(double *m_lhsA, 
                        double *m_lhsB, double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT)
{
  int t_k = blockDim.y * blockIdx.y + threadIdx.y;
  int mn = t_k / work_num_item;
  int k = t_k % work_num_item;
  int m = mn / 5;
  int n = mn % 5;
  int j = blockDim.x * blockIdx.x + threadIdx.x+1;


  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || m >= 5) return;

  // front alignment
  if (split_flag) k += 2;
  else k += work_base;

  int isize;


  // for purpose of memory coalescing modify index from k,j,i,m,n to m,n,k,i,j
#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]


  isize = gp0 - 1 ;


  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in x direction
  //---------------------------------------------------------------------
  
  lhsA(m, n, k, 0, j-1) = 0.0;
  lhsB(m, n, k, 0, j-1) = (m==n)?1.0:0.0;
  lhsC(m, n, k, 0, j-1) = 0.0;

  lhsA(m, n, k, isize, j-1) = 0.0;
  lhsB(m, n, k, isize, j-1) = (m==n)?1.0:0.0;
  lhsC(m, n, k, isize, j-1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC
}




//---------------------------------------------------------------------
// This function computes the left hand side in the xi-direction
//---------------------------------------------------------------------
__device__ 
static void compute_fjac_x(double fjac[5][5], 
                           double t_u[5], 
                           double rho_i,
                           double qs, double square, 
                           double c1, double c2)
{
  double tmp1, tmp2;

  //---------------------------------------------------------------------
  // determine a (labeled f) and n jacobians
  //---------------------------------------------------------------------
  tmp1 = rho_i;
  tmp2 = tmp1 * tmp1;

  fjac[0][0] = 0.0;
  fjac[1][0] = 1.0;
  fjac[2][0] = 0.0;
  fjac[3][0] = 0.0;
  fjac[4][0] = 0.0;

  fjac[0][1] = -(t_u[1] * tmp2 * t_u[1])
    + c2 * qs;
  fjac[1][1] = ( 2.0 - c2 ) * ( t_u[1] / t_u[0] );
  fjac[2][1] = - c2 * ( t_u[2] * tmp1 );
  fjac[3][1] = - c2 * ( t_u[3] * tmp1 );
  fjac[4][1] = c2;

  fjac[0][2] = - ( t_u[1]*t_u[2] ) * tmp2;
  fjac[1][2] = t_u[2] * tmp1;
  fjac[2][2] = t_u[1] * tmp1;
  fjac[3][2] = 0.0;
  fjac[4][2] = 0.0;

  fjac[0][3] = - ( t_u[1]*t_u[3] ) * tmp2;
  fjac[1][3] = t_u[3] * tmp1;
  fjac[2][3] = 0.0;
  fjac[3][3] = t_u[1] * tmp1;
  fjac[4][3] = 0.0;

  fjac[0][4] = ( c2 * 2.0 * square - c1 * t_u[4] )
    * ( t_u[1] * tmp2 );
  fjac[1][4] = c1 *  t_u[4] * tmp1 
    - c2 * ( t_u[1]*t_u[1] * tmp2 + qs );
  fjac[2][4] = - c2 * ( t_u[2]*t_u[1] ) * tmp2;
  fjac[3][4] = - c2 * ( t_u[3]*t_u[1] ) * tmp2;
  fjac[4][4] = c1 * ( t_u[1] * tmp1 );

}


__device__ 
static void compute_njac_x(double njac[5][5], 
                           double t_u[5],
                           double rho_i, double c3c4, 
                           double con43, double c1345)
{

  double tmp1, tmp2, tmp3;

  //---------------------------------------------------------------------
  // determine a (labeled f) and n jacobians
  //---------------------------------------------------------------------
  tmp1 = rho_i;
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  njac[0][0] = 0.0;
  njac[1][0] = 0.0;
  njac[2][0] = 0.0;
  njac[3][0] = 0.0;
  njac[4][0] = 0.0;

  njac[0][1] = - con43 * c3c4 * tmp2 * t_u[1];
  njac[1][1] =   con43 * c3c4 * tmp1;
  njac[2][1] =   0.0;
  njac[3][1] =   0.0;
  njac[4][1] =   0.0;

  njac[0][2] = - c3c4 * tmp2 * t_u[2];
  njac[1][2] =   0.0;
  njac[2][2] =   c3c4 * tmp1;
  njac[3][2] =   0.0;
  njac[4][2] =   0.0;

  njac[0][3] = - c3c4 * tmp2 * t_u[3];
  njac[1][3] =   0.0;
  njac[2][3] =   0.0;
  njac[3][3] =   c3c4 * tmp1;
  njac[4][3] =   0.0;

  njac[0][4] = - ( con43 * c3c4
      - c1345 ) * tmp3 * (t_u[1]*t_u[1])
    - ( c3c4 - c1345 ) * tmp3 * (t_u[2]*t_u[2])
    - ( c3c4 - c1345 ) * tmp3 * (t_u[3]*t_u[3])
    - c1345 * tmp2 * t_u[4];

  njac[1][4] = ( con43 * c3c4
      - c1345 ) * tmp2 * t_u[1];
  njac[2][4] = ( c3c4 - c1345 ) * tmp2 * t_u[2];
  njac[3][4] = ( c3c4 - c1345 ) * tmp2 * t_u[3];
  njac[4][4] = ( c1345 ) * tmp1;

}

__launch_bounds__(MAX_THREAD_DIM_1)
__global__ 
void k_x_solve2_fullopt(double *m_qs, double *m_rho_i,
                        double *m_square, double *m_u, 
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        double dx1, double dx2, 
                        double dx3, double dx4, 
                        double dx5,
                        double c1, double c2, 
                        double tx1, double tx2, 
                        double con43,
                        double c3c4, double c1345, double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y+1;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;

  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;


   double (* qs)[JMAXP+1][IMAXP+1]
    = ( double (*) [JMAXP+1][IMAXP+1])m_qs;
   double (* rho_i)[JMAXP+1][IMAXP+1]
    = ( double (*) [JMAXP+1][IMAXP+1])m_rho_i;
   double (* square)[JMAXP+1][IMAXP+1]
    = ( double (*) [JMAXP+1][IMAXP+1]) m_square; 
   double (* u)[JMAXP+1][IMAXP+1][5]
    = ( double (*) [JMAXP+1][IMAXP+1][5])m_u;

  // for purpose of memory coalescing modify index from k,j,i,m,n to m,n,k,i,j
#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]

  double fjac[5][5];
  double njac[5][5];

  double t_u[5];

  int m;

  double tmp1, tmp2;


  tmp1 = dt * tx1;
  tmp2 = dt * tx2;


  for (m = 0; m < 5; m++) t_u[m] = u[k][j][i-1][m];
  compute_fjac_x(fjac, t_u, rho_i[k][j][i-1], qs[k][j][i-1], square[k][j][i-1], c1, c2);
  compute_njac_x(njac, t_u, rho_i[k][j][i-1], c3c4, con43, c1345);


  lhsA(0, 0, k, i, j-1) = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dx1;
  lhsA(1, 0, k, i, j-1) = - tmp2 * fjac[1][0] - tmp1 * njac[1][0];
  lhsA(2, 0, k, i, j-1) = - tmp2 * fjac[2][0] - tmp1 * njac[2][0];
  lhsA(3, 0, k, i, j-1) = - tmp2 * fjac[3][0] - tmp1 * njac[3][0];
  lhsA(4, 0, k, i, j-1) = - tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    
  lhsA(0, 1, k, i, j-1) = - tmp2 * fjac[0][1] - tmp1 * njac[0][1];
  lhsA(1, 1, k, i, j-1) = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dx2;
  lhsA(2, 1, k, i, j-1) = - tmp2 * fjac[2][1] - tmp1 * njac[2][1];
  lhsA(3, 1, k, i, j-1) = - tmp2 * fjac[3][1] - tmp1 * njac[3][1];
  lhsA(4, 1, k, i, j-1) = - tmp2 * fjac[4][1] - tmp1 * njac[4][1];
                                                                    
  lhsA(0, 2, k, i, j-1) = - tmp2 * fjac[0][2] - tmp1 * njac[0][2];
  lhsA(1, 2, k, i, j-1) = - tmp2 * fjac[1][2] - tmp1 * njac[1][2];
  lhsA(2, 2, k, i, j-1) = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dx3;
  lhsA(3, 2, k, i, j-1) = - tmp2 * fjac[3][2] - tmp1 * njac[3][2];
  lhsA(4, 2, k, i, j-1) = - tmp2 * fjac[4][2] - tmp1 * njac[4][2];
                                                                    
  lhsA(0, 3, k, i, j-1) = - tmp2 * fjac[0][3] - tmp1 * njac[0][3];
  lhsA(1, 3, k, i, j-1) = - tmp2 * fjac[1][3] - tmp1 * njac[1][3];
  lhsA(2, 3, k, i, j-1) = - tmp2 * fjac[2][3] - tmp1 * njac[2][3];
  lhsA(3, 3, k, i, j-1) = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dx4;
  lhsA(4, 3, k, i, j-1) = - tmp2 * fjac[4][3] - tmp1 * njac[4][3];
                                                                    
  lhsA(0, 4, k, i, j-1) = - tmp2 * fjac[0][4] - tmp1 * njac[0][4];
  lhsA(1, 4, k, i, j-1) = - tmp2 * fjac[1][4] - tmp1 * njac[1][4];
  lhsA(2, 4, k, i, j-1) = - tmp2 * fjac[2][4] - tmp1 * njac[2][4];
  lhsA(3, 4, k, i, j-1) = - tmp2 * fjac[3][4] - tmp1 * njac[3][4];
  lhsA(4, 4, k, i, j-1) = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dx5;






  for (m = 0; m < 5; m++) t_u[m] = u[k][j][i][m];
  compute_njac_x(fjac, t_u, rho_i[k][j][i], c3c4, con43, c1345);


  lhsB(0, 0, k, i, j-1) = 1.0 + tmp1 * 2.0 * fjac[0][0] + tmp1 * 2.0 * dx1;
  lhsB(1, 0, k, i, j-1) = tmp1 * 2.0 * fjac[1][0];
  lhsB(2, 0, k, i, j-1) = tmp1 * 2.0 * fjac[2][0];
  lhsB(3, 0, k, i, j-1) = tmp1 * 2.0 * fjac[3][0];
  lhsB(4, 0, k, i, j-1) = tmp1 * 2.0 * fjac[4][0];

  lhsB(0, 1, k, i, j-1) = tmp1 * 2.0 * fjac[0][1];
  lhsB(1, 1, k, i, j-1) = 1.0 + tmp1 * 2.0 * fjac[1][1] + tmp1 * 2.0 * dx2;
  lhsB(2, 1, k, i, j-1) = tmp1 * 2.0 * fjac[2][1];
  lhsB(3, 1, k, i, j-1) = tmp1 * 2.0 * fjac[3][1];
  lhsB(4, 1, k, i, j-1) = tmp1 * 2.0 * fjac[4][1];

  lhsB(0, 2, k, i, j-1) = tmp1 * 2.0 * fjac[0][2];
  lhsB(1, 2, k, i, j-1) = tmp1 * 2.0 * fjac[1][2];
  lhsB(2, 2, k, i, j-1) = 1.0 + tmp1 * 2.0 * fjac[2][2] + tmp1 * 2.0 * dx3;
  lhsB(3, 2, k, i, j-1) = tmp1 * 2.0 * fjac[3][2];
  lhsB(4, 2, k, i, j-1) = tmp1 * 2.0 * fjac[4][2];

  lhsB(0, 3, k, i, j-1) = tmp1 * 2.0 * fjac[0][3];
  lhsB(1, 3, k, i, j-1) = tmp1 * 2.0 * fjac[1][3];
  lhsB(2, 3, k, i, j-1) = tmp1 * 2.0 * fjac[2][3];
  lhsB(3, 3, k, i, j-1) = 1.0 + tmp1 * 2.0 * fjac[3][3] + tmp1 * 2.0 * dx4;
  lhsB(4, 3, k, i, j-1) = tmp1 * 2.0 * fjac[4][3];

  lhsB(0, 4, k, i, j-1) = tmp1 * 2.0 * fjac[0][4];
  lhsB(1, 4, k, i, j-1) = tmp1 * 2.0 * fjac[1][4];
  lhsB(2, 4, k, i, j-1) = tmp1 * 2.0 * fjac[2][4];
  lhsB(3, 4, k, i, j-1) = tmp1 * 2.0 * fjac[3][4];
  lhsB(4, 4, k, i, j-1) = 1.0 + tmp1 * 2.0 * fjac[4][4] + tmp1 * 2.0 * dx5;






  for (m = 0; m < 5; m++) t_u[m] = u[k][j][i+1][m];
  compute_fjac_x(fjac, t_u, rho_i[k][j][i+1], qs[k][j][i+1], square[k][j][i+1], c1, c2);
  compute_njac_x(njac, t_u, rho_i[k][j][i+1], c3c4, con43, c1345);

  lhsC(0, 0, k, i, j-1) = tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dx1;
  lhsC(1, 0, k, i, j-1) = tmp2 * fjac[1][0] - tmp1 * njac[1][0];
  lhsC(2, 0, k, i, j-1) = tmp2 * fjac[2][0] - tmp1 * njac[2][0];
  lhsC(3, 0, k, i, j-1) = tmp2 * fjac[3][0] - tmp1 * njac[3][0];
  lhsC(4, 0, k, i, j-1) = tmp2 * fjac[4][0] - tmp1 * njac[4][0];
                                                                   
  lhsC(0, 1, k, i, j-1) = tmp2 * fjac[0][1] - tmp1 * njac[0][1];
  lhsC(1, 1, k, i, j-1) = tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dx2;
  lhsC(2, 1, k, i, j-1) = tmp2 * fjac[2][1] - tmp1 * njac[2][1];
  lhsC(3, 1, k, i, j-1) = tmp2 * fjac[3][1] - tmp1 * njac[3][1];
  lhsC(4, 1, k, i, j-1) = tmp2 * fjac[4][1] - tmp1 * njac[4][1];
                                                                   
  lhsC(0, 2, k, i, j-1) = tmp2 * fjac[0][2] - tmp1 * njac[0][2];
  lhsC(1, 2, k, i, j-1) = tmp2 * fjac[1][2] - tmp1 * njac[1][2];
  lhsC(2, 2, k, i, j-1) = tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dx3;
  lhsC(3, 2, k, i, j-1) = tmp2 * fjac[3][2] - tmp1 * njac[3][2];
  lhsC(4, 2, k, i, j-1) = tmp2 * fjac[4][2] - tmp1 * njac[4][2];
                                                                   
  lhsC(0, 3, k, i, j-1) = tmp2 * fjac[0][3] - tmp1 * njac[0][3];
  lhsC(1, 3, k, i, j-1) = tmp2 * fjac[1][3] - tmp1 * njac[1][3];
  lhsC(2, 3, k, i, j-1) = tmp2 * fjac[2][3] - tmp1 * njac[2][3];
  lhsC(3, 3, k, i, j-1) = tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dx4;
  lhsC(4, 3, k, i, j-1) = tmp2 * fjac[4][3] - tmp1 * njac[4][3];
                                                                   
  lhsC(0, 4, k, i, j-1) = tmp2 * fjac[0][4] - tmp1 * njac[0][4];
  lhsC(1, 4, k, i, j-1) = tmp2 * fjac[1][4] - tmp1 * njac[1][4];
  lhsC(2, 4, k, i, j-1) = tmp2 * fjac[2][4] - tmp1 * njac[2][4];
  lhsC(3, 4, k, i, j-1) = tmp2 * fjac[3][4] - tmp1 * njac[3][4];
  lhsC(4, 4, k, i, j-1) = tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dx5;

#undef lhsA
#undef lhsB
#undef lhsC

}



//---------------------------------------------------------------------
// This function computes the left hand side in the xi-direction
//---------------------------------------------------------------------
__global__ 
void k_x_solve3_fullopt(double *m_rhs,
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT)
{
  extern __shared__ double tmp_l_lhs[];
  double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];

  int k = (blockDim.y * blockIdx.y + threadIdx.y)/5;
  int m = (blockDim.y * blockIdx.y + threadIdx.y)%5;
  int j = blockDim.x * blockIdx.x + threadIdx.x+1;
  int dummy = 0;
  int l_j = threadIdx.x;
  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || j > gp1-2) dummy = 1;

  if (split_flag) k += 2;
  else k += work_base;

  int i, n, p,isize;

   double (* rhs)[JMAXP+1][IMAXP+1][5]
    = ( double (*) [JMAXP+1][IMAXP+1][5])m_rhs;


  // for purpose of memory coalescing modify index from k,j,i,m,n to m,n,k,i,j
#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]

   double (* tmp2_l_lhs)[3][5][5] = ( double(*) [3][5][5])tmp_l_lhs;
   double (* l_lhs)[5][5] = tmp2_l_lhs[l_j];
   double (* tmp2_l_r)[2][5] = ( double(*)[2][5])tmp_l_r;
   double (* l_r)[5] = tmp2_l_r[l_j]; 

  double pivot, coeff;

  isize = gp0 - 1 ;

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(IMAX) and rhs'(IMAX) will be sent to next cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------


  // load data
  if (!dummy) {
    for (p = 0; p < 5; p++) {
      l_lhs[BB][p][m] = lhsB(p, m, k, 0, j-1);
      l_lhs[CC][p][m] = lhsC(p, m, k, 0, j-1);
    }

    l_r[1][m] = rhs[k][j][0][m];
  } 
  __syncthreads();

  //---------------------------------------------------------------------
  // multiply c[k][j][0] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  for (p = 0; p < 5; p++) {
    if (!dummy) {
      pivot = 1.00/l_lhs[BB][p][p];
      if (m > p && m < 5)   l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
      if (m < 5)    l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;
      if (p == m)   l_r[1][p] = l_r[1][p]*pivot;
    }
    //barrier
    __syncthreads();

    if (!dummy) {
      if (p != m) {
        coeff = l_lhs[BB][p][m];
        for (n = p+1; n < 5; n++) 
          l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
        for (n = 0; n < 5; n++) 
          l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];
        l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
      }
    } 
    //barrier
    __syncthreads();
  }

  //update data
  if (!dummy)
    rhs[k][j][0][m] = l_r[1][m];


  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (i = 1; i <= isize-1; i++) {

    if (!dummy) {
      for (n = 0; n < 5; n++) {
        l_lhs[AA][n][m] = lhsA(n, m, k, i, j-1);
        l_lhs[BB][n][m] = lhsB(n, m, k, i, j-1);
      }
      l_r[0][m] = l_r[1][m];
      l_r[1][m] = rhs[k][j][i][m];

    }
    __syncthreads();

    //-------------------------------------------------------------------
    // rhs(i) = rhs(i) - A*rhs(i-1)
    //-------------------------------------------------------------------
    if (!dummy) {
      l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
        - l_lhs[AA][1][m]*l_r[0][1]
        - l_lhs[AA][2][m]*l_r[0][2]
        - l_lhs[AA][3][m]*l_r[0][3]
        - l_lhs[AA][4][m]*l_r[0][4];
    }


    //-------------------------------------------------------------------
    // B(i) = B(i) - C(i-1)*A(i)
    //-------------------------------------------------------------------
    if (!dummy) {
      for (p = 0; p < 5; p++) {
        l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
          - l_lhs[AA][1][p]*l_lhs[CC][m][1]
          - l_lhs[AA][2][p]*l_lhs[CC][m][2]
          - l_lhs[AA][3][p]*l_lhs[CC][m][3]
          - l_lhs[AA][4][p]*l_lhs[CC][m][4];

      }
    }
    __syncthreads();



    if (!dummy) {
      for (n = 0; n < 5; n++) l_lhs[CC][n][m] = lhsC(n, m, k, i, j-1);
    }
    __syncthreads();


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
    //-------------------------------------------------------------------
    for (p = 0; p < 5; p++) {
      if (!dummy) {
        pivot = 1.00/l_lhs[BB][p][p];
        if(m > p)   l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
        l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;
        if(p == m)    l_r[1][p] = l_r[1][p]*pivot;
      }

      //barrier
      __syncthreads();
      if (!dummy) {
        if (p != m) {
          coeff = l_lhs[BB][p][m];
          for (n = p+1; n < 5; n++) l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
          for (n = 0; n < 5; n++) l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];
          l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
        }
      } 

      //barrier
      __syncthreads();
    }


    if (!dummy) {
      for (n = 0; n < 5; n++) 
        lhsC(n, m, k, i, j-1) = l_lhs[CC][n][m];

      rhs[k][j][i][m] = l_r[1][m];
    }
  }




  if (!dummy) {
    for (n = 0; n < 5; n++) {
      l_lhs[AA][n][m] = lhsA(n, m, k, i, j-1);
      l_lhs[BB][n][m] = lhsB(n, m, k, i, j-1);
    }
    l_r[0][m] = l_r[1][m];
    l_r[1][m] = rhs[k][j][i][m];

  }
  __syncthreads();




  //---------------------------------------------------------------------
  // rhs(isize) = rhs(isize) - A*rhs(isize-1)
  //---------------------------------------------------------------------
  if (!dummy) {

    l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
      - l_lhs[AA][1][m]*l_r[0][1]
      - l_lhs[AA][2][m]*l_r[0][2]
      - l_lhs[AA][3][m]*l_r[0][3]
      - l_lhs[AA][4][m]*l_r[0][3];

  }


  //---------------------------------------------------------------------
  // B(isize) = B(isize) - C(isize-1)*A(isize)
  //---------------------------------------------------------------------
  if (!dummy) {
    for (p = 0; p < 5; p++) {
      l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
        - l_lhs[AA][1][p]*l_lhs[CC][m][1]
        - l_lhs[AA][2][p]*l_lhs[CC][m][2]
        - l_lhs[AA][3][p]*l_lhs[CC][m][3]
        - l_lhs[AA][4][p]*l_lhs[CC][m][4];
    }
  }

  //---------------------------------------------------------------------
  // multiply rhs() by b_inverse() and copy to rhs
  //---------------------------------------------------------------------
  for (p = 0; p < 5; p++) {
    if (!dummy) {
      pivot = 1.00/l_lhs[BB][p][p];
      if (m > p && m < 5) l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
      if (p == m) l_r[1][p] = l_r[1][p]*pivot;
    }
    //barrier
    __syncthreads();

    if (!dummy) {
      if (p != m) {
        coeff = l_lhs[BB][p][m];
        for (n = p+1; n < 5; n++) 
          l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
        l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];
      }
    } 
    //barrier
    __syncthreads();

  }


  if (!dummy)
    rhs[k][j][i][m] = l_r[1][m];

  __syncthreads();

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(isize)=rhs(isize)
  // else assume U(isize) is loaded in un pack backsub_info
  // so just use it
  // after u(istart) will be sent to next cell
  //---------------------------------------------------------------------
  for (i = isize-1; i >= 0; i--) {
    if (!dummy) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhsC(n, m, k, i, j-1)*rhs[k][j][i+1][n];
      } 
    }
    __syncthreads();
  }

#undef lhsA
#undef lhsB
#undef lhsC
}

__global__ 
void k_y_solve1_fullopt(double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT)
{
  int t_k = blockDim.y * blockIdx.y + threadIdx.y;
  int k = t_k % work_num_item;
  int mn = t_k / work_num_item;
  int m = mn / 5;
  int n = mn % 5;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;

  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || i > gp0-2 || m >= 5) return;

  if (split_flag) k += 2;
  else k += work_base;

  int jsize;


#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]



  jsize = gp1 - 1;


  //---------------------------------------------------------------------
  // now joacobians set, so form left hand side in y direction
  //---------------------------------------------------------------------

  lhsA(m, n, k, 0, i-1) = 0.0;
  lhsB(m, n, k, 0, i-1) = (m==n)?1.0:0.0;
  lhsC(m, n, k, 0, i-1) = 0.0;

  lhsA(m, n, k, jsize, i-1) = 0.0;
  lhsB(m, n, k, jsize, i-1) = (m==n)?1.0:0.0;
  lhsC(m, n, k, jsize, i-1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC
}



__device__ 
static void compute_fjac_y(double fjac[5][5], 
                           double t_u[5],
                           double rho_i, 
                           double square, 
                           double qs, double c1, 
                           double c2)
{

  double tmp1, tmp2;

  tmp1 = rho_i;
  tmp2 = tmp1 * tmp1;

  fjac[0][0] = 0.0;
  fjac[1][0] = 0.0;
  fjac[2][0] = 1.0;
  fjac[3][0] = 0.0;
  fjac[4][0] = 0.0;

  fjac[0][1] = - ( t_u[1]*t_u[2] ) * tmp2;
  fjac[1][1] = t_u[2] * tmp1;
  fjac[2][1] = t_u[1] * tmp1;
  fjac[3][1] = 0.0;
  fjac[4][1] = 0.0;

  fjac[0][2] = - ( t_u[2]*t_u[2]*tmp2)
    + c2 * qs;
  fjac[1][2] = - c2 *  t_u[1] * tmp1;
  fjac[2][2] = ( 2.0 - c2 ) *  t_u[2] * tmp1;
  fjac[3][2] = - c2 * t_u[3] * tmp1;
  fjac[4][2] = c2;

  fjac[0][3] = - ( t_u[2]*t_u[3] ) * tmp2;
  fjac[1][3] = 0.0;
  fjac[2][3] = t_u[3] * tmp1;
  fjac[3][3] = t_u[2] * tmp1;
  fjac[4][3] = 0.0;

  fjac[0][4] = ( c2 * 2.0 * square - c1 * t_u[4] )
    * t_u[2] * tmp2;
  fjac[1][4] = - c2 * t_u[1]*t_u[2] * tmp2;
  fjac[2][4] = c1 * t_u[4] * tmp1 
    - c2 * ( qs + t_u[2]*t_u[2] * tmp2 );
  fjac[3][4] = - c2 * ( t_u[2]*t_u[3] ) * tmp2;
  fjac[4][4] = c1 * t_u[2] * tmp1;


}

__device__ 
static void compute_njac_y(double njac[5][5], 
                           double t_u[5],
                           double rho_i, double c3c4, 
                           double con43, double c1345)
{

  double tmp1, tmp2, tmp3;

  tmp1 = rho_i;
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  njac[0][0] = 0.0;
  njac[1][0] = 0.0;
  njac[2][0] = 0.0;
  njac[3][0] = 0.0;
  njac[4][0] = 0.0;

  njac[0][1] = - c3c4 * tmp2 * t_u[1];
  njac[1][1] =   c3c4 * tmp1;
  njac[2][1] =   0.0;
  njac[3][1] =   0.0;
  njac[4][1] =   0.0;

  njac[0][2] = - con43 * c3c4 * tmp2 * t_u[2];
  njac[1][2] =   0.0;
  njac[2][2] =   con43 * c3c4 * tmp1;
  njac[3][2] =   0.0;
  njac[4][2] =   0.0;

  njac[0][3] = - c3c4 * tmp2 * t_u[3];
  njac[1][3] =   0.0;
  njac[2][3] =   0.0;
  njac[3][3] =   c3c4 * tmp1;
  njac[4][3] =   0.0;

  njac[0][4] = - (  c3c4
      - c1345 ) * tmp3 * (t_u[1]*t_u[1])
    - ( con43 * c3c4
        - c1345 ) * tmp3 * (t_u[2]*t_u[2])
    - ( c3c4 - c1345 ) * tmp3 * (t_u[3]*t_u[3])
    - c1345 * tmp2 * t_u[4];

  njac[1][4] = (  c3c4 - c1345 ) * tmp2 * t_u[1];
  njac[2][4] = ( con43 * c3c4 - c1345 ) * tmp2 * t_u[2];
  njac[3][4] = ( c3c4 - c1345 ) * tmp2 * t_u[3];
  njac[4][4] = ( c1345 ) * tmp1;


}


__launch_bounds__(min(MAX_THREAD_BLOCK_SIZE, MAX_THREAD_DIM_0))
__global__ 
void k_y_solve2_fullopt(double *m_qs, double *m_rho_i,
                        double *m_square, double *m_u, 
                        double *m_lhsA, double *m_lhsB, double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        double dy1, double dy2, 
                        double dy3, double dy4, 
                        double dy5,
                        double c1, double c2, 
                        double ty1, double ty2, 
                        double con43,
                        double c3c4, double c1345, 
                        double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT)
{

  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item  || j > gp1-2 || i > gp0-2) return;

  if (split_flag) k += 2;
  else k += work_base;

  int m;
  double tmp1, tmp2;

   double (* qs)[JMAXP+1][IMAXP+1]
    = ( double (*) [JMAXP+1][IMAXP+1])m_qs;
   double (* rho_i)[JMAXP+1][IMAXP+1]
    = ( double (*) [JMAXP+1][IMAXP+1])m_rho_i;
   double (* square)[JMAXP+1][IMAXP+1]
    = ( double (*) [JMAXP+1][IMAXP+1]) m_square; 
   double (* u)[JMAXP+1][IMAXP+1][5]
    = ( double (*) [JMAXP+1][IMAXP+1][5])m_u;

#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]


  double fjac[5][5], njac[5][5];

  double t_u[5];



  //---------------------------------------------------------------------
  // Compute the indices for storing the tri-diagonal matrix;
  // determine a (labeled f) and n jacobians for cell c
  //---------------------------------------------------------------------

  tmp1 = dt * ty1;
  tmp2 = dt * ty2;



  for (m = 0; m < 5; m++) t_u[m] = u[k][j-1][i][m];
  compute_fjac_y(fjac, t_u, rho_i[k][j-1][i], square[k][j-1][i], qs[k][j-1][i], c1, c2);
  compute_njac_y(njac, t_u, rho_i[k][j-1][i], c3c4, con43, c1345);

  // in "i-1", -1 is for reducing memory usage
  
  lhsA(0, 0, k, j, i-1) = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dy1; 
  lhsA(1, 0, k, j, i-1) = - tmp2 * fjac[1][0] - tmp1 * njac[1][0];
  lhsA(2, 0, k, j, i-1) = - tmp2 * fjac[2][0] - tmp1 * njac[2][0];
  lhsA(3, 0, k, j, i-1) = - tmp2 * fjac[3][0] - tmp1 * njac[3][0];
  lhsA(4, 0, k, j, i-1) = - tmp2 * fjac[4][0] - tmp1 * njac[4][0];

  lhsA(0, 1, k, j, i-1) = - tmp2 * fjac[0][1] - tmp1 * njac[0][1];
  lhsA(1, 1, k, j, i-1) = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dy2;
  lhsA(2, 1, k, j, i-1) = - tmp2 * fjac[2][1] - tmp1 * njac[2][1];
  lhsA(3, 1, k, j, i-1) = - tmp2 * fjac[3][1] - tmp1 * njac[3][1];
  lhsA(4, 1, k, j, i-1) = - tmp2 * fjac[4][1] - tmp1 * njac[4][1];

  lhsA(0, 2, k, j, i-1) = - tmp2 * fjac[0][2] - tmp1 * njac[0][2];
  lhsA(1, 2, k, j, i-1) = - tmp2 * fjac[1][2] - tmp1 * njac[1][2];
  lhsA(2, 2, k, j, i-1) = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dy3;
  lhsA(3, 2, k, j, i-1) = - tmp2 * fjac[3][2] - tmp1 * njac[3][2];
  lhsA(4, 2, k, j, i-1) = - tmp2 * fjac[4][2] - tmp1 * njac[4][2];

  lhsA(0, 3, k, j, i-1) = - tmp2 * fjac[0][3] - tmp1 * njac[0][3];
  lhsA(1, 3, k, j, i-1) = - tmp2 * fjac[1][3] - tmp1 * njac[1][3];
  lhsA(2, 3, k, j, i-1) = - tmp2 * fjac[2][3] - tmp1 * njac[2][3];
  lhsA(3, 3, k, j, i-1) = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dy4;
  lhsA(4, 3, k, j, i-1) = - tmp2 * fjac[4][3] - tmp1 * njac[4][3];

  lhsA(0, 4, k, j, i-1) = - tmp2 * fjac[0][4] - tmp1 * njac[0][4];
  lhsA(1, 4, k, j, i-1) = - tmp2 * fjac[1][4] - tmp1 * njac[1][4];
  lhsA(2, 4, k, j, i-1) = - tmp2 * fjac[2][4] - tmp1 * njac[2][4];
  lhsA(3, 4, k, j, i-1) = - tmp2 * fjac[3][4] - tmp1 * njac[3][4];
  lhsA(4, 4, k, j, i-1) = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dy5;



  for (m = 0; m < 5; m++) t_u[m] = u[k][j][i][m];
  compute_njac_y(njac, t_u, rho_i[k][j][i], c3c4, con43, c1345);



  lhsB(0, 0, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * dy1;
  lhsB(1, 0, k, j, i-1) = tmp1 * 2.0 * njac[1][0];
  lhsB(2, 0, k, j, i-1) = tmp1 * 2.0 * njac[2][0];
  lhsB(3, 0, k, j, i-1) = tmp1 * 2.0 * njac[3][0];
  lhsB(4, 0, k, j, i-1) = tmp1 * 2.0 * njac[4][0];

  lhsB(0, 1, k, j, i-1) = tmp1 * 2.0 * njac[0][1];
  lhsB(1, 1, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * dy2;
  lhsB(2, 1, k, j, i-1) = tmp1 * 2.0 * njac[2][1];
  lhsB(3, 1, k, j, i-1) = tmp1 * 2.0 * njac[3][1];
  lhsB(4, 1, k, j, i-1) = tmp1 * 2.0 * njac[4][1];

  lhsB(0, 2, k, j, i-1) = tmp1 * 2.0 * njac[0][2];
  lhsB(1, 2, k, j, i-1) = tmp1 * 2.0 * njac[1][2];
  lhsB(2, 2, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * dy3;
  lhsB(3, 2, k, j, i-1) = tmp1 * 2.0 * njac[3][2];
  lhsB(4, 2, k, j, i-1) = tmp1 * 2.0 * njac[4][2];

  lhsB(0, 3, k, j, i-1) = tmp1 * 2.0 * njac[0][3];
  lhsB(1, 3, k, j, i-1) = tmp1 * 2.0 * njac[1][3];
  lhsB(2, 3, k, j, i-1) = tmp1 * 2.0 * njac[2][3];
  lhsB(3, 3, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * dy4;
  lhsB(4, 3, k, j, i-1) = tmp1 * 2.0 * njac[4][3];

  lhsB(0, 4, k, j, i-1) = tmp1 * 2.0 * njac[0][4];
  lhsB(1, 4, k, j, i-1) = tmp1 * 2.0 * njac[1][4];
  lhsB(2, 4, k, j, i-1) = tmp1 * 2.0 * njac[2][4];
  lhsB(3, 4, k, j, i-1) = tmp1 * 2.0 * njac[3][4];
  lhsB(4, 4, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * dy5;



  for (m = 0; m < 5; m++) t_u[m] = u[k][j+1][i][m];

  compute_fjac_y(fjac, t_u, rho_i[k][j+1][i], square[k][j+1][i], qs[k][j+1][i], c1, c2);
  compute_njac_y(njac, t_u, rho_i[k][j+1][i], c3c4, con43, c1345);



  lhsC(0, 0, k, j, i-1) =  tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dy1;
  lhsC(1, 0, k, j, i-1) =  tmp2 * fjac[1][0] - tmp1 * njac[1][0];
  lhsC(2, 0, k, j, i-1) =  tmp2 * fjac[2][0] - tmp1 * njac[2][0];
  lhsC(3, 0, k, j, i-1) =  tmp2 * fjac[3][0] - tmp1 * njac[3][0];
  lhsC(4, 0, k, j, i-1) =  tmp2 * fjac[4][0] - tmp1 * njac[4][0];

  lhsC(0, 1, k, j, i-1) =  tmp2 * fjac[0][1] - tmp1 * njac[0][1];
  lhsC(1, 1, k, j, i-1) =  tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dy2;
  lhsC(2, 1, k, j, i-1) =  tmp2 * fjac[2][1] - tmp1 * njac[2][1];
  lhsC(3, 1, k, j, i-1) =  tmp2 * fjac[3][1] - tmp1 * njac[3][1];
  lhsC(4, 1, k, j, i-1) =  tmp2 * fjac[4][1] - tmp1 * njac[4][1];

  lhsC(0, 2, k, j, i-1) =  tmp2 * fjac[0][2] - tmp1 * njac[0][2];
  lhsC(1, 2, k, j, i-1) =  tmp2 * fjac[1][2] - tmp1 * njac[1][2];
  lhsC(2, 2, k, j, i-1) =  tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dy3;
  lhsC(3, 2, k, j, i-1) =  tmp2 * fjac[3][2] - tmp1 * njac[3][2];
  lhsC(4, 2, k, j, i-1) =  tmp2 * fjac[4][2] - tmp1 * njac[4][2];

  lhsC(0, 3, k, j, i-1) =  tmp2 * fjac[0][3] - tmp1 * njac[0][3];
  lhsC(1, 3, k, j, i-1) =  tmp2 * fjac[1][3] - tmp1 * njac[1][3];
  lhsC(2, 3, k, j, i-1) =  tmp2 * fjac[2][3] - tmp1 * njac[2][3];
  lhsC(3, 3, k, j, i-1) =  tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dy4;
  lhsC(4, 3, k, j, i-1) =  tmp2 * fjac[4][3] - tmp1 * njac[4][3];

  lhsC(0, 4, k, j, i-1) =  tmp2 * fjac[0][4] - tmp1 * njac[0][4];
  lhsC(1, 4, k, j, i-1) =  tmp2 * fjac[1][4] - tmp1 * njac[1][4];
  lhsC(2, 4, k, j, i-1) =  tmp2 * fjac[2][4] - tmp1 * njac[2][4];
  lhsC(3, 4, k, j, i-1) =  tmp2 * fjac[3][4] - tmp1 * njac[3][4];
  lhsC(4, 4, k, j, i-1) =  tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dy5;


#undef lhsA
#undef lhsB
#undef lhsC

}





__global__ 
void k_y_solve3_fullopt(double *m_rhs,
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT)
{
  extern __shared__ double tmp_l_lhs[];
  double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];


  int k = (blockDim.y * blockIdx.y + threadIdx.y)/5;
  int m = (blockDim.y * blockIdx.y + threadIdx.y)%5;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;
  int dummy = 0;
  int l_i = threadIdx.x;
  if (k+work_base < 1 || k+work_base > gp2-2 || k >= work_num_item || i > gp0-2) dummy = 1;

  if (split_flag) k += 2;
  else k += work_base;

  int j, n, p, jsize;

   double (* rhs)[JMAXP+1][IMAXP+1][5]
    = ( double (*) [JMAXP+1][IMAXP+1][5])m_rhs;

#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) *  WORK_NUM_ITEM_DEFAULT + (c)) * (PROBLEM_SIZE+1) + (d)) * (PROBLEM_SIZE-1) + (e)]

   double (* tmp2_l_lhs)[3][5][5] = ( double(*) [3][5][5])tmp_l_lhs;
   double (* l_lhs)[5][5] = tmp2_l_lhs[l_i];
   double (* tmp2_l_r)[2][5] = ( double(*)[2][5])tmp_l_r;
   double (* l_r)[5] = tmp2_l_r[l_i]; 

  double pivot, coeff;

  jsize = gp1 - 1;



  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(JMAX) and rhs'(JMAX) will be sent to next cell
  //---------------------------------------------------------------------
  

  // load data
  if (!dummy) {
    for (p = 0; p < 5; p++) {
      l_lhs[BB][p][m] = lhsB(p, m, k, 0, i-1);
      l_lhs[CC][p][m] = lhsC(p, m, k, 0, i-1);
    }

    l_r[1][m] = rhs[k][0][i][m];
  } 
  __syncthreads();



  //---------------------------------------------------------------------
  // multiply c[k][0][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  for (p = 0; p < 5; p++) {
    if (!dummy) {
      pivot = 1.00/l_lhs[BB][p][p];
      if (m > p && m < 5)   l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
      if (m < 5)    l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;
      if (p == m)   l_r[1][p] = l_r[1][p]*pivot;
    }
    //barrier
    __syncthreads();

    if (!dummy) {
      if (p != m) {
        coeff = l_lhs[BB][p][m];
        for (n = p+1; n < 5; n++) 
          l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
        for (n = 0; n < 5; n++) 
          l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];
        l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
      }
    } 
    //barrier
    __syncthreads();
  }



  //update data
  if (!dummy)
    rhs[k][0][i][m] = l_r[1][m];


  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (j = 1; j <= jsize-1; j++) {

    // load data
    if (!dummy) {
      for (n = 0; n < 5; n++) {
        l_lhs[AA][n][m] = lhsA(n, m, k, j, i-1);
        l_lhs[BB][n][m] = lhsB(n, m, k, j, i-1);
      }
      l_r[0][m] = l_r[1][m];
      l_r[1][m] = rhs[k][j][i][m];

    }
    __syncthreads();


    //-------------------------------------------------------------------
    // subtract A*lhs_vector(j-1) from lhs_vector(j)
    // 
    // rhs(j) = rhs(j) - A*rhs(j-1)
    //-------------------------------------------------------------------

    if (!dummy) {
      l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
        - l_lhs[AA][1][m]*l_r[0][1]
        - l_lhs[AA][2][m]*l_r[0][2]
        - l_lhs[AA][3][m]*l_r[0][3]
        - l_lhs[AA][4][m]*l_r[0][4];
    }   

    //-------------------------------------------------------------------
    // B(j) = B(j) - C(j-1)*A(j)
    //-------------------------------------------------------------------

    if (!dummy) {
      for (p = 0; p < 5; p++) {
        l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
          - l_lhs[AA][1][p]*l_lhs[CC][m][1]
          - l_lhs[AA][2][p]*l_lhs[CC][m][2]
          - l_lhs[AA][3][p]*l_lhs[CC][m][3]
          - l_lhs[AA][4][p]*l_lhs[CC][m][4];
      }
    }
    __syncthreads();




    // update data
    if (!dummy) {
      for (n = 0; n < 5; n++) l_lhs[CC][n][m] = lhsC(n, m, k, j, i-1);
    }
    __syncthreads();


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
    //-------------------------------------------------------------------
    for (p = 0; p < 5; p++) {

      if (!dummy) {
        pivot = 1.00/l_lhs[BB][p][p];
        if(m > p) l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
        l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;
        if(p == m) l_r[1][p] = l_r[1][p]*pivot;
      }

      //barrier
      __syncthreads();
      if (!dummy) {
        if (p != m) {
          coeff = l_lhs[BB][p][m];
          for (n = p+1; n < 5; n++) 
            l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
          for (n = 0; n < 5; n++) 
            l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];
          l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
        }
      } 

      //barrier
      __syncthreads();

    }




    // Update data from local memory to global memory
    if (!dummy)
    {
      for (n = 0; n < 5; n++) {
        lhsC(n, m, k, j, i-1) = l_lhs[CC][n][m];
      }
      rhs[k][j][i][m] = l_r[1][m];
    }
  

  }


  // load data
  if (!dummy) {
    for (n = 0; n < 5; n++) {
      l_lhs[AA][n][m] = lhsA(n, m, k, j, i-1);
      l_lhs[BB][n][m] = lhsB(n, m, k, j, i-1);
    }
    l_r[0][m] = l_r[1][m];
    l_r[1][m] = rhs[k][j][i][m];

  }
  __syncthreads();






  //---------------------------------------------------------------------
  // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
  //---------------------------------------------------------------------
  if (!dummy) {
    l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
      - l_lhs[AA][1][m]*l_r[0][1]
      - l_lhs[AA][2][m]*l_r[0][2]
      - l_lhs[AA][3][m]*l_r[0][3]
      - l_lhs[AA][4][m]*l_r[0][4];

  }

  //---------------------------------------------------------------------
  // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
  // matmul_sub(AA,i,jsize,k,c,
  // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
  //---------------------------------------------------------------------

  if (!dummy) {
    for (p = 0; p < 5; p++) {
      l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
        - l_lhs[AA][1][p]*l_lhs[CC][m][1]
        - l_lhs[AA][2][p]*l_lhs[CC][m][2]
        - l_lhs[AA][3][p]*l_lhs[CC][m][3]
        - l_lhs[AA][4][p]*l_lhs[CC][m][4];
    }
  }
  
  

  
  

  //---------------------------------------------------------------------
  // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
  //---------------------------------------------------------------------
  //binvrhs_p( lhs[jsize][BB], rhs[k][jsize][i], dummy, m);

  for (p = 0; p < 5; p++) {
    if (!dummy) {
      pivot = 1.00/l_lhs[BB][p][p];
      if (m > p && m < 5)   l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
      if (p == m)   l_r[1][p] = l_r[1][p]*pivot;
    }
    //barrier
    __syncthreads();

    if (!dummy) {
      if (p != m) {
        coeff = l_lhs[BB][p][m];
        for (n = p+1; n < 5; n++) 
          l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
        l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
      }
    } 
    //barrier
    __syncthreads();

  }

  if (!dummy)
    rhs[k][j][i][m] = l_r[1][m];

  __syncthreads();






  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(jsize)=rhs(jsize)
  // else assume U(jsize) is loaded in un pack backsub_info
  // so just use it
  // after u(jstart) will be sent to next cell
  //---------------------------------------------------------------------

  for (j = jsize-1; j >= 0; j--) {
    if (!dummy) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhsC(n, m, k, j, i-1)*rhs[k][j+1][i][n];
      }
    }
    __syncthreads();
  }

#undef lhsA
#undef lhsB
#undef lhsC

}












//---------------------------------------------------------------------
// This function computes the left hand side for the three z-factors   
//---------------------------------------------------------------------

__global__ 
void k_z_solve_data_gen_fullopt(double *m_u,
                                double *m_square, double *m_qs,
                                int gp0, int gp1, int gp2,
                                int work_base, 
                                int work_num_item, 
                                int split_flag, 
                                int WORK_NUM_ITEM_DEFAULT_J)
{

  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;

  if (k > gp2-1 || j+work_base < 1 || j+work_base > gp1-2 || j >= work_num_item || i > gp0-2) return;


#define qs(a, b, c) m_qs[((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)]
#define square(a, b, c) m_square[((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)]
#define u(a, b, c, d) m_u[(((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)) * 5 + (d)]

  double rho_inv;
  double t_u[4];
  int m;

  for (m = 0; m < 4; m++) {
    t_u[m] = u(k, j, i, m);
  }

  rho_inv = 1.0/t_u[0];
  square(k, j, i) = 0.5* (
      t_u[1]*t_u[1] + 
      t_u[2]*t_u[2] +
      t_u[3]*t_u[3] ) * rho_inv;
  qs(k, j, i) = square(k, j, i) * rho_inv;

#undef qs
#undef square
#undef u
}





//---------------------------------------------------------------------
// This function computes the left hand side for the three z-factors   
//---------------------------------------------------------------------
__global__ 
void k_z_solve1_fullopt(double *m_lhsA, 
                        double *m_lhsB, double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT_J)
{


  int t_j = blockDim.y * blockIdx.y + threadIdx.y;
  int j = t_j % work_num_item;
  int mn = t_j / work_num_item;
  int m = mn / 5;
  int n = mn % 5;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;
  
  if (j+work_base < 1 || j+work_base > gp1-2 || j >= work_num_item || i > gp0-2 || m >= 5) return;

  if (!split_flag) j += work_base;

  int ksize;

#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]


  ksize = gp2 - 1;


  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in z direction
  //---------------------------------------------------------------------

  lhsA(m, n, 0, j, i-1) = 0.0;
  lhsB(m, n, 0, j, i-1) = (m==n)?1.0:0.0;
  lhsC(m, n, 0, j, i-1) = 0.0;

  lhsA(m, n, ksize, j, i-1) = 0.0;
  lhsB(m, n, ksize, j, i-1) = (m==n)?1.0:0.0;
  lhsC(m, n, ksize, j, i-1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC

}




//---------------------------------------------------------------------
// This function computes the left hand side for the three z-factors   
//---------------------------------------------------------------------
__device__ 
static void compute_fjac_z(double l_fjac[5][5], double t_u[5],
      double square, double qs, double c1, double c2)
{

  double tmp1, tmp2;

  tmp1 = 1.0 / t_u[0];
  tmp2 = tmp1 * tmp1;

  l_fjac[0][0] = 0.0;
  l_fjac[1][0] = 0.0;
  l_fjac[2][0] = 0.0;
  l_fjac[3][0] = 1.0;
  l_fjac[4][0] = 0.0;

  l_fjac[0][1] = - ( t_u[1]*t_u[3] ) * tmp2;
  l_fjac[1][1] = t_u[3] * tmp1;
  l_fjac[2][1] = 0.0;
  l_fjac[3][1] = t_u[1] * tmp1;
  l_fjac[4][1] = 0.0;

  l_fjac[0][2] = - ( t_u[2]*t_u[3] ) * tmp2;
  l_fjac[1][2] = 0.0;
  l_fjac[2][2] = t_u[3] * tmp1;
  l_fjac[3][2] = t_u[2] * tmp1;
  l_fjac[4][2] = 0.0;

  l_fjac[0][3] = - (t_u[3]*t_u[3] * tmp2 ) 
    + c2 * qs;
  l_fjac[1][3] = - c2 *  t_u[1] * tmp1;
  l_fjac[2][3] = - c2 *  t_u[2] * tmp1;
  l_fjac[3][3] = ( 2.0 - c2 ) *  t_u[3] * tmp1;
  l_fjac[4][3] = c2;

  l_fjac[0][4] = ( c2 * 2.0 * square - c1 * t_u[4] )
    * t_u[3] * tmp2;
  l_fjac[1][4] = - c2 * ( t_u[1]*t_u[3] ) * tmp2;
  l_fjac[2][4] = - c2 * ( t_u[2]*t_u[3] ) * tmp2;
  l_fjac[3][4] = c1 * ( t_u[4] * tmp1 )
    - c2 * ( qs + t_u[3]*t_u[3] * tmp2 );
  l_fjac[4][4] = c1 * t_u[3] * tmp1;

}


__device__ 
static void compute_njac_z(double l_njac[5][5], double t_u[5],
      double c3c4, double c1345, double con43, double c3, double c4)
{

  double tmp1, tmp2, tmp3;

  tmp1 = 1.0 / t_u[0];
  tmp2 = tmp1 * tmp1;
  tmp3 = tmp1 * tmp2;

  l_njac[0][0] = 0.0;
  l_njac[1][0] = 0.0;
  l_njac[2][0] = 0.0;
  l_njac[3][0] = 0.0;
  l_njac[4][0] = 0.0;

  l_njac[0][1] = - c3c4 * tmp2 * t_u[1];
  l_njac[1][1] =   c3c4 * tmp1;
  l_njac[2][1] =   0.0;
  l_njac[3][1] =   0.0;
  l_njac[4][1] =   0.0;

  l_njac[0][2] = - c3c4 * tmp2 * t_u[2];
  l_njac[1][2] =   0.0;
  l_njac[2][2] =   c3c4 * tmp1;
  l_njac[3][2] =   0.0;
  l_njac[4][2] =   0.0;

  l_njac[0][3] = - con43 * c3c4 * tmp2 * t_u[3];
  l_njac[1][3] =   0.0;
  l_njac[2][3] =   0.0;
  l_njac[3][3] =   con43 * c3 * c4 * tmp1;
  l_njac[4][3] =   0.0;

  l_njac[0][4] = - (  c3c4
      - c1345 ) * tmp3 * (t_u[1]*t_u[1])
    - ( c3c4 - c1345 ) * tmp3 * (t_u[2]*t_u[2])
    - ( con43 * c3c4
        - c1345 ) * tmp3 * (t_u[3]*t_u[3])
    - c1345 * tmp2 * t_u[4];

  l_njac[1][4] = (  c3c4 - c1345 ) * tmp2 * t_u[1];
  l_njac[2][4] = (  c3c4 - c1345 ) * tmp2 * t_u[2];
  l_njac[3][4] = ( con43 * c3c4
      - c1345 ) * tmp2 * t_u[3];
  l_njac[4][4] = ( c1345 )* tmp1;

}



__launch_bounds__(min(MAX_THREAD_BLOCK_SIZE, MAX_THREAD_DIM_0))
__global__ 
void k_z_solve2_fullopt(double *m_qs, double *m_square, 
                        double *m_u,
                        double *m_lhsA, double *m_lhsB, 
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        double dz1, double dz2, 
                        double dz3, double dz4, 
                        double dz5,
                        double c1, double c2, 
                        double c3, double c4,
                        double tz1, double tz2, 
                        double con43,
                        double c3c4, double c1345, 
                        double dt,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT_J)
{

  int k = blockDim.z * blockIdx.z + threadIdx.z + 1;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

  if (k > gp2-2 || j+work_base < 1 || j+work_base > gp1-2 || j >= work_num_item || i > gp0-2 ) return;

  if (!split_flag) j += work_base;

  int m;
  double tmp1, tmp2;



#define qs(a, b, c)         m_qs[((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)]
#define square(a, b, c) m_square[((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)]
#define u(a, b, c, d)        m_u[(((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)) * 5 + (d)]

#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]


  double fjac[5][5];
  double njac[5][5];

  double t_u[5];

  //---------------------------------------------------------------------
  // Compute the indices for storing the block-diagonal matrix;
  // determine c (labeled f) and s jacobians
  //---------------------------------------------------------------------


  tmp1 = dt * tz1;
  tmp2 = dt * tz2;

  for (m = 0; m < 5; m++) t_u[m] = u(k-1, j, i, m);

  compute_fjac_z(fjac, t_u, square(k-1, j, i), qs(k-1, j, i), c1, c2);
  compute_njac_z(njac, t_u, c3c4, c1345, con43, c3, c4);

  lhsA(0, 0, k, j, i-1) = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dz1; 
  lhsA(1, 0, k, j, i-1) = - tmp2 * fjac[1][0] - tmp1 * njac[1][0];
  lhsA(2, 0, k, j, i-1) = - tmp2 * fjac[2][0] - tmp1 * njac[2][0];
  lhsA(3, 0, k, j, i-1) = - tmp2 * fjac[3][0] - tmp1 * njac[3][0];
  lhsA(4, 0, k, j, i-1) = - tmp2 * fjac[4][0] - tmp1 * njac[4][0];
              
  lhsA(0, 1, k, j, i-1) = - tmp2 * fjac[0][1] - tmp1 * njac[0][1];
  lhsA(1, 1, k, j, i-1) = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dz2;
  lhsA(2, 1, k, j, i-1) = - tmp2 * fjac[2][1] - tmp1 * njac[2][1];
  lhsA(3, 1, k, j, i-1) = - tmp2 * fjac[3][1] - tmp1 * njac[3][1];
  lhsA(4, 1, k, j, i-1) = - tmp2 * fjac[4][1] - tmp1 * njac[4][1];
              
  lhsA(0, 2, k, j, i-1) = - tmp2 * fjac[0][2] - tmp1 * njac[0][2];
  lhsA(1, 2, k, j, i-1) = - tmp2 * fjac[1][2] - tmp1 * njac[1][2];
  lhsA(2, 2, k, j, i-1) = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dz3;
  lhsA(3, 2, k, j, i-1) = - tmp2 * fjac[3][2] - tmp1 * njac[3][2];
  lhsA(4, 2, k, j, i-1) = - tmp2 * fjac[4][2] - tmp1 * njac[4][2];
              
  lhsA(0, 3, k, j, i-1) = - tmp2 * fjac[0][3] - tmp1 * njac[0][3];
  lhsA(1, 3, k, j, i-1) = - tmp2 * fjac[1][3] - tmp1 * njac[1][3];
  lhsA(2, 3, k, j, i-1) = - tmp2 * fjac[2][3] - tmp1 * njac[2][3];
  lhsA(3, 3, k, j, i-1) = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dz4;
  lhsA(4, 3, k, j, i-1) = - tmp2 * fjac[4][3] - tmp1 * njac[4][3];
              
  lhsA(0, 4, k, j, i-1) = - tmp2 * fjac[0][4] - tmp1 * njac[0][4];
  lhsA(1, 4, k, j, i-1) = - tmp2 * fjac[1][4] - tmp1 * njac[1][4];
  lhsA(2, 4, k, j, i-1) = - tmp2 * fjac[2][4] - tmp1 * njac[2][4];
  lhsA(3, 4, k, j, i-1) = - tmp2 * fjac[3][4] - tmp1 * njac[3][4];
  lhsA(4, 4, k, j, i-1) = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dz5;

  for (m = 0; m < 5; m++) t_u[m] = u(k, j, i, m);

  compute_njac_z(njac, t_u, c3c4, c1345, con43, c3, c4);


  lhsB(0, 0, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * dz1;
  lhsB(1, 0, k, j, i-1) = tmp1 * 2.0 * njac[1][0];
  lhsB(2, 0, k, j, i-1) = tmp1 * 2.0 * njac[2][0];
  lhsB(3, 0, k, j, i-1) = tmp1 * 2.0 * njac[3][0];
  lhsB(4, 0, k, j, i-1) = tmp1 * 2.0 * njac[4][0];

  lhsB(0, 1, k, j, i-1) = tmp1 * 2.0 * njac[0][1];
  lhsB(1, 1, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * dz2;
  lhsB(2, 1, k, j, i-1) = tmp1 * 2.0 * njac[2][1];
  lhsB(3, 1, k, j, i-1) = tmp1 * 2.0 * njac[3][1];
  lhsB(4, 1, k, j, i-1) = tmp1 * 2.0 * njac[4][1];

  lhsB(0, 2, k, j, i-1) = tmp1 * 2.0 * njac[0][2];
  lhsB(1, 2, k, j, i-1) = tmp1 * 2.0 * njac[1][2];
  lhsB(2, 2, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * dz3;
  lhsB(3, 2, k, j, i-1) = tmp1 * 2.0 * njac[3][2];
  lhsB(4, 2, k, j, i-1) = tmp1 * 2.0 * njac[4][2];

  lhsB(0, 3, k, j, i-1) = tmp1 * 2.0 * njac[0][3];
  lhsB(1, 3, k, j, i-1) = tmp1 * 2.0 * njac[1][3];
  lhsB(2, 3, k, j, i-1) = tmp1 * 2.0 * njac[2][3];
  lhsB(3, 3, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * dz4;
  lhsB(4, 3, k, j, i-1) = tmp1 * 2.0 * njac[4][3];

  lhsB(0, 4, k, j, i-1) = tmp1 * 2.0 * njac[0][4];
  lhsB(1, 4, k, j, i-1) = tmp1 * 2.0 * njac[1][4];
  lhsB(2, 4, k, j, i-1) = tmp1 * 2.0 * njac[2][4];
  lhsB(3, 4, k, j, i-1) = tmp1 * 2.0 * njac[3][4];
  lhsB(4, 4, k, j, i-1) = 1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * dz5;




  for (m = 0; m < 5; m++) t_u[m] = u(k+1, j, i, m);

  compute_fjac_z(fjac, t_u, square(k+1, j, i), qs(k+1, j, i), c1, c2);
  compute_njac_z(njac, t_u, c3c4, c1345, con43, c3, c4);


  lhsC(0, 0, k, j, i-1) =  tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dz1;
  lhsC(1, 0, k, j, i-1) =  tmp2 * fjac[1][0] - tmp1 * njac[1][0];
  lhsC(2, 0, k, j, i-1) =  tmp2 * fjac[2][0] - tmp1 * njac[2][0];
  lhsC(3, 0, k, j, i-1) =  tmp2 * fjac[3][0] - tmp1 * njac[3][0];
  lhsC(4, 0, k, j, i-1) =  tmp2 * fjac[4][0] - tmp1 * njac[4][0];
              
  lhsC(0, 1, k, j, i-1) =  tmp2 * fjac[0][1] - tmp1 * njac[0][1];
  lhsC(1, 1, k, j, i-1) =  tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dz2;
  lhsC(2, 1, k, j, i-1) =  tmp2 * fjac[2][1] - tmp1 * njac[2][1];
  lhsC(3, 1, k, j, i-1) =  tmp2 * fjac[3][1] - tmp1 * njac[3][1];
  lhsC(4, 1, k, j, i-1) =  tmp2 * fjac[4][1] - tmp1 * njac[4][1];
              
  lhsC(0, 2, k, j, i-1) =  tmp2 * fjac[0][2] - tmp1 * njac[0][2];
  lhsC(1, 2, k, j, i-1) =  tmp2 * fjac[1][2] - tmp1 * njac[1][2];
  lhsC(2, 2, k, j, i-1) =  tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dz3;
  lhsC(3, 2, k, j, i-1) =  tmp2 * fjac[3][2] - tmp1 * njac[3][2];
  lhsC(4, 2, k, j, i-1) =  tmp2 * fjac[4][2] - tmp1 * njac[4][2];
              
  lhsC(0, 3, k, j, i-1) =  tmp2 * fjac[0][3] - tmp1 * njac[0][3];
  lhsC(1, 3, k, j, i-1) =  tmp2 * fjac[1][3] - tmp1 * njac[1][3];
  lhsC(2, 3, k, j, i-1) =  tmp2 * fjac[2][3] - tmp1 * njac[2][3];
  lhsC(3, 3, k, j, i-1) =  tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dz4;
  lhsC(4, 3, k, j, i-1) =  tmp2 * fjac[4][3] - tmp1 * njac[4][3];
              
  lhsC(0, 4, k, j, i-1) =  tmp2 * fjac[0][4] - tmp1 * njac[0][4];
  lhsC(1, 4, k, j, i-1) =  tmp2 * fjac[1][4] - tmp1 * njac[1][4];
  lhsC(2, 4, k, j, i-1) =  tmp2 * fjac[2][4] - tmp1 * njac[2][4];
  lhsC(3, 4, k, j, i-1) =  tmp2 * fjac[3][4] - tmp1 * njac[3][4];
  lhsC(4, 4, k, j, i-1) =  tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dz5;


#undef qs
#undef square
#undef u
#undef lhsA
#undef lhsB
#undef lhsC
}



//---------------------------------------------------------------------
// This function computes the left hand side for the three z-factors   
//---------------------------------------------------------------------
__global__ 
void k_z_solve3_fullopt(double *m_rhs,
                        double *m_lhsA, double *m_lhsB,  
                        double *m_lhsC,
                        int gp0, int gp1, int gp2,
                        int work_base, 
                        int work_num_item, 
                        int split_flag, 
                        int WORK_NUM_ITEM_DEFAULT_J)
{
  extern __shared__ double tmp_l_lhs[];
  double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];

  int t_j = blockDim.y * blockIdx.y + threadIdx.y;
  int j = t_j / 5;
  int m = t_j % 5;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;
  int l_i = threadIdx.x;
  int dummy = 0;
  
  if (j+work_base < 1 || j+work_base > gp1 - 2 || j >= work_num_item || i > gp0 - 2 ) dummy = 1;

  if (!split_flag) j += work_base;

  int k, n, p, ksize;

#define rhs(a, b, c, d) m_rhs[(((a) * WORK_NUM_ITEM_DEFAULT_J + (b)) * (IMAXP+1) + (c)) * 5 + (d)]

#define lhsA(a, b, c, d, e) m_lhsA[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsB(a, b, c, d, e) m_lhsB[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]
#define lhsC(a, b, c, d, e) m_lhsC[((((a) * 5 + (b)) * (PROBLEM_SIZE+1) + (c)) * WORK_NUM_ITEM_DEFAULT_J + (d)) * (PROBLEM_SIZE-1) + (e)]



   double (* tmp2_l_lhs)[3][5][5] = ( double(*) [3][5][5])tmp_l_lhs;
   double (* l_lhs)[5][5] = tmp2_l_lhs[l_i];
   double (* tmp2_l_r)[2][5] = ( double(*)[2][5])tmp_l_r;
   double (* l_r)[5] = tmp2_l_r[l_i]; 

  double pivot, coeff;

  ksize = gp2 - 1;

  //---------------------------------------------------------------------
  // Compute the indices for storing the block-diagonal matrix;
  // determine c (labeled f) and s jacobians
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // performs guaussian elimination on this cell.
  // 
  // assumes that unpacking routines for non-first cells 
  // preload C' and rhs' from previous cell.
  // 
  // assumed send happens outside this routine, but that
  // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------


  //load data
  if (!dummy){
    for(p = 0; p < 5; p++){
      l_lhs[BB][p][m] = lhsB(p, m, 0, j, i-1);
      l_lhs[CC][p][m] = lhsC(p, m, 0, j, i-1);
    }
    l_r[1][m] = rhs(0, j, i, m);
  }
  __syncthreads();

  //---------------------------------------------------------------------
  // multiply c[0][j][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  for (p = 0; p < 5; p++) {
    if (!dummy) {
      pivot = 1.00/l_lhs[BB][p][p];
      if(m > p && m < 5)    l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
      if(m < 5)     l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;
      if(p == m)    l_r[1][p] = l_r[1][p]*pivot;
    }
    //barrier
    __syncthreads();

    if (!dummy) {
      if (p != m) {
        coeff = l_lhs[BB][p][m];
        for(n = p+1; n < 5; n++)  
          l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
        for(n = 0; n < 5; n++) 
          l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];
        l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
      }
    } 
    //barrier
    __syncthreads();
  }




  // update data
  if (!dummy)
    rhs(0, j, i, m) = l_r[1][m];






  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (k = 1; k <= ksize-1; k++) {
    // load data
    if (!dummy) {
      for (n = 0; n < 5; n++) {
        l_lhs[AA][n][m] = lhsA(n, m, k, j, i-1);
        l_lhs[BB][n][m] = lhsB(n, m, k, j, i-1);
      }
      l_r[0][m] = l_r[1][m];
      l_r[1][m] = rhs(k, j, i, m);
    }
    __syncthreads();

    //-------------------------------------------------------------------
    // subtract A*lhs_vector(k-1) from lhs_vector(k)
    // 
    // rhs(k) = rhs(k) - A*rhs(k-1)
    //-------------------------------------------------------------------


    if (!dummy) {
      l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
        - l_lhs[AA][1][m]*l_r[0][1]
        - l_lhs[AA][2][m]*l_r[0][2]
        - l_lhs[AA][3][m]*l_r[0][3]
        - l_lhs[AA][4][m]*l_r[0][4];
    }





    //-------------------------------------------------------------------
    // B(k) = B(k) - C(k-1)*A(k)
    // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
    //-------------------------------------------------------------------

    if (!dummy) {
      for (p = 0; p < 5; p++) {
        l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
          - l_lhs[AA][1][p]*l_lhs[CC][m][1]
          - l_lhs[AA][2][p]*l_lhs[CC][m][2]
          - l_lhs[AA][3][p]*l_lhs[CC][m][3]
          - l_lhs[AA][4][p]*l_lhs[CC][m][4];

      }
    }
    __syncthreads();




    // load data
    if (!dummy) {
      for (n = 0; n < 5; n++) l_lhs[CC][n][m] = lhsC(n, m, k, j, i-1);
    }
    __syncthreads();


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
    //-------------------------------------------------------------------


    for (p = 0; p < 5; p++) {

      if (!dummy) {
        pivot = 1.00/l_lhs[BB][p][p];
        if(m > p && m < 5)    l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
        if(m < 5)     l_lhs[CC][m][p] = l_lhs[CC][m][p]*pivot;
        if(p == m)    l_r[1][p] = l_r[1][p]*pivot;
      }
      //barrier
      __syncthreads();


      if (!dummy) {
        if (p != m) {
          coeff = l_lhs[BB][p][m];
          for(n = p+1; n < 5; n++)  l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
          for(n = 0; n < 5; n++) l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff*l_lhs[CC][n][p];
          l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
        }
      } 
      //barrier
      __syncthreads();
    }



    // update data
    if (!dummy){
      for (n = 0; n < 5; n++) {
        lhsC(n, m, k, j, i-1) = l_lhs[CC][n][m];
      }
      rhs(k, j, i, m) = l_r[1][m];
    }


  }

  


  //---------------------------------------------------------------------
  // Now finish up special cases for last cell
  //---------------------------------------------------------------------
  
  // load data
  if (!dummy) {
    for (n = 0; n < 5; n++) {
      l_lhs[AA][n][m] = lhsA(n, m, k, j, i-1);
      l_lhs[BB][n][m] = lhsB(n, m, k, j, i-1);
    }
    l_r[0][m] = l_r[1][m];
    l_r[1][m] = rhs(k, j, i, m);
  }
  __syncthreads();


  //---------------------------------------------------------------------
  // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
  //---------------------------------------------------------------------
  if (!dummy) {
    l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m]*l_r[0][0]
      - l_lhs[AA][1][m]*l_r[0][1]
      - l_lhs[AA][2][m]*l_r[0][2]
      - l_lhs[AA][3][m]*l_r[0][3]
      - l_lhs[AA][4][m]*l_r[0][4];
  }




  //---------------------------------------------------------------------
  // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
  // matmul_sub(AA,i,j,ksize,c,
  // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
  //---------------------------------------------------------------------
  if (!dummy) {
    for (p = 0; p < 5; p++) {
      l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p]*l_lhs[CC][m][0]
        - l_lhs[AA][1][p]*l_lhs[CC][m][1]
        - l_lhs[AA][2][p]*l_lhs[CC][m][2]
        - l_lhs[AA][3][p]*l_lhs[CC][m][3]
        - l_lhs[AA][4][p]*l_lhs[CC][m][4];

    }
  }


  



  //---------------------------------------------------------------------
  // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
  //---------------------------------------------------------------------

  for (p = 0; p < 5; p++) {
    if (!dummy) {
      pivot = 1.00/l_lhs[BB][p][p];
      if (m > p && m < 5)   l_lhs[BB][m][p] = l_lhs[BB][m][p]*pivot;
      if (p == m)   l_r[1][p] = l_r[1][p]*pivot;
    }
    //barrier
    __syncthreads();

    if (!dummy) {
      if (p != m) {
        coeff = l_lhs[BB][p][m];
        for (n = p+1; n < 5; n++) 
          l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff*l_lhs[BB][n][p];
        l_r[1][m] = l_r[1][m] - coeff*l_r[1][p];  
      }
    } 
    //barrier
    __syncthreads();
  }


  // update data
  if (!dummy)
    rhs(k, j, i, m) = l_r[1][m];

  __syncthreads();


  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(ksize)=rhs(ksize)
  // else assume U(ksize) is loaded in un pack backsub_info
  // so just use it
  // after u(kstart) will be sent to next cell
  //---------------------------------------------------------------------

  for (k = ksize-1; k >= 0; k--) {
    if (!dummy) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs(k, j, i, m) = rhs(k, j, i, m) 
          - lhsC(n, m, k, j, i-1)*rhs(k+1, j, i, n);
      }
    }
    __syncthreads();
  }
#undef rhs
#undef lhsA
#undef lhsB
#undef lhsC
}
