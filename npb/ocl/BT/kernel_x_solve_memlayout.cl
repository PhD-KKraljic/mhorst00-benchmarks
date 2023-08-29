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
// This function computes the left hand side in the xi-direction
//---------------------------------------------------------------------

inline void lhsinit_x(__global double lhs[][3][5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1], int size, int k, int j)
{
  int i, m, n;

  i = size;
  //---------------------------------------------------------------------
  // zero the whole left hand side for starters
  //---------------------------------------------------------------------
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[0][0][n][m][k][j] = 0.0;
      lhs[0][1][n][m][k][j] = 0.0;
      lhs[0][2][n][m][k][j] = 0.0;
      lhs[i][0][n][m][k][j] = 0.0;
      lhs[i][1][n][m][k][j] = 0.0;
      lhs[i][2][n][m][k][j] = 0.0;
    }
  }

  //---------------------------------------------------------------------
  // next, set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs[0][1][m][m][k][j] = 1.0;
    lhs[i][1][m][m][k][j] = 1.0;
  }
}

inline void binvcrhs_x(__global double lhs[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1],
                       __global double c[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1], 
                       __global double r[5], 
                       int k, int j)
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0][k][j];
  lhs[1][0][k][j] = lhs[1][0][k][j]*pivot;
  lhs[2][0][k][j] = lhs[2][0][k][j]*pivot;
  lhs[3][0][k][j] = lhs[3][0][k][j]*pivot;
  lhs[4][0][k][j] = lhs[4][0][k][j]*pivot;
  c[0][0][k][j] = c[0][0][k][j]*pivot;
  c[1][0][k][j] = c[1][0][k][j]*pivot;
  c[2][0][k][j] = c[2][0][k][j]*pivot;
  c[3][0][k][j] = c[3][0][k][j]*pivot;
  c[4][0][k][j] = c[4][0][k][j]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1][k][j];
  lhs[1][1][k][j]= lhs[1][1][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][1][k][j]= lhs[2][1][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][1][k][j]= lhs[3][1][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][1][k][j]= lhs[4][1][k][j] - coeff*lhs[4][0][k][j];
  c[0][1][k][j] = c[0][1][k][j] - coeff*c[0][0][k][j];
  c[1][1][k][j] = c[1][1][k][j] - coeff*c[1][0][k][j];
  c[2][1][k][j] = c[2][1][k][j] - coeff*c[2][0][k][j];
  c[3][1][k][j] = c[3][1][k][j] - coeff*c[3][0][k][j];
  c[4][1][k][j] = c[4][1][k][j] - coeff*c[4][0][k][j];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2][k][j];
  lhs[1][2][k][j]= lhs[1][2][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][2][k][j]= lhs[2][2][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][2][k][j]= lhs[3][2][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][2][k][j]= lhs[4][2][k][j] - coeff*lhs[4][0][k][j];
  c[0][2][k][j] = c[0][2][k][j] - coeff*c[0][0][k][j];
  c[1][2][k][j] = c[1][2][k][j] - coeff*c[1][0][k][j];
  c[2][2][k][j] = c[2][2][k][j] - coeff*c[2][0][k][j];
  c[3][2][k][j] = c[3][2][k][j] - coeff*c[3][0][k][j];
  c[4][2][k][j] = c[4][2][k][j] - coeff*c[4][0][k][j];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3][k][j];
  lhs[1][3][k][j]= lhs[1][3][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][3][k][j]= lhs[2][3][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][3][k][j]= lhs[3][3][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][3][k][j]= lhs[4][3][k][j] - coeff*lhs[4][0][k][j];
  c[0][3][k][j] = c[0][3][k][j] - coeff*c[0][0][k][j];
  c[1][3][k][j] = c[1][3][k][j] - coeff*c[1][0][k][j];
  c[2][3][k][j] = c[2][3][k][j] - coeff*c[2][0][k][j];
  c[3][3][k][j] = c[3][3][k][j] - coeff*c[3][0][k][j];
  c[4][3][k][j] = c[4][3][k][j] - coeff*c[4][0][k][j];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4][k][j];
  lhs[1][4][k][j]= lhs[1][4][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][4][k][j]= lhs[2][4][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][4][k][j]= lhs[3][4][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][0][k][j];
  c[0][4][k][j] = c[0][4][k][j] - coeff*c[0][0][k][j];
  c[1][4][k][j] = c[1][4][k][j] - coeff*c[1][0][k][j];
  c[2][4][k][j] = c[2][4][k][j] - coeff*c[2][0][k][j];
  c[3][4][k][j] = c[3][4][k][j] - coeff*c[3][0][k][j];
  c[4][4][k][j] = c[4][4][k][j] - coeff*c[4][0][k][j];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1][k][j];
  lhs[2][1][k][j] = lhs[2][1][k][j]*pivot;
  lhs[3][1][k][j] = lhs[3][1][k][j]*pivot;
  lhs[4][1][k][j] = lhs[4][1][k][j]*pivot;
  c[0][1][k][j] = c[0][1][k][j]*pivot;
  c[1][1][k][j] = c[1][1][k][j]*pivot;
  c[2][1][k][j] = c[2][1][k][j]*pivot;
  c[3][1][k][j] = c[3][1][k][j]*pivot;
  c[4][1][k][j] = c[4][1][k][j]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0][k][j];
  lhs[2][0][k][j]= lhs[2][0][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][0][k][j]= lhs[3][0][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][0][k][j]= lhs[4][0][k][j] - coeff*lhs[4][1][k][j];
  c[0][0][k][j] = c[0][0][k][j] - coeff*c[0][1][k][j];
  c[1][0][k][j] = c[1][0][k][j] - coeff*c[1][1][k][j];
  c[2][0][k][j] = c[2][0][k][j] - coeff*c[2][1][k][j];
  c[3][0][k][j] = c[3][0][k][j] - coeff*c[3][1][k][j];
  c[4][0][k][j] = c[4][0][k][j] - coeff*c[4][1][k][j];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2][k][j];
  lhs[2][2][k][j]= lhs[2][2][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][2][k][j]= lhs[3][2][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][2][k][j]= lhs[4][2][k][j] - coeff*lhs[4][1][k][j];
  c[0][2][k][j] = c[0][2][k][j] - coeff*c[0][1][k][j];
  c[1][2][k][j] = c[1][2][k][j] - coeff*c[1][1][k][j];
  c[2][2][k][j] = c[2][2][k][j] - coeff*c[2][1][k][j];
  c[3][2][k][j] = c[3][2][k][j] - coeff*c[3][1][k][j];
  c[4][2][k][j] = c[4][2][k][j] - coeff*c[4][1][k][j];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3][k][j];
  lhs[2][3][k][j]= lhs[2][3][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][3][k][j]= lhs[3][3][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][3][k][j]= lhs[4][3][k][j] - coeff*lhs[4][1][k][j];
  c[0][3][k][j] = c[0][3][k][j] - coeff*c[0][1][k][j];
  c[1][3][k][j] = c[1][3][k][j] - coeff*c[1][1][k][j];
  c[2][3][k][j] = c[2][3][k][j] - coeff*c[2][1][k][j];
  c[3][3][k][j] = c[3][3][k][j] - coeff*c[3][1][k][j];
  c[4][3][k][j] = c[4][3][k][j] - coeff*c[4][1][k][j];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4][k][j];
  lhs[2][4][k][j]= lhs[2][4][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][4][k][j]= lhs[3][4][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][1][k][j];
  c[0][4][k][j] = c[0][4][k][j] - coeff*c[0][1][k][j];
  c[1][4][k][j] = c[1][4][k][j] - coeff*c[1][1][k][j];
  c[2][4][k][j] = c[2][4][k][j] - coeff*c[2][1][k][j];
  c[3][4][k][j] = c[3][4][k][j] - coeff*c[3][1][k][j];
  c[4][4][k][j] = c[4][4][k][j] - coeff*c[4][1][k][j];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2][k][j];
  lhs[3][2][k][j] = lhs[3][2][k][j]*pivot;
  lhs[4][2][k][j] = lhs[4][2][k][j]*pivot;
  c[0][2][k][j] = c[0][2][k][j]*pivot;
  c[1][2][k][j] = c[1][2][k][j]*pivot;
  c[2][2][k][j] = c[2][2][k][j]*pivot;
  c[3][2][k][j] = c[3][2][k][j]*pivot;
  c[4][2][k][j] = c[4][2][k][j]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0][k][j];
  lhs[3][0][k][j]= lhs[3][0][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][0][k][j]= lhs[4][0][k][j] - coeff*lhs[4][2][k][j];
  c[0][0][k][j] = c[0][0][k][j] - coeff*c[0][2][k][j];
  c[1][0][k][j] = c[1][0][k][j] - coeff*c[1][2][k][j];
  c[2][0][k][j] = c[2][0][k][j] - coeff*c[2][2][k][j];
  c[3][0][k][j] = c[3][0][k][j] - coeff*c[3][2][k][j];
  c[4][0][k][j] = c[4][0][k][j] - coeff*c[4][2][k][j];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1][k][j];
  lhs[3][1][k][j]= lhs[3][1][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][1][k][j]= lhs[4][1][k][j] - coeff*lhs[4][2][k][j];
  c[0][1][k][j] = c[0][1][k][j] - coeff*c[0][2][k][j];
  c[1][1][k][j] = c[1][1][k][j] - coeff*c[1][2][k][j];
  c[2][1][k][j] = c[2][1][k][j] - coeff*c[2][2][k][j];
  c[3][1][k][j] = c[3][1][k][j] - coeff*c[3][2][k][j];
  c[4][1][k][j] = c[4][1][k][j] - coeff*c[4][2][k][j];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3][k][j];
  lhs[3][3][k][j]= lhs[3][3][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][3][k][j]= lhs[4][3][k][j] - coeff*lhs[4][2][k][j];
  c[0][3][k][j] = c[0][3][k][j] - coeff*c[0][2][k][j];
  c[1][3][k][j] = c[1][3][k][j] - coeff*c[1][2][k][j];
  c[2][3][k][j] = c[2][3][k][j] - coeff*c[2][2][k][j];
  c[3][3][k][j] = c[3][3][k][j] - coeff*c[3][2][k][j];
  c[4][3][k][j] = c[4][3][k][j] - coeff*c[4][2][k][j];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4][k][j];
  lhs[3][4][k][j]= lhs[3][4][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][2][k][j];
  c[0][4][k][j] = c[0][4][k][j] - coeff*c[0][2][k][j];
  c[1][4][k][j] = c[1][4][k][j] - coeff*c[1][2][k][j];
  c[2][4][k][j] = c[2][4][k][j] - coeff*c[2][2][k][j];
  c[3][4][k][j] = c[3][4][k][j] - coeff*c[3][2][k][j];
  c[4][4][k][j] = c[4][4][k][j] - coeff*c[4][2][k][j];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3][k][j];
  lhs[4][3][k][j] = lhs[4][3][k][j]*pivot;
  c[0][3][k][j] = c[0][3][k][j]*pivot;
  c[1][3][k][j] = c[1][3][k][j]*pivot;
  c[2][3][k][j] = c[2][3][k][j]*pivot;
  c[3][3][k][j] = c[3][3][k][j]*pivot;
  c[4][3][k][j] = c[4][3][k][j]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0][k][j];
  lhs[4][0][k][j]= lhs[4][0][k][j] - coeff*lhs[4][3][k][j];
  c[0][0][k][j] = c[0][0][k][j] - coeff*c[0][3][k][j];
  c[1][0][k][j] = c[1][0][k][j] - coeff*c[1][3][k][j];
  c[2][0][k][j] = c[2][0][k][j] - coeff*c[2][3][k][j];
  c[3][0][k][j] = c[3][0][k][j] - coeff*c[3][3][k][j];
  c[4][0][k][j] = c[4][0][k][j] - coeff*c[4][3][k][j];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1][k][j];
  lhs[4][1][k][j]= lhs[4][1][k][j] - coeff*lhs[4][3][k][j];
  c[0][1][k][j] = c[0][1][k][j] - coeff*c[0][3][k][j];
  c[1][1][k][j] = c[1][1][k][j] - coeff*c[1][3][k][j];
  c[2][1][k][j] = c[2][1][k][j] - coeff*c[2][3][k][j];
  c[3][1][k][j] = c[3][1][k][j] - coeff*c[3][3][k][j];
  c[4][1][k][j] = c[4][1][k][j] - coeff*c[4][3][k][j];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2][k][j];
  lhs[4][2][k][j]= lhs[4][2][k][j] - coeff*lhs[4][3][k][j];
  c[0][2][k][j] = c[0][2][k][j] - coeff*c[0][3][k][j];
  c[1][2][k][j] = c[1][2][k][j] - coeff*c[1][3][k][j];
  c[2][2][k][j] = c[2][2][k][j] - coeff*c[2][3][k][j];
  c[3][2][k][j] = c[3][2][k][j] - coeff*c[3][3][k][j];
  c[4][2][k][j] = c[4][2][k][j] - coeff*c[4][3][k][j];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][3][k][j];
  c[0][4][k][j] = c[0][4][k][j] - coeff*c[0][3][k][j];
  c[1][4][k][j] = c[1][4][k][j] - coeff*c[1][3][k][j];
  c[2][4][k][j] = c[2][4][k][j] - coeff*c[2][3][k][j];
  c[3][4][k][j] = c[3][4][k][j] - coeff*c[3][3][k][j];
  c[4][4][k][j] = c[4][4][k][j] - coeff*c[4][3][k][j];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4][k][j];
  c[0][4][k][j] = c[0][4][k][j]*pivot;
  c[1][4][k][j] = c[1][4][k][j]*pivot;
  c[2][4][k][j] = c[2][4][k][j]*pivot;
  c[3][4][k][j] = c[3][4][k][j]*pivot;
  c[4][4][k][j] = c[4][4][k][j]*pivot;
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0][k][j];
  c[0][0][k][j] = c[0][0][k][j] - coeff*c[0][4][k][j];
  c[1][0][k][j] = c[1][0][k][j] - coeff*c[1][4][k][j];
  c[2][0][k][j] = c[2][0][k][j] - coeff*c[2][4][k][j];
  c[3][0][k][j] = c[3][0][k][j] - coeff*c[3][4][k][j];
  c[4][0][k][j] = c[4][0][k][j] - coeff*c[4][4][k][j];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1][k][j];
  c[0][1][k][j] = c[0][1][k][j] - coeff*c[0][4][k][j];
  c[1][1][k][j] = c[1][1][k][j] - coeff*c[1][4][k][j];
  c[2][1][k][j] = c[2][1][k][j] - coeff*c[2][4][k][j];
  c[3][1][k][j] = c[3][1][k][j] - coeff*c[3][4][k][j];
  c[4][1][k][j] = c[4][1][k][j] - coeff*c[4][4][k][j];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2][k][j];
  c[0][2][k][j] = c[0][2][k][j] - coeff*c[0][4][k][j];
  c[1][2][k][j] = c[1][2][k][j] - coeff*c[1][4][k][j];
  c[2][2][k][j] = c[2][2][k][j] - coeff*c[2][4][k][j];
  c[3][2][k][j] = c[3][2][k][j] - coeff*c[3][4][k][j];
  c[4][2][k][j] = c[4][2][k][j] - coeff*c[4][4][k][j];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3][k][j];
  c[0][3][k][j] = c[0][3][k][j] - coeff*c[0][4][k][j];
  c[1][3][k][j] = c[1][3][k][j] - coeff*c[1][4][k][j];
  c[2][3][k][j] = c[2][3][k][j] - coeff*c[2][4][k][j];
  c[3][3][k][j] = c[3][3][k][j] - coeff*c[3][4][k][j];
  c[4][3][k][j] = c[4][3][k][j] - coeff*c[4][4][k][j];
  r[3]   = r[3]   - coeff*r[4];
}

inline void matvec_sub_x(__global double ablock[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1],
                         __global double avec[5], 
                         __global double bvec[5],
                         int k, int j)
{
  //---------------------------------------------------------------------
  // rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] 
  // $                  - lhs[ia][ablock][0][i]*
  //---------------------------------------------------------------------
  bvec[0] = bvec[0] - ablock[0][0][k][j]*avec[0]
                    - ablock[1][0][k][j]*avec[1]
                    - ablock[2][0][k][j]*avec[2]
                    - ablock[3][0][k][j]*avec[3]
                    - ablock[4][0][k][j]*avec[4];
  bvec[1] = bvec[1] - ablock[0][1][k][j]*avec[0]
                    - ablock[1][1][k][j]*avec[1]
                    - ablock[2][1][k][j]*avec[2]
                    - ablock[3][1][k][j]*avec[3]
                    - ablock[4][1][k][j]*avec[4];
  bvec[2] = bvec[2] - ablock[0][2][k][j]*avec[0]
                    - ablock[1][2][k][j]*avec[1]
                    - ablock[2][2][k][j]*avec[2]
                    - ablock[3][2][k][j]*avec[3]
                    - ablock[4][2][k][j]*avec[4];
  bvec[3] = bvec[3] - ablock[0][3][k][j]*avec[0]
                    - ablock[1][3][k][j]*avec[1]
                    - ablock[2][3][k][j]*avec[2]
                    - ablock[3][3][k][j]*avec[3]
                    - ablock[4][3][k][j]*avec[4];
  bvec[4] = bvec[4] - ablock[0][4][k][j]*avec[0]
                    - ablock[1][4][k][j]*avec[1]
                    - ablock[2][4][k][j]*avec[2]
                    - ablock[3][4][k][j]*avec[3]
                    - ablock[4][4][k][j]*avec[4];
}

inline void matmul_sub_x(__global double ablock[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1],
                         __global double bblock[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1],
                         __global double cblock[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1],
                         int k, int j)
{
  cblock[0][0][k][j] = cblock[0][0][k][j] - ablock[0][0][k][j]*bblock[0][0][k][j]
                              - ablock[1][0][k][j]*bblock[0][1][k][j]
                              - ablock[2][0][k][j]*bblock[0][2][k][j]
                              - ablock[3][0][k][j]*bblock[0][3][k][j]
                              - ablock[4][0][k][j]*bblock[0][4][k][j];
  cblock[0][1][k][j] = cblock[0][1][k][j] - ablock[0][1][k][j]*bblock[0][0][k][j]
                              - ablock[1][1][k][j]*bblock[0][1][k][j]
                              - ablock[2][1][k][j]*bblock[0][2][k][j]
                              - ablock[3][1][k][j]*bblock[0][3][k][j]
                              - ablock[4][1][k][j]*bblock[0][4][k][j];
  cblock[0][2][k][j] = cblock[0][2][k][j] - ablock[0][2][k][j]*bblock[0][0][k][j]
                              - ablock[1][2][k][j]*bblock[0][1][k][j]
                              - ablock[2][2][k][j]*bblock[0][2][k][j]
                              - ablock[3][2][k][j]*bblock[0][3][k][j]
                              - ablock[4][2][k][j]*bblock[0][4][k][j];
  cblock[0][3][k][j] = cblock[0][3][k][j] - ablock[0][3][k][j]*bblock[0][0][k][j]
                              - ablock[1][3][k][j]*bblock[0][1][k][j]
                              - ablock[2][3][k][j]*bblock[0][2][k][j]
                              - ablock[3][3][k][j]*bblock[0][3][k][j]
                              - ablock[4][3][k][j]*bblock[0][4][k][j];
  cblock[0][4][k][j] = cblock[0][4][k][j] - ablock[0][4][k][j]*bblock[0][0][k][j]
                              - ablock[1][4][k][j]*bblock[0][1][k][j]
                              - ablock[2][4][k][j]*bblock[0][2][k][j]
                              - ablock[3][4][k][j]*bblock[0][3][k][j]
                              - ablock[4][4][k][j]*bblock[0][4][k][j];
  cblock[1][0][k][j] = cblock[1][0][k][j] - ablock[0][0][k][j]*bblock[1][0][k][j]
                              - ablock[1][0][k][j]*bblock[1][1][k][j]
                              - ablock[2][0][k][j]*bblock[1][2][k][j]
                              - ablock[3][0][k][j]*bblock[1][3][k][j]
                              - ablock[4][0][k][j]*bblock[1][4][k][j];
  cblock[1][1][k][j] = cblock[1][1][k][j] - ablock[0][1][k][j]*bblock[1][0][k][j]
                              - ablock[1][1][k][j]*bblock[1][1][k][j]
                              - ablock[2][1][k][j]*bblock[1][2][k][j]
                              - ablock[3][1][k][j]*bblock[1][3][k][j]
                              - ablock[4][1][k][j]*bblock[1][4][k][j];
  cblock[1][2][k][j] = cblock[1][2][k][j] - ablock[0][2][k][j]*bblock[1][0][k][j]
                              - ablock[1][2][k][j]*bblock[1][1][k][j]
                              - ablock[2][2][k][j]*bblock[1][2][k][j]
                              - ablock[3][2][k][j]*bblock[1][3][k][j]
                              - ablock[4][2][k][j]*bblock[1][4][k][j];
  cblock[1][3][k][j] = cblock[1][3][k][j] - ablock[0][3][k][j]*bblock[1][0][k][j]
                              - ablock[1][3][k][j]*bblock[1][1][k][j]
                              - ablock[2][3][k][j]*bblock[1][2][k][j]
                              - ablock[3][3][k][j]*bblock[1][3][k][j]
                              - ablock[4][3][k][j]*bblock[1][4][k][j];
  cblock[1][4][k][j] = cblock[1][4][k][j] - ablock[0][4][k][j]*bblock[1][0][k][j]
                              - ablock[1][4][k][j]*bblock[1][1][k][j]
                              - ablock[2][4][k][j]*bblock[1][2][k][j]
                              - ablock[3][4][k][j]*bblock[1][3][k][j]
                              - ablock[4][4][k][j]*bblock[1][4][k][j];
  cblock[2][0][k][j] = cblock[2][0][k][j] - ablock[0][0][k][j]*bblock[2][0][k][j]
                              - ablock[1][0][k][j]*bblock[2][1][k][j]
                              - ablock[2][0][k][j]*bblock[2][2][k][j]
                              - ablock[3][0][k][j]*bblock[2][3][k][j]
                              - ablock[4][0][k][j]*bblock[2][4][k][j];
  cblock[2][1][k][j] = cblock[2][1][k][j] - ablock[0][1][k][j]*bblock[2][0][k][j]
                              - ablock[1][1][k][j]*bblock[2][1][k][j]
                              - ablock[2][1][k][j]*bblock[2][2][k][j]
                              - ablock[3][1][k][j]*bblock[2][3][k][j]
                              - ablock[4][1][k][j]*bblock[2][4][k][j];
  cblock[2][2][k][j] = cblock[2][2][k][j] - ablock[0][2][k][j]*bblock[2][0][k][j]
                              - ablock[1][2][k][j]*bblock[2][1][k][j]
                              - ablock[2][2][k][j]*bblock[2][2][k][j]
                              - ablock[3][2][k][j]*bblock[2][3][k][j]
                              - ablock[4][2][k][j]*bblock[2][4][k][j];
  cblock[2][3][k][j] = cblock[2][3][k][j] - ablock[0][3][k][j]*bblock[2][0][k][j]
                              - ablock[1][3][k][j]*bblock[2][1][k][j]
                              - ablock[2][3][k][j]*bblock[2][2][k][j]
                              - ablock[3][3][k][j]*bblock[2][3][k][j]
                              - ablock[4][3][k][j]*bblock[2][4][k][j];
  cblock[2][4][k][j] = cblock[2][4][k][j] - ablock[0][4][k][j]*bblock[2][0][k][j]
                              - ablock[1][4][k][j]*bblock[2][1][k][j]
                              - ablock[2][4][k][j]*bblock[2][2][k][j]
                              - ablock[3][4][k][j]*bblock[2][3][k][j]
                              - ablock[4][4][k][j]*bblock[2][4][k][j];
  cblock[3][0][k][j] = cblock[3][0][k][j] - ablock[0][0][k][j]*bblock[3][0][k][j]
                              - ablock[1][0][k][j]*bblock[3][1][k][j]
                              - ablock[2][0][k][j]*bblock[3][2][k][j]
                              - ablock[3][0][k][j]*bblock[3][3][k][j]
                              - ablock[4][0][k][j]*bblock[3][4][k][j];
  cblock[3][1][k][j] = cblock[3][1][k][j] - ablock[0][1][k][j]*bblock[3][0][k][j]
                              - ablock[1][1][k][j]*bblock[3][1][k][j]
                              - ablock[2][1][k][j]*bblock[3][2][k][j]
                              - ablock[3][1][k][j]*bblock[3][3][k][j]
                              - ablock[4][1][k][j]*bblock[3][4][k][j];
  cblock[3][2][k][j] = cblock[3][2][k][j] - ablock[0][2][k][j]*bblock[3][0][k][j]
                              - ablock[1][2][k][j]*bblock[3][1][k][j]
                              - ablock[2][2][k][j]*bblock[3][2][k][j]
                              - ablock[3][2][k][j]*bblock[3][3][k][j]
                              - ablock[4][2][k][j]*bblock[3][4][k][j];
  cblock[3][3][k][j] = cblock[3][3][k][j] - ablock[0][3][k][j]*bblock[3][0][k][j]
                              - ablock[1][3][k][j]*bblock[3][1][k][j]
                              - ablock[2][3][k][j]*bblock[3][2][k][j]
                              - ablock[3][3][k][j]*bblock[3][3][k][j]
                              - ablock[4][3][k][j]*bblock[3][4][k][j];
  cblock[3][4][k][j] = cblock[3][4][k][j] - ablock[0][4][k][j]*bblock[3][0][k][j]
                              - ablock[1][4][k][j]*bblock[3][1][k][j]
                              - ablock[2][4][k][j]*bblock[3][2][k][j]
                              - ablock[3][4][k][j]*bblock[3][3][k][j]
                              - ablock[4][4][k][j]*bblock[3][4][k][j];
  cblock[4][0][k][j] = cblock[4][0][k][j] - ablock[0][0][k][j]*bblock[4][0][k][j]
                              - ablock[1][0][k][j]*bblock[4][1][k][j]
                              - ablock[2][0][k][j]*bblock[4][2][k][j]
                              - ablock[3][0][k][j]*bblock[4][3][k][j]
                              - ablock[4][0][k][j]*bblock[4][4][k][j];
  cblock[4][1][k][j] = cblock[4][1][k][j] - ablock[0][1][k][j]*bblock[4][0][k][j]
                              - ablock[1][1][k][j]*bblock[4][1][k][j]
                              - ablock[2][1][k][j]*bblock[4][2][k][j]
                              - ablock[3][1][k][j]*bblock[4][3][k][j]
                              - ablock[4][1][k][j]*bblock[4][4][k][j];
  cblock[4][2][k][j] = cblock[4][2][k][j] - ablock[0][2][k][j]*bblock[4][0][k][j]
                              - ablock[1][2][k][j]*bblock[4][1][k][j]
                              - ablock[2][2][k][j]*bblock[4][2][k][j]
                              - ablock[3][2][k][j]*bblock[4][3][k][j]
                              - ablock[4][2][k][j]*bblock[4][4][k][j];
  cblock[4][3][k][j] = cblock[4][3][k][j] - ablock[0][3][k][j]*bblock[4][0][k][j]
                              - ablock[1][3][k][j]*bblock[4][1][k][j]
                              - ablock[2][3][k][j]*bblock[4][2][k][j]
                              - ablock[3][3][k][j]*bblock[4][3][k][j]
                              - ablock[4][3][k][j]*bblock[4][4][k][j];
  cblock[4][4][k][j] = cblock[4][4][k][j] - ablock[0][4][k][j]*bblock[4][0][k][j]
                              - ablock[1][4][k][j]*bblock[4][1][k][j]
                              - ablock[2][4][k][j]*bblock[4][2][k][j]
                              - ablock[3][4][k][j]*bblock[4][3][k][j]
                              - ablock[4][4][k][j]*bblock[4][4][k][j];
}

inline void binvrhs_x(__global double lhs[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1],
                      __global double r[5],
                      int k, int j)
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0][k][j];
  lhs[1][0][k][j] = lhs[1][0][k][j]*pivot;
  lhs[2][0][k][j] = lhs[2][0][k][j]*pivot;
  lhs[3][0][k][j] = lhs[3][0][k][j]*pivot;
  lhs[4][0][k][j] = lhs[4][0][k][j]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1][k][j];
  lhs[1][1][k][j]= lhs[1][1][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][1][k][j]= lhs[2][1][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][1][k][j]= lhs[3][1][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][1][k][j]= lhs[4][1][k][j] - coeff*lhs[4][0][k][j];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2][k][j];
  lhs[1][2][k][j]= lhs[1][2][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][2][k][j]= lhs[2][2][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][2][k][j]= lhs[3][2][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][2][k][j]= lhs[4][2][k][j] - coeff*lhs[4][0][k][j];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3][k][j];
  lhs[1][3][k][j]= lhs[1][3][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][3][k][j]= lhs[2][3][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][3][k][j]= lhs[3][3][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][3][k][j]= lhs[4][3][k][j] - coeff*lhs[4][0][k][j];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4][k][j];
  lhs[1][4][k][j]= lhs[1][4][k][j] - coeff*lhs[1][0][k][j];
  lhs[2][4][k][j]= lhs[2][4][k][j] - coeff*lhs[2][0][k][j];
  lhs[3][4][k][j]= lhs[3][4][k][j] - coeff*lhs[3][0][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][0][k][j];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1][k][j];
  lhs[2][1][k][j] = lhs[2][1][k][j]*pivot;
  lhs[3][1][k][j] = lhs[3][1][k][j]*pivot;
  lhs[4][1][k][j] = lhs[4][1][k][j]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0][k][j];
  lhs[2][0][k][j]= lhs[2][0][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][0][k][j]= lhs[3][0][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][0][k][j]= lhs[4][0][k][j] - coeff*lhs[4][1][k][j];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2][k][j];
  lhs[2][2][k][j]= lhs[2][2][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][2][k][j]= lhs[3][2][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][2][k][j]= lhs[4][2][k][j] - coeff*lhs[4][1][k][j];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3][k][j];
  lhs[2][3][k][j]= lhs[2][3][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][3][k][j]= lhs[3][3][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][3][k][j]= lhs[4][3][k][j] - coeff*lhs[4][1][k][j];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4][k][j];
  lhs[2][4][k][j]= lhs[2][4][k][j] - coeff*lhs[2][1][k][j];
  lhs[3][4][k][j]= lhs[3][4][k][j] - coeff*lhs[3][1][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][1][k][j];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2][k][j];
  lhs[3][2][k][j] = lhs[3][2][k][j]*pivot;
  lhs[4][2][k][j] = lhs[4][2][k][j]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0][k][j];
  lhs[3][0][k][j]= lhs[3][0][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][0][k][j]= lhs[4][0][k][j] - coeff*lhs[4][2][k][j];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1][k][j];
  lhs[3][1][k][j]= lhs[3][1][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][1][k][j]= lhs[4][1][k][j] - coeff*lhs[4][2][k][j];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3][k][j];
  lhs[3][3][k][j]= lhs[3][3][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][3][k][j]= lhs[4][3][k][j] - coeff*lhs[4][2][k][j];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4][k][j];
  lhs[3][4][k][j]= lhs[3][4][k][j] - coeff*lhs[3][2][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][2][k][j];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3][k][j];
  lhs[4][3][k][j] = lhs[4][3][k][j]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0][k][j];
  lhs[4][0][k][j]= lhs[4][0][k][j] - coeff*lhs[4][3][k][j];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1][k][j];
  lhs[4][1][k][j]= lhs[4][1][k][j] - coeff*lhs[4][3][k][j];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2][k][j];
  lhs[4][2][k][j]= lhs[4][2][k][j] - coeff*lhs[4][3][k][j];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4][k][j];
  lhs[4][4][k][j]= lhs[4][4][k][j] - coeff*lhs[4][3][k][j];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4][k][j];
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0][k][j];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1][k][j];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2][k][j];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3][k][j];
  r[3]   = r[3]   - coeff*r[4];
}

__kernel void x_solve_memlayout(__global double *m_qs,
                                __global double *m_rho_i,
                                __global double *m_square,
                                __global double *m_u,
                                __global double *m_rhs,
                                __global double *m_lhs, 
                                __global double *m_fjac,
                                __global double *m_njac,
                                int gp0, int gp1, int gp2,
                                double dx1, double dx2, 
                                double dx3, double dx4, 
                                double dx5,
                                double c1, double c2,
                                double tx1, double tx2,
                                double con43, double c3c4,
                                double c1345, double dt,
                                int work_base, 
                                int work_num_item, 
                                int split_flag)
{
  int k = get_global_id(1);
  int j = get_global_id(0) + 1;
  int i, m, n;

  if (k + work_base < 1
      || k + work_base > gp2-2
      || k >= work_num_item
      || j > gp1-2)
    return;

  if (split_flag) k += 2;
  else k += work_base;

  __global double (* qs)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1])m_qs;
  __global double (* rho_i)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1])m_rho_i;
  __global double (* square)[JMAXP+1][IMAXP+1]
    = (__global double (*) [JMAXP+1][IMAXP+1]) m_square; 
  __global double (* u)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5])m_u;
  __global double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (__global double (*) [JMAXP+1][IMAXP+1][5])m_rhs;

  __global double (* lhs)[3][5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1]
    = (__global double (*)[3][5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1])m_lhs;


  __global double (* fjac)[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1]
    = (__global double (*)[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1])m_fjac;
  __global double (* njac)[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1]
    = (__global double (*)[5][5][WORK_NUM_ITEM_DEFAULT][PROBLEM_SIZE-1])m_njac;

  double tmp1, tmp2, tmp3;
  int isize = gp0 - 1;

  for (i = 0; i <= isize; i++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[i][0][0][k][j] = 0.0;
    fjac[i][1][0][k][j] = 1.0;
    fjac[i][2][0][k][j] = 0.0;
    fjac[i][3][0][k][j] = 0.0;
    fjac[i][4][0][k][j] = 0.0;

    fjac[i][0][1][k][j] = -(u[k][j][i][1] * tmp2 * u[k][j][i][1])
      + c2 * qs[k][j][i];
    fjac[i][1][1][k][j] = ( 2.0 - c2 ) * ( u[k][j][i][1] / u[k][j][i][0] );
    fjac[i][2][1][k][j] = - c2 * ( u[k][j][i][2] * tmp1 );
    fjac[i][3][1][k][j] = - c2 * ( u[k][j][i][3] * tmp1 );
    fjac[i][4][1][k][j] = c2;

    fjac[i][0][2][k][j] = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac[i][1][2][k][j] = u[k][j][i][2] * tmp1;
    fjac[i][2][2][k][j] = u[k][j][i][1] * tmp1;
    fjac[i][3][2][k][j] = 0.0;
    fjac[i][4][2][k][j] = 0.0;

    fjac[i][0][3][k][j] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[i][1][3][k][j] = u[k][j][i][3] * tmp1;
    fjac[i][2][3][k][j] = 0.0;
    fjac[i][3][3][k][j] = u[k][j][i][1] * tmp1;
    fjac[i][4][3][k][j] = 0.0;

    fjac[i][0][4][k][j] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * ( u[k][j][i][1] * tmp2 );
    fjac[i][1][4][k][j] = c1 *  u[k][j][i][4] * tmp1 
      - c2 * ( u[k][j][i][1]*u[k][j][i][1] * tmp2 + qs[k][j][i] );
    fjac[i][2][4][k][j] = - c2 * ( u[k][j][i][2]*u[k][j][i][1] ) * tmp2;
    fjac[i][3][4][k][j] = - c2 * ( u[k][j][i][3]*u[k][j][i][1] ) * tmp2;
    fjac[i][4][4][k][j] = c1 * ( u[k][j][i][1] * tmp1 );

    njac[i][0][0][k][j] = 0.0;
    njac[i][1][0][k][j] = 0.0;
    njac[i][2][0][k][j] = 0.0;
    njac[i][3][0][k][j] = 0.0;
    njac[i][4][0][k][j] = 0.0;

    njac[i][0][1][k][j] = - con43 * c3c4 * tmp2 * u[k][j][i][1];
    njac[i][1][1][k][j] =   con43 * c3c4 * tmp1;
    njac[i][2][1][k][j] =   0.0;
    njac[i][3][1][k][j] =   0.0;
    njac[i][4][1][k][j] =   0.0;

    njac[i][0][2][k][j] = - c3c4 * tmp2 * u[k][j][i][2];
    njac[i][1][2][k][j] =   0.0;
    njac[i][2][2][k][j] =   c3c4 * tmp1;
    njac[i][3][2][k][j] =   0.0;
    njac[i][4][2][k][j] =   0.0;

    njac[i][0][3][k][j] = - c3c4 * tmp2 * u[k][j][i][3];
    njac[i][1][3][k][j] =   0.0;
    njac[i][2][3][k][j] =   0.0;
    njac[i][3][3][k][j] =   c3c4 * tmp1;
    njac[i][4][3][k][j] =   0.0;

    njac[i][0][4][k][j] = - ( con43 * c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[i][1][4][k][j] = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][1];
    njac[i][2][4][k][j] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[i][3][4][k][j] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac[i][4][4][k][j] = ( c1345 ) * tmp1;
  }
  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in x direction
  //---------------------------------------------------------------------
  lhsinit_x(lhs, isize, k, j);
  for (i = 1; i <= isize-1; i++) {
    tmp1 = dt * tx1;
    tmp2 = dt * tx2;

    lhs[i][AA][0][0][k][j] = - tmp2 * fjac[i-1][0][0][k][j]
      - tmp1 * njac[i-1][0][0][k][j]
      - tmp1 * dx1; 
    lhs[i][AA][1][0][k][j] = - tmp2 * fjac[i-1][1][0][k][j]
      - tmp1 * njac[i-1][1][0][k][j];
    lhs[i][AA][2][0][k][j] = - tmp2 * fjac[i-1][2][0][k][j]
      - tmp1 * njac[i-1][2][0][k][j];
    lhs[i][AA][3][0][k][j] = - tmp2 * fjac[i-1][3][0][k][j]
      - tmp1 * njac[i-1][3][0][k][j];
    lhs[i][AA][4][0][k][j] = - tmp2 * fjac[i-1][4][0][k][j]
      - tmp1 * njac[i-1][4][0][k][j];

    lhs[i][AA][0][1][k][j] = - tmp2 * fjac[i-1][0][1][k][j]
      - tmp1 * njac[i-1][0][1][k][j];
    lhs[i][AA][1][1][k][j] = - tmp2 * fjac[i-1][1][1][k][j]
      - tmp1 * njac[i-1][1][1][k][j]
      - tmp1 * dx2;
    lhs[i][AA][2][1][k][j] = - tmp2 * fjac[i-1][2][1][k][j]
      - tmp1 * njac[i-1][2][1][k][j];
    lhs[i][AA][3][1][k][j] = - tmp2 * fjac[i-1][3][1][k][j]
      - tmp1 * njac[i-1][3][1][k][j];
    lhs[i][AA][4][1][k][j] = - tmp2 * fjac[i-1][4][1][k][j]
      - tmp1 * njac[i-1][4][1][k][j];

    lhs[i][AA][0][2][k][j] = - tmp2 * fjac[i-1][0][2][k][j]
      - tmp1 * njac[i-1][0][2][k][j];
    lhs[i][AA][1][2][k][j] = - tmp2 * fjac[i-1][1][2][k][j]
      - tmp1 * njac[i-1][1][2][k][j];
    lhs[i][AA][2][2][k][j] = - tmp2 * fjac[i-1][2][2][k][j]
      - tmp1 * njac[i-1][2][2][k][j]
      - tmp1 * dx3;
    lhs[i][AA][3][2][k][j] = - tmp2 * fjac[i-1][3][2][k][j]
      - tmp1 * njac[i-1][3][2][k][j];
    lhs[i][AA][4][2][k][j] = - tmp2 * fjac[i-1][4][2][k][j]
      - tmp1 * njac[i-1][4][2][k][j];

    lhs[i][AA][0][3][k][j] = - tmp2 * fjac[i-1][0][3][k][j]
      - tmp1 * njac[i-1][0][3][k][j];
    lhs[i][AA][1][3][k][j] = - tmp2 * fjac[i-1][1][3][k][j]
      - tmp1 * njac[i-1][1][3][k][j];
    lhs[i][AA][2][3][k][j] = - tmp2 * fjac[i-1][2][3][k][j]
      - tmp1 * njac[i-1][2][3][k][j];
    lhs[i][AA][3][3][k][j] = - tmp2 * fjac[i-1][3][3][k][j]
      - tmp1 * njac[i-1][3][3][k][j]
      - tmp1 * dx4;
    lhs[i][AA][4][3][k][j] = - tmp2 * fjac[i-1][4][3][k][j]
      - tmp1 * njac[i-1][4][3][k][j];

    lhs[i][AA][0][4][k][j] = - tmp2 * fjac[i-1][0][4][k][j]
      - tmp1 * njac[i-1][0][4][k][j];
    lhs[i][AA][1][4][k][j] = - tmp2 * fjac[i-1][1][4][k][j]
      - tmp1 * njac[i-1][1][4][k][j];
    lhs[i][AA][2][4][k][j] = - tmp2 * fjac[i-1][2][4][k][j]
      - tmp1 * njac[i-1][2][4][k][j];
    lhs[i][AA][3][4][k][j] = - tmp2 * fjac[i-1][3][4][k][j]
      - tmp1 * njac[i-1][3][4][k][j];
    lhs[i][AA][4][4][k][j] = - tmp2 * fjac[i-1][4][4][k][j]
      - tmp1 * njac[i-1][4][4][k][j]
      - tmp1 * dx5;

    lhs[i][BB][0][0][k][j] = 1.0
      + tmp1 * 2.0 * njac[i][0][0][k][j]
      + tmp1 * 2.0 * dx1;
    lhs[i][BB][1][0][k][j] = tmp1 * 2.0 * njac[i][1][0][k][j];
    lhs[i][BB][2][0][k][j] = tmp1 * 2.0 * njac[i][2][0][k][j];
    lhs[i][BB][3][0][k][j] = tmp1 * 2.0 * njac[i][3][0][k][j];
    lhs[i][BB][4][0][k][j] = tmp1 * 2.0 * njac[i][4][0][k][j];

    lhs[i][BB][0][1][k][j] = tmp1 * 2.0 * njac[i][0][1][k][j];
    lhs[i][BB][1][1][k][j] = 1.0
      + tmp1 * 2.0 * njac[i][1][1][k][j]
      + tmp1 * 2.0 * dx2;
    lhs[i][BB][2][1][k][j] = tmp1 * 2.0 * njac[i][2][1][k][j];
    lhs[i][BB][3][1][k][j] = tmp1 * 2.0 * njac[i][3][1][k][j];
    lhs[i][BB][4][1][k][j] = tmp1 * 2.0 * njac[i][4][1][k][j];

    lhs[i][BB][0][2][k][j] = tmp1 * 2.0 * njac[i][0][2][k][j];
    lhs[i][BB][1][2][k][j] = tmp1 * 2.0 * njac[i][1][2][k][j];
    lhs[i][BB][2][2][k][j] = 1.0
      + tmp1 * 2.0 * njac[i][2][2][k][j]
      + tmp1 * 2.0 * dx3;
    lhs[i][BB][3][2][k][j] = tmp1 * 2.0 * njac[i][3][2][k][j];
    lhs[i][BB][4][2][k][j] = tmp1 * 2.0 * njac[i][4][2][k][j];

    lhs[i][BB][0][3][k][j] = tmp1 * 2.0 * njac[i][0][3][k][j];
    lhs[i][BB][1][3][k][j] = tmp1 * 2.0 * njac[i][1][3][k][j];
    lhs[i][BB][2][3][k][j] = tmp1 * 2.0 * njac[i][2][3][k][j];
    lhs[i][BB][3][3][k][j] = 1.0
      + tmp1 * 2.0 * njac[i][3][3][k][j]
      + tmp1 * 2.0 * dx4;
    lhs[i][BB][4][3][k][j] = tmp1 * 2.0 * njac[i][4][3][k][j];

    lhs[i][BB][0][4][k][j] = tmp1 * 2.0 * njac[i][0][4][k][j];
    lhs[i][BB][1][4][k][j] = tmp1 * 2.0 * njac[i][1][4][k][j];
    lhs[i][BB][2][4][k][j] = tmp1 * 2.0 * njac[i][2][4][k][j];
    lhs[i][BB][3][4][k][j] = tmp1 * 2.0 * njac[i][3][4][k][j];
    lhs[i][BB][4][4][k][j] = 1.0
      + tmp1 * 2.0 * njac[i][4][4][k][j]
      + tmp1 * 2.0 * dx5;

    lhs[i][CC][0][0][k][j] =  tmp2 * fjac[i+1][0][0][k][j]
      - tmp1 * njac[i+1][0][0][k][j]
      - tmp1 * dx1;
    lhs[i][CC][1][0][k][j] =  tmp2 * fjac[i+1][1][0][k][j]
      - tmp1 * njac[i+1][1][0][k][j];
    lhs[i][CC][2][0][k][j] =  tmp2 * fjac[i+1][2][0][k][j]
      - tmp1 * njac[i+1][2][0][k][j];
    lhs[i][CC][3][0][k][j] =  tmp2 * fjac[i+1][3][0][k][j]
      - tmp1 * njac[i+1][3][0][k][j];
    lhs[i][CC][4][0][k][j] =  tmp2 * fjac[i+1][4][0][k][j]
      - tmp1 * njac[i+1][4][0][k][j];

    lhs[i][CC][0][1][k][j] =  tmp2 * fjac[i+1][0][1][k][j]
      - tmp1 * njac[i+1][0][1][k][j];
    lhs[i][CC][1][1][k][j] =  tmp2 * fjac[i+1][1][1][k][j]
      - tmp1 * njac[i+1][1][1][k][j]
      - tmp1 * dx2;
    lhs[i][CC][2][1][k][j] =  tmp2 * fjac[i+1][2][1][k][j]
      - tmp1 * njac[i+1][2][1][k][j];
    lhs[i][CC][3][1][k][j] =  tmp2 * fjac[i+1][3][1][k][j]
      - tmp1 * njac[i+1][3][1][k][j];
    lhs[i][CC][4][1][k][j] =  tmp2 * fjac[i+1][4][1][k][j]
      - tmp1 * njac[i+1][4][1][k][j];

    lhs[i][CC][0][2][k][j] =  tmp2 * fjac[i+1][0][2][k][j]
      - tmp1 * njac[i+1][0][2][k][j];
    lhs[i][CC][1][2][k][j] =  tmp2 * fjac[i+1][1][2][k][j]
      - tmp1 * njac[i+1][1][2][k][j];
    lhs[i][CC][2][2][k][j] =  tmp2 * fjac[i+1][2][2][k][j]
      - tmp1 * njac[i+1][2][2][k][j]
      - tmp1 * dx3;
    lhs[i][CC][3][2][k][j] =  tmp2 * fjac[i+1][3][2][k][j]
      - tmp1 * njac[i+1][3][2][k][j];
    lhs[i][CC][4][2][k][j] =  tmp2 * fjac[i+1][4][2][k][j]
      - tmp1 * njac[i+1][4][2][k][j];

    lhs[i][CC][0][3][k][j] =  tmp2 * fjac[i+1][0][3][k][j]
      - tmp1 * njac[i+1][0][3][k][j];
    lhs[i][CC][1][3][k][j] =  tmp2 * fjac[i+1][1][3][k][j]
      - tmp1 * njac[i+1][1][3][k][j];
    lhs[i][CC][2][3][k][j] =  tmp2 * fjac[i+1][2][3][k][j]
      - tmp1 * njac[i+1][2][3][k][j];
    lhs[i][CC][3][3][k][j] =  tmp2 * fjac[i+1][3][3][k][j]
      - tmp1 * njac[i+1][3][3][k][j]
      - tmp1 * dx4;
    lhs[i][CC][4][3][k][j] =  tmp2 * fjac[i+1][4][3][k][j]
      - tmp1 * njac[i+1][4][3][k][j];

    lhs[i][CC][0][4][k][j] =  tmp2 * fjac[i+1][0][4][k][j]
      - tmp1 * njac[i+1][0][4][k][j];
    lhs[i][CC][1][4][k][j] =  tmp2 * fjac[i+1][1][4][k][j]
      - tmp1 * njac[i+1][1][4][k][j];
    lhs[i][CC][2][4][k][j] =  tmp2 * fjac[i+1][2][4][k][j]
      - tmp1 * njac[i+1][2][4][k][j];
    lhs[i][CC][3][4][k][j] =  tmp2 * fjac[i+1][3][4][k][j]
      - tmp1 * njac[i+1][3][4][k][j];
    lhs[i][CC][4][4][k][j] =  tmp2 * fjac[i+1][4][4][k][j]
      - tmp1 * njac[i+1][4][4][k][j]
      - tmp1 * dx5;
  }


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

  //---------------------------------------------------------------------
  // multiply c[k][j][0] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  binvcrhs_x( lhs[0][BB], lhs[0][CC], rhs[k][j][0], k, j);

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (i = 1; i <= isize-1; i++) {
    //-------------------------------------------------------------------
    // rhs(i) = rhs(i) - A*rhs(i-1)
    //-------------------------------------------------------------------
    matvec_sub_x(lhs[i][AA], rhs[k][j][i-1], rhs[k][j][i], k, j);

    //-------------------------------------------------------------------
    // B(i) = B(i) - C(i-1)*A(i)
    //-------------------------------------------------------------------
    matmul_sub_x(lhs[i][AA], lhs[i-1][CC], lhs[i][BB], k, j);


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs_x( lhs[i][BB], lhs[i][CC], rhs[k][j][i], k, j);
  }

  //---------------------------------------------------------------------
  // rhs(isize) = rhs(isize) - A*rhs(isize-1)
  //---------------------------------------------------------------------
  matvec_sub_x(lhs[isize][AA], rhs[k][j][isize-1], rhs[k][j][isize], k, j);

  //---------------------------------------------------------------------
  // B(isize) = B(isize) - C(isize-1)*A(isize)
  //---------------------------------------------------------------------
  matmul_sub_x(lhs[isize][AA], lhs[isize-1][CC], lhs[isize][BB], k, j);

  //---------------------------------------------------------------------
  // multiply rhs() by b_inverse() and copy to rhs
  //---------------------------------------------------------------------
  binvrhs_x( lhs[isize][BB], rhs[k][j][isize] , k, j);

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(isize)=rhs(isize)
  // else assume U(isize) is loaded in un pack backsub_info
  // so just use it
  // after u(istart) will be sent to next cell
  //---------------------------------------------------------------------
  for (i = isize-1; i >=0; i--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhs[i][CC][n][m][k][j]*rhs[k][j][i+1][n];
      }
    }
  }
}
