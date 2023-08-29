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

inline void lhsinit_z(__global double lhs[][3][5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1], 
                      int size,
                      int j, int i)
{
  int k, m, n;

  k = size;
  //---------------------------------------------------------------------
  // zero the whole left hand side for starters
  //---------------------------------------------------------------------
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[0][0][n][m][j][i] = 0.0;
      lhs[0][1][n][m][j][i] = 0.0;
      lhs[0][2][n][m][j][i] = 0.0;
      lhs[k][0][n][m][j][i] = 0.0;
      lhs[k][1][n][m][j][i] = 0.0;
      lhs[k][2][n][m][j][i] = 0.0;
    }
  }

  //---------------------------------------------------------------------
  // next, set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs[0][1][m][m][j][i] = 1.0;
    lhs[k][1][m][m][j][i] = 1.0;
  }
}

inline void binvcrhs_z(__global double lhs[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1], 
                       __global double c[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1], 
                       __global double r[5],
                       int j, int i)
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0][j][i];
  lhs[1][0][j][i] = lhs[1][0][j][i]*pivot;
  lhs[2][0][j][i] = lhs[2][0][j][i]*pivot;
  lhs[3][0][j][i] = lhs[3][0][j][i]*pivot;
  lhs[4][0][j][i] = lhs[4][0][j][i]*pivot;
  c[0][0][j][i] = c[0][0][j][i]*pivot;
  c[1][0][j][i] = c[1][0][j][i]*pivot;
  c[2][0][j][i] = c[2][0][j][i]*pivot;
  c[3][0][j][i] = c[3][0][j][i]*pivot;
  c[4][0][j][i] = c[4][0][j][i]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1][j][i];
  lhs[1][1][j][i]= lhs[1][1][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][1][j][i]= lhs[2][1][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][1][j][i]= lhs[3][1][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][1][j][i]= lhs[4][1][j][i] - coeff*lhs[4][0][j][i];
  c[0][1][j][i] = c[0][1][j][i] - coeff*c[0][0][j][i];
  c[1][1][j][i] = c[1][1][j][i] - coeff*c[1][0][j][i];
  c[2][1][j][i] = c[2][1][j][i] - coeff*c[2][0][j][i];
  c[3][1][j][i] = c[3][1][j][i] - coeff*c[3][0][j][i];
  c[4][1][j][i] = c[4][1][j][i] - coeff*c[4][0][j][i];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2][j][i];
  lhs[1][2][j][i]= lhs[1][2][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][2][j][i]= lhs[2][2][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][2][j][i]= lhs[3][2][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][2][j][i]= lhs[4][2][j][i] - coeff*lhs[4][0][j][i];
  c[0][2][j][i] = c[0][2][j][i] - coeff*c[0][0][j][i];
  c[1][2][j][i] = c[1][2][j][i] - coeff*c[1][0][j][i];
  c[2][2][j][i] = c[2][2][j][i] - coeff*c[2][0][j][i];
  c[3][2][j][i] = c[3][2][j][i] - coeff*c[3][0][j][i];
  c[4][2][j][i] = c[4][2][j][i] - coeff*c[4][0][j][i];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3][j][i];
  lhs[1][3][j][i]= lhs[1][3][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][3][j][i]= lhs[2][3][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][3][j][i]= lhs[3][3][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][3][j][i]= lhs[4][3][j][i] - coeff*lhs[4][0][j][i];
  c[0][3][j][i] = c[0][3][j][i] - coeff*c[0][0][j][i];
  c[1][3][j][i] = c[1][3][j][i] - coeff*c[1][0][j][i];
  c[2][3][j][i] = c[2][3][j][i] - coeff*c[2][0][j][i];
  c[3][3][j][i] = c[3][3][j][i] - coeff*c[3][0][j][i];
  c[4][3][j][i] = c[4][3][j][i] - coeff*c[4][0][j][i];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4][j][i];
  lhs[1][4][j][i]= lhs[1][4][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][4][j][i]= lhs[2][4][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][4][j][i]= lhs[3][4][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][0][j][i];
  c[0][4][j][i] = c[0][4][j][i] - coeff*c[0][0][j][i];
  c[1][4][j][i] = c[1][4][j][i] - coeff*c[1][0][j][i];
  c[2][4][j][i] = c[2][4][j][i] - coeff*c[2][0][j][i];
  c[3][4][j][i] = c[3][4][j][i] - coeff*c[3][0][j][i];
  c[4][4][j][i] = c[4][4][j][i] - coeff*c[4][0][j][i];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1][j][i];
  lhs[2][1][j][i] = lhs[2][1][j][i]*pivot;
  lhs[3][1][j][i] = lhs[3][1][j][i]*pivot;
  lhs[4][1][j][i] = lhs[4][1][j][i]*pivot;
  c[0][1][j][i] = c[0][1][j][i]*pivot;
  c[1][1][j][i] = c[1][1][j][i]*pivot;
  c[2][1][j][i] = c[2][1][j][i]*pivot;
  c[3][1][j][i] = c[3][1][j][i]*pivot;
  c[4][1][j][i] = c[4][1][j][i]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0][j][i];
  lhs[2][0][j][i]= lhs[2][0][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][0][j][i]= lhs[3][0][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][0][j][i]= lhs[4][0][j][i] - coeff*lhs[4][1][j][i];
  c[0][0][j][i] = c[0][0][j][i] - coeff*c[0][1][j][i];
  c[1][0][j][i] = c[1][0][j][i] - coeff*c[1][1][j][i];
  c[2][0][j][i] = c[2][0][j][i] - coeff*c[2][1][j][i];
  c[3][0][j][i] = c[3][0][j][i] - coeff*c[3][1][j][i];
  c[4][0][j][i] = c[4][0][j][i] - coeff*c[4][1][j][i];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2][j][i];
  lhs[2][2][j][i]= lhs[2][2][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][2][j][i]= lhs[3][2][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][2][j][i]= lhs[4][2][j][i] - coeff*lhs[4][1][j][i];
  c[0][2][j][i] = c[0][2][j][i] - coeff*c[0][1][j][i];
  c[1][2][j][i] = c[1][2][j][i] - coeff*c[1][1][j][i];
  c[2][2][j][i] = c[2][2][j][i] - coeff*c[2][1][j][i];
  c[3][2][j][i] = c[3][2][j][i] - coeff*c[3][1][j][i];
  c[4][2][j][i] = c[4][2][j][i] - coeff*c[4][1][j][i];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3][j][i];
  lhs[2][3][j][i]= lhs[2][3][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][3][j][i]= lhs[3][3][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][3][j][i]= lhs[4][3][j][i] - coeff*lhs[4][1][j][i];
  c[0][3][j][i] = c[0][3][j][i] - coeff*c[0][1][j][i];
  c[1][3][j][i] = c[1][3][j][i] - coeff*c[1][1][j][i];
  c[2][3][j][i] = c[2][3][j][i] - coeff*c[2][1][j][i];
  c[3][3][j][i] = c[3][3][j][i] - coeff*c[3][1][j][i];
  c[4][3][j][i] = c[4][3][j][i] - coeff*c[4][1][j][i];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4][j][i];
  lhs[2][4][j][i]= lhs[2][4][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][4][j][i]= lhs[3][4][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][1][j][i];
  c[0][4][j][i] = c[0][4][j][i] - coeff*c[0][1][j][i];
  c[1][4][j][i] = c[1][4][j][i] - coeff*c[1][1][j][i];
  c[2][4][j][i] = c[2][4][j][i] - coeff*c[2][1][j][i];
  c[3][4][j][i] = c[3][4][j][i] - coeff*c[3][1][j][i];
  c[4][4][j][i] = c[4][4][j][i] - coeff*c[4][1][j][i];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2][j][i];
  lhs[3][2][j][i] = lhs[3][2][j][i]*pivot;
  lhs[4][2][j][i] = lhs[4][2][j][i]*pivot;
  c[0][2][j][i] = c[0][2][j][i]*pivot;
  c[1][2][j][i] = c[1][2][j][i]*pivot;
  c[2][2][j][i] = c[2][2][j][i]*pivot;
  c[3][2][j][i] = c[3][2][j][i]*pivot;
  c[4][2][j][i] = c[4][2][j][i]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0][j][i];
  lhs[3][0][j][i]= lhs[3][0][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][0][j][i]= lhs[4][0][j][i] - coeff*lhs[4][2][j][i];
  c[0][0][j][i] = c[0][0][j][i] - coeff*c[0][2][j][i];
  c[1][0][j][i] = c[1][0][j][i] - coeff*c[1][2][j][i];
  c[2][0][j][i] = c[2][0][j][i] - coeff*c[2][2][j][i];
  c[3][0][j][i] = c[3][0][j][i] - coeff*c[3][2][j][i];
  c[4][0][j][i] = c[4][0][j][i] - coeff*c[4][2][j][i];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1][j][i];
  lhs[3][1][j][i]= lhs[3][1][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][1][j][i]= lhs[4][1][j][i] - coeff*lhs[4][2][j][i];
  c[0][1][j][i] = c[0][1][j][i] - coeff*c[0][2][j][i];
  c[1][1][j][i] = c[1][1][j][i] - coeff*c[1][2][j][i];
  c[2][1][j][i] = c[2][1][j][i] - coeff*c[2][2][j][i];
  c[3][1][j][i] = c[3][1][j][i] - coeff*c[3][2][j][i];
  c[4][1][j][i] = c[4][1][j][i] - coeff*c[4][2][j][i];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3][j][i];
  lhs[3][3][j][i]= lhs[3][3][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][3][j][i]= lhs[4][3][j][i] - coeff*lhs[4][2][j][i];
  c[0][3][j][i] = c[0][3][j][i] - coeff*c[0][2][j][i];
  c[1][3][j][i] = c[1][3][j][i] - coeff*c[1][2][j][i];
  c[2][3][j][i] = c[2][3][j][i] - coeff*c[2][2][j][i];
  c[3][3][j][i] = c[3][3][j][i] - coeff*c[3][2][j][i];
  c[4][3][j][i] = c[4][3][j][i] - coeff*c[4][2][j][i];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4][j][i];
  lhs[3][4][j][i]= lhs[3][4][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][2][j][i];
  c[0][4][j][i] = c[0][4][j][i] - coeff*c[0][2][j][i];
  c[1][4][j][i] = c[1][4][j][i] - coeff*c[1][2][j][i];
  c[2][4][j][i] = c[2][4][j][i] - coeff*c[2][2][j][i];
  c[3][4][j][i] = c[3][4][j][i] - coeff*c[3][2][j][i];
  c[4][4][j][i] = c[4][4][j][i] - coeff*c[4][2][j][i];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3][j][i];
  lhs[4][3][j][i] = lhs[4][3][j][i]*pivot;
  c[0][3][j][i] = c[0][3][j][i]*pivot;
  c[1][3][j][i] = c[1][3][j][i]*pivot;
  c[2][3][j][i] = c[2][3][j][i]*pivot;
  c[3][3][j][i] = c[3][3][j][i]*pivot;
  c[4][3][j][i] = c[4][3][j][i]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0][j][i];
  lhs[4][0][j][i]= lhs[4][0][j][i] - coeff*lhs[4][3][j][i];
  c[0][0][j][i] = c[0][0][j][i] - coeff*c[0][3][j][i];
  c[1][0][j][i] = c[1][0][j][i] - coeff*c[1][3][j][i];
  c[2][0][j][i] = c[2][0][j][i] - coeff*c[2][3][j][i];
  c[3][0][j][i] = c[3][0][j][i] - coeff*c[3][3][j][i];
  c[4][0][j][i] = c[4][0][j][i] - coeff*c[4][3][j][i];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1][j][i];
  lhs[4][1][j][i]= lhs[4][1][j][i] - coeff*lhs[4][3][j][i];
  c[0][1][j][i] = c[0][1][j][i] - coeff*c[0][3][j][i];
  c[1][1][j][i] = c[1][1][j][i] - coeff*c[1][3][j][i];
  c[2][1][j][i] = c[2][1][j][i] - coeff*c[2][3][j][i];
  c[3][1][j][i] = c[3][1][j][i] - coeff*c[3][3][j][i];
  c[4][1][j][i] = c[4][1][j][i] - coeff*c[4][3][j][i];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2][j][i];
  lhs[4][2][j][i]= lhs[4][2][j][i] - coeff*lhs[4][3][j][i];
  c[0][2][j][i] = c[0][2][j][i] - coeff*c[0][3][j][i];
  c[1][2][j][i] = c[1][2][j][i] - coeff*c[1][3][j][i];
  c[2][2][j][i] = c[2][2][j][i] - coeff*c[2][3][j][i];
  c[3][2][j][i] = c[3][2][j][i] - coeff*c[3][3][j][i];
  c[4][2][j][i] = c[4][2][j][i] - coeff*c[4][3][j][i];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][3][j][i];
  c[0][4][j][i] = c[0][4][j][i] - coeff*c[0][3][j][i];
  c[1][4][j][i] = c[1][4][j][i] - coeff*c[1][3][j][i];
  c[2][4][j][i] = c[2][4][j][i] - coeff*c[2][3][j][i];
  c[3][4][j][i] = c[3][4][j][i] - coeff*c[3][3][j][i];
  c[4][4][j][i] = c[4][4][j][i] - coeff*c[4][3][j][i];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4][j][i];
  c[0][4][j][i] = c[0][4][j][i]*pivot;
  c[1][4][j][i] = c[1][4][j][i]*pivot;
  c[2][4][j][i] = c[2][4][j][i]*pivot;
  c[3][4][j][i] = c[3][4][j][i]*pivot;
  c[4][4][j][i] = c[4][4][j][i]*pivot;
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0][j][i];
  c[0][0][j][i] = c[0][0][j][i] - coeff*c[0][4][j][i];
  c[1][0][j][i] = c[1][0][j][i] - coeff*c[1][4][j][i];
  c[2][0][j][i] = c[2][0][j][i] - coeff*c[2][4][j][i];
  c[3][0][j][i] = c[3][0][j][i] - coeff*c[3][4][j][i];
  c[4][0][j][i] = c[4][0][j][i] - coeff*c[4][4][j][i];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1][j][i];
  c[0][1][j][i] = c[0][1][j][i] - coeff*c[0][4][j][i];
  c[1][1][j][i] = c[1][1][j][i] - coeff*c[1][4][j][i];
  c[2][1][j][i] = c[2][1][j][i] - coeff*c[2][4][j][i];
  c[3][1][j][i] = c[3][1][j][i] - coeff*c[3][4][j][i];
  c[4][1][j][i] = c[4][1][j][i] - coeff*c[4][4][j][i];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2][j][i];
  c[0][2][j][i] = c[0][2][j][i] - coeff*c[0][4][j][i];
  c[1][2][j][i] = c[1][2][j][i] - coeff*c[1][4][j][i];
  c[2][2][j][i] = c[2][2][j][i] - coeff*c[2][4][j][i];
  c[3][2][j][i] = c[3][2][j][i] - coeff*c[3][4][j][i];
  c[4][2][j][i] = c[4][2][j][i] - coeff*c[4][4][j][i];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3][j][i];
  c[0][3][j][i] = c[0][3][j][i] - coeff*c[0][4][j][i];
  c[1][3][j][i] = c[1][3][j][i] - coeff*c[1][4][j][i];
  c[2][3][j][i] = c[2][3][j][i] - coeff*c[2][4][j][i];
  c[3][3][j][i] = c[3][3][j][i] - coeff*c[3][4][j][i];
  c[4][3][j][i] = c[4][3][j][i] - coeff*c[4][4][j][i];
  r[3]   = r[3]   - coeff*r[4];
}

inline void matvec_sub_z(__global double ablock[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1], 
                         __global double avec[5], 
                         __global double bvec[5],
                         int j, int i)
{
  //---------------------------------------------------------------------
  // rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] 
  // $                  - lhs[ia][ablock][0][i]*
  //---------------------------------------------------------------------
  bvec[0] = bvec[0] - ablock[0][0][j][i]*avec[0]
                    - ablock[1][0][j][i]*avec[1]
                    - ablock[2][0][j][i]*avec[2]
                    - ablock[3][0][j][i]*avec[3]
                    - ablock[4][0][j][i]*avec[4];
  bvec[1] = bvec[1] - ablock[0][1][j][i]*avec[0]
                    - ablock[1][1][j][i]*avec[1]
                    - ablock[2][1][j][i]*avec[2]
                    - ablock[3][1][j][i]*avec[3]
                    - ablock[4][1][j][i]*avec[4];
  bvec[2] = bvec[2] - ablock[0][2][j][i]*avec[0]
                    - ablock[1][2][j][i]*avec[1]
                    - ablock[2][2][j][i]*avec[2]
                    - ablock[3][2][j][i]*avec[3]
                    - ablock[4][2][j][i]*avec[4];
  bvec[3] = bvec[3] - ablock[0][3][j][i]*avec[0]
                    - ablock[1][3][j][i]*avec[1]
                    - ablock[2][3][j][i]*avec[2]
                    - ablock[3][3][j][i]*avec[3]
                    - ablock[4][3][j][i]*avec[4];
  bvec[4] = bvec[4] - ablock[0][4][j][i]*avec[0]
                    - ablock[1][4][j][i]*avec[1]
                    - ablock[2][4][j][i]*avec[2]
                    - ablock[3][4][j][i]*avec[3]
                    - ablock[4][4][j][i]*avec[4];
}

inline void matmul_sub_z(__global double ablock[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1], 
                         __global double bblock[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1], 
                         __global double cblock[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1],
                         int j, int i)
{
  cblock[0][0][j][i] = cblock[0][0][j][i] - ablock[0][0][j][i]*bblock[0][0][j][i]
                              - ablock[1][0][j][i]*bblock[0][1][j][i]
                              - ablock[2][0][j][i]*bblock[0][2][j][i]
                              - ablock[3][0][j][i]*bblock[0][3][j][i]
                              - ablock[4][0][j][i]*bblock[0][4][j][i];
  cblock[0][1][j][i] = cblock[0][1][j][i] - ablock[0][1][j][i]*bblock[0][0][j][i]
                              - ablock[1][1][j][i]*bblock[0][1][j][i]
                              - ablock[2][1][j][i]*bblock[0][2][j][i]
                              - ablock[3][1][j][i]*bblock[0][3][j][i]
                              - ablock[4][1][j][i]*bblock[0][4][j][i];
  cblock[0][2][j][i] = cblock[0][2][j][i] - ablock[0][2][j][i]*bblock[0][0][j][i]
                              - ablock[1][2][j][i]*bblock[0][1][j][i]
                              - ablock[2][2][j][i]*bblock[0][2][j][i]
                              - ablock[3][2][j][i]*bblock[0][3][j][i]
                              - ablock[4][2][j][i]*bblock[0][4][j][i];
  cblock[0][3][j][i] = cblock[0][3][j][i] - ablock[0][3][j][i]*bblock[0][0][j][i]
                              - ablock[1][3][j][i]*bblock[0][1][j][i]
                              - ablock[2][3][j][i]*bblock[0][2][j][i]
                              - ablock[3][3][j][i]*bblock[0][3][j][i]
                              - ablock[4][3][j][i]*bblock[0][4][j][i];
  cblock[0][4][j][i] = cblock[0][4][j][i] - ablock[0][4][j][i]*bblock[0][0][j][i]
                              - ablock[1][4][j][i]*bblock[0][1][j][i]
                              - ablock[2][4][j][i]*bblock[0][2][j][i]
                              - ablock[3][4][j][i]*bblock[0][3][j][i]
                              - ablock[4][4][j][i]*bblock[0][4][j][i];
  cblock[1][0][j][i] = cblock[1][0][j][i] - ablock[0][0][j][i]*bblock[1][0][j][i]
                              - ablock[1][0][j][i]*bblock[1][1][j][i]
                              - ablock[2][0][j][i]*bblock[1][2][j][i]
                              - ablock[3][0][j][i]*bblock[1][3][j][i]
                              - ablock[4][0][j][i]*bblock[1][4][j][i];
  cblock[1][1][j][i] = cblock[1][1][j][i] - ablock[0][1][j][i]*bblock[1][0][j][i]
                              - ablock[1][1][j][i]*bblock[1][1][j][i]
                              - ablock[2][1][j][i]*bblock[1][2][j][i]
                              - ablock[3][1][j][i]*bblock[1][3][j][i]
                              - ablock[4][1][j][i]*bblock[1][4][j][i];
  cblock[1][2][j][i] = cblock[1][2][j][i] - ablock[0][2][j][i]*bblock[1][0][j][i]
                              - ablock[1][2][j][i]*bblock[1][1][j][i]
                              - ablock[2][2][j][i]*bblock[1][2][j][i]
                              - ablock[3][2][j][i]*bblock[1][3][j][i]
                              - ablock[4][2][j][i]*bblock[1][4][j][i];
  cblock[1][3][j][i] = cblock[1][3][j][i] - ablock[0][3][j][i]*bblock[1][0][j][i]
                              - ablock[1][3][j][i]*bblock[1][1][j][i]
                              - ablock[2][3][j][i]*bblock[1][2][j][i]
                              - ablock[3][3][j][i]*bblock[1][3][j][i]
                              - ablock[4][3][j][i]*bblock[1][4][j][i];
  cblock[1][4][j][i] = cblock[1][4][j][i] - ablock[0][4][j][i]*bblock[1][0][j][i]
                              - ablock[1][4][j][i]*bblock[1][1][j][i]
                              - ablock[2][4][j][i]*bblock[1][2][j][i]
                              - ablock[3][4][j][i]*bblock[1][3][j][i]
                              - ablock[4][4][j][i]*bblock[1][4][j][i];
  cblock[2][0][j][i] = cblock[2][0][j][i] - ablock[0][0][j][i]*bblock[2][0][j][i]
                              - ablock[1][0][j][i]*bblock[2][1][j][i]
                              - ablock[2][0][j][i]*bblock[2][2][j][i]
                              - ablock[3][0][j][i]*bblock[2][3][j][i]
                              - ablock[4][0][j][i]*bblock[2][4][j][i];
  cblock[2][1][j][i] = cblock[2][1][j][i] - ablock[0][1][j][i]*bblock[2][0][j][i]
                              - ablock[1][1][j][i]*bblock[2][1][j][i]
                              - ablock[2][1][j][i]*bblock[2][2][j][i]
                              - ablock[3][1][j][i]*bblock[2][3][j][i]
                              - ablock[4][1][j][i]*bblock[2][4][j][i];
  cblock[2][2][j][i] = cblock[2][2][j][i] - ablock[0][2][j][i]*bblock[2][0][j][i]
                              - ablock[1][2][j][i]*bblock[2][1][j][i]
                              - ablock[2][2][j][i]*bblock[2][2][j][i]
                              - ablock[3][2][j][i]*bblock[2][3][j][i]
                              - ablock[4][2][j][i]*bblock[2][4][j][i];
  cblock[2][3][j][i] = cblock[2][3][j][i] - ablock[0][3][j][i]*bblock[2][0][j][i]
                              - ablock[1][3][j][i]*bblock[2][1][j][i]
                              - ablock[2][3][j][i]*bblock[2][2][j][i]
                              - ablock[3][3][j][i]*bblock[2][3][j][i]
                              - ablock[4][3][j][i]*bblock[2][4][j][i];
  cblock[2][4][j][i] = cblock[2][4][j][i] - ablock[0][4][j][i]*bblock[2][0][j][i]
                              - ablock[1][4][j][i]*bblock[2][1][j][i]
                              - ablock[2][4][j][i]*bblock[2][2][j][i]
                              - ablock[3][4][j][i]*bblock[2][3][j][i]
                              - ablock[4][4][j][i]*bblock[2][4][j][i];
  cblock[3][0][j][i] = cblock[3][0][j][i] - ablock[0][0][j][i]*bblock[3][0][j][i]
                              - ablock[1][0][j][i]*bblock[3][1][j][i]
                              - ablock[2][0][j][i]*bblock[3][2][j][i]
                              - ablock[3][0][j][i]*bblock[3][3][j][i]
                              - ablock[4][0][j][i]*bblock[3][4][j][i];
  cblock[3][1][j][i] = cblock[3][1][j][i] - ablock[0][1][j][i]*bblock[3][0][j][i]
                              - ablock[1][1][j][i]*bblock[3][1][j][i]
                              - ablock[2][1][j][i]*bblock[3][2][j][i]
                              - ablock[3][1][j][i]*bblock[3][3][j][i]
                              - ablock[4][1][j][i]*bblock[3][4][j][i];
  cblock[3][2][j][i] = cblock[3][2][j][i] - ablock[0][2][j][i]*bblock[3][0][j][i]
                              - ablock[1][2][j][i]*bblock[3][1][j][i]
                              - ablock[2][2][j][i]*bblock[3][2][j][i]
                              - ablock[3][2][j][i]*bblock[3][3][j][i]
                              - ablock[4][2][j][i]*bblock[3][4][j][i];
  cblock[3][3][j][i] = cblock[3][3][j][i] - ablock[0][3][j][i]*bblock[3][0][j][i]
                              - ablock[1][3][j][i]*bblock[3][1][j][i]
                              - ablock[2][3][j][i]*bblock[3][2][j][i]
                              - ablock[3][3][j][i]*bblock[3][3][j][i]
                              - ablock[4][3][j][i]*bblock[3][4][j][i];
  cblock[3][4][j][i] = cblock[3][4][j][i] - ablock[0][4][j][i]*bblock[3][0][j][i]
                              - ablock[1][4][j][i]*bblock[3][1][j][i]
                              - ablock[2][4][j][i]*bblock[3][2][j][i]
                              - ablock[3][4][j][i]*bblock[3][3][j][i]
                              - ablock[4][4][j][i]*bblock[3][4][j][i];
  cblock[4][0][j][i] = cblock[4][0][j][i] - ablock[0][0][j][i]*bblock[4][0][j][i]
                              - ablock[1][0][j][i]*bblock[4][1][j][i]
                              - ablock[2][0][j][i]*bblock[4][2][j][i]
                              - ablock[3][0][j][i]*bblock[4][3][j][i]
                              - ablock[4][0][j][i]*bblock[4][4][j][i];
  cblock[4][1][j][i] = cblock[4][1][j][i] - ablock[0][1][j][i]*bblock[4][0][j][i]
                              - ablock[1][1][j][i]*bblock[4][1][j][i]
                              - ablock[2][1][j][i]*bblock[4][2][j][i]
                              - ablock[3][1][j][i]*bblock[4][3][j][i]
                              - ablock[4][1][j][i]*bblock[4][4][j][i];
  cblock[4][2][j][i] = cblock[4][2][j][i] - ablock[0][2][j][i]*bblock[4][0][j][i]
                              - ablock[1][2][j][i]*bblock[4][1][j][i]
                              - ablock[2][2][j][i]*bblock[4][2][j][i]
                              - ablock[3][2][j][i]*bblock[4][3][j][i]
                              - ablock[4][2][j][i]*bblock[4][4][j][i];
  cblock[4][3][j][i] = cblock[4][3][j][i] - ablock[0][3][j][i]*bblock[4][0][j][i]
                              - ablock[1][3][j][i]*bblock[4][1][j][i]
                              - ablock[2][3][j][i]*bblock[4][2][j][i]
                              - ablock[3][3][j][i]*bblock[4][3][j][i]
                              - ablock[4][3][j][i]*bblock[4][4][j][i];
  cblock[4][4][j][i] = cblock[4][4][j][i] - ablock[0][4][j][i]*bblock[4][0][j][i]
                              - ablock[1][4][j][i]*bblock[4][1][j][i]
                              - ablock[2][4][j][i]*bblock[4][2][j][i]
                              - ablock[3][4][j][i]*bblock[4][3][j][i]
                              - ablock[4][4][j][i]*bblock[4][4][j][i];
}

inline void binvrhs_z(__global double lhs[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1],
                      __global double r[5],
                      int j, int i)
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0][j][i];
  lhs[1][0][j][i] = lhs[1][0][j][i]*pivot;
  lhs[2][0][j][i] = lhs[2][0][j][i]*pivot;
  lhs[3][0][j][i] = lhs[3][0][j][i]*pivot;
  lhs[4][0][j][i] = lhs[4][0][j][i]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1][j][i];
  lhs[1][1][j][i]= lhs[1][1][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][1][j][i]= lhs[2][1][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][1][j][i]= lhs[3][1][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][1][j][i]= lhs[4][1][j][i] - coeff*lhs[4][0][j][i];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2][j][i];
  lhs[1][2][j][i]= lhs[1][2][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][2][j][i]= lhs[2][2][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][2][j][i]= lhs[3][2][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][2][j][i]= lhs[4][2][j][i] - coeff*lhs[4][0][j][i];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3][j][i];
  lhs[1][3][j][i]= lhs[1][3][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][3][j][i]= lhs[2][3][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][3][j][i]= lhs[3][3][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][3][j][i]= lhs[4][3][j][i] - coeff*lhs[4][0][j][i];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4][j][i];
  lhs[1][4][j][i]= lhs[1][4][j][i] - coeff*lhs[1][0][j][i];
  lhs[2][4][j][i]= lhs[2][4][j][i] - coeff*lhs[2][0][j][i];
  lhs[3][4][j][i]= lhs[3][4][j][i] - coeff*lhs[3][0][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][0][j][i];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1][j][i];
  lhs[2][1][j][i] = lhs[2][1][j][i]*pivot;
  lhs[3][1][j][i] = lhs[3][1][j][i]*pivot;
  lhs[4][1][j][i] = lhs[4][1][j][i]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0][j][i];
  lhs[2][0][j][i]= lhs[2][0][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][0][j][i]= lhs[3][0][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][0][j][i]= lhs[4][0][j][i] - coeff*lhs[4][1][j][i];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2][j][i];
  lhs[2][2][j][i]= lhs[2][2][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][2][j][i]= lhs[3][2][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][2][j][i]= lhs[4][2][j][i] - coeff*lhs[4][1][j][i];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3][j][i];
  lhs[2][3][j][i]= lhs[2][3][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][3][j][i]= lhs[3][3][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][3][j][i]= lhs[4][3][j][i] - coeff*lhs[4][1][j][i];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4][j][i];
  lhs[2][4][j][i]= lhs[2][4][j][i] - coeff*lhs[2][1][j][i];
  lhs[3][4][j][i]= lhs[3][4][j][i] - coeff*lhs[3][1][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][1][j][i];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2][j][i];
  lhs[3][2][j][i] = lhs[3][2][j][i]*pivot;
  lhs[4][2][j][i] = lhs[4][2][j][i]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0][j][i];
  lhs[3][0][j][i]= lhs[3][0][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][0][j][i]= lhs[4][0][j][i] - coeff*lhs[4][2][j][i];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1][j][i];
  lhs[3][1][j][i]= lhs[3][1][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][1][j][i]= lhs[4][1][j][i] - coeff*lhs[4][2][j][i];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3][j][i];
  lhs[3][3][j][i]= lhs[3][3][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][3][j][i]= lhs[4][3][j][i] - coeff*lhs[4][2][j][i];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4][j][i];
  lhs[3][4][j][i]= lhs[3][4][j][i] - coeff*lhs[3][2][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][2][j][i];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3][j][i];
  lhs[4][3][j][i] = lhs[4][3][j][i]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0][j][i];
  lhs[4][0][j][i]= lhs[4][0][j][i] - coeff*lhs[4][3][j][i];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1][j][i];
  lhs[4][1][j][i]= lhs[4][1][j][i] - coeff*lhs[4][3][j][i];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2][j][i];
  lhs[4][2][j][i]= lhs[4][2][j][i] - coeff*lhs[4][3][j][i];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4][j][i];
  lhs[4][4][j][i]= lhs[4][4][j][i] - coeff*lhs[4][3][j][i];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4][j][i];
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0][j][i];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1][j][i];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2][j][i];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3][j][i];
  r[3]   = r[3]   - coeff*r[4];
}


__kernel void z_solve_data_gen_memlayout(__global double *m_u,
                                         __global double *m_square,
                                         __global double *m_qs,
                                         int gp0, int gp1, int gp2,
                                         int work_base,
                                         int work_num_item,
                                         int split_flag)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0)+1;

  if (k > gp2-1 || j+work_base < 1 || j+work_base > gp1-2 || j >= work_num_item || i > gp0-2) return;

  __global double (* qs)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1])m_qs;
  __global double (* square)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1]) m_square; 
  __global double (* u)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1][5]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1][5])m_u;

  double rho_inv;

  rho_inv = 1.0/u[k][j][i][0];
  square[k][j][i] = 0.5* (
      u[k][j][i][1]*u[k][j][i][1] + 
      u[k][j][i][2]*u[k][j][i][2] +
      u[k][j][i][3]*u[k][j][i][3] ) * rho_inv;
  qs[k][j][i] = square[k][j][i] * rho_inv;

}

__kernel void z_solve_memlayout(__global double *m_qs,
                                __global double *m_square,
                                __global double *m_u,
                                __global double *m_rhs,
                                __global double *m_lhs,
                                __global double *m_fjac,
                                __global double *m_njac,
                                int gp0, int gp1, int gp2,
                                double dz1, double dz2,
                                double dz3, double dz4,
                                double dz5,
                                double c1, double c2,
                                double c3, double c4,
                                double tz1, double tz2,
                                double con43, double c3c4,
                                double c1345, double dt,
                                int work_base,
                                int work_num_item,
                                int split_flag)
{
  int j = get_global_id(1);
  int i = get_global_id(0) + 1;
  int ksize = gp2-1;
  int k, m, n;
  double tmp1, tmp2, tmp3;

  if (j + work_base < 1
      || j + work_base > gp1-2
      || j >= work_num_item
      || i > gp0-2)
    return;

  if (!split_flag) j += work_base;

  __global double (* qs)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1])m_qs;
  __global double (* square)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1]) m_square; 
  __global double (* u)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1][5]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1][5])m_u;
  __global double (* rhs)[WORK_NUM_ITEM_DEFAULT_J][IMAXP+1][5]
    = (__global double (*) [WORK_NUM_ITEM_DEFAULT_J][IMAXP+1][5])m_rhs;


  __global double (* lhs)[3][5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1]
    = (__global double (*)[3][5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1])m_lhs;

  __global double (* fjac)[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1]
    = (__global double (*)[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1])m_fjac;
  __global double (* njac)[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1]
    = (__global double (*)[5][5][WORK_NUM_ITEM_DEFAULT_J][PROBLEM_SIZE-1])m_njac;

  for (k = 0; k <= ksize; k++) {
    tmp1 = 1.0 / u[k][j][i][0];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[k][0][0][j][i] = 0.0;
    fjac[k][1][0][j][i] = 0.0;
    fjac[k][2][0][j][i] = 0.0;
    fjac[k][3][0][j][i] = 1.0;
    fjac[k][4][0][j][i] = 0.0;

    fjac[k][0][1][j][i] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[k][1][1][j][i] = u[k][j][i][3] * tmp1;
    fjac[k][2][1][j][i] = 0.0;
    fjac[k][3][1][j][i] = u[k][j][i][1] * tmp1;
    fjac[k][4][1][j][i] = 0.0;

    fjac[k][0][2][j][i] = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[k][1][2][j][i] = 0.0;
    fjac[k][2][2][j][i] = u[k][j][i][3] * tmp1;
    fjac[k][3][2][j][i] = u[k][j][i][2] * tmp1;
    fjac[k][4][2][j][i] = 0.0;

    fjac[k][0][3][j][i] = - (u[k][j][i][3]*u[k][j][i][3] * tmp2 ) 
      + c2 * qs[k][j][i];
    fjac[k][1][3][j][i] = - c2 *  u[k][j][i][1] * tmp1;
    fjac[k][2][3][j][i] = - c2 *  u[k][j][i][2] * tmp1;
    fjac[k][3][3][j][i] = ( 2.0 - c2 ) *  u[k][j][i][3] * tmp1;
    fjac[k][4][3][j][i] = c2;

    fjac[k][0][4][j][i] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * u[k][j][i][3] * tmp2;
    fjac[k][1][4][j][i] = - c2 * ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[k][2][4][j][i] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[k][3][4][j][i] = c1 * ( u[k][j][i][4] * tmp1 )
      - c2 * ( qs[k][j][i] + u[k][j][i][3]*u[k][j][i][3] * tmp2 );
    fjac[k][4][4][j][i] = c1 * u[k][j][i][3] * tmp1;

    njac[k][0][0][j][i] = 0.0;
    njac[k][1][0][j][i] = 0.0;
    njac[k][2][0][j][i] = 0.0;
    njac[k][3][0][j][i] = 0.0;
    njac[k][4][0][j][i] = 0.0;

    njac[k][0][1][j][i] = - c3c4 * tmp2 * u[k][j][i][1];
    njac[k][1][1][j][i] =   c3c4 * tmp1;
    njac[k][2][1][j][i] =   0.0;
    njac[k][3][1][j][i] =   0.0;
    njac[k][4][1][j][i] =   0.0;

    njac[k][0][2][j][i] = - c3c4 * tmp2 * u[k][j][i][2];
    njac[k][1][2][j][i] =   0.0;
    njac[k][2][2][j][i] =   c3c4 * tmp1;
    njac[k][3][2][j][i] =   0.0;
    njac[k][4][2][j][i] =   0.0;

    njac[k][0][3][j][i] = - con43 * c3c4 * tmp2 * u[k][j][i][3];
    njac[k][1][3][j][i] =   0.0;
    njac[k][2][3][j][i] =   0.0;
    njac[k][3][3][j][i] =   con43 * c3 * c4 * tmp1;
    njac[k][4][3][j][i] =   0.0;

    njac[k][0][4][j][i] = - (  c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( con43 * c3c4
          - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[k][1][4][j][i] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
    njac[k][2][4][j][i] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[k][3][4][j][i] = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][3];
    njac[k][4][4][j][i] = ( c1345 )* tmp1;
  }

  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in z direction
  //---------------------------------------------------------------------
  lhsinit_z(lhs, ksize, j, i);
  for (k = 1; k <= ksize-1; k++) {
    tmp1 = dt * tz1;
    tmp2 = dt * tz2;

    lhs[k][AA][0][0][j][i] = - tmp2 * fjac[k-1][0][0][j][i]
      - tmp1 * njac[k-1][0][0][j][i]
      - tmp1 * dz1; 
    lhs[k][AA][1][0][j][i] = - tmp2 * fjac[k-1][1][0][j][i]
      - tmp1 * njac[k-1][1][0][j][i];
    lhs[k][AA][2][0][j][i] = - tmp2 * fjac[k-1][2][0][j][i]
      - tmp1 * njac[k-1][2][0][j][i];
    lhs[k][AA][3][0][j][i] = - tmp2 * fjac[k-1][3][0][j][i]
      - tmp1 * njac[k-1][3][0][j][i];
    lhs[k][AA][4][0][j][i] = - tmp2 * fjac[k-1][4][0][j][i]
      - tmp1 * njac[k-1][4][0][j][i];

    lhs[k][AA][0][1][j][i] = - tmp2 * fjac[k-1][0][1][j][i]
      - tmp1 * njac[k-1][0][1][j][i];
    lhs[k][AA][1][1][j][i] = - tmp2 * fjac[k-1][1][1][j][i]
      - tmp1 * njac[k-1][1][1][j][i]
      - tmp1 * dz2;
    lhs[k][AA][2][1][j][i] = - tmp2 * fjac[k-1][2][1][j][i]
      - tmp1 * njac[k-1][2][1][j][i];
    lhs[k][AA][3][1][j][i] = - tmp2 * fjac[k-1][3][1][j][i]
      - tmp1 * njac[k-1][3][1][j][i];
    lhs[k][AA][4][1][j][i] = - tmp2 * fjac[k-1][4][1][j][i]
      - tmp1 * njac[k-1][4][1][j][i];

    lhs[k][AA][0][2][j][i] = - tmp2 * fjac[k-1][0][2][j][i]
      - tmp1 * njac[k-1][0][2][j][i];
    lhs[k][AA][1][2][j][i] = - tmp2 * fjac[k-1][1][2][j][i]
      - tmp1 * njac[k-1][1][2][j][i];
    lhs[k][AA][2][2][j][i] = - tmp2 * fjac[k-1][2][2][j][i]
      - tmp1 * njac[k-1][2][2][j][i]
      - tmp1 * dz3;
    lhs[k][AA][3][2][j][i] = - tmp2 * fjac[k-1][3][2][j][i]
      - tmp1 * njac[k-1][3][2][j][i];
    lhs[k][AA][4][2][j][i] = - tmp2 * fjac[k-1][4][2][j][i]
      - tmp1 * njac[k-1][4][2][j][i];

    lhs[k][AA][0][3][j][i] = - tmp2 * fjac[k-1][0][3][j][i]
      - tmp1 * njac[k-1][0][3][j][i];
    lhs[k][AA][1][3][j][i] = - tmp2 * fjac[k-1][1][3][j][i]
      - tmp1 * njac[k-1][1][3][j][i];
    lhs[k][AA][2][3][j][i] = - tmp2 * fjac[k-1][2][3][j][i]
      - tmp1 * njac[k-1][2][3][j][i];
    lhs[k][AA][3][3][j][i] = - tmp2 * fjac[k-1][3][3][j][i]
      - tmp1 * njac[k-1][3][3][j][i]
      - tmp1 * dz4;
    lhs[k][AA][4][3][j][i] = - tmp2 * fjac[k-1][4][3][j][i]
      - tmp1 * njac[k-1][4][3][j][i];

    lhs[k][AA][0][4][j][i] = - tmp2 * fjac[k-1][0][4][j][i]
      - tmp1 * njac[k-1][0][4][j][i];
    lhs[k][AA][1][4][j][i] = - tmp2 * fjac[k-1][1][4][j][i]
      - tmp1 * njac[k-1][1][4][j][i];
    lhs[k][AA][2][4][j][i] = - tmp2 * fjac[k-1][2][4][j][i]
      - tmp1 * njac[k-1][2][4][j][i];
    lhs[k][AA][3][4][j][i] = - tmp2 * fjac[k-1][3][4][j][i]
      - tmp1 * njac[k-1][3][4][j][i];
    lhs[k][AA][4][4][j][i] = - tmp2 * fjac[k-1][4][4][j][i]
      - tmp1 * njac[k-1][4][4][j][i]
      - tmp1 * dz5;

    lhs[k][BB][0][0][j][i] = 1.0
      + tmp1 * 2.0 * njac[k][0][0][j][i]
      + tmp1 * 2.0 * dz1;
    lhs[k][BB][1][0][j][i] = tmp1 * 2.0 * njac[k][1][0][j][i];
    lhs[k][BB][2][0][j][i] = tmp1 * 2.0 * njac[k][2][0][j][i];
    lhs[k][BB][3][0][j][i] = tmp1 * 2.0 * njac[k][3][0][j][i];
    lhs[k][BB][4][0][j][i] = tmp1 * 2.0 * njac[k][4][0][j][i];

    lhs[k][BB][0][1][j][i] = tmp1 * 2.0 * njac[k][0][1][j][i];
    lhs[k][BB][1][1][j][i] = 1.0
      + tmp1 * 2.0 * njac[k][1][1][j][i]
      + tmp1 * 2.0 * dz2;
    lhs[k][BB][2][1][j][i] = tmp1 * 2.0 * njac[k][2][1][j][i];
    lhs[k][BB][3][1][j][i] = tmp1 * 2.0 * njac[k][3][1][j][i];
    lhs[k][BB][4][1][j][i] = tmp1 * 2.0 * njac[k][4][1][j][i];

    lhs[k][BB][0][2][j][i] = tmp1 * 2.0 * njac[k][0][2][j][i];
    lhs[k][BB][1][2][j][i] = tmp1 * 2.0 * njac[k][1][2][j][i];
    lhs[k][BB][2][2][j][i] = 1.0
      + tmp1 * 2.0 * njac[k][2][2][j][i]
      + tmp1 * 2.0 * dz3;
    lhs[k][BB][3][2][j][i] = tmp1 * 2.0 * njac[k][3][2][j][i];
    lhs[k][BB][4][2][j][i] = tmp1 * 2.0 * njac[k][4][2][j][i];

    lhs[k][BB][0][3][j][i] = tmp1 * 2.0 * njac[k][0][3][j][i];
    lhs[k][BB][1][3][j][i] = tmp1 * 2.0 * njac[k][1][3][j][i];
    lhs[k][BB][2][3][j][i] = tmp1 * 2.0 * njac[k][2][3][j][i];
    lhs[k][BB][3][3][j][i] = 1.0
      + tmp1 * 2.0 * njac[k][3][3][j][i]
      + tmp1 * 2.0 * dz4;
    lhs[k][BB][4][3][j][i] = tmp1 * 2.0 * njac[k][4][3][j][i];

    lhs[k][BB][0][4][j][i] = tmp1 * 2.0 * njac[k][0][4][j][i];
    lhs[k][BB][1][4][j][i] = tmp1 * 2.0 * njac[k][1][4][j][i];
    lhs[k][BB][2][4][j][i] = tmp1 * 2.0 * njac[k][2][4][j][i];
    lhs[k][BB][3][4][j][i] = tmp1 * 2.0 * njac[k][3][4][j][i];
    lhs[k][BB][4][4][j][i] = 1.0
      + tmp1 * 2.0 * njac[k][4][4][j][i] 
      + tmp1 * 2.0 * dz5;

    lhs[k][CC][0][0][j][i] =  tmp2 * fjac[k+1][0][0][j][i]
      - tmp1 * njac[k+1][0][0][j][i]
      - tmp1 * dz1;
    lhs[k][CC][1][0][j][i] =  tmp2 * fjac[k+1][1][0][j][i]
      - tmp1 * njac[k+1][1][0][j][i];
    lhs[k][CC][2][0][j][i] =  tmp2 * fjac[k+1][2][0][j][i]
      - tmp1 * njac[k+1][2][0][j][i];
    lhs[k][CC][3][0][j][i] =  tmp2 * fjac[k+1][3][0][j][i]
      - tmp1 * njac[k+1][3][0][j][i];
    lhs[k][CC][4][0][j][i] =  tmp2 * fjac[k+1][4][0][j][i]
      - tmp1 * njac[k+1][4][0][j][i];

    lhs[k][CC][0][1][j][i] =  tmp2 * fjac[k+1][0][1][j][i]
      - tmp1 * njac[k+1][0][1][j][i];
    lhs[k][CC][1][1][j][i] =  tmp2 * fjac[k+1][1][1][j][i]
      - tmp1 * njac[k+1][1][1][j][i]
      - tmp1 * dz2;
    lhs[k][CC][2][1][j][i] =  tmp2 * fjac[k+1][2][1][j][i]
      - tmp1 * njac[k+1][2][1][j][i];
    lhs[k][CC][3][1][j][i] =  tmp2 * fjac[k+1][3][1][j][i]
      - tmp1 * njac[k+1][3][1][j][i];
    lhs[k][CC][4][1][j][i] =  tmp2 * fjac[k+1][4][1][j][i]
      - tmp1 * njac[k+1][4][1][j][i];

    lhs[k][CC][0][2][j][i] =  tmp2 * fjac[k+1][0][2][j][i]
      - tmp1 * njac[k+1][0][2][j][i];
    lhs[k][CC][1][2][j][i] =  tmp2 * fjac[k+1][1][2][j][i]
      - tmp1 * njac[k+1][1][2][j][i];
    lhs[k][CC][2][2][j][i] =  tmp2 * fjac[k+1][2][2][j][i]
      - tmp1 * njac[k+1][2][2][j][i]
      - tmp1 * dz3;
    lhs[k][CC][3][2][j][i] =  tmp2 * fjac[k+1][3][2][j][i]
      - tmp1 * njac[k+1][3][2][j][i];
    lhs[k][CC][4][2][j][i] =  tmp2 * fjac[k+1][4][2][j][i]
      - tmp1 * njac[k+1][4][2][j][i];

    lhs[k][CC][0][3][j][i] =  tmp2 * fjac[k+1][0][3][j][i]
      - tmp1 * njac[k+1][0][3][j][i];
    lhs[k][CC][1][3][j][i] =  tmp2 * fjac[k+1][1][3][j][i]
      - tmp1 * njac[k+1][1][3][j][i];
    lhs[k][CC][2][3][j][i] =  tmp2 * fjac[k+1][2][3][j][i]
      - tmp1 * njac[k+1][2][3][j][i];
    lhs[k][CC][3][3][j][i] =  tmp2 * fjac[k+1][3][3][j][i]
      - tmp1 * njac[k+1][3][3][j][i]
      - tmp1 * dz4;
    lhs[k][CC][4][3][j][i] =  tmp2 * fjac[k+1][4][3][j][i]
      - tmp1 * njac[k+1][4][3][j][i];

    lhs[k][CC][0][4][j][i] =  tmp2 * fjac[k+1][0][4][j][i]
      - tmp1 * njac[k+1][0][4][j][i];
    lhs[k][CC][1][4][j][i] =  tmp2 * fjac[k+1][1][4][j][i]
      - tmp1 * njac[k+1][1][4][j][i];
    lhs[k][CC][2][4][j][i] =  tmp2 * fjac[k+1][2][4][j][i]
      - tmp1 * njac[k+1][2][4][j][i];
    lhs[k][CC][3][4][j][i] =  tmp2 * fjac[k+1][3][4][j][i]
      - tmp1 * njac[k+1][3][4][j][i];
    lhs[k][CC][4][4][j][i] =  tmp2 * fjac[k+1][4][4][j][i]
      - tmp1 * njac[k+1][4][4][j][i]
      - tmp1 * dz5;
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
  // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // outer most do loops - sweeping in i direction
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[0][j][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  binvcrhs_z( lhs[0][BB], lhs[0][CC], rhs[0][j][i], j, i);

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (k = 1; k <= ksize-1; k++) {
    //-------------------------------------------------------------------
    // subtract A*lhs_vector(k-1) from lhs_vector(k)
    // 
    // rhs(k) = rhs(k) - A*rhs(k-1)
    //-------------------------------------------------------------------
    matvec_sub_z(lhs[k][AA], rhs[k-1][j][i], rhs[k][j][i], j, i);

    //-------------------------------------------------------------------
    // B(k) = B(k) - C(k-1)*A(k)
    // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
    //-------------------------------------------------------------------
    matmul_sub_z(lhs[k][AA], lhs[k-1][CC], lhs[k][BB], j, i);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs_z( lhs[k][BB], lhs[k][CC], rhs[k][j][i], j, i);
  }

  //---------------------------------------------------------------------
  // Now finish up special cases for last cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
  //---------------------------------------------------------------------
  matvec_sub_z(lhs[ksize][AA], rhs[ksize-1][j][i], rhs[ksize][j][i], j, i);

  //---------------------------------------------------------------------
  // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
  // matmul_sub(AA,i,j,ksize,c,
  // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
  //---------------------------------------------------------------------
  matmul_sub_z(lhs[ksize][AA], lhs[ksize-1][CC], lhs[ksize][BB], j, i);

  //---------------------------------------------------------------------
  // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
  //---------------------------------------------------------------------
  binvrhs_z( lhs[ksize][BB], rhs[ksize][j][i], j, i);

  //---------------------------------------------------------------------
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(ksize)=rhs(ksize)
  // else assume U(ksize) is loaded in un pack backsub_info
  // so just use it
  // after u(kstart) will be sent to next cell
  //---------------------------------------------------------------------

  for (k = ksize-1; k >= 0; k--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhs[k][CC][n][m][j][i]*rhs[k+1][j][i][n];
      }
    }
  }
}
