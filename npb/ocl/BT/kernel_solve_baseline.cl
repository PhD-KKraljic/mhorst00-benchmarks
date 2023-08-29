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

inline void lhsinit(__global double lhs[][3][5][5], int size)
{
  int i, m, n;

  i = size;
  //---------------------------------------------------------------------
  // zero the whole left hand side for starters
  //---------------------------------------------------------------------
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      lhs[0][0][n][m] = 0.0;
      lhs[0][1][n][m] = 0.0;
      lhs[0][2][n][m] = 0.0;
      lhs[i][0][n][m] = 0.0;
      lhs[i][1][n][m] = 0.0;
      lhs[i][2][n][m] = 0.0;
    }
  }

  //---------------------------------------------------------------------
  // next, set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    lhs[0][1][m][m] = 1.0;
    lhs[i][1][m][m] = 1.0;
  }
}

inline void binvcrhs(__global double lhs[5][5], __global double c[5][5], __global double r[5])
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0];
  lhs[1][0] = lhs[1][0]*pivot;
  lhs[2][0] = lhs[2][0]*pivot;
  lhs[3][0] = lhs[3][0]*pivot;
  lhs[4][0] = lhs[4][0]*pivot;
  c[0][0] = c[0][0]*pivot;
  c[1][0] = c[1][0]*pivot;
  c[2][0] = c[2][0]*pivot;
  c[3][0] = c[3][0]*pivot;
  c[4][0] = c[4][0]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1];
  lhs[1][1]= lhs[1][1] - coeff*lhs[1][0];
  lhs[2][1]= lhs[2][1] - coeff*lhs[2][0];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][0];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][0];
  c[0][1] = c[0][1] - coeff*c[0][0];
  c[1][1] = c[1][1] - coeff*c[1][0];
  c[2][1] = c[2][1] - coeff*c[2][0];
  c[3][1] = c[3][1] - coeff*c[3][0];
  c[4][1] = c[4][1] - coeff*c[4][0];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2];
  lhs[1][2]= lhs[1][2] - coeff*lhs[1][0];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][0];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][0];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][0];
  c[0][2] = c[0][2] - coeff*c[0][0];
  c[1][2] = c[1][2] - coeff*c[1][0];
  c[2][2] = c[2][2] - coeff*c[2][0];
  c[3][2] = c[3][2] - coeff*c[3][0];
  c[4][2] = c[4][2] - coeff*c[4][0];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3];
  lhs[1][3]= lhs[1][3] - coeff*lhs[1][0];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][0];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][0];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][0];
  c[0][3] = c[0][3] - coeff*c[0][0];
  c[1][3] = c[1][3] - coeff*c[1][0];
  c[2][3] = c[2][3] - coeff*c[2][0];
  c[3][3] = c[3][3] - coeff*c[3][0];
  c[4][3] = c[4][3] - coeff*c[4][0];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4];
  lhs[1][4]= lhs[1][4] - coeff*lhs[1][0];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][0];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][0];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][0];
  c[0][4] = c[0][4] - coeff*c[0][0];
  c[1][4] = c[1][4] - coeff*c[1][0];
  c[2][4] = c[2][4] - coeff*c[2][0];
  c[3][4] = c[3][4] - coeff*c[3][0];
  c[4][4] = c[4][4] - coeff*c[4][0];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1];
  lhs[2][1] = lhs[2][1]*pivot;
  lhs[3][1] = lhs[3][1]*pivot;
  lhs[4][1] = lhs[4][1]*pivot;
  c[0][1] = c[0][1]*pivot;
  c[1][1] = c[1][1]*pivot;
  c[2][1] = c[2][1]*pivot;
  c[3][1] = c[3][1]*pivot;
  c[4][1] = c[4][1]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0];
  lhs[2][0]= lhs[2][0] - coeff*lhs[2][1];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][1];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][1];
  c[0][0] = c[0][0] - coeff*c[0][1];
  c[1][0] = c[1][0] - coeff*c[1][1];
  c[2][0] = c[2][0] - coeff*c[2][1];
  c[3][0] = c[3][0] - coeff*c[3][1];
  c[4][0] = c[4][0] - coeff*c[4][1];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][1];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][1];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][1];
  c[0][2] = c[0][2] - coeff*c[0][1];
  c[1][2] = c[1][2] - coeff*c[1][1];
  c[2][2] = c[2][2] - coeff*c[2][1];
  c[3][2] = c[3][2] - coeff*c[3][1];
  c[4][2] = c[4][2] - coeff*c[4][1];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][1];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][1];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][1];
  c[0][3] = c[0][3] - coeff*c[0][1];
  c[1][3] = c[1][3] - coeff*c[1][1];
  c[2][3] = c[2][3] - coeff*c[2][1];
  c[3][3] = c[3][3] - coeff*c[3][1];
  c[4][3] = c[4][3] - coeff*c[4][1];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][1];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][1];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][1];
  c[0][4] = c[0][4] - coeff*c[0][1];
  c[1][4] = c[1][4] - coeff*c[1][1];
  c[2][4] = c[2][4] - coeff*c[2][1];
  c[3][4] = c[3][4] - coeff*c[3][1];
  c[4][4] = c[4][4] - coeff*c[4][1];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2];
  lhs[3][2] = lhs[3][2]*pivot;
  lhs[4][2] = lhs[4][2]*pivot;
  c[0][2] = c[0][2]*pivot;
  c[1][2] = c[1][2]*pivot;
  c[2][2] = c[2][2]*pivot;
  c[3][2] = c[3][2]*pivot;
  c[4][2] = c[4][2]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][2];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][2];
  c[0][0] = c[0][0] - coeff*c[0][2];
  c[1][0] = c[1][0] - coeff*c[1][2];
  c[2][0] = c[2][0] - coeff*c[2][2];
  c[3][0] = c[3][0] - coeff*c[3][2];
  c[4][0] = c[4][0] - coeff*c[4][2];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][2];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][2];
  c[0][1] = c[0][1] - coeff*c[0][2];
  c[1][1] = c[1][1] - coeff*c[1][2];
  c[2][1] = c[2][1] - coeff*c[2][2];
  c[3][1] = c[3][1] - coeff*c[3][2];
  c[4][1] = c[4][1] - coeff*c[4][2];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][2];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][2];
  c[0][3] = c[0][3] - coeff*c[0][2];
  c[1][3] = c[1][3] - coeff*c[1][2];
  c[2][3] = c[2][3] - coeff*c[2][2];
  c[3][3] = c[3][3] - coeff*c[3][2];
  c[4][3] = c[4][3] - coeff*c[4][2];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][2];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][2];
  c[0][4] = c[0][4] - coeff*c[0][2];
  c[1][4] = c[1][4] - coeff*c[1][2];
  c[2][4] = c[2][4] - coeff*c[2][2];
  c[3][4] = c[3][4] - coeff*c[3][2];
  c[4][4] = c[4][4] - coeff*c[4][2];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3];
  lhs[4][3] = lhs[4][3]*pivot;
  c[0][3] = c[0][3]*pivot;
  c[1][3] = c[1][3]*pivot;
  c[2][3] = c[2][3]*pivot;
  c[3][3] = c[3][3]*pivot;
  c[4][3] = c[4][3]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][3];
  c[0][0] = c[0][0] - coeff*c[0][3];
  c[1][0] = c[1][0] - coeff*c[1][3];
  c[2][0] = c[2][0] - coeff*c[2][3];
  c[3][0] = c[3][0] - coeff*c[3][3];
  c[4][0] = c[4][0] - coeff*c[4][3];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][3];
  c[0][1] = c[0][1] - coeff*c[0][3];
  c[1][1] = c[1][1] - coeff*c[1][3];
  c[2][1] = c[2][1] - coeff*c[2][3];
  c[3][1] = c[3][1] - coeff*c[3][3];
  c[4][1] = c[4][1] - coeff*c[4][3];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][3];
  c[0][2] = c[0][2] - coeff*c[0][3];
  c[1][2] = c[1][2] - coeff*c[1][3];
  c[2][2] = c[2][2] - coeff*c[2][3];
  c[3][2] = c[3][2] - coeff*c[3][3];
  c[4][2] = c[4][2] - coeff*c[4][3];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][3];
  c[0][4] = c[0][4] - coeff*c[0][3];
  c[1][4] = c[1][4] - coeff*c[1][3];
  c[2][4] = c[2][4] - coeff*c[2][3];
  c[3][4] = c[3][4] - coeff*c[3][3];
  c[4][4] = c[4][4] - coeff*c[4][3];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4];
  c[0][4] = c[0][4]*pivot;
  c[1][4] = c[1][4]*pivot;
  c[2][4] = c[2][4]*pivot;
  c[3][4] = c[3][4]*pivot;
  c[4][4] = c[4][4]*pivot;
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0];
  c[0][0] = c[0][0] - coeff*c[0][4];
  c[1][0] = c[1][0] - coeff*c[1][4];
  c[2][0] = c[2][0] - coeff*c[2][4];
  c[3][0] = c[3][0] - coeff*c[3][4];
  c[4][0] = c[4][0] - coeff*c[4][4];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1];
  c[0][1] = c[0][1] - coeff*c[0][4];
  c[1][1] = c[1][1] - coeff*c[1][4];
  c[2][1] = c[2][1] - coeff*c[2][4];
  c[3][1] = c[3][1] - coeff*c[3][4];
  c[4][1] = c[4][1] - coeff*c[4][4];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2];
  c[0][2] = c[0][2] - coeff*c[0][4];
  c[1][2] = c[1][2] - coeff*c[1][4];
  c[2][2] = c[2][2] - coeff*c[2][4];
  c[3][2] = c[3][2] - coeff*c[3][4];
  c[4][2] = c[4][2] - coeff*c[4][4];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3];
  c[0][3] = c[0][3] - coeff*c[0][4];
  c[1][3] = c[1][3] - coeff*c[1][4];
  c[2][3] = c[2][3] - coeff*c[2][4];
  c[3][3] = c[3][3] - coeff*c[3][4];
  c[4][3] = c[4][3] - coeff*c[4][4];
  r[3]   = r[3]   - coeff*r[4];
}

inline void matvec_sub(__global double ablock[5][5], __global double avec[5], __global double bvec[5])
{
  //---------------------------------------------------------------------
  // rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] 
  // $                  - lhs[ia][ablock][0][i]*
  //---------------------------------------------------------------------
  bvec[0] = bvec[0] - ablock[0][0]*avec[0]
                    - ablock[1][0]*avec[1]
                    - ablock[2][0]*avec[2]
                    - ablock[3][0]*avec[3]
                    - ablock[4][0]*avec[4];
  bvec[1] = bvec[1] - ablock[0][1]*avec[0]
                    - ablock[1][1]*avec[1]
                    - ablock[2][1]*avec[2]
                    - ablock[3][1]*avec[3]
                    - ablock[4][1]*avec[4];
  bvec[2] = bvec[2] - ablock[0][2]*avec[0]
                    - ablock[1][2]*avec[1]
                    - ablock[2][2]*avec[2]
                    - ablock[3][2]*avec[3]
                    - ablock[4][2]*avec[4];
  bvec[3] = bvec[3] - ablock[0][3]*avec[0]
                    - ablock[1][3]*avec[1]
                    - ablock[2][3]*avec[2]
                    - ablock[3][3]*avec[3]
                    - ablock[4][3]*avec[4];
  bvec[4] = bvec[4] - ablock[0][4]*avec[0]
                    - ablock[1][4]*avec[1]
                    - ablock[2][4]*avec[2]
                    - ablock[3][4]*avec[3]
                    - ablock[4][4]*avec[4];
}

inline void matmul_sub(__global double ablock[5][5], __global double bblock[5][5], __global double cblock[5][5])
{
  cblock[0][0] = cblock[0][0] - ablock[0][0]*bblock[0][0]
                              - ablock[1][0]*bblock[0][1]
                              - ablock[2][0]*bblock[0][2]
                              - ablock[3][0]*bblock[0][3]
                              - ablock[4][0]*bblock[0][4];
  cblock[0][1] = cblock[0][1] - ablock[0][1]*bblock[0][0]
                              - ablock[1][1]*bblock[0][1]
                              - ablock[2][1]*bblock[0][2]
                              - ablock[3][1]*bblock[0][3]
                              - ablock[4][1]*bblock[0][4];
  cblock[0][2] = cblock[0][2] - ablock[0][2]*bblock[0][0]
                              - ablock[1][2]*bblock[0][1]
                              - ablock[2][2]*bblock[0][2]
                              - ablock[3][2]*bblock[0][3]
                              - ablock[4][2]*bblock[0][4];
  cblock[0][3] = cblock[0][3] - ablock[0][3]*bblock[0][0]
                              - ablock[1][3]*bblock[0][1]
                              - ablock[2][3]*bblock[0][2]
                              - ablock[3][3]*bblock[0][3]
                              - ablock[4][3]*bblock[0][4];
  cblock[0][4] = cblock[0][4] - ablock[0][4]*bblock[0][0]
                              - ablock[1][4]*bblock[0][1]
                              - ablock[2][4]*bblock[0][2]
                              - ablock[3][4]*bblock[0][3]
                              - ablock[4][4]*bblock[0][4];
  cblock[1][0] = cblock[1][0] - ablock[0][0]*bblock[1][0]
                              - ablock[1][0]*bblock[1][1]
                              - ablock[2][0]*bblock[1][2]
                              - ablock[3][0]*bblock[1][3]
                              - ablock[4][0]*bblock[1][4];
  cblock[1][1] = cblock[1][1] - ablock[0][1]*bblock[1][0]
                              - ablock[1][1]*bblock[1][1]
                              - ablock[2][1]*bblock[1][2]
                              - ablock[3][1]*bblock[1][3]
                              - ablock[4][1]*bblock[1][4];
  cblock[1][2] = cblock[1][2] - ablock[0][2]*bblock[1][0]
                              - ablock[1][2]*bblock[1][1]
                              - ablock[2][2]*bblock[1][2]
                              - ablock[3][2]*bblock[1][3]
                              - ablock[4][2]*bblock[1][4];
  cblock[1][3] = cblock[1][3] - ablock[0][3]*bblock[1][0]
                              - ablock[1][3]*bblock[1][1]
                              - ablock[2][3]*bblock[1][2]
                              - ablock[3][3]*bblock[1][3]
                              - ablock[4][3]*bblock[1][4];
  cblock[1][4] = cblock[1][4] - ablock[0][4]*bblock[1][0]
                              - ablock[1][4]*bblock[1][1]
                              - ablock[2][4]*bblock[1][2]
                              - ablock[3][4]*bblock[1][3]
                              - ablock[4][4]*bblock[1][4];
  cblock[2][0] = cblock[2][0] - ablock[0][0]*bblock[2][0]
                              - ablock[1][0]*bblock[2][1]
                              - ablock[2][0]*bblock[2][2]
                              - ablock[3][0]*bblock[2][3]
                              - ablock[4][0]*bblock[2][4];
  cblock[2][1] = cblock[2][1] - ablock[0][1]*bblock[2][0]
                              - ablock[1][1]*bblock[2][1]
                              - ablock[2][1]*bblock[2][2]
                              - ablock[3][1]*bblock[2][3]
                              - ablock[4][1]*bblock[2][4];
  cblock[2][2] = cblock[2][2] - ablock[0][2]*bblock[2][0]
                              - ablock[1][2]*bblock[2][1]
                              - ablock[2][2]*bblock[2][2]
                              - ablock[3][2]*bblock[2][3]
                              - ablock[4][2]*bblock[2][4];
  cblock[2][3] = cblock[2][3] - ablock[0][3]*bblock[2][0]
                              - ablock[1][3]*bblock[2][1]
                              - ablock[2][3]*bblock[2][2]
                              - ablock[3][3]*bblock[2][3]
                              - ablock[4][3]*bblock[2][4];
  cblock[2][4] = cblock[2][4] - ablock[0][4]*bblock[2][0]
                              - ablock[1][4]*bblock[2][1]
                              - ablock[2][4]*bblock[2][2]
                              - ablock[3][4]*bblock[2][3]
                              - ablock[4][4]*bblock[2][4];
  cblock[3][0] = cblock[3][0] - ablock[0][0]*bblock[3][0]
                              - ablock[1][0]*bblock[3][1]
                              - ablock[2][0]*bblock[3][2]
                              - ablock[3][0]*bblock[3][3]
                              - ablock[4][0]*bblock[3][4];
  cblock[3][1] = cblock[3][1] - ablock[0][1]*bblock[3][0]
                              - ablock[1][1]*bblock[3][1]
                              - ablock[2][1]*bblock[3][2]
                              - ablock[3][1]*bblock[3][3]
                              - ablock[4][1]*bblock[3][4];
  cblock[3][2] = cblock[3][2] - ablock[0][2]*bblock[3][0]
                              - ablock[1][2]*bblock[3][1]
                              - ablock[2][2]*bblock[3][2]
                              - ablock[3][2]*bblock[3][3]
                              - ablock[4][2]*bblock[3][4];
  cblock[3][3] = cblock[3][3] - ablock[0][3]*bblock[3][0]
                              - ablock[1][3]*bblock[3][1]
                              - ablock[2][3]*bblock[3][2]
                              - ablock[3][3]*bblock[3][3]
                              - ablock[4][3]*bblock[3][4];
  cblock[3][4] = cblock[3][4] - ablock[0][4]*bblock[3][0]
                              - ablock[1][4]*bblock[3][1]
                              - ablock[2][4]*bblock[3][2]
                              - ablock[3][4]*bblock[3][3]
                              - ablock[4][4]*bblock[3][4];
  cblock[4][0] = cblock[4][0] - ablock[0][0]*bblock[4][0]
                              - ablock[1][0]*bblock[4][1]
                              - ablock[2][0]*bblock[4][2]
                              - ablock[3][0]*bblock[4][3]
                              - ablock[4][0]*bblock[4][4];
  cblock[4][1] = cblock[4][1] - ablock[0][1]*bblock[4][0]
                              - ablock[1][1]*bblock[4][1]
                              - ablock[2][1]*bblock[4][2]
                              - ablock[3][1]*bblock[4][3]
                              - ablock[4][1]*bblock[4][4];
  cblock[4][2] = cblock[4][2] - ablock[0][2]*bblock[4][0]
                              - ablock[1][2]*bblock[4][1]
                              - ablock[2][2]*bblock[4][2]
                              - ablock[3][2]*bblock[4][3]
                              - ablock[4][2]*bblock[4][4];
  cblock[4][3] = cblock[4][3] - ablock[0][3]*bblock[4][0]
                              - ablock[1][3]*bblock[4][1]
                              - ablock[2][3]*bblock[4][2]
                              - ablock[3][3]*bblock[4][3]
                              - ablock[4][3]*bblock[4][4];
  cblock[4][4] = cblock[4][4] - ablock[0][4]*bblock[4][0]
                              - ablock[1][4]*bblock[4][1]
                              - ablock[2][4]*bblock[4][2]
                              - ablock[3][4]*bblock[4][3]
                              - ablock[4][4]*bblock[4][4];
}

inline void binvrhs(__global double lhs[5][5], __global double r[5])
{
  double pivot, coeff;

  pivot = 1.00/lhs[0][0];
  lhs[1][0] = lhs[1][0]*pivot;
  lhs[2][0] = lhs[2][0]*pivot;
  lhs[3][0] = lhs[3][0]*pivot;
  lhs[4][0] = lhs[4][0]*pivot;
  r[0]   = r[0]  *pivot;

  coeff = lhs[0][1];
  lhs[1][1]= lhs[1][1] - coeff*lhs[1][0];
  lhs[2][1]= lhs[2][1] - coeff*lhs[2][0];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][0];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][0];
  r[1]   = r[1]   - coeff*r[0];

  coeff = lhs[0][2];
  lhs[1][2]= lhs[1][2] - coeff*lhs[1][0];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][0];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][0];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][0];
  r[2]   = r[2]   - coeff*r[0];

  coeff = lhs[0][3];
  lhs[1][3]= lhs[1][3] - coeff*lhs[1][0];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][0];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][0];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][0];
  r[3]   = r[3]   - coeff*r[0];

  coeff = lhs[0][4];
  lhs[1][4]= lhs[1][4] - coeff*lhs[1][0];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][0];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][0];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][0];
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/lhs[1][1];
  lhs[2][1] = lhs[2][1]*pivot;
  lhs[3][1] = lhs[3][1]*pivot;
  lhs[4][1] = lhs[4][1]*pivot;
  r[1]   = r[1]  *pivot;

  coeff = lhs[1][0];
  lhs[2][0]= lhs[2][0] - coeff*lhs[2][1];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][1];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][1];
  r[0]   = r[0]   - coeff*r[1];

  coeff = lhs[1][2];
  lhs[2][2]= lhs[2][2] - coeff*lhs[2][1];
  lhs[3][2]= lhs[3][2] - coeff*lhs[3][1];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][1];
  r[2]   = r[2]   - coeff*r[1];

  coeff = lhs[1][3];
  lhs[2][3]= lhs[2][3] - coeff*lhs[2][1];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][1];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][1];
  r[3]   = r[3]   - coeff*r[1];

  coeff = lhs[1][4];
  lhs[2][4]= lhs[2][4] - coeff*lhs[2][1];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][1];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][1];
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/lhs[2][2];
  lhs[3][2] = lhs[3][2]*pivot;
  lhs[4][2] = lhs[4][2]*pivot;
  r[2]   = r[2]  *pivot;

  coeff = lhs[2][0];
  lhs[3][0]= lhs[3][0] - coeff*lhs[3][2];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][2];
  r[0]   = r[0]   - coeff*r[2];

  coeff = lhs[2][1];
  lhs[3][1]= lhs[3][1] - coeff*lhs[3][2];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][2];
  r[1]   = r[1]   - coeff*r[2];

  coeff = lhs[2][3];
  lhs[3][3]= lhs[3][3] - coeff*lhs[3][2];
  lhs[4][3]= lhs[4][3] - coeff*lhs[4][2];
  r[3]   = r[3]   - coeff*r[2];

  coeff = lhs[2][4];
  lhs[3][4]= lhs[3][4] - coeff*lhs[3][2];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][2];
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/lhs[3][3];
  lhs[4][3] = lhs[4][3]*pivot;
  r[3]   = r[3]  *pivot;

  coeff = lhs[3][0];
  lhs[4][0]= lhs[4][0] - coeff*lhs[4][3];
  r[0]   = r[0]   - coeff*r[3];

  coeff = lhs[3][1];
  lhs[4][1]= lhs[4][1] - coeff*lhs[4][3];
  r[1]   = r[1]   - coeff*r[3];

  coeff = lhs[3][2];
  lhs[4][2]= lhs[4][2] - coeff*lhs[4][3];
  r[2]   = r[2]   - coeff*r[3];

  coeff = lhs[3][4];
  lhs[4][4]= lhs[4][4] - coeff*lhs[4][3];
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/lhs[4][4];
  r[4]   = r[4]  *pivot;

  coeff = lhs[4][0];
  r[0]   = r[0]   - coeff*r[4];

  coeff = lhs[4][1];
  r[1]   = r[1]   - coeff*r[4];

  coeff = lhs[4][2];
  r[2]   = r[2]   - coeff*r[4];

  coeff = lhs[4][3];
  r[3]   = r[3]   - coeff*r[4];
}

__kernel void x_solve_baseline(__global double *m_qs,
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

  __global double (* g_lhs)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][3][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][3][5][5])m_lhs;

  __global double (* lhs)[3][5][5] = g_lhs[k][j];

  __global double (* g_fjac)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5])m_fjac;
  __global double (* g_njac)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5])m_njac;

  __global double (* fjac)[5][5] = g_fjac[k][j];
  __global double (* njac)[5][5] = g_njac[k][j];

  double tmp1, tmp2, tmp3;
  int isize = gp0 - 1;

  for (i = 0; i <= isize; i++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[i][0][0] = 0.0;
    fjac[i][1][0] = 1.0;
    fjac[i][2][0] = 0.0;
    fjac[i][3][0] = 0.0;
    fjac[i][4][0] = 0.0;

    fjac[i][0][1] = -(u[k][j][i][1] * tmp2 * u[k][j][i][1])
      + c2 * qs[k][j][i];
    fjac[i][1][1] = ( 2.0 - c2 ) * ( u[k][j][i][1] / u[k][j][i][0] );
    fjac[i][2][1] = - c2 * ( u[k][j][i][2] * tmp1 );
    fjac[i][3][1] = - c2 * ( u[k][j][i][3] * tmp1 );
    fjac[i][4][1] = c2;

    fjac[i][0][2] = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac[i][1][2] = u[k][j][i][2] * tmp1;
    fjac[i][2][2] = u[k][j][i][1] * tmp1;
    fjac[i][3][2] = 0.0;
    fjac[i][4][2] = 0.0;

    fjac[i][0][3] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[i][1][3] = u[k][j][i][3] * tmp1;
    fjac[i][2][3] = 0.0;
    fjac[i][3][3] = u[k][j][i][1] * tmp1;
    fjac[i][4][3] = 0.0;

    fjac[i][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * ( u[k][j][i][1] * tmp2 );
    fjac[i][1][4] = c1 *  u[k][j][i][4] * tmp1 
      - c2 * ( u[k][j][i][1]*u[k][j][i][1] * tmp2 + qs[k][j][i] );
    fjac[i][2][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][1] ) * tmp2;
    fjac[i][3][4] = - c2 * ( u[k][j][i][3]*u[k][j][i][1] ) * tmp2;
    fjac[i][4][4] = c1 * ( u[k][j][i][1] * tmp1 );

    njac[i][0][0] = 0.0;
    njac[i][1][0] = 0.0;
    njac[i][2][0] = 0.0;
    njac[i][3][0] = 0.0;
    njac[i][4][0] = 0.0;

    njac[i][0][1] = - con43 * c3c4 * tmp2 * u[k][j][i][1];
    njac[i][1][1] =   con43 * c3c4 * tmp1;
    njac[i][2][1] =   0.0;
    njac[i][3][1] =   0.0;
    njac[i][4][1] =   0.0;

    njac[i][0][2] = - c3c4 * tmp2 * u[k][j][i][2];
    njac[i][1][2] =   0.0;
    njac[i][2][2] =   c3c4 * tmp1;
    njac[i][3][2] =   0.0;
    njac[i][4][2] =   0.0;

    njac[i][0][3] = - c3c4 * tmp2 * u[k][j][i][3];
    njac[i][1][3] =   0.0;
    njac[i][2][3] =   0.0;
    njac[i][3][3] =   c3c4 * tmp1;
    njac[i][4][3] =   0.0;

    njac[i][0][4] = - ( con43 * c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[i][1][4] = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][1];
    njac[i][2][4] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[i][3][4] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac[i][4][4] = ( c1345 ) * tmp1;
  }
  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in x direction
  //---------------------------------------------------------------------
  lhsinit(lhs, isize);
  for (i = 1; i <= isize-1; i++) {
    tmp1 = dt * tx1;
    tmp2 = dt * tx2;

    lhs[i][AA][0][0] = - tmp2 * fjac[i-1][0][0]
      - tmp1 * njac[i-1][0][0]
      - tmp1 * dx1; 
    lhs[i][AA][1][0] = - tmp2 * fjac[i-1][1][0]
      - tmp1 * njac[i-1][1][0];
    lhs[i][AA][2][0] = - tmp2 * fjac[i-1][2][0]
      - tmp1 * njac[i-1][2][0];
    lhs[i][AA][3][0] = - tmp2 * fjac[i-1][3][0]
      - tmp1 * njac[i-1][3][0];
    lhs[i][AA][4][0] = - tmp2 * fjac[i-1][4][0]
      - tmp1 * njac[i-1][4][0];

    lhs[i][AA][0][1] = - tmp2 * fjac[i-1][0][1]
      - tmp1 * njac[i-1][0][1];
    lhs[i][AA][1][1] = - tmp2 * fjac[i-1][1][1]
      - tmp1 * njac[i-1][1][1]
      - tmp1 * dx2;
    lhs[i][AA][2][1] = - tmp2 * fjac[i-1][2][1]
      - tmp1 * njac[i-1][2][1];
    lhs[i][AA][3][1] = - tmp2 * fjac[i-1][3][1]
      - tmp1 * njac[i-1][3][1];
    lhs[i][AA][4][1] = - tmp2 * fjac[i-1][4][1]
      - tmp1 * njac[i-1][4][1];

    lhs[i][AA][0][2] = - tmp2 * fjac[i-1][0][2]
      - tmp1 * njac[i-1][0][2];
    lhs[i][AA][1][2] = - tmp2 * fjac[i-1][1][2]
      - tmp1 * njac[i-1][1][2];
    lhs[i][AA][2][2] = - tmp2 * fjac[i-1][2][2]
      - tmp1 * njac[i-1][2][2]
      - tmp1 * dx3;
    lhs[i][AA][3][2] = - tmp2 * fjac[i-1][3][2]
      - tmp1 * njac[i-1][3][2];
    lhs[i][AA][4][2] = - tmp2 * fjac[i-1][4][2]
      - tmp1 * njac[i-1][4][2];

    lhs[i][AA][0][3] = - tmp2 * fjac[i-1][0][3]
      - tmp1 * njac[i-1][0][3];
    lhs[i][AA][1][3] = - tmp2 * fjac[i-1][1][3]
      - tmp1 * njac[i-1][1][3];
    lhs[i][AA][2][3] = - tmp2 * fjac[i-1][2][3]
      - tmp1 * njac[i-1][2][3];
    lhs[i][AA][3][3] = - tmp2 * fjac[i-1][3][3]
      - tmp1 * njac[i-1][3][3]
      - tmp1 * dx4;
    lhs[i][AA][4][3] = - tmp2 * fjac[i-1][4][3]
      - tmp1 * njac[i-1][4][3];

    lhs[i][AA][0][4] = - tmp2 * fjac[i-1][0][4]
      - tmp1 * njac[i-1][0][4];
    lhs[i][AA][1][4] = - tmp2 * fjac[i-1][1][4]
      - tmp1 * njac[i-1][1][4];
    lhs[i][AA][2][4] = - tmp2 * fjac[i-1][2][4]
      - tmp1 * njac[i-1][2][4];
    lhs[i][AA][3][4] = - tmp2 * fjac[i-1][3][4]
      - tmp1 * njac[i-1][3][4];
    lhs[i][AA][4][4] = - tmp2 * fjac[i-1][4][4]
      - tmp1 * njac[i-1][4][4]
      - tmp1 * dx5;

    lhs[i][BB][0][0] = 1.0
      + tmp1 * 2.0 * njac[i][0][0]
      + tmp1 * 2.0 * dx1;
    lhs[i][BB][1][0] = tmp1 * 2.0 * njac[i][1][0];
    lhs[i][BB][2][0] = tmp1 * 2.0 * njac[i][2][0];
    lhs[i][BB][3][0] = tmp1 * 2.0 * njac[i][3][0];
    lhs[i][BB][4][0] = tmp1 * 2.0 * njac[i][4][0];

    lhs[i][BB][0][1] = tmp1 * 2.0 * njac[i][0][1];
    lhs[i][BB][1][1] = 1.0
      + tmp1 * 2.0 * njac[i][1][1]
      + tmp1 * 2.0 * dx2;
    lhs[i][BB][2][1] = tmp1 * 2.0 * njac[i][2][1];
    lhs[i][BB][3][1] = tmp1 * 2.0 * njac[i][3][1];
    lhs[i][BB][4][1] = tmp1 * 2.0 * njac[i][4][1];

    lhs[i][BB][0][2] = tmp1 * 2.0 * njac[i][0][2];
    lhs[i][BB][1][2] = tmp1 * 2.0 * njac[i][1][2];
    lhs[i][BB][2][2] = 1.0
      + tmp1 * 2.0 * njac[i][2][2]
      + tmp1 * 2.0 * dx3;
    lhs[i][BB][3][2] = tmp1 * 2.0 * njac[i][3][2];
    lhs[i][BB][4][2] = tmp1 * 2.0 * njac[i][4][2];

    lhs[i][BB][0][3] = tmp1 * 2.0 * njac[i][0][3];
    lhs[i][BB][1][3] = tmp1 * 2.0 * njac[i][1][3];
    lhs[i][BB][2][3] = tmp1 * 2.0 * njac[i][2][3];
    lhs[i][BB][3][3] = 1.0
      + tmp1 * 2.0 * njac[i][3][3]
      + tmp1 * 2.0 * dx4;
    lhs[i][BB][4][3] = tmp1 * 2.0 * njac[i][4][3];

    lhs[i][BB][0][4] = tmp1 * 2.0 * njac[i][0][4];
    lhs[i][BB][1][4] = tmp1 * 2.0 * njac[i][1][4];
    lhs[i][BB][2][4] = tmp1 * 2.0 * njac[i][2][4];
    lhs[i][BB][3][4] = tmp1 * 2.0 * njac[i][3][4];
    lhs[i][BB][4][4] = 1.0
      + tmp1 * 2.0 * njac[i][4][4]
      + tmp1 * 2.0 * dx5;

    lhs[i][CC][0][0] =  tmp2 * fjac[i+1][0][0]
      - tmp1 * njac[i+1][0][0]
      - tmp1 * dx1;
    lhs[i][CC][1][0] =  tmp2 * fjac[i+1][1][0]
      - tmp1 * njac[i+1][1][0];
    lhs[i][CC][2][0] =  tmp2 * fjac[i+1][2][0]
      - tmp1 * njac[i+1][2][0];
    lhs[i][CC][3][0] =  tmp2 * fjac[i+1][3][0]
      - tmp1 * njac[i+1][3][0];
    lhs[i][CC][4][0] =  tmp2 * fjac[i+1][4][0]
      - tmp1 * njac[i+1][4][0];

    lhs[i][CC][0][1] =  tmp2 * fjac[i+1][0][1]
      - tmp1 * njac[i+1][0][1];
    lhs[i][CC][1][1] =  tmp2 * fjac[i+1][1][1]
      - tmp1 * njac[i+1][1][1]
      - tmp1 * dx2;
    lhs[i][CC][2][1] =  tmp2 * fjac[i+1][2][1]
      - tmp1 * njac[i+1][2][1];
    lhs[i][CC][3][1] =  tmp2 * fjac[i+1][3][1]
      - tmp1 * njac[i+1][3][1];
    lhs[i][CC][4][1] =  tmp2 * fjac[i+1][4][1]
      - tmp1 * njac[i+1][4][1];

    lhs[i][CC][0][2] =  tmp2 * fjac[i+1][0][2]
      - tmp1 * njac[i+1][0][2];
    lhs[i][CC][1][2] =  tmp2 * fjac[i+1][1][2]
      - tmp1 * njac[i+1][1][2];
    lhs[i][CC][2][2] =  tmp2 * fjac[i+1][2][2]
      - tmp1 * njac[i+1][2][2]
      - tmp1 * dx3;
    lhs[i][CC][3][2] =  tmp2 * fjac[i+1][3][2]
      - tmp1 * njac[i+1][3][2];
    lhs[i][CC][4][2] =  tmp2 * fjac[i+1][4][2]
      - tmp1 * njac[i+1][4][2];

    lhs[i][CC][0][3] =  tmp2 * fjac[i+1][0][3]
      - tmp1 * njac[i+1][0][3];
    lhs[i][CC][1][3] =  tmp2 * fjac[i+1][1][3]
      - tmp1 * njac[i+1][1][3];
    lhs[i][CC][2][3] =  tmp2 * fjac[i+1][2][3]
      - tmp1 * njac[i+1][2][3];
    lhs[i][CC][3][3] =  tmp2 * fjac[i+1][3][3]
      - tmp1 * njac[i+1][3][3]
      - tmp1 * dx4;
    lhs[i][CC][4][3] =  tmp2 * fjac[i+1][4][3]
      - tmp1 * njac[i+1][4][3];

    lhs[i][CC][0][4] =  tmp2 * fjac[i+1][0][4]
      - tmp1 * njac[i+1][0][4];
    lhs[i][CC][1][4] =  tmp2 * fjac[i+1][1][4]
      - tmp1 * njac[i+1][1][4];
    lhs[i][CC][2][4] =  tmp2 * fjac[i+1][2][4]
      - tmp1 * njac[i+1][2][4];
    lhs[i][CC][3][4] =  tmp2 * fjac[i+1][3][4]
      - tmp1 * njac[i+1][3][4];
    lhs[i][CC][4][4] =  tmp2 * fjac[i+1][4][4]
      - tmp1 * njac[i+1][4][4]
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
  binvcrhs( lhs[0][BB], lhs[0][CC], rhs[k][j][0] );

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (i = 1; i <= isize-1; i++) {
    //-------------------------------------------------------------------
    // rhs(i) = rhs(i) - A*rhs(i-1)
    //-------------------------------------------------------------------
    matvec_sub(lhs[i][AA], rhs[k][j][i-1], rhs[k][j][i]);

    //-------------------------------------------------------------------
    // B(i) = B(i) - C(i-1)*A(i)
    //-------------------------------------------------------------------
    matmul_sub(lhs[i][AA], lhs[i-1][CC], lhs[i][BB]);


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs( lhs[i][BB], lhs[i][CC], rhs[k][j][i] );
  }

  //---------------------------------------------------------------------
  // rhs(isize) = rhs(isize) - A*rhs(isize-1)
  //---------------------------------------------------------------------
  matvec_sub(lhs[isize][AA], rhs[k][j][isize-1], rhs[k][j][isize]);

  //---------------------------------------------------------------------
  // B(isize) = B(isize) - C(isize-1)*A(isize)
  //---------------------------------------------------------------------
  matmul_sub(lhs[isize][AA], lhs[isize-1][CC], lhs[isize][BB]);

  //---------------------------------------------------------------------
  // multiply rhs() by b_inverse() and copy to rhs
  //---------------------------------------------------------------------
  binvrhs( lhs[isize][BB], rhs[k][j][isize] );

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
          - lhs[i][CC][n][m]*rhs[k][j][i+1][n];
      }
    }
  }
}

__kernel void y_solve_baseline(__global double *m_qs,
                               __global double *m_rho_i,
                               __global double *m_square,
                               __global double *m_u,
                               __global double *m_rhs,
                               __global double *m_lhs, 
                               __global double *m_fjac,
                               __global double *m_njac,
                               int gp0, int gp1, int gp2,
                               double dy1, double dy2,
                               double dy3, double dy4,
                               double dy5,
                               double c1, double c2,
                               double ty1, double ty2,
                               double con43, double c3c4,
                               double c1345, double dt,
                               int work_base,
                               int work_num_item,
                               int split_flag)
{
  int k = get_global_id(1);
  int i = get_global_id(0) + 1;
  int m, n, jsize;
  double tmp1, tmp2, tmp3;
  int j;

  if (k + work_base < 1
      || k + work_base > gp2-2
      || k >= work_num_item
      || i > gp0-2)
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

  __global double (* g_lhs)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][3][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][3][5][5])m_lhs;

  __global double (* lhs)[3][5][5] = g_lhs[k][i];

  __global double (* g_fjac)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5])m_fjac;
  __global double (* g_njac)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5])m_njac;

  __global double (* fjac)[5][5] = g_fjac[k][i];
  __global double (* njac)[5][5] = g_njac[k][i];


  jsize = gp1-1;

  for (j = 0; j <= jsize; j++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[j][0][0] = 0.0;
    fjac[j][1][0] = 0.0;
    fjac[j][2][0] = 1.0;
    fjac[j][3][0] = 0.0;
    fjac[j][4][0] = 0.0;

    fjac[j][0][1] = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac[j][1][1] = u[k][j][i][2] * tmp1;
    fjac[j][2][1] = u[k][j][i][1] * tmp1;
    fjac[j][3][1] = 0.0;
    fjac[j][4][1] = 0.0;

    fjac[j][0][2] = - ( u[k][j][i][2]*u[k][j][i][2]*tmp2)
      + c2 * qs[k][j][i];
    fjac[j][1][2] = - c2 *  u[k][j][i][1] * tmp1;
    fjac[j][2][2] = ( 2.0 - c2 ) *  u[k][j][i][2] * tmp1;
    fjac[j][3][2] = - c2 * u[k][j][i][3] * tmp1;
    fjac[j][4][2] = c2;

    fjac[j][0][3] = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[j][1][3] = 0.0;
    fjac[j][2][3] = u[k][j][i][3] * tmp1;
    fjac[j][3][3] = u[k][j][i][2] * tmp1;
    fjac[j][4][3] = 0.0;

    fjac[j][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * u[k][j][i][2] * tmp2;
    fjac[j][1][4] = - c2 * u[k][j][i][1]*u[k][j][i][2] * tmp2;
    fjac[j][2][4] = c1 * u[k][j][i][4] * tmp1 
      - c2 * ( qs[k][j][i] + u[k][j][i][2]*u[k][j][i][2] * tmp2 );
    fjac[j][3][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[j][4][4] = c1 * u[k][j][i][2] * tmp1;

    njac[j][0][0] = 0.0;
    njac[j][1][0] = 0.0;
    njac[j][2][0] = 0.0;
    njac[j][3][0] = 0.0;
    njac[j][4][0] = 0.0;

    njac[j][0][1] = - c3c4 * tmp2 * u[k][j][i][1];
    njac[j][1][1] =   c3c4 * tmp1;
    njac[j][2][1] =   0.0;
    njac[j][3][1] =   0.0;
    njac[j][4][1] =   0.0;

    njac[j][0][2] = - con43 * c3c4 * tmp2 * u[k][j][i][2];
    njac[j][1][2] =   0.0;
    njac[j][2][2] =   con43 * c3c4 * tmp1;
    njac[j][3][2] =   0.0;
    njac[j][4][2] =   0.0;

    njac[j][0][3] = - c3c4 * tmp2 * u[k][j][i][3];
    njac[j][1][3] =   0.0;
    njac[j][2][3] =   0.0;
    njac[j][3][3] =   c3c4 * tmp1;
    njac[j][4][3] =   0.0;

    njac[j][0][4] = - (  c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( con43 * c3c4
          - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[j][1][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
    njac[j][2][4] = ( con43 * c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[j][3][4] = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac[j][4][4] = ( c1345 ) * tmp1;
  }

  //---------------------------------------------------------------------
  // now joacobians set, so form left hand side in y direction
  //---------------------------------------------------------------------
  lhsinit(lhs, jsize);
  for (j = 1; j <= jsize-1; j++) {
    tmp1 = dt * ty1;
    tmp2 = dt * ty2;

    lhs[j][AA][0][0] = - tmp2 * fjac[j-1][0][0]
      - tmp1 * njac[j-1][0][0]
      - tmp1 * dy1; 
    lhs[j][AA][1][0] = - tmp2 * fjac[j-1][1][0]
      - tmp1 * njac[j-1][1][0];
    lhs[j][AA][2][0] = - tmp2 * fjac[j-1][2][0]
      - tmp1 * njac[j-1][2][0];
    lhs[j][AA][3][0] = - tmp2 * fjac[j-1][3][0]
      - tmp1 * njac[j-1][3][0];
    lhs[j][AA][4][0] = - tmp2 * fjac[j-1][4][0]
      - tmp1 * njac[j-1][4][0];

    lhs[j][AA][0][1] = - tmp2 * fjac[j-1][0][1]
      - tmp1 * njac[j-1][0][1];
    lhs[j][AA][1][1] = - tmp2 * fjac[j-1][1][1]
      - tmp1 * njac[j-1][1][1]
      - tmp1 * dy2;
    lhs[j][AA][2][1] = - tmp2 * fjac[j-1][2][1]
      - tmp1 * njac[j-1][2][1];
    lhs[j][AA][3][1] = - tmp2 * fjac[j-1][3][1]
      - tmp1 * njac[j-1][3][1];
    lhs[j][AA][4][1] = - tmp2 * fjac[j-1][4][1]
      - tmp1 * njac[j-1][4][1];

    lhs[j][AA][0][2] = - tmp2 * fjac[j-1][0][2]
      - tmp1 * njac[j-1][0][2];
    lhs[j][AA][1][2] = - tmp2 * fjac[j-1][1][2]
      - tmp1 * njac[j-1][1][2];
    lhs[j][AA][2][2] = - tmp2 * fjac[j-1][2][2]
      - tmp1 * njac[j-1][2][2]
      - tmp1 * dy3;
    lhs[j][AA][3][2] = - tmp2 * fjac[j-1][3][2]
      - tmp1 * njac[j-1][3][2];
    lhs[j][AA][4][2] = - tmp2 * fjac[j-1][4][2]
      - tmp1 * njac[j-1][4][2];

    lhs[j][AA][0][3] = - tmp2 * fjac[j-1][0][3]
      - tmp1 * njac[j-1][0][3];
    lhs[j][AA][1][3] = - tmp2 * fjac[j-1][1][3]
      - tmp1 * njac[j-1][1][3];
    lhs[j][AA][2][3] = - tmp2 * fjac[j-1][2][3]
      - tmp1 * njac[j-1][2][3];
    lhs[j][AA][3][3] = - tmp2 * fjac[j-1][3][3]
      - tmp1 * njac[j-1][3][3]
      - tmp1 * dy4;
    lhs[j][AA][4][3] = - tmp2 * fjac[j-1][4][3]
      - tmp1 * njac[j-1][4][3];

    lhs[j][AA][0][4] = - tmp2 * fjac[j-1][0][4]
      - tmp1 * njac[j-1][0][4];
    lhs[j][AA][1][4] = - tmp2 * fjac[j-1][1][4]
      - tmp1 * njac[j-1][1][4];
    lhs[j][AA][2][4] = - tmp2 * fjac[j-1][2][4]
      - tmp1 * njac[j-1][2][4];
    lhs[j][AA][3][4] = - tmp2 * fjac[j-1][3][4]
      - tmp1 * njac[j-1][3][4];
    lhs[j][AA][4][4] = - tmp2 * fjac[j-1][4][4]
      - tmp1 * njac[j-1][4][4]
      - tmp1 * dy5;

    lhs[j][BB][0][0] = 1.0
      + tmp1 * 2.0 * njac[j][0][0]
      + tmp1 * 2.0 * dy1;
    lhs[j][BB][1][0] = tmp1 * 2.0 * njac[j][1][0];
    lhs[j][BB][2][0] = tmp1 * 2.0 * njac[j][2][0];
    lhs[j][BB][3][0] = tmp1 * 2.0 * njac[j][3][0];
    lhs[j][BB][4][0] = tmp1 * 2.0 * njac[j][4][0];

    lhs[j][BB][0][1] = tmp1 * 2.0 * njac[j][0][1];
    lhs[j][BB][1][1] = 1.0
      + tmp1 * 2.0 * njac[j][1][1]
      + tmp1 * 2.0 * dy2;
    lhs[j][BB][2][1] = tmp1 * 2.0 * njac[j][2][1];
    lhs[j][BB][3][1] = tmp1 * 2.0 * njac[j][3][1];
    lhs[j][BB][4][1] = tmp1 * 2.0 * njac[j][4][1];

    lhs[j][BB][0][2] = tmp1 * 2.0 * njac[j][0][2];
    lhs[j][BB][1][2] = tmp1 * 2.0 * njac[j][1][2];
    lhs[j][BB][2][2] = 1.0
      + tmp1 * 2.0 * njac[j][2][2]
      + tmp1 * 2.0 * dy3;
    lhs[j][BB][3][2] = tmp1 * 2.0 * njac[j][3][2];
    lhs[j][BB][4][2] = tmp1 * 2.0 * njac[j][4][2];

    lhs[j][BB][0][3] = tmp1 * 2.0 * njac[j][0][3];
    lhs[j][BB][1][3] = tmp1 * 2.0 * njac[j][1][3];
    lhs[j][BB][2][3] = tmp1 * 2.0 * njac[j][2][3];
    lhs[j][BB][3][3] = 1.0
      + tmp1 * 2.0 * njac[j][3][3]
      + tmp1 * 2.0 * dy4;
    lhs[j][BB][4][3] = tmp1 * 2.0 * njac[j][4][3];

    lhs[j][BB][0][4] = tmp1 * 2.0 * njac[j][0][4];
    lhs[j][BB][1][4] = tmp1 * 2.0 * njac[j][1][4];
    lhs[j][BB][2][4] = tmp1 * 2.0 * njac[j][2][4];
    lhs[j][BB][3][4] = tmp1 * 2.0 * njac[j][3][4];
    lhs[j][BB][4][4] = 1.0
      + tmp1 * 2.0 * njac[j][4][4] 
      + tmp1 * 2.0 * dy5;

    lhs[j][CC][0][0] =  tmp2 * fjac[j+1][0][0]
      - tmp1 * njac[j+1][0][0]
      - tmp1 * dy1;
    lhs[j][CC][1][0] =  tmp2 * fjac[j+1][1][0]
      - tmp1 * njac[j+1][1][0];
    lhs[j][CC][2][0] =  tmp2 * fjac[j+1][2][0]
      - tmp1 * njac[j+1][2][0];
    lhs[j][CC][3][0] =  tmp2 * fjac[j+1][3][0]
      - tmp1 * njac[j+1][3][0];
    lhs[j][CC][4][0] =  tmp2 * fjac[j+1][4][0]
      - tmp1 * njac[j+1][4][0];

    lhs[j][CC][0][1] =  tmp2 * fjac[j+1][0][1]
      - tmp1 * njac[j+1][0][1];
    lhs[j][CC][1][1] =  tmp2 * fjac[j+1][1][1]
      - tmp1 * njac[j+1][1][1]
      - tmp1 * dy2;
    lhs[j][CC][2][1] =  tmp2 * fjac[j+1][2][1]
      - tmp1 * njac[j+1][2][1];
    lhs[j][CC][3][1] =  tmp2 * fjac[j+1][3][1]
      - tmp1 * njac[j+1][3][1];
    lhs[j][CC][4][1] =  tmp2 * fjac[j+1][4][1]
      - tmp1 * njac[j+1][4][1];

    lhs[j][CC][0][2] =  tmp2 * fjac[j+1][0][2]
      - tmp1 * njac[j+1][0][2];
    lhs[j][CC][1][2] =  tmp2 * fjac[j+1][1][2]
      - tmp1 * njac[j+1][1][2];
    lhs[j][CC][2][2] =  tmp2 * fjac[j+1][2][2]
      - tmp1 * njac[j+1][2][2]
      - tmp1 * dy3;
    lhs[j][CC][3][2] =  tmp2 * fjac[j+1][3][2]
      - tmp1 * njac[j+1][3][2];
    lhs[j][CC][4][2] =  tmp2 * fjac[j+1][4][2]
      - tmp1 * njac[j+1][4][2];

    lhs[j][CC][0][3] =  tmp2 * fjac[j+1][0][3]
      - tmp1 * njac[j+1][0][3];
    lhs[j][CC][1][3] =  tmp2 * fjac[j+1][1][3]
      - tmp1 * njac[j+1][1][3];
    lhs[j][CC][2][3] =  tmp2 * fjac[j+1][2][3]
      - tmp1 * njac[j+1][2][3];
    lhs[j][CC][3][3] =  tmp2 * fjac[j+1][3][3]
      - tmp1 * njac[j+1][3][3]
      - tmp1 * dy4;
    lhs[j][CC][4][3] =  tmp2 * fjac[j+1][4][3]
      - tmp1 * njac[j+1][4][3];

    lhs[j][CC][0][4] =  tmp2 * fjac[j+1][0][4]
      - tmp1 * njac[j+1][0][4];
    lhs[j][CC][1][4] =  tmp2 * fjac[j+1][1][4]
      - tmp1 * njac[j+1][1][4];
    lhs[j][CC][2][4] =  tmp2 * fjac[j+1][2][4]
      - tmp1 * njac[j+1][2][4];
    lhs[j][CC][3][4] =  tmp2 * fjac[j+1][3][4]
      - tmp1 * njac[j+1][3][4];
    lhs[j][CC][4][4] =  tmp2 * fjac[j+1][4][4]
      - tmp1 * njac[j+1][4][4]
      - tmp1 * dy5;
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
  // c'(JMAX) and rhs'(JMAX) will be sent to next cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // multiply c[k][0][i] by b_inverse and copy back to c
  // multiply rhs(0) by b_inverse(0) and copy to rhs
  //---------------------------------------------------------------------
  binvcrhs( lhs[0][BB], lhs[0][CC], rhs[k][0][i] );

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (j = 1; j <= jsize-1; j++) {
    //-------------------------------------------------------------------
    // subtract A*lhs_vector(j-1) from lhs_vector(j)
    // 
    // rhs(j) = rhs(j) - A*rhs(j-1)
    //-------------------------------------------------------------------
    matvec_sub(lhs[j][AA], rhs[k][j-1][i], rhs[k][j][i]);

    //-------------------------------------------------------------------
    // B(j) = B(j) - C(j-1)*A(j)
    //-------------------------------------------------------------------
    matmul_sub(lhs[j][AA], lhs[j-1][CC], lhs[j][BB]);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs( lhs[j][BB], lhs[j][CC], rhs[k][j][i] );
  }

  //---------------------------------------------------------------------
  // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
  //---------------------------------------------------------------------
  matvec_sub(lhs[jsize][AA], rhs[k][jsize-1][i], rhs[k][jsize][i]);

  //---------------------------------------------------------------------
  // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
  // matmul_sub(AA,i,jsize,k,c,
  // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
  //---------------------------------------------------------------------
  matmul_sub(lhs[jsize][AA], lhs[jsize-1][CC], lhs[jsize][BB]);

  //---------------------------------------------------------------------
  // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
  //---------------------------------------------------------------------
  binvrhs( lhs[jsize][BB], rhs[k][jsize][i] );

  //---------------------------------------------------------------------
  // back solve: if last cell, then generate U(jsize)=rhs(jsize)
  // else assume U(jsize) is loaded in un pack backsub_info
  // so just use it
  // after u(jstart) will be sent to next cell
  //---------------------------------------------------------------------
  for (j = jsize-1; j >= 0; j--) {
    for (m = 0; m < BLOCK_SIZE; m++) {
      for (n = 0; n < BLOCK_SIZE; n++) {
        rhs[k][j][i][m] = rhs[k][j][i][m] 
          - lhs[j][CC][n][m]*rhs[k][j+1][i][n];
      }
    }
  }
}


__kernel void z_solve_data_gen_baseline(__global double *m_u,
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

__kernel void z_solve_baseline(__global double *m_qs,
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

  __global double (* g_lhs)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][3][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][3][5][5])m_lhs;

  __global double (* lhs)[3][5][5] = g_lhs[j][i];

  __global double (* g_fjac)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5])m_fjac;
  __global double (* g_njac)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5]
    = (__global double (*)[PROBLEM_SIZE-1][PROBLEM_SIZE+1][5][5])m_njac;

  __global double (* fjac)[5][5] = g_fjac[j][i];
  __global double (* njac)[5][5] = g_njac[j][i];

  for (k = 0; k <= ksize; k++) {
    tmp1 = 1.0 / u[k][j][i][0];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac[k][0][0] = 0.0;
    fjac[k][1][0] = 0.0;
    fjac[k][2][0] = 0.0;
    fjac[k][3][0] = 1.0;
    fjac[k][4][0] = 0.0;

    fjac[k][0][1] = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[k][1][1] = u[k][j][i][3] * tmp1;
    fjac[k][2][1] = 0.0;
    fjac[k][3][1] = u[k][j][i][1] * tmp1;
    fjac[k][4][1] = 0.0;

    fjac[k][0][2] = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[k][1][2] = 0.0;
    fjac[k][2][2] = u[k][j][i][3] * tmp1;
    fjac[k][3][2] = u[k][j][i][2] * tmp1;
    fjac[k][4][2] = 0.0;

    fjac[k][0][3] = - (u[k][j][i][3]*u[k][j][i][3] * tmp2 ) 
      + c2 * qs[k][j][i];
    fjac[k][1][3] = - c2 *  u[k][j][i][1] * tmp1;
    fjac[k][2][3] = - c2 *  u[k][j][i][2] * tmp1;
    fjac[k][3][3] = ( 2.0 - c2 ) *  u[k][j][i][3] * tmp1;
    fjac[k][4][3] = c2;

    fjac[k][0][4] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * u[k][j][i][3] * tmp2;
    fjac[k][1][4] = - c2 * ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac[k][2][4] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac[k][3][4] = c1 * ( u[k][j][i][4] * tmp1 )
      - c2 * ( qs[k][j][i] + u[k][j][i][3]*u[k][j][i][3] * tmp2 );
    fjac[k][4][4] = c1 * u[k][j][i][3] * tmp1;

    njac[k][0][0] = 0.0;
    njac[k][1][0] = 0.0;
    njac[k][2][0] = 0.0;
    njac[k][3][0] = 0.0;
    njac[k][4][0] = 0.0;

    njac[k][0][1] = - c3c4 * tmp2 * u[k][j][i][1];
    njac[k][1][1] =   c3c4 * tmp1;
    njac[k][2][1] =   0.0;
    njac[k][3][1] =   0.0;
    njac[k][4][1] =   0.0;

    njac[k][0][2] = - c3c4 * tmp2 * u[k][j][i][2];
    njac[k][1][2] =   0.0;
    njac[k][2][2] =   c3c4 * tmp1;
    njac[k][3][2] =   0.0;
    njac[k][4][2] =   0.0;

    njac[k][0][3] = - con43 * c3c4 * tmp2 * u[k][j][i][3];
    njac[k][1][3] =   0.0;
    njac[k][2][3] =   0.0;
    njac[k][3][3] =   con43 * c3 * c4 * tmp1;
    njac[k][4][3] =   0.0;

    njac[k][0][4] = - (  c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( con43 * c3c4
          - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac[k][1][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
    njac[k][2][4] = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac[k][3][4] = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][3];
    njac[k][4][4] = ( c1345 )* tmp1;
  }

  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in z direction
  //---------------------------------------------------------------------
  lhsinit(lhs, ksize);
  for (k = 1; k <= ksize-1; k++) {
    tmp1 = dt * tz1;
    tmp2 = dt * tz2;

    lhs[k][AA][0][0] = - tmp2 * fjac[k-1][0][0]
      - tmp1 * njac[k-1][0][0]
      - tmp1 * dz1; 
    lhs[k][AA][1][0] = - tmp2 * fjac[k-1][1][0]
      - tmp1 * njac[k-1][1][0];
    lhs[k][AA][2][0] = - tmp2 * fjac[k-1][2][0]
      - tmp1 * njac[k-1][2][0];
    lhs[k][AA][3][0] = - tmp2 * fjac[k-1][3][0]
      - tmp1 * njac[k-1][3][0];
    lhs[k][AA][4][0] = - tmp2 * fjac[k-1][4][0]
      - tmp1 * njac[k-1][4][0];

    lhs[k][AA][0][1] = - tmp2 * fjac[k-1][0][1]
      - tmp1 * njac[k-1][0][1];
    lhs[k][AA][1][1] = - tmp2 * fjac[k-1][1][1]
      - tmp1 * njac[k-1][1][1]
      - tmp1 * dz2;
    lhs[k][AA][2][1] = - tmp2 * fjac[k-1][2][1]
      - tmp1 * njac[k-1][2][1];
    lhs[k][AA][3][1] = - tmp2 * fjac[k-1][3][1]
      - tmp1 * njac[k-1][3][1];
    lhs[k][AA][4][1] = - tmp2 * fjac[k-1][4][1]
      - tmp1 * njac[k-1][4][1];

    lhs[k][AA][0][2] = - tmp2 * fjac[k-1][0][2]
      - tmp1 * njac[k-1][0][2];
    lhs[k][AA][1][2] = - tmp2 * fjac[k-1][1][2]
      - tmp1 * njac[k-1][1][2];
    lhs[k][AA][2][2] = - tmp2 * fjac[k-1][2][2]
      - tmp1 * njac[k-1][2][2]
      - tmp1 * dz3;
    lhs[k][AA][3][2] = - tmp2 * fjac[k-1][3][2]
      - tmp1 * njac[k-1][3][2];
    lhs[k][AA][4][2] = - tmp2 * fjac[k-1][4][2]
      - tmp1 * njac[k-1][4][2];

    lhs[k][AA][0][3] = - tmp2 * fjac[k-1][0][3]
      - tmp1 * njac[k-1][0][3];
    lhs[k][AA][1][3] = - tmp2 * fjac[k-1][1][3]
      - tmp1 * njac[k-1][1][3];
    lhs[k][AA][2][3] = - tmp2 * fjac[k-1][2][3]
      - tmp1 * njac[k-1][2][3];
    lhs[k][AA][3][3] = - tmp2 * fjac[k-1][3][3]
      - tmp1 * njac[k-1][3][3]
      - tmp1 * dz4;
    lhs[k][AA][4][3] = - tmp2 * fjac[k-1][4][3]
      - tmp1 * njac[k-1][4][3];

    lhs[k][AA][0][4] = - tmp2 * fjac[k-1][0][4]
      - tmp1 * njac[k-1][0][4];
    lhs[k][AA][1][4] = - tmp2 * fjac[k-1][1][4]
      - tmp1 * njac[k-1][1][4];
    lhs[k][AA][2][4] = - tmp2 * fjac[k-1][2][4]
      - tmp1 * njac[k-1][2][4];
    lhs[k][AA][3][4] = - tmp2 * fjac[k-1][3][4]
      - tmp1 * njac[k-1][3][4];
    lhs[k][AA][4][4] = - tmp2 * fjac[k-1][4][4]
      - tmp1 * njac[k-1][4][4]
      - tmp1 * dz5;

    lhs[k][BB][0][0] = 1.0
      + tmp1 * 2.0 * njac[k][0][0]
      + tmp1 * 2.0 * dz1;
    lhs[k][BB][1][0] = tmp1 * 2.0 * njac[k][1][0];
    lhs[k][BB][2][0] = tmp1 * 2.0 * njac[k][2][0];
    lhs[k][BB][3][0] = tmp1 * 2.0 * njac[k][3][0];
    lhs[k][BB][4][0] = tmp1 * 2.0 * njac[k][4][0];

    lhs[k][BB][0][1] = tmp1 * 2.0 * njac[k][0][1];
    lhs[k][BB][1][1] = 1.0
      + tmp1 * 2.0 * njac[k][1][1]
      + tmp1 * 2.0 * dz2;
    lhs[k][BB][2][1] = tmp1 * 2.0 * njac[k][2][1];
    lhs[k][BB][3][1] = tmp1 * 2.0 * njac[k][3][1];
    lhs[k][BB][4][1] = tmp1 * 2.0 * njac[k][4][1];

    lhs[k][BB][0][2] = tmp1 * 2.0 * njac[k][0][2];
    lhs[k][BB][1][2] = tmp1 * 2.0 * njac[k][1][2];
    lhs[k][BB][2][2] = 1.0
      + tmp1 * 2.0 * njac[k][2][2]
      + tmp1 * 2.0 * dz3;
    lhs[k][BB][3][2] = tmp1 * 2.0 * njac[k][3][2];
    lhs[k][BB][4][2] = tmp1 * 2.0 * njac[k][4][2];

    lhs[k][BB][0][3] = tmp1 * 2.0 * njac[k][0][3];
    lhs[k][BB][1][3] = tmp1 * 2.0 * njac[k][1][3];
    lhs[k][BB][2][3] = tmp1 * 2.0 * njac[k][2][3];
    lhs[k][BB][3][3] = 1.0
      + tmp1 * 2.0 * njac[k][3][3]
      + tmp1 * 2.0 * dz4;
    lhs[k][BB][4][3] = tmp1 * 2.0 * njac[k][4][3];

    lhs[k][BB][0][4] = tmp1 * 2.0 * njac[k][0][4];
    lhs[k][BB][1][4] = tmp1 * 2.0 * njac[k][1][4];
    lhs[k][BB][2][4] = tmp1 * 2.0 * njac[k][2][4];
    lhs[k][BB][3][4] = tmp1 * 2.0 * njac[k][3][4];
    lhs[k][BB][4][4] = 1.0
      + tmp1 * 2.0 * njac[k][4][4] 
      + tmp1 * 2.0 * dz5;

    lhs[k][CC][0][0] =  tmp2 * fjac[k+1][0][0]
      - tmp1 * njac[k+1][0][0]
      - tmp1 * dz1;
    lhs[k][CC][1][0] =  tmp2 * fjac[k+1][1][0]
      - tmp1 * njac[k+1][1][0];
    lhs[k][CC][2][0] =  tmp2 * fjac[k+1][2][0]
      - tmp1 * njac[k+1][2][0];
    lhs[k][CC][3][0] =  tmp2 * fjac[k+1][3][0]
      - tmp1 * njac[k+1][3][0];
    lhs[k][CC][4][0] =  tmp2 * fjac[k+1][4][0]
      - tmp1 * njac[k+1][4][0];

    lhs[k][CC][0][1] =  tmp2 * fjac[k+1][0][1]
      - tmp1 * njac[k+1][0][1];
    lhs[k][CC][1][1] =  tmp2 * fjac[k+1][1][1]
      - tmp1 * njac[k+1][1][1]
      - tmp1 * dz2;
    lhs[k][CC][2][1] =  tmp2 * fjac[k+1][2][1]
      - tmp1 * njac[k+1][2][1];
    lhs[k][CC][3][1] =  tmp2 * fjac[k+1][3][1]
      - tmp1 * njac[k+1][3][1];
    lhs[k][CC][4][1] =  tmp2 * fjac[k+1][4][1]
      - tmp1 * njac[k+1][4][1];

    lhs[k][CC][0][2] =  tmp2 * fjac[k+1][0][2]
      - tmp1 * njac[k+1][0][2];
    lhs[k][CC][1][2] =  tmp2 * fjac[k+1][1][2]
      - tmp1 * njac[k+1][1][2];
    lhs[k][CC][2][2] =  tmp2 * fjac[k+1][2][2]
      - tmp1 * njac[k+1][2][2]
      - tmp1 * dz3;
    lhs[k][CC][3][2] =  tmp2 * fjac[k+1][3][2]
      - tmp1 * njac[k+1][3][2];
    lhs[k][CC][4][2] =  tmp2 * fjac[k+1][4][2]
      - tmp1 * njac[k+1][4][2];

    lhs[k][CC][0][3] =  tmp2 * fjac[k+1][0][3]
      - tmp1 * njac[k+1][0][3];
    lhs[k][CC][1][3] =  tmp2 * fjac[k+1][1][3]
      - tmp1 * njac[k+1][1][3];
    lhs[k][CC][2][3] =  tmp2 * fjac[k+1][2][3]
      - tmp1 * njac[k+1][2][3];
    lhs[k][CC][3][3] =  tmp2 * fjac[k+1][3][3]
      - tmp1 * njac[k+1][3][3]
      - tmp1 * dz4;
    lhs[k][CC][4][3] =  tmp2 * fjac[k+1][4][3]
      - tmp1 * njac[k+1][4][3];

    lhs[k][CC][0][4] =  tmp2 * fjac[k+1][0][4]
      - tmp1 * njac[k+1][0][4];
    lhs[k][CC][1][4] =  tmp2 * fjac[k+1][1][4]
      - tmp1 * njac[k+1][1][4];
    lhs[k][CC][2][4] =  tmp2 * fjac[k+1][2][4]
      - tmp1 * njac[k+1][2][4];
    lhs[k][CC][3][4] =  tmp2 * fjac[k+1][3][4]
      - tmp1 * njac[k+1][3][4];
    lhs[k][CC][4][4] =  tmp2 * fjac[k+1][4][4]
      - tmp1 * njac[k+1][4][4]
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
  binvcrhs( lhs[0][BB], lhs[0][CC], rhs[0][j][i] );

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
    matvec_sub(lhs[k][AA], rhs[k-1][j][i], rhs[k][j][i]);

    //-------------------------------------------------------------------
    // B(k) = B(k) - C(k-1)*A(k)
    // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
    //-------------------------------------------------------------------
    matmul_sub(lhs[k][AA], lhs[k-1][CC], lhs[k][BB]);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs( lhs[k][BB], lhs[k][CC], rhs[k][j][i] );
  }

  //---------------------------------------------------------------------
  // Now finish up special cases for last cell
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
  //---------------------------------------------------------------------
  matvec_sub(lhs[ksize][AA], rhs[ksize-1][j][i], rhs[ksize][j][i]);

  //---------------------------------------------------------------------
  // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
  // matmul_sub(AA,i,j,ksize,c,
  // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
  //---------------------------------------------------------------------
  matmul_sub(lhs[ksize][AA], lhs[ksize-1][CC], lhs[ksize][BB]);

  //---------------------------------------------------------------------
  // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
  //---------------------------------------------------------------------
  binvrhs( lhs[ksize][BB], rhs[ksize][j][i] );

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
          - lhs[k][CC][n][m]*rhs[k+1][j][i][n];
      }
    }
  }
}

