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

__device__
void lhsinit_x(double *lhs, 
               int size, int k, int j, int WORK_NUM_ITEM_DEFAULT)
{
#define t_lhs(a, b, c, d, e, f) lhs[(((((a)*3 + (b))*5 + (c))*5 + (d))*WORK_NUM_ITEM_DEFAULT + (e))*(PROBLEM_SIZE-1) + (f)]
  int i, m, n;

  i = size;
  //---------------------------------------------------------------------
  // zero the whole left hand side for starters
  //---------------------------------------------------------------------
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      t_lhs(0, 0, n, m, k, j) = 0.0;
      t_lhs(0, 1, n, m, k, j) = 0.0;
      t_lhs(0, 2, n, m, k, j) = 0.0;
      t_lhs(i, 0, n, m, k, j) = 0.0;
      t_lhs(i, 1, n, m, k, j) = 0.0;
      t_lhs(i, 2, n, m, k, j) = 0.0;

    }
  }

  //---------------------------------------------------------------------
  // next, set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    t_lhs(0, 1, m, m, k, j) = 1.0;
    t_lhs(i, 1, m, m, k, j) = 1.0;

#undef t_lhs
  }
}

__device__
void binvcrhs_x(double *lhs,
                double *c, 
                double r[5], 
                int k, int j, int WORK_NUM_ITEM_DEFAULT)
{
#define t_lhs(a, b, c, d) lhs[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
#define t_c(a, b, d, e) c[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (d))*(PROBLEM_SIZE-1) + (e)]
  double pivot, coeff;

  pivot = 1.00/t_lhs(0, 0, k, j);
  t_lhs(1, 0, k, j) = t_lhs(1, 0, k, j)*pivot;
  t_lhs(2, 0, k, j) = t_lhs(2, 0, k, j)*pivot;
  t_lhs(3, 0, k, j) = t_lhs(3, 0, k, j)*pivot;
  t_lhs(4, 0, k, j) = t_lhs(4, 0, k, j)*pivot;
  t_c(0, 0, k, j) = t_c(0, 0, k, j)*pivot;
  t_c(1, 0, k, j) = t_c(1, 0, k, j)*pivot;
  t_c(2, 0, k, j) = t_c(2, 0, k, j)*pivot;
  t_c(3, 0, k, j) = t_c(3, 0, k, j)*pivot;
  t_c(4, 0, k, j) = t_c(4, 0, k, j)*pivot;
  r[0]   = r[0]  *pivot;

  coeff = t_lhs(0, 1, k, j);
  t_lhs(1, 1, k, j)= t_lhs(1, 1, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 1, k, j)= t_lhs(2, 1, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 1, k, j)= t_lhs(3, 1, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 1, k, j)= t_lhs(4, 1, k, j) - coeff*t_lhs(4, 0, k, j);
  t_c(0, 1, k, j) = t_c(0, 1, k, j) - coeff*t_c(0, 0, k, j);
  t_c(1, 1, k, j) = t_c(1, 1, k, j) - coeff*t_c(1, 0, k, j);
  t_c(2, 1, k, j) = t_c(2, 1, k, j) - coeff*t_c(2, 0, k, j);
  t_c(3, 1, k, j) = t_c(3, 1, k, j) - coeff*t_c(3, 0, k, j);
  t_c(4, 1, k, j) = t_c(4, 1, k, j) - coeff*t_c(4, 0, k, j);
  r[1]   = r[1]   - coeff*r[0];

  coeff = t_lhs(0, 2, k, j);
  t_lhs(1, 2, k, j)= t_lhs(1, 2, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 2, k, j)= t_lhs(2, 2, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 2, k, j)= t_lhs(3, 2, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 2, k, j)= t_lhs(4, 2, k, j) - coeff*t_lhs(4, 0, k, j);
  t_c(0, 2, k, j) = t_c(0, 2, k, j) - coeff*t_c(0, 0, k, j);
  t_c(1, 2, k, j) = t_c(1, 2, k, j) - coeff*t_c(1, 0, k, j);
  t_c(2, 2, k, j) = t_c(2, 2, k, j) - coeff*t_c(2, 0, k, j);
  t_c(3, 2, k, j) = t_c(3, 2, k, j) - coeff*t_c(3, 0, k, j);
  t_c(4, 2, k, j) = t_c(4, 2, k, j) - coeff*t_c(4, 0, k, j);
  r[2]   = r[2]   - coeff*r[0];

  coeff = t_lhs(0, 3, k, j);
  t_lhs(1, 3, k, j)= t_lhs(1, 3, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 3, k, j)= t_lhs(2, 3, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 3, k, j)= t_lhs(3, 3, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 3, k, j)= t_lhs(4, 3, k, j) - coeff*t_lhs(4, 0, k, j);
  t_c(0, 3, k, j) = t_c(0, 3, k, j) - coeff*t_c(0, 0, k, j);
  t_c(1, 3, k, j) = t_c(1, 3, k, j) - coeff*t_c(1, 0, k, j);
  t_c(2, 3, k, j) = t_c(2, 3, k, j) - coeff*t_c(2, 0, k, j);
  t_c(3, 3, k, j) = t_c(3, 3, k, j) - coeff*t_c(3, 0, k, j);
  t_c(4, 3, k, j) = t_c(4, 3, k, j) - coeff*t_c(4, 0, k, j);
  r[3]   = r[3]   - coeff*r[0];

  coeff = t_lhs(0, 4, k, j);
  t_lhs(1, 4, k, j)= t_lhs(1, 4, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 4, k, j)= t_lhs(2, 4, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 4, k, j)= t_lhs(3, 4, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 0, k, j);
  t_c(0, 4, k, j) = t_c(0, 4, k, j) - coeff*t_c(0, 0, k, j);
  t_c(1, 4, k, j) = t_c(1, 4, k, j) - coeff*t_c(1, 0, k, j);
  t_c(2, 4, k, j) = t_c(2, 4, k, j) - coeff*t_c(2, 0, k, j);
  t_c(3, 4, k, j) = t_c(3, 4, k, j) - coeff*t_c(3, 0, k, j);
  t_c(4, 4, k, j) = t_c(4, 4, k, j) - coeff*t_c(4, 0, k, j);
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/t_lhs(1, 1, k, j);
  t_lhs(2, 1, k, j) = t_lhs(2, 1, k, j)*pivot;
  t_lhs(3, 1, k, j) = t_lhs(3, 1, k, j)*pivot;
  t_lhs(4, 1, k, j) = t_lhs(4, 1, k, j)*pivot;
  t_c(0, 1, k, j) = t_c(0, 1, k, j)*pivot;
  t_c(1, 1, k, j) = t_c(1, 1, k, j)*pivot;
  t_c(2, 1, k, j) = t_c(2, 1, k, j)*pivot;
  t_c(3, 1, k, j) = t_c(3, 1, k, j)*pivot;
  t_c(4, 1, k, j) = t_c(4, 1, k, j)*pivot;
  r[1]   = r[1]  *pivot;

  coeff = t_lhs(1, 0, k, j);
  t_lhs(2, 0, k, j)= t_lhs(2, 0, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 0, k, j)= t_lhs(3, 0, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 0, k, j)= t_lhs(4, 0, k, j) - coeff*t_lhs(4, 1, k, j);
  t_c(0, 0, k, j) = t_c(0, 0, k, j) - coeff*t_c(0, 1, k, j);
  t_c(1, 0, k, j) = t_c(1, 0, k, j) - coeff*t_c(1, 1, k, j);
  t_c(2, 0, k, j) = t_c(2, 0, k, j) - coeff*t_c(2, 1, k, j);
  t_c(3, 0, k, j) = t_c(3, 0, k, j) - coeff*t_c(3, 1, k, j);
  t_c(4, 0, k, j) = t_c(4, 0, k, j) - coeff*t_c(4, 1, k, j);
  r[0]   = r[0]   - coeff*r[1];

  coeff = t_lhs(1, 2, k, j);
  t_lhs(2, 2, k, j)= t_lhs(2, 2, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 2, k, j)= t_lhs(3, 2, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 2, k, j)= t_lhs(4, 2, k, j) - coeff*t_lhs(4, 1, k, j);
  t_c(0, 2, k, j) = t_c(0, 2, k, j) - coeff*t_c(0, 1, k, j);
  t_c(1, 2, k, j) = t_c(1, 2, k, j) - coeff*t_c(1, 1, k, j);
  t_c(2, 2, k, j) = t_c(2, 2, k, j) - coeff*t_c(2, 1, k, j);
  t_c(3, 2, k, j) = t_c(3, 2, k, j) - coeff*t_c(3, 1, k, j);
  t_c(4, 2, k, j) = t_c(4, 2, k, j) - coeff*t_c(4, 1, k, j);
  r[2]   = r[2]   - coeff*r[1];

  coeff = t_lhs(1, 3, k, j);
  t_lhs(2, 3, k, j)= t_lhs(2, 3, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 3, k, j)= t_lhs(3, 3, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 3, k, j)= t_lhs(4, 3, k, j) - coeff*t_lhs(4, 1, k, j);
  t_c(0, 3, k, j) = t_c(0, 3, k, j) - coeff*t_c(0, 1, k, j);
  t_c(1, 3, k, j) = t_c(1, 3, k, j) - coeff*t_c(1, 1, k, j);
  t_c(2, 3, k, j) = t_c(2, 3, k, j) - coeff*t_c(2, 1, k, j);
  t_c(3, 3, k, j) = t_c(3, 3, k, j) - coeff*t_c(3, 1, k, j);
  t_c(4, 3, k, j) = t_c(4, 3, k, j) - coeff*t_c(4, 1, k, j);
  r[3]   = r[3]   - coeff*r[1];

  coeff = t_lhs(1, 4, k, j);
  t_lhs(2, 4, k, j)= t_lhs(2, 4, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 4, k, j)= t_lhs(3, 4, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 1, k, j);
  t_c(0, 4, k, j) = t_c(0, 4, k, j) - coeff*t_c(0, 1, k, j);
  t_c(1, 4, k, j) = t_c(1, 4, k, j) - coeff*t_c(1, 1, k, j);
  t_c(2, 4, k, j) = t_c(2, 4, k, j) - coeff*t_c(2, 1, k, j);
  t_c(3, 4, k, j) = t_c(3, 4, k, j) - coeff*t_c(3, 1, k, j);
  t_c(4, 4, k, j) = t_c(4, 4, k, j) - coeff*t_c(4, 1, k, j);
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/t_lhs(2, 2, k, j);
  t_lhs(3, 2, k, j) = t_lhs(3, 2, k, j)*pivot;
  t_lhs(4, 2, k, j) = t_lhs(4, 2, k, j)*pivot;
  t_c(0, 2, k, j) = t_c(0, 2, k, j)*pivot;
  t_c(1, 2, k, j) = t_c(1, 2, k, j)*pivot;
  t_c(2, 2, k, j) = t_c(2, 2, k, j)*pivot;
  t_c(3, 2, k, j) = t_c(3, 2, k, j)*pivot;
  t_c(4, 2, k, j) = t_c(4, 2, k, j)*pivot;
  r[2]   = r[2]  *pivot;

  coeff = t_lhs(2, 0, k, j);
  t_lhs(3, 0, k, j)= t_lhs(3, 0, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 0, k, j)= t_lhs(4, 0, k, j) - coeff*t_lhs(4, 2, k, j);
  t_c(0, 0, k, j) = t_c(0, 0, k, j) - coeff*t_c(0, 2, k, j);
  t_c(1, 0, k, j) = t_c(1, 0, k, j) - coeff*t_c(1, 2, k, j);
  t_c(2, 0, k, j) = t_c(2, 0, k, j) - coeff*t_c(2, 2, k, j);
  t_c(3, 0, k, j) = t_c(3, 0, k, j) - coeff*t_c(3, 2, k, j);
  t_c(4, 0, k, j) = t_c(4, 0, k, j) - coeff*t_c(4, 2, k, j);
  r[0]   = r[0]   - coeff*r[2];

  coeff = t_lhs(2, 1, k, j);
  t_lhs(3, 1, k, j)= t_lhs(3, 1, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 1, k, j)= t_lhs(4, 1, k, j) - coeff*t_lhs(4, 2, k, j);
  t_c(0, 1, k, j) = t_c(0, 1, k, j) - coeff*t_c(0, 2, k, j);
  t_c(1, 1, k, j) = t_c(1, 1, k, j) - coeff*t_c(1, 2, k, j);
  t_c(2, 1, k, j) = t_c(2, 1, k, j) - coeff*t_c(2, 2, k, j);
  t_c(3, 1, k, j) = t_c(3, 1, k, j) - coeff*t_c(3, 2, k, j);
  t_c(4, 1, k, j) = t_c(4, 1, k, j) - coeff*t_c(4, 2, k, j);
  r[1]   = r[1]   - coeff*r[2];

  coeff = t_lhs(2, 3, k, j);
  t_lhs(3, 3, k, j)= t_lhs(3, 3, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 3, k, j)= t_lhs(4, 3, k, j) - coeff*t_lhs(4, 2, k, j);
  t_c(0, 3, k, j) = t_c(0, 3, k, j) - coeff*t_c(0, 2, k, j);
  t_c(1, 3, k, j) = t_c(1, 3, k, j) - coeff*t_c(1, 2, k, j);
  t_c(2, 3, k, j) = t_c(2, 3, k, j) - coeff*t_c(2, 2, k, j);
  t_c(3, 3, k, j) = t_c(3, 3, k, j) - coeff*t_c(3, 2, k, j);
  t_c(4, 3, k, j) = t_c(4, 3, k, j) - coeff*t_c(4, 2, k, j);
  r[3]   = r[3]   - coeff*r[2];

  coeff = t_lhs(2, 4, k, j);
  t_lhs(3, 4, k, j)= t_lhs(3, 4, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 2, k, j);
  t_c(0, 4, k, j) = t_c(0, 4, k, j) - coeff*t_c(0, 2, k, j);
  t_c(1, 4, k, j) = t_c(1, 4, k, j) - coeff*t_c(1, 2, k, j);
  t_c(2, 4, k, j) = t_c(2, 4, k, j) - coeff*t_c(2, 2, k, j);
  t_c(3, 4, k, j) = t_c(3, 4, k, j) - coeff*t_c(3, 2, k, j);
  t_c(4, 4, k, j) = t_c(4, 4, k, j) - coeff*t_c(4, 2, k, j);
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/t_lhs(3, 3, k, j);
  t_lhs(4, 3, k, j) = t_lhs(4, 3, k, j)*pivot;
  t_c(0, 3, k, j) = t_c(0, 3, k, j)*pivot;
  t_c(1, 3, k, j) = t_c(1, 3, k, j)*pivot;
  t_c(2, 3, k, j) = t_c(2, 3, k, j)*pivot;
  t_c(3, 3, k, j) = t_c(3, 3, k, j)*pivot;
  t_c(4, 3, k, j) = t_c(4, 3, k, j)*pivot;
  r[3]   = r[3]  *pivot;

  coeff = t_lhs(3, 0, k, j);
  t_lhs(4, 0, k, j)= t_lhs(4, 0, k, j) - coeff*t_lhs(4, 3, k, j);
  t_c(0, 0, k, j) = t_c(0, 0, k, j) - coeff*t_c(0, 3, k, j);
  t_c(1, 0, k, j) = t_c(1, 0, k, j) - coeff*t_c(1, 3, k, j);
  t_c(2, 0, k, j) = t_c(2, 0, k, j) - coeff*t_c(2, 3, k, j);
  t_c(3, 0, k, j) = t_c(3, 0, k, j) - coeff*t_c(3, 3, k, j);
  t_c(4, 0, k, j) = t_c(4, 0, k, j) - coeff*t_c(4, 3, k, j);
  r[0]   = r[0]   - coeff*r[3];

  coeff = t_lhs(3, 1, k, j);
  t_lhs(4, 1, k, j)= t_lhs(4, 1, k, j) - coeff*t_lhs(4, 3, k, j);
  t_c(0, 1, k, j) = t_c(0, 1, k, j) - coeff*t_c(0, 3, k, j);
  t_c(1, 1, k, j) = t_c(1, 1, k, j) - coeff*t_c(1, 3, k, j);
  t_c(2, 1, k, j) = t_c(2, 1, k, j) - coeff*t_c(2, 3, k, j);
  t_c(3, 1, k, j) = t_c(3, 1, k, j) - coeff*t_c(3, 3, k, j);
  t_c(4, 1, k, j) = t_c(4, 1, k, j) - coeff*t_c(4, 3, k, j);
  r[1]   = r[1]   - coeff*r[3];

  coeff = t_lhs(3, 2, k, j);
  t_lhs(4, 2, k, j)= t_lhs(4, 2, k, j) - coeff*t_lhs(4, 3, k, j);
  t_c(0, 2, k, j) = t_c(0, 2, k, j) - coeff*t_c(0, 3, k, j);
  t_c(1, 2, k, j) = t_c(1, 2, k, j) - coeff*t_c(1, 3, k, j);
  t_c(2, 2, k, j) = t_c(2, 2, k, j) - coeff*t_c(2, 3, k, j);
  t_c(3, 2, k, j) = t_c(3, 2, k, j) - coeff*t_c(3, 3, k, j);
  t_c(4, 2, k, j) = t_c(4, 2, k, j) - coeff*t_c(4, 3, k, j);
  r[2]   = r[2]   - coeff*r[3];

  coeff = t_lhs(3, 4, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 3, k, j);
  t_c(0, 4, k, j) = t_c(0, 4, k, j) - coeff*t_c(0, 3, k, j);
  t_c(1, 4, k, j) = t_c(1, 4, k, j) - coeff*t_c(1, 3, k, j);
  t_c(2, 4, k, j) = t_c(2, 4, k, j) - coeff*t_c(2, 3, k, j);
  t_c(3, 4, k, j) = t_c(3, 4, k, j) - coeff*t_c(3, 3, k, j);
  t_c(4, 4, k, j) = t_c(4, 4, k, j) - coeff*t_c(4, 3, k, j);
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/t_lhs(4, 4, k, j);
  t_c(0, 4, k, j) = t_c(0, 4, k, j)*pivot;
  t_c(1, 4, k, j) = t_c(1, 4, k, j)*pivot;
  t_c(2, 4, k, j) = t_c(2, 4, k, j)*pivot;
  t_c(3, 4, k, j) = t_c(3, 4, k, j)*pivot;
  t_c(4, 4, k, j) = t_c(4, 4, k, j)*pivot;
  r[4]   = r[4]  *pivot;

  coeff = t_lhs(4, 0, k, j);
  t_c(0, 0, k, j) = t_c(0, 0, k, j) - coeff*t_c(0, 4, k, j);
  t_c(1, 0, k, j) = t_c(1, 0, k, j) - coeff*t_c(1, 4, k, j);
  t_c(2, 0, k, j) = t_c(2, 0, k, j) - coeff*t_c(2, 4, k, j);
  t_c(3, 0, k, j) = t_c(3, 0, k, j) - coeff*t_c(3, 4, k, j);
  t_c(4, 0, k, j) = t_c(4, 0, k, j) - coeff*t_c(4, 4, k, j);
  r[0]   = r[0]   - coeff*r[4];

  coeff = t_lhs(4, 1, k, j);
  t_c(0, 1, k, j) = t_c(0, 1, k, j) - coeff*t_c(0, 4, k, j);
  t_c(1, 1, k, j) = t_c(1, 1, k, j) - coeff*t_c(1, 4, k, j);
  t_c(2, 1, k, j) = t_c(2, 1, k, j) - coeff*t_c(2, 4, k, j);
  t_c(3, 1, k, j) = t_c(3, 1, k, j) - coeff*t_c(3, 4, k, j);
  t_c(4, 1, k, j) = t_c(4, 1, k, j) - coeff*t_c(4, 4, k, j);
  r[1]   = r[1]   - coeff*r[4];

  coeff = t_lhs(4, 2, k, j);
  t_c(0, 2, k, j) = t_c(0, 2, k, j) - coeff*t_c(0, 4, k, j);
  t_c(1, 2, k, j) = t_c(1, 2, k, j) - coeff*t_c(1, 4, k, j);
  t_c(2, 2, k, j) = t_c(2, 2, k, j) - coeff*t_c(2, 4, k, j);
  t_c(3, 2, k, j) = t_c(3, 2, k, j) - coeff*t_c(3, 4, k, j);
  t_c(4, 2, k, j) = t_c(4, 2, k, j) - coeff*t_c(4, 4, k, j);
  r[2]   = r[2]   - coeff*r[4];

  coeff = t_lhs(4, 3, k, j);
  t_c(0, 3, k, j) = t_c(0, 3, k, j) - coeff*t_c(0, 4, k, j);
  t_c(1, 3, k, j) = t_c(1, 3, k, j) - coeff*t_c(1, 4, k, j);
  t_c(2, 3, k, j) = t_c(2, 3, k, j) - coeff*t_c(2, 4, k, j);
  t_c(3, 3, k, j) = t_c(3, 3, k, j) - coeff*t_c(3, 4, k, j);
  t_c(4, 3, k, j) = t_c(4, 3, k, j) - coeff*t_c(4, 4, k, j);
  r[3]   = r[3]   - coeff*r[4];

#undef t_lhs
#undef t_c
}

__device__
void matvec_sub_x(double *ablock,
                  double avec[5], 
                  double bvec[5],
                  int k, int j, int WORK_NUM_ITEM_DEFAULT)
{ 
#define t_ablock(a, b, c, d) ablock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
  //---------------------------------------------------------------------
  // rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] 
  // $                  - lhs[ia][ablock][0][i]*
  //---------------------------------------------------------------------
  bvec[0] = bvec[0] - t_ablock(0, 0, k, j)*avec[0]
                    - t_ablock(1, 0, k, j)*avec[1]
                    - t_ablock(2, 0, k, j)*avec[2]
                    - t_ablock(3, 0, k, j)*avec[3]
                    - t_ablock(4, 0, k, j)*avec[4];
  bvec[1] = bvec[1] - t_ablock(0, 1, k, j)*avec[0]
                    - t_ablock(1, 1, k, j)*avec[1]
                    - t_ablock(2, 1, k, j)*avec[2]
                    - t_ablock(3, 1, k, j)*avec[3]
                    - t_ablock(4, 1, k, j)*avec[4];
  bvec[2] = bvec[2] - t_ablock(0, 2, k, j)*avec[0]
                    - t_ablock(1, 2, k, j)*avec[1]
                    - t_ablock(2, 2, k, j)*avec[2]
                    - t_ablock(3, 2, k, j)*avec[3]
                    - t_ablock(4, 2, k, j)*avec[4];
  bvec[3] = bvec[3] - t_ablock(0, 3, k, j)*avec[0]
                    - t_ablock(1, 3, k, j)*avec[1]
                    - t_ablock(2, 3, k, j)*avec[2]
                    - t_ablock(3, 3, k, j)*avec[3]
                    - t_ablock(4, 3, k, j)*avec[4];
  bvec[4] = bvec[4] - t_ablock(0, 4, k, j)*avec[0]
                    - t_ablock(1, 4, k, j)*avec[1]
                    - t_ablock(2, 4, k, j)*avec[2]
                    - t_ablock(3, 4, k, j)*avec[3]
                    - t_ablock(4, 4, k, j)*avec[4];
#undef t_ablock
}

__device__
void matmul_sub_x(double *ablock,
                  double *bblock,
                  double *cblock,
                  int k, int j, int WORK_NUM_ITEM_DEFAULT)
{
#define t_ablock(a, b, c, d) ablock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
#define t_bblock(a, b, c, d) bblock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
#define t_cblock(a, b, c, d) cblock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]

  t_cblock(0, 0, k, j) = t_cblock(0, 0, k, j) - t_ablock(0, 0, k, j)*t_bblock(0, 0, k, j)
                              - t_ablock(1, 0, k, j)*t_bblock(0, 1, k, j)
                              - t_ablock(2, 0, k, j)*t_bblock(0, 2, k, j)
                              - t_ablock(3, 0, k, j)*t_bblock(0, 3, k, j)
                              - t_ablock(4, 0, k, j)*t_bblock(0, 4, k, j);
  t_cblock(0, 1, k, j) = t_cblock(0, 1, k, j) - t_ablock(0, 1, k, j)*t_bblock(0, 0, k, j)
                              - t_ablock(1, 1, k, j)*t_bblock(0, 1, k, j)
                              - t_ablock(2, 1, k, j)*t_bblock(0, 2, k, j)
                              - t_ablock(3, 1, k, j)*t_bblock(0, 3, k, j)
                              - t_ablock(4, 1, k, j)*t_bblock(0, 4, k, j);
  t_cblock(0, 2, k, j) = t_cblock(0, 2, k, j) - t_ablock(0, 2, k, j)*t_bblock(0, 0, k, j)
                              - t_ablock(1, 2, k, j)*t_bblock(0, 1, k, j)
                              - t_ablock(2, 2, k, j)*t_bblock(0, 2, k, j)
                              - t_ablock(3, 2, k, j)*t_bblock(0, 3, k, j)
                              - t_ablock(4, 2, k, j)*t_bblock(0, 4, k, j);
  t_cblock(0, 3, k, j) = t_cblock(0, 3, k, j) - t_ablock(0, 3, k, j)*t_bblock(0, 0, k, j)
                              - t_ablock(1, 3, k, j)*t_bblock(0, 1, k, j)
                              - t_ablock(2, 3, k, j)*t_bblock(0, 2, k, j)
                              - t_ablock(3, 3, k, j)*t_bblock(0, 3, k, j)
                              - t_ablock(4, 3, k, j)*t_bblock(0, 4, k, j);
  t_cblock(0, 4, k, j) = t_cblock(0, 4, k, j) - t_ablock(0, 4, k, j)*t_bblock(0, 0, k, j)
                              - t_ablock(1, 4, k, j)*t_bblock(0, 1, k, j)
                              - t_ablock(2, 4, k, j)*t_bblock(0, 2, k, j)
                              - t_ablock(3, 4, k, j)*t_bblock(0, 3, k, j)
                              - t_ablock(4, 4, k, j)*t_bblock(0, 4, k, j);
  t_cblock(1, 0, k, j) = t_cblock(1, 0, k, j) - t_ablock(0, 0, k, j)*t_bblock(1, 0, k, j)
                              - t_ablock(1, 0, k, j)*t_bblock(1, 1, k, j)
                              - t_ablock(2, 0, k, j)*t_bblock(1, 2, k, j)
                              - t_ablock(3, 0, k, j)*t_bblock(1, 3, k, j)
                              - t_ablock(4, 0, k, j)*t_bblock(1, 4, k, j);
  t_cblock(1, 1, k, j) = t_cblock(1, 1, k, j) - t_ablock(0, 1, k, j)*t_bblock(1, 0, k, j)
                              - t_ablock(1, 1, k, j)*t_bblock(1, 1, k, j)
                              - t_ablock(2, 1, k, j)*t_bblock(1, 2, k, j)
                              - t_ablock(3, 1, k, j)*t_bblock(1, 3, k, j)
                              - t_ablock(4, 1, k, j)*t_bblock(1, 4, k, j);
  t_cblock(1, 2, k, j) = t_cblock(1, 2, k, j) - t_ablock(0, 2, k, j)*t_bblock(1, 0, k, j)
                              - t_ablock(1, 2, k, j)*t_bblock(1, 1, k, j)
                              - t_ablock(2, 2, k, j)*t_bblock(1, 2, k, j)
                              - t_ablock(3, 2, k, j)*t_bblock(1, 3, k, j)
                              - t_ablock(4, 2, k, j)*t_bblock(1, 4, k, j);
  t_cblock(1, 3, k, j) = t_cblock(1, 3, k, j) - t_ablock(0, 3, k, j)*t_bblock(1, 0, k, j)
                              - t_ablock(1, 3, k, j)*t_bblock(1, 1, k, j)
                              - t_ablock(2, 3, k, j)*t_bblock(1, 2, k, j)
                              - t_ablock(3, 3, k, j)*t_bblock(1, 3, k, j)
                              - t_ablock(4, 3, k, j)*t_bblock(1, 4, k, j);
  t_cblock(1, 4, k, j) = t_cblock(1, 4, k, j) - t_ablock(0, 4, k, j)*t_bblock(1, 0, k, j)
                              - t_ablock(1, 4, k, j)*t_bblock(1, 1, k, j)
                              - t_ablock(2, 4, k, j)*t_bblock(1, 2, k, j)
                              - t_ablock(3, 4, k, j)*t_bblock(1, 3, k, j)
                              - t_ablock(4, 4, k, j)*t_bblock(1, 4, k, j);
  t_cblock(2, 0, k, j) = t_cblock(2, 0, k, j) - t_ablock(0, 0, k, j)*t_bblock(2, 0, k, j)
                              - t_ablock(1, 0, k, j)*t_bblock(2, 1, k, j)
                              - t_ablock(2, 0, k, j)*t_bblock(2, 2, k, j)
                              - t_ablock(3, 0, k, j)*t_bblock(2, 3, k, j)
                              - t_ablock(4, 0, k, j)*t_bblock(2, 4, k, j);
  t_cblock(2, 1, k, j) = t_cblock(2, 1, k, j) - t_ablock(0, 1, k, j)*t_bblock(2, 0, k, j)
                              - t_ablock(1, 1, k, j)*t_bblock(2, 1, k, j)
                              - t_ablock(2, 1, k, j)*t_bblock(2, 2, k, j)
                              - t_ablock(3, 1, k, j)*t_bblock(2, 3, k, j)
                              - t_ablock(4, 1, k, j)*t_bblock(2, 4, k, j);
  t_cblock(2, 2, k, j) = t_cblock(2, 2, k, j) - t_ablock(0, 2, k, j)*t_bblock(2, 0, k, j)
                              - t_ablock(1, 2, k, j)*t_bblock(2, 1, k, j)
                              - t_ablock(2, 2, k, j)*t_bblock(2, 2, k, j)
                              - t_ablock(3, 2, k, j)*t_bblock(2, 3, k, j)
                              - t_ablock(4, 2, k, j)*t_bblock(2, 4, k, j);
  t_cblock(2, 3, k, j) = t_cblock(2, 3, k, j) - t_ablock(0, 3, k, j)*t_bblock(2, 0, k, j)
                              - t_ablock(1, 3, k, j)*t_bblock(2, 1, k, j)
                              - t_ablock(2, 3, k, j)*t_bblock(2, 2, k, j)
                              - t_ablock(3, 3, k, j)*t_bblock(2, 3, k, j)
                              - t_ablock(4, 3, k, j)*t_bblock(2, 4, k, j);
  t_cblock(2, 4, k, j) = t_cblock(2, 4, k, j) - t_ablock(0, 4, k, j)*t_bblock(2, 0, k, j)
                              - t_ablock(1, 4, k, j)*t_bblock(2, 1, k, j)
                              - t_ablock(2, 4, k, j)*t_bblock(2, 2, k, j)
                              - t_ablock(3, 4, k, j)*t_bblock(2, 3, k, j)
                              - t_ablock(4, 4, k, j)*t_bblock(2, 4, k, j);
  t_cblock(3, 0, k, j) = t_cblock(3, 0, k, j) - t_ablock(0, 0, k, j)*t_bblock(3, 0, k, j)
                              - t_ablock(1, 0, k, j)*t_bblock(3, 1, k, j)
                              - t_ablock(2, 0, k, j)*t_bblock(3, 2, k, j)
                              - t_ablock(3, 0, k, j)*t_bblock(3, 3, k, j)
                              - t_ablock(4, 0, k, j)*t_bblock(3, 4, k, j);
  t_cblock(3, 1, k, j) = t_cblock(3, 1, k, j) - t_ablock(0, 1, k, j)*t_bblock(3, 0, k, j)
                              - t_ablock(1, 1, k, j)*t_bblock(3, 1, k, j)
                              - t_ablock(2, 1, k, j)*t_bblock(3, 2, k, j)
                              - t_ablock(3, 1, k, j)*t_bblock(3, 3, k, j)
                              - t_ablock(4, 1, k, j)*t_bblock(3, 4, k, j);
  t_cblock(3, 2, k, j) = t_cblock(3, 2, k, j) - t_ablock(0, 2, k, j)*t_bblock(3, 0, k, j)
                              - t_ablock(1, 2, k, j)*t_bblock(3, 1, k, j)
                              - t_ablock(2, 2, k, j)*t_bblock(3, 2, k, j)
                              - t_ablock(3, 2, k, j)*t_bblock(3, 3, k, j)
                              - t_ablock(4, 2, k, j)*t_bblock(3, 4, k, j);
  t_cblock(3, 3, k, j) = t_cblock(3, 3, k, j) - t_ablock(0, 3, k, j)*t_bblock(3, 0, k, j)
                              - t_ablock(1, 3, k, j)*t_bblock(3, 1, k, j)
                              - t_ablock(2, 3, k, j)*t_bblock(3, 2, k, j)
                              - t_ablock(3, 3, k, j)*t_bblock(3, 3, k, j)
                              - t_ablock(4, 3, k, j)*t_bblock(3, 4, k, j);
  t_cblock(3, 4, k, j) = t_cblock(3, 4, k, j) - t_ablock(0, 4, k, j)*t_bblock(3, 0, k, j)
                              - t_ablock(1, 4, k, j)*t_bblock(3, 1, k, j)
                              - t_ablock(2, 4, k, j)*t_bblock(3, 2, k, j)
                              - t_ablock(3, 4, k, j)*t_bblock(3, 3, k, j)
                              - t_ablock(4, 4, k, j)*t_bblock(3, 4, k, j);
  t_cblock(4, 0, k, j) = t_cblock(4, 0, k, j) - t_ablock(0, 0, k, j)*t_bblock(4, 0, k, j)
                              - t_ablock(1, 0, k, j)*t_bblock(4, 1, k, j)
                              - t_ablock(2, 0, k, j)*t_bblock(4, 2, k, j)
                              - t_ablock(3, 0, k, j)*t_bblock(4, 3, k, j)
                              - t_ablock(4, 0, k, j)*t_bblock(4, 4, k, j);
  t_cblock(4, 1, k, j) = t_cblock(4, 1, k, j) - t_ablock(0, 1, k, j)*t_bblock(4, 0, k, j)
                              - t_ablock(1, 1, k, j)*t_bblock(4, 1, k, j)
                              - t_ablock(2, 1, k, j)*t_bblock(4, 2, k, j)
                              - t_ablock(3, 1, k, j)*t_bblock(4, 3, k, j)
                              - t_ablock(4, 1, k, j)*t_bblock(4, 4, k, j);
  t_cblock(4, 2, k, j) = t_cblock(4, 2, k, j) - t_ablock(0, 2, k, j)*t_bblock(4, 0, k, j)
                              - t_ablock(1, 2, k, j)*t_bblock(4, 1, k, j)
                              - t_ablock(2, 2, k, j)*t_bblock(4, 2, k, j)
                              - t_ablock(3, 2, k, j)*t_bblock(4, 3, k, j)
                              - t_ablock(4, 2, k, j)*t_bblock(4, 4, k, j);
  t_cblock(4, 3, k, j) = t_cblock(4, 3, k, j) - t_ablock(0, 3, k, j)*t_bblock(4, 0, k, j)
                              - t_ablock(1, 3, k, j)*t_bblock(4, 1, k, j)
                              - t_ablock(2, 3, k, j)*t_bblock(4, 2, k, j)
                              - t_ablock(3, 3, k, j)*t_bblock(4, 3, k, j)
                              - t_ablock(4, 3, k, j)*t_bblock(4, 4, k, j);
  t_cblock(4, 4, k, j) = t_cblock(4, 4, k, j) - t_ablock(0, 4, k, j)*t_bblock(4, 0, k, j)
                              - t_ablock(1, 4, k, j)*t_bblock(4, 1, k, j)
                              - t_ablock(2, 4, k, j)*t_bblock(4, 2, k, j)
                              - t_ablock(3, 4, k, j)*t_bblock(4, 3, k, j)
                              - t_ablock(4, 4, k, j)*t_bblock(4, 4, k, j);
#undef t_ablock
#undef t_bblock
#undef t_cblock
}

__device__
void binvrhs_x(double *lhs,
              double r[5],
              int k, int j, int WORK_NUM_ITEM_DEFAULT)
{
#define t_lhs(a, b, c, d) lhs[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
  double pivot, coeff;

  pivot = 1.00/t_lhs(0, 0, k, j);
  t_lhs(1, 0, k, j) = t_lhs(1, 0, k, j)*pivot;
  t_lhs(2, 0, k, j) = t_lhs(2, 0, k, j)*pivot;
  t_lhs(3, 0, k, j) = t_lhs(3, 0, k, j)*pivot;
  t_lhs(4, 0, k, j) = t_lhs(4, 0, k, j)*pivot;
  r[0]   = r[0]  *pivot;

  coeff = t_lhs(0, 1, k, j);
  t_lhs(1, 1, k, j)= t_lhs(1, 1, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 1, k, j)= t_lhs(2, 1, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 1, k, j)= t_lhs(3, 1, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 1, k, j)= t_lhs(4, 1, k, j) - coeff*t_lhs(4, 0, k, j);
  r[1]   = r[1]   - coeff*r[0];

  coeff = t_lhs(0, 2, k, j);
  t_lhs(1, 2, k, j)= t_lhs(1, 2, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 2, k, j)= t_lhs(2, 2, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 2, k, j)= t_lhs(3, 2, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 2, k, j)= t_lhs(4, 2, k, j) - coeff*t_lhs(4, 0, k, j);
  r[2]   = r[2]   - coeff*r[0];

  coeff = t_lhs(0, 3, k, j);
  t_lhs(1, 3, k, j)= t_lhs(1, 3, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 3, k, j)= t_lhs(2, 3, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 3, k, j)= t_lhs(3, 3, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 3, k, j)= t_lhs(4, 3, k, j) - coeff*t_lhs(4, 0, k, j);
  r[3]   = r[3]   - coeff*r[0];

  coeff = t_lhs(0, 4, k, j);
  t_lhs(1, 4, k, j)= t_lhs(1, 4, k, j) - coeff*t_lhs(1, 0, k, j);
  t_lhs(2, 4, k, j)= t_lhs(2, 4, k, j) - coeff*t_lhs(2, 0, k, j);
  t_lhs(3, 4, k, j)= t_lhs(3, 4, k, j) - coeff*t_lhs(3, 0, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 0, k, j);
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/t_lhs(1, 1, k, j);
  t_lhs(2, 1, k, j) = t_lhs(2, 1, k, j)*pivot;
  t_lhs(3, 1, k, j) = t_lhs(3, 1, k, j)*pivot;
  t_lhs(4, 1, k, j) = t_lhs(4, 1, k, j)*pivot;
  r[1]   = r[1]  *pivot;

  coeff = t_lhs(1, 0, k, j);
  t_lhs(2, 0, k, j)= t_lhs(2, 0, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 0, k, j)= t_lhs(3, 0, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 0, k, j)= t_lhs(4, 0, k, j) - coeff*t_lhs(4, 1, k, j);
  r[0]   = r[0]   - coeff*r[1];

  coeff = t_lhs(1, 2, k, j);
  t_lhs(2, 2, k, j)= t_lhs(2, 2, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 2, k, j)= t_lhs(3, 2, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 2, k, j)= t_lhs(4, 2, k, j) - coeff*t_lhs(4, 1, k, j);
  r[2]   = r[2]   - coeff*r[1];

  coeff = t_lhs(1, 3, k, j);
  t_lhs(2, 3, k, j)= t_lhs(2, 3, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 3, k, j)= t_lhs(3, 3, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 3, k, j)= t_lhs(4, 3, k, j) - coeff*t_lhs(4, 1, k, j);
  r[3]   = r[3]   - coeff*r[1];

  coeff = t_lhs(1, 4, k, j);
  t_lhs(2, 4, k, j)= t_lhs(2, 4, k, j) - coeff*t_lhs(2, 1, k, j);
  t_lhs(3, 4, k, j)= t_lhs(3, 4, k, j) - coeff*t_lhs(3, 1, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 1, k, j);
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/t_lhs(2, 2, k, j);
  t_lhs(3, 2, k, j) = t_lhs(3, 2, k, j)*pivot;
  t_lhs(4, 2, k, j) = t_lhs(4, 2, k, j)*pivot;
  r[2]   = r[2]  *pivot;

  coeff = t_lhs(2, 0, k, j);
  t_lhs(3, 0, k, j)= t_lhs(3, 0, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 0, k, j)= t_lhs(4, 0, k, j) - coeff*t_lhs(4, 2, k, j);
  r[0]   = r[0]   - coeff*r[2];

  coeff = t_lhs(2, 1, k, j);
  t_lhs(3, 1, k, j)= t_lhs(3, 1, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 1, k, j)= t_lhs(4, 1, k, j) - coeff*t_lhs(4, 2, k, j);
  r[1]   = r[1]   - coeff*r[2];

  coeff = t_lhs(2, 3, k, j);
  t_lhs(3, 3, k, j)= t_lhs(3, 3, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 3, k, j)= t_lhs(4, 3, k, j) - coeff*t_lhs(4, 2, k, j);
  r[3]   = r[3]   - coeff*r[2];

  coeff = t_lhs(2, 4, k, j);
  t_lhs(3, 4, k, j)= t_lhs(3, 4, k, j) - coeff*t_lhs(3, 2, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 2, k, j);
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/t_lhs(3, 3, k, j);
  t_lhs(4, 3, k, j) = t_lhs(4, 3, k, j)*pivot;
  r[3]   = r[3]  *pivot;

  coeff = t_lhs(3, 0, k, j);
  t_lhs(4, 0, k, j)= t_lhs(4, 0, k, j) - coeff*t_lhs(4, 3, k, j);
  r[0]   = r[0]   - coeff*r[3];

  coeff = t_lhs(3, 1, k, j);
  t_lhs(4, 1, k, j)= t_lhs(4, 1, k, j) - coeff*t_lhs(4, 3, k, j);
  r[1]   = r[1]   - coeff*r[3];

  coeff = t_lhs(3, 2, k, j);
  t_lhs(4, 2, k, j)= t_lhs(4, 2, k, j) - coeff*t_lhs(4, 3, k, j);
  r[2]   = r[2]   - coeff*r[3];

  coeff = t_lhs(3, 4, k, j);
  t_lhs(4, 4, k, j)= t_lhs(4, 4, k, j) - coeff*t_lhs(4, 3, k, j);
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/t_lhs(4, 4, k, j);
  r[4]   = r[4]  *pivot;

  coeff = t_lhs(4, 0, k, j);
  r[0]   = r[0]   - coeff*r[4];

  coeff = t_lhs(4, 1, k, j);
  r[1]   = r[1]   - coeff*r[4];

  coeff = t_lhs(4, 2, k, j);
  r[2]   = r[2]   - coeff*r[4];

  coeff = t_lhs(4, 3, k, j);
  r[3]   = r[3]   - coeff*r[4];
#undef t_lhs
}

__launch_bounds__(min(PROBLEM_SIZE-2, MAX_THREAD_DIM_0))
__global__
void k_x_solve_memlayout(double *m_qs,
                         double *m_rho_i,
                         double *m_square,
                         double *m_u,
                         double *m_rhs,
                         double *m_lhs, 
                         double *m_fjac,
                         double *m_njac,
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
                         int split_flag,
                         int WORK_NUM_ITEM_DEFAULT)
{
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int i, m, n;

  if (k + work_base < 1
      || k + work_base > gp2-2
      || k >= work_num_item
      || j > gp1-2)
    return;

  if (split_flag) k += 2;
  else k += work_base;

  double (* qs)[JMAXP+1][IMAXP+1]
    = (double (*) [JMAXP+1][IMAXP+1])m_qs;
  double (* rho_i)[JMAXP+1][IMAXP+1]
    = (double (*) [JMAXP+1][IMAXP+1])m_rho_i;
  double (* square)[JMAXP+1][IMAXP+1]
    = (double (*) [JMAXP+1][IMAXP+1]) m_square; 
  double (* u)[JMAXP+1][IMAXP+1][5]
    = (double (*) [JMAXP+1][IMAXP+1][5])m_u;
  double (* rhs)[JMAXP+1][IMAXP+1][5]
    = (double (*) [JMAXP+1][IMAXP+1][5])m_rhs;

#define lhs(a, b, c, d, e, f) m_lhs[(((((a)*3 + (b))*5 + (c))*5 + (d))*WORK_NUM_ITEM_DEFAULT + (e))*(PROBLEM_SIZE-1) + (f)]
#define fjac(a, b, c, d, e) m_fjac[((((a)*5 + (b))*5 + (c))*WORK_NUM_ITEM_DEFAULT + (d))*(PROBLEM_SIZE-1) + (e)]
#define njac(a, b, c, d, e) m_njac[((((a)*5 + (b))*5 + (c))*WORK_NUM_ITEM_DEFAULT + (d))*(PROBLEM_SIZE-1) + (e)]

  double tmp1, tmp2, tmp3;
  int isize = gp0 - 1;

  for (i = 0; i <= isize; i++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac(i, 0, 0, k, j) = 0.0;
    fjac(i, 1, 0, k, j) = 1.0;
    fjac(i, 2, 0, k, j) = 0.0;
    fjac(i, 3, 0, k, j) = 0.0;
    fjac(i, 4, 0, k, j) = 0.0;

    fjac(i, 0, 1, k, j) = -(u[k][j][i][1] * tmp2 * u[k][j][i][1])
      + c2 * qs[k][j][i];
    fjac(i, 1, 1, k, j) = ( 2.0 - c2 ) * ( u[k][j][i][1] / u[k][j][i][0] );
    fjac(i, 2, 1, k, j) = - c2 * ( u[k][j][i][2] * tmp1 );
    fjac(i, 3, 1, k, j) = - c2 * ( u[k][j][i][3] * tmp1 );
    fjac(i, 4, 1, k, j) = c2;

    fjac(i, 0, 2, k, j) = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac(i, 1, 2, k, j) = u[k][j][i][2] * tmp1;
    fjac(i, 2, 2, k, j) = u[k][j][i][1] * tmp1;
    fjac(i, 3, 2, k, j) = 0.0;
    fjac(i, 4, 2, k, j) = 0.0;

    fjac(i, 0, 3, k, j) = - ( u[k][j][i][1]*u[k][j][i][3] ) * tmp2;
    fjac(i, 1, 3, k, j) = u[k][j][i][3] * tmp1;
    fjac(i, 2, 3, k, j) = 0.0;
    fjac(i, 3, 3, k, j) = u[k][j][i][1] * tmp1;
    fjac(i, 4, 3, k, j) = 0.0;

    fjac(i, 0, 4, k, j) = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * ( u[k][j][i][1] * tmp2 );
    fjac(i, 1, 4, k, j) = c1 *  u[k][j][i][4] * tmp1 
      - c2 * ( u[k][j][i][1]*u[k][j][i][1] * tmp2 + qs[k][j][i] );
    fjac(i, 2, 4, k, j) = - c2 * ( u[k][j][i][2]*u[k][j][i][1] ) * tmp2;
    fjac(i, 3, 4, k, j) = - c2 * ( u[k][j][i][3]*u[k][j][i][1] ) * tmp2;
    fjac(i, 4, 4, k, j) = c1 * ( u[k][j][i][1] * tmp1 );

    njac(i, 0, 0, k, j) = 0.0;
    njac(i, 1, 0, k, j) = 0.0;
    njac(i, 2, 0, k, j) = 0.0;
    njac(i, 3, 0, k, j) = 0.0;
    njac(i, 4, 0, k, j) = 0.0;

    njac(i, 0, 1, k, j) = - con43 * c3c4 * tmp2 * u[k][j][i][1];
    njac(i, 1, 1, k, j) =   con43 * c3c4 * tmp1;
    njac(i, 2, 1, k, j) =   0.0;
    njac(i, 3, 1, k, j) =   0.0;
    njac(i, 4, 1, k, j) =   0.0;

    njac(i, 0, 2, k, j) = - c3c4 * tmp2 * u[k][j][i][2];
    njac(i, 1, 2, k, j) =   0.0;
    njac(i, 2, 2, k, j) =   c3c4 * tmp1;
    njac(i, 3, 2, k, j) =   0.0;
    njac(i, 4, 2, k, j) =   0.0;

    njac(i, 0, 3, k, j) = - c3c4 * tmp2 * u[k][j][i][3];
    njac(i, 1, 3, k, j) =   0.0;
    njac(i, 2, 3, k, j) =   0.0;
    njac(i, 3, 3, k, j) =   c3c4 * tmp1;
    njac(i, 4, 3, k, j) =   0.0;

    njac(i, 0, 4, k, j) = - ( con43 * c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac(i, 1, 4, k, j) = ( con43 * c3c4
        - c1345 ) * tmp2 * u[k][j][i][1];
    njac(i, 2, 4, k, j) = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac(i, 3, 4, k, j) = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac(i, 4, 4, k, j) = ( c1345 ) * tmp1;
  }
  //---------------------------------------------------------------------
  // now jacobians set, so form left hand side in x direction
  //---------------------------------------------------------------------
  lhsinit_x(&lhs(0, 0, 0, 0, 0, 0), isize, k, j, WORK_NUM_ITEM_DEFAULT);
  for (i = 1; i <= isize-1; i++) {
    tmp1 = dt * tx1;
    tmp2 = dt * tx2;

    lhs(i, AA, 0, 0, k, j) = - tmp2 * fjac(i-1, 0, 0, k, j)
      - tmp1 * njac(i-1, 0, 0, k, j)
      - tmp1 * dx1; 
    lhs(i, AA, 1, 0, k, j) = - tmp2 * fjac(i-1, 1, 0, k, j)
      - tmp1 * njac(i-1, 1, 0, k, j);
    lhs(i, AA, 2, 0, k, j) = - tmp2 * fjac(i-1, 2, 0, k, j)
      - tmp1 * njac(i-1, 2, 0, k, j);
    lhs(i, AA, 3, 0, k, j) = - tmp2 * fjac(i-1, 3, 0, k, j)
      - tmp1 * njac(i-1, 3, 0, k, j);
    lhs(i, AA, 4, 0, k, j) = - tmp2 * fjac(i-1, 4, 0, k, j)
      - tmp1 * njac(i-1, 4, 0, k, j);

    lhs(i, AA, 0, 1, k, j) = - tmp2 * fjac(i-1, 0, 1, k, j)
      - tmp1 * njac(i-1, 0, 1, k, j);
    lhs(i, AA, 1, 1, k, j) = - tmp2 * fjac(i-1, 1, 1, k, j)
      - tmp1 * njac(i-1, 1, 1, k, j)
      - tmp1 * dx2;
    lhs(i, AA, 2, 1, k, j) = - tmp2 * fjac(i-1, 2, 1, k, j)
      - tmp1 * njac(i-1, 2, 1, k, j);
    lhs(i, AA, 3, 1, k, j) = - tmp2 * fjac(i-1, 3, 1, k, j)
      - tmp1 * njac(i-1, 3, 1, k, j);
    lhs(i, AA, 4, 1, k, j) = - tmp2 * fjac(i-1, 4, 1, k, j)
      - tmp1 * njac(i-1, 4, 1, k, j);

    lhs(i, AA, 0, 2, k, j) = - tmp2 * fjac(i-1, 0, 2, k, j)
      - tmp1 * njac(i-1, 0, 2, k, j);
    lhs(i, AA, 1, 2, k, j) = - tmp2 * fjac(i-1, 1, 2, k, j)
      - tmp1 * njac(i-1, 1, 2, k, j);
    lhs(i, AA, 2, 2, k, j) = - tmp2 * fjac(i-1, 2, 2, k, j)
      - tmp1 * njac(i-1, 2, 2, k, j)
      - tmp1 * dx3;
    lhs(i, AA, 3, 2, k, j) = - tmp2 * fjac(i-1, 3, 2, k, j)
      - tmp1 * njac(i-1, 3, 2, k, j);
    lhs(i, AA, 4, 2, k, j) = - tmp2 * fjac(i-1, 4, 2, k, j)
      - tmp1 * njac(i-1, 4, 2, k, j);

    lhs(i, AA, 0, 3, k, j) = - tmp2 * fjac(i-1, 0, 3, k, j)
      - tmp1 * njac(i-1, 0, 3, k, j);
    lhs(i, AA, 1, 3, k, j) = - tmp2 * fjac(i-1, 1, 3, k, j)
      - tmp1 * njac(i-1, 1, 3, k, j);
    lhs(i, AA, 2, 3, k, j) = - tmp2 * fjac(i-1, 2, 3, k, j)
      - tmp1 * njac(i-1, 2, 3, k, j);
    lhs(i, AA, 3, 3, k, j) = - tmp2 * fjac(i-1, 3, 3, k, j)
      - tmp1 * njac(i-1, 3, 3, k, j)
      - tmp1 * dx4;
    lhs(i, AA, 4, 3, k, j) = - tmp2 * fjac(i-1, 4, 3, k, j)
      - tmp1 * njac(i-1, 4, 3, k, j);

    lhs(i, AA, 0, 4, k, j) = - tmp2 * fjac(i-1, 0, 4, k, j)
      - tmp1 * njac(i-1, 0, 4, k, j);
    lhs(i, AA, 1, 4, k, j) = - tmp2 * fjac(i-1, 1, 4, k, j)
      - tmp1 * njac(i-1, 1, 4, k, j);
    lhs(i, AA, 2, 4, k, j) = - tmp2 * fjac(i-1, 2, 4, k, j)
      - tmp1 * njac(i-1, 2, 4, k, j);
    lhs(i, AA, 3, 4, k, j) = - tmp2 * fjac(i-1, 3, 4, k, j)
      - tmp1 * njac(i-1, 3, 4, k, j);
    lhs(i, AA, 4, 4, k, j) = - tmp2 * fjac(i-1, 4, 4, k, j)
      - tmp1 * njac(i-1, 4, 4, k, j)
      - tmp1 * dx5;

    lhs(i, BB, 0, 0, k, j) = 1.0
      + tmp1 * 2.0 * njac(i, 0, 0, k, j)
      + tmp1 * 2.0 * dx1;
    lhs(i, BB, 1, 0, k, j) = tmp1 * 2.0 * njac(i, 1, 0, k, j);
    lhs(i, BB, 2, 0, k, j) = tmp1 * 2.0 * njac(i, 2, 0, k, j);
    lhs(i, BB, 3, 0, k, j) = tmp1 * 2.0 * njac(i, 3, 0, k, j);
    lhs(i, BB, 4, 0, k, j) = tmp1 * 2.0 * njac(i, 4, 0, k, j);

    lhs(i, BB, 0, 1, k, j) = tmp1 * 2.0 * njac(i, 0, 1, k, j);
    lhs(i, BB, 1, 1, k, j) = 1.0
      + tmp1 * 2.0 * njac(i, 1, 1, k, j)
      + tmp1 * 2.0 * dx2;
    lhs(i, BB, 2, 1, k, j) = tmp1 * 2.0 * njac(i, 2, 1, k, j);
    lhs(i, BB, 3, 1, k, j) = tmp1 * 2.0 * njac(i, 3, 1, k, j);
    lhs(i, BB, 4, 1, k, j) = tmp1 * 2.0 * njac(i, 4, 1, k, j);

    lhs(i, BB, 0, 2, k, j) = tmp1 * 2.0 * njac(i, 0, 2, k, j);
    lhs(i, BB, 1, 2, k, j) = tmp1 * 2.0 * njac(i, 1, 2, k, j);
    lhs(i, BB, 2, 2, k, j) = 1.0
      + tmp1 * 2.0 * njac(i, 2, 2, k, j)
      + tmp1 * 2.0 * dx3;
    lhs(i, BB, 3, 2, k, j) = tmp1 * 2.0 * njac(i, 3, 2, k, j);
    lhs(i, BB, 4, 2, k, j) = tmp1 * 2.0 * njac(i, 4, 2, k, j);

    lhs(i, BB, 0, 3, k, j) = tmp1 * 2.0 * njac(i, 0, 3, k, j);
    lhs(i, BB, 1, 3, k, j) = tmp1 * 2.0 * njac(i, 1, 3, k, j);
    lhs(i, BB, 2, 3, k, j) = tmp1 * 2.0 * njac(i, 2, 3, k, j);
    lhs(i, BB, 3, 3, k, j) = 1.0
      + tmp1 * 2.0 * njac(i, 3, 3, k, j)
      + tmp1 * 2.0 * dx4;
    lhs(i, BB, 4, 3, k, j) = tmp1 * 2.0 * njac(i, 4, 3, k, j);

    lhs(i, BB, 0, 4, k, j) = tmp1 * 2.0 * njac(i, 0, 4, k, j);
    lhs(i, BB, 1, 4, k, j) = tmp1 * 2.0 * njac(i, 1, 4, k, j);
    lhs(i, BB, 2, 4, k, j) = tmp1 * 2.0 * njac(i, 2, 4, k, j);
    lhs(i, BB, 3, 4, k, j) = tmp1 * 2.0 * njac(i, 3, 4, k, j);
    lhs(i, BB, 4, 4, k, j) = 1.0
      + tmp1 * 2.0 * njac(i, 4, 4, k, j)
      + tmp1 * 2.0 * dx5;

    lhs(i, CC, 0, 0, k, j) =  tmp2 * fjac(i+1, 0, 0, k, j)
      - tmp1 * njac(i+1, 0, 0, k, j)
      - tmp1 * dx1;
    lhs(i, CC, 1, 0, k, j) =  tmp2 * fjac(i+1, 1, 0, k, j)
      - tmp1 * njac(i+1, 1, 0, k, j);
    lhs(i, CC, 2, 0, k, j) =  tmp2 * fjac(i+1, 2, 0, k, j)
      - tmp1 * njac(i+1, 2, 0, k, j);
    lhs(i, CC, 3, 0, k, j) =  tmp2 * fjac(i+1, 3, 0, k, j)
      - tmp1 * njac(i+1, 3, 0, k, j);
    lhs(i, CC, 4, 0, k, j) =  tmp2 * fjac(i+1, 4, 0, k, j)
      - tmp1 * njac(i+1, 4, 0, k, j);

    lhs(i, CC, 0, 1, k, j) =  tmp2 * fjac(i+1, 0, 1, k, j)
      - tmp1 * njac(i+1, 0, 1, k, j);
    lhs(i, CC, 1, 1, k, j) =  tmp2 * fjac(i+1, 1, 1, k, j)
      - tmp1 * njac(i+1, 1, 1, k, j)
      - tmp1 * dx2;
    lhs(i, CC, 2, 1, k, j) =  tmp2 * fjac(i+1, 2, 1, k, j)
      - tmp1 * njac(i+1, 2, 1, k, j);
    lhs(i, CC, 3, 1, k, j) =  tmp2 * fjac(i+1, 3, 1, k, j)
      - tmp1 * njac(i+1, 3, 1, k, j);
    lhs(i, CC, 4, 1, k, j) =  tmp2 * fjac(i+1, 4, 1, k, j)
      - tmp1 * njac(i+1, 4, 1, k, j);

    lhs(i, CC, 0, 2, k, j) =  tmp2 * fjac(i+1, 0, 2, k, j)
      - tmp1 * njac(i+1, 0, 2, k, j);
    lhs(i, CC, 1, 2, k, j) =  tmp2 * fjac(i+1, 1, 2, k, j)
      - tmp1 * njac(i+1, 1, 2, k, j);
    lhs(i, CC, 2, 2, k, j) =  tmp2 * fjac(i+1, 2, 2, k, j)
      - tmp1 * njac(i+1, 2, 2, k, j)
      - tmp1 * dx3;
    lhs(i, CC, 3, 2, k, j) =  tmp2 * fjac(i+1, 3, 2, k, j)
      - tmp1 * njac(i+1, 3, 2, k, j);
    lhs(i, CC, 4, 2, k, j) =  tmp2 * fjac(i+1, 4, 2, k, j)
      - tmp1 * njac(i+1, 4, 2, k, j);

    lhs(i, CC, 0, 3, k, j) =  tmp2 * fjac(i+1, 0, 3, k, j)
      - tmp1 * njac(i+1, 0, 3, k, j);
    lhs(i, CC, 1, 3, k, j) =  tmp2 * fjac(i+1, 1, 3, k, j)
      - tmp1 * njac(i+1, 1, 3, k, j);
    lhs(i, CC, 2, 3, k, j) =  tmp2 * fjac(i+1, 2, 3, k, j)
      - tmp1 * njac(i+1, 2, 3, k, j);
    lhs(i, CC, 3, 3, k, j) =  tmp2 * fjac(i+1, 3, 3, k, j)
      - tmp1 * njac(i+1, 3, 3, k, j)
      - tmp1 * dx4;
    lhs(i, CC, 4, 3, k, j) =  tmp2 * fjac(i+1, 4, 3, k, j)
      - tmp1 * njac(i+1, 4, 3, k, j);

    lhs(i, CC, 0, 4, k, j) =  tmp2 * fjac(i+1, 0, 4, k, j)
      - tmp1 * njac(i+1, 0, 4, k, j);
    lhs(i, CC, 1, 4, k, j) =  tmp2 * fjac(i+1, 1, 4, k, j)
      - tmp1 * njac(i+1, 1, 4, k, j);
    lhs(i, CC, 2, 4, k, j) =  tmp2 * fjac(i+1, 2, 4, k, j)
      - tmp1 * njac(i+1, 2, 4, k, j);
    lhs(i, CC, 3, 4, k, j) =  tmp2 * fjac(i+1, 3, 4, k, j)
      - tmp1 * njac(i+1, 3, 4, k, j);
    lhs(i, CC, 4, 4, k, j) =  tmp2 * fjac(i+1, 4, 4, k, j)
      - tmp1 * njac(i+1, 4, 4, k, j)
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
  binvcrhs_x(&lhs(0, BB, 0, 0, 0, 0), &lhs(0, CC, 0, 0, 0, 0), rhs[k][j][0], k, j, WORK_NUM_ITEM_DEFAULT);

  //---------------------------------------------------------------------
  // begin inner most do loop
  // do all the elements of the cell unless last 
  //---------------------------------------------------------------------
  for (i = 1; i <= isize-1; i++) {
    //-------------------------------------------------------------------
    // rhs(i) = rhs(i) - A*rhs(i-1)
    //-------------------------------------------------------------------
    matvec_sub_x(&lhs(i, AA, 0, 0, 0, 0), rhs[k][j][i-1], rhs[k][j][i], k, j, WORK_NUM_ITEM_DEFAULT);

    //-------------------------------------------------------------------
    // B(i) = B(i) - C(i-1)*A(i)
    //-------------------------------------------------------------------
    matmul_sub_x(&lhs(i, AA, 0, 0, 0, 0), &lhs(i-1, CC, 0, 0, 0, 0), &lhs(i, BB, 0, 0, 0, 0), k, j, WORK_NUM_ITEM_DEFAULT);


    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs_x(&lhs(i, BB, 0, 0, 0, 0), &lhs(i, CC, 0, 0, 0, 0), rhs[k][j][i], k, j, WORK_NUM_ITEM_DEFAULT);
  }

  //---------------------------------------------------------------------
  // rhs(isize) = rhs(isize) - A*rhs(isize-1)
  //---------------------------------------------------------------------
  matvec_sub_x(&lhs(isize, AA, 0, 0, 0, 0), rhs[k][j][isize-1], rhs[k][j][isize], k, j, WORK_NUM_ITEM_DEFAULT);

  //---------------------------------------------------------------------
  // B(isize) = B(isize) - C(isize-1)*A(isize)
  //---------------------------------------------------------------------
  matmul_sub_x(&lhs(isize, AA, 0, 0, 0, 0), &lhs(isize-1, CC, 0, 0, 0, 0), &lhs(isize, BB, 0, 0, 0, 0), k, j, WORK_NUM_ITEM_DEFAULT);

  //---------------------------------------------------------------------
  // multiply rhs() by b_inverse() and copy to rhs
  //---------------------------------------------------------------------
  binvrhs_x(&lhs(isize, BB, 0, 0, 0, 0), rhs[k][j][isize] , k, j, WORK_NUM_ITEM_DEFAULT);

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
          - lhs(i, CC, n, m, k, j)*rhs[k][j][i+1][n];
      }
    }
  }

#undef lhs
#undef mfjac
#undef njac
}
