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
void lhsinit_y(double *lhs, 
               int size, int k, int i, int WORK_NUM_ITEM_DEFAULT)
{
#define t_lhs(a, b, c, d, e, f) lhs[(((((a)*3 + (b))*5 + (c))*5 + (d))*WORK_NUM_ITEM_DEFAULT + (e))*(PROBLEM_SIZE-1) + (f)]
  int j, m, n;

  j = size;
  //---------------------------------------------------------------------
  // zero the whole left hand side for starters
  //---------------------------------------------------------------------
  for (n = 0; n < 5; n++) {
    for (m = 0; m < 5; m++) {
      t_lhs(0, 0, n, m, k, i) = 0.0;
      t_lhs(0, 1, n, m, k, i) = 0.0;
      t_lhs(0, 2, n, m, k, i) = 0.0;
      t_lhs(j, 0, n, m, k, i) = 0.0;
      t_lhs(j, 1, n, m, k, i) = 0.0;
      t_lhs(j, 2, n, m, k, i) = 0.0;
    }
  }

  //---------------------------------------------------------------------
  // next, set all diagonal values to 1. This is overkill, but convenient
  //---------------------------------------------------------------------
  for (m = 0; m < 5; m++) {
    t_lhs(0, 1, m, m, k, i) = 1.0;
    t_lhs(j, 1, m, m, k, i) = 1.0;
  }
#undef t_lhs
}

__device__
void binvcrhs_y(double *lhs, 
                double *c, 
                double r[5],
                int k, int i, int WORK_NUM_ITEM_DEFAULT)
{
#define t_lhs(a, b, c, d) lhs[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
#define t_c(a, b, d, e) c[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (d))*(PROBLEM_SIZE-1) + (e)]

  double pivot, coeff;

  pivot = 1.00/t_lhs(0, 0, k, i);
  t_lhs(1, 0, k, i) = t_lhs(1, 0, k, i)*pivot;
  t_lhs(2, 0, k, i) = t_lhs(2, 0, k, i)*pivot;
  t_lhs(3, 0, k, i) = t_lhs(3, 0, k, i)*pivot;
  t_lhs(4, 0, k, i) = t_lhs(4, 0, k, i)*pivot;
  t_c(0, 0, k, i) = t_c(0, 0, k, i)*pivot;
  t_c(1, 0, k, i) = t_c(1, 0, k, i)*pivot;
  t_c(2, 0, k, i) = t_c(2, 0, k, i)*pivot;
  t_c(3, 0, k, i) = t_c(3, 0, k, i)*pivot;
  t_c(4, 0, k, i) = t_c(4, 0, k, i)*pivot;
  r[0]   = r[0]  *pivot;

  coeff = t_lhs(0, 1, k, i);
  t_lhs(1, 1, k, i)= t_lhs(1, 1, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 1, k, i)= t_lhs(2, 1, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 1, k, i)= t_lhs(3, 1, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 1, k, i)= t_lhs(4, 1, k, i) - coeff*t_lhs(4, 0, k, i);
  t_c(0, 1, k, i) = t_c(0, 1, k, i) - coeff*t_c(0, 0, k, i);
  t_c(1, 1, k, i) = t_c(1, 1, k, i) - coeff*t_c(1, 0, k, i);
  t_c(2, 1, k, i) = t_c(2, 1, k, i) - coeff*t_c(2, 0, k, i);
  t_c(3, 1, k, i) = t_c(3, 1, k, i) - coeff*t_c(3, 0, k, i);
  t_c(4, 1, k, i) = t_c(4, 1, k, i) - coeff*t_c(4, 0, k, i);
  r[1]   = r[1]   - coeff*r[0];

  coeff = t_lhs(0, 2, k, i);
  t_lhs(1, 2, k, i)= t_lhs(1, 2, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 2, k, i)= t_lhs(2, 2, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 2, k, i)= t_lhs(3, 2, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 2, k, i)= t_lhs(4, 2, k, i) - coeff*t_lhs(4, 0, k, i);
  t_c(0, 2, k, i) = t_c(0, 2, k, i) - coeff*t_c(0, 0, k, i);
  t_c(1, 2, k, i) = t_c(1, 2, k, i) - coeff*t_c(1, 0, k, i);
  t_c(2, 2, k, i) = t_c(2, 2, k, i) - coeff*t_c(2, 0, k, i);
  t_c(3, 2, k, i) = t_c(3, 2, k, i) - coeff*t_c(3, 0, k, i);
  t_c(4, 2, k, i) = t_c(4, 2, k, i) - coeff*t_c(4, 0, k, i);
  r[2]   = r[2]   - coeff*r[0];

  coeff = t_lhs(0, 3, k, i);
  t_lhs(1, 3, k, i)= t_lhs(1, 3, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 3, k, i)= t_lhs(2, 3, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 3, k, i)= t_lhs(3, 3, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 3, k, i)= t_lhs(4, 3, k, i) - coeff*t_lhs(4, 0, k, i);
  t_c(0, 3, k, i) = t_c(0, 3, k, i) - coeff*t_c(0, 0, k, i);
  t_c(1, 3, k, i) = t_c(1, 3, k, i) - coeff*t_c(1, 0, k, i);
  t_c(2, 3, k, i) = t_c(2, 3, k, i) - coeff*t_c(2, 0, k, i);
  t_c(3, 3, k, i) = t_c(3, 3, k, i) - coeff*t_c(3, 0, k, i);
  t_c(4, 3, k, i) = t_c(4, 3, k, i) - coeff*t_c(4, 0, k, i);
  r[3]   = r[3]   - coeff*r[0];

  coeff = t_lhs(0, 4, k, i);
  t_lhs(1, 4, k, i)= t_lhs(1, 4, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 4, k, i)= t_lhs(2, 4, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 4, k, i)= t_lhs(3, 4, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 0, k, i);
  t_c(0, 4, k, i) = t_c(0, 4, k, i) - coeff*t_c(0, 0, k, i);
  t_c(1, 4, k, i) = t_c(1, 4, k, i) - coeff*t_c(1, 0, k, i);
  t_c(2, 4, k, i) = t_c(2, 4, k, i) - coeff*t_c(2, 0, k, i);
  t_c(3, 4, k, i) = t_c(3, 4, k, i) - coeff*t_c(3, 0, k, i);
  t_c(4, 4, k, i) = t_c(4, 4, k, i) - coeff*t_c(4, 0, k, i);
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/t_lhs(1, 1, k, i);
  t_lhs(2, 1, k, i) = t_lhs(2, 1, k, i)*pivot;
  t_lhs(3, 1, k, i) = t_lhs(3, 1, k, i)*pivot;
  t_lhs(4, 1, k, i) = t_lhs(4, 1, k, i)*pivot;
  t_c(0, 1, k, i) = t_c(0, 1, k, i)*pivot;
  t_c(1, 1, k, i) = t_c(1, 1, k, i)*pivot;
  t_c(2, 1, k, i) = t_c(2, 1, k, i)*pivot;
  t_c(3, 1, k, i) = t_c(3, 1, k, i)*pivot;
  t_c(4, 1, k, i) = t_c(4, 1, k, i)*pivot;
  r[1]   = r[1]  *pivot;

  coeff = t_lhs(1, 0, k, i);
  t_lhs(2, 0, k, i)= t_lhs(2, 0, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 0, k, i)= t_lhs(3, 0, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 0, k, i)= t_lhs(4, 0, k, i) - coeff*t_lhs(4, 1, k, i);
  t_c(0, 0, k, i) = t_c(0, 0, k, i) - coeff*t_c(0, 1, k, i);
  t_c(1, 0, k, i) = t_c(1, 0, k, i) - coeff*t_c(1, 1, k, i);
  t_c(2, 0, k, i) = t_c(2, 0, k, i) - coeff*t_c(2, 1, k, i);
  t_c(3, 0, k, i) = t_c(3, 0, k, i) - coeff*t_c(3, 1, k, i);
  t_c(4, 0, k, i) = t_c(4, 0, k, i) - coeff*t_c(4, 1, k, i);
  r[0]   = r[0]   - coeff*r[1];

  coeff = t_lhs(1, 2, k, i);
  t_lhs(2, 2, k, i)= t_lhs(2, 2, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 2, k, i)= t_lhs(3, 2, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 2, k, i)= t_lhs(4, 2, k, i) - coeff*t_lhs(4, 1, k, i);
  t_c(0, 2, k, i) = t_c(0, 2, k, i) - coeff*t_c(0, 1, k, i);
  t_c(1, 2, k, i) = t_c(1, 2, k, i) - coeff*t_c(1, 1, k, i);
  t_c(2, 2, k, i) = t_c(2, 2, k, i) - coeff*t_c(2, 1, k, i);
  t_c(3, 2, k, i) = t_c(3, 2, k, i) - coeff*t_c(3, 1, k, i);
  t_c(4, 2, k, i) = t_c(4, 2, k, i) - coeff*t_c(4, 1, k, i);
  r[2]   = r[2]   - coeff*r[1];

  coeff = t_lhs(1, 3, k, i);
  t_lhs(2, 3, k, i)= t_lhs(2, 3, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 3, k, i)= t_lhs(3, 3, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 3, k, i)= t_lhs(4, 3, k, i) - coeff*t_lhs(4, 1, k, i);
  t_c(0, 3, k, i) = t_c(0, 3, k, i) - coeff*t_c(0, 1, k, i);
  t_c(1, 3, k, i) = t_c(1, 3, k, i) - coeff*t_c(1, 1, k, i);
  t_c(2, 3, k, i) = t_c(2, 3, k, i) - coeff*t_c(2, 1, k, i);
  t_c(3, 3, k, i) = t_c(3, 3, k, i) - coeff*t_c(3, 1, k, i);
  t_c(4, 3, k, i) = t_c(4, 3, k, i) - coeff*t_c(4, 1, k, i);
  r[3]   = r[3]   - coeff*r[1];

  coeff = t_lhs(1, 4, k, i);
  t_lhs(2, 4, k, i)= t_lhs(2, 4, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 4, k, i)= t_lhs(3, 4, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 1, k, i);
  t_c(0, 4, k, i) = t_c(0, 4, k, i) - coeff*t_c(0, 1, k, i);
  t_c(1, 4, k, i) = t_c(1, 4, k, i) - coeff*t_c(1, 1, k, i);
  t_c(2, 4, k, i) = t_c(2, 4, k, i) - coeff*t_c(2, 1, k, i);
  t_c(3, 4, k, i) = t_c(3, 4, k, i) - coeff*t_c(3, 1, k, i);
  t_c(4, 4, k, i) = t_c(4, 4, k, i) - coeff*t_c(4, 1, k, i);
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/t_lhs(2, 2, k, i);
  t_lhs(3, 2, k, i) = t_lhs(3, 2, k, i)*pivot;
  t_lhs(4, 2, k, i) = t_lhs(4, 2, k, i)*pivot;
  t_c(0, 2, k, i) = t_c(0, 2, k, i)*pivot;
  t_c(1, 2, k, i) = t_c(1, 2, k, i)*pivot;
  t_c(2, 2, k, i) = t_c(2, 2, k, i)*pivot;
  t_c(3, 2, k, i) = t_c(3, 2, k, i)*pivot;
  t_c(4, 2, k, i) = t_c(4, 2, k, i)*pivot;
  r[2]   = r[2]  *pivot;

  coeff = t_lhs(2, 0, k, i);
  t_lhs(3, 0, k, i)= t_lhs(3, 0, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 0, k, i)= t_lhs(4, 0, k, i) - coeff*t_lhs(4, 2, k, i);
  t_c(0, 0, k, i) = t_c(0, 0, k, i) - coeff*t_c(0, 2, k, i);
  t_c(1, 0, k, i) = t_c(1, 0, k, i) - coeff*t_c(1, 2, k, i);
  t_c(2, 0, k, i) = t_c(2, 0, k, i) - coeff*t_c(2, 2, k, i);
  t_c(3, 0, k, i) = t_c(3, 0, k, i) - coeff*t_c(3, 2, k, i);
  t_c(4, 0, k, i) = t_c(4, 0, k, i) - coeff*t_c(4, 2, k, i);
  r[0]   = r[0]   - coeff*r[2];

  coeff = t_lhs(2, 1, k, i);
  t_lhs(3, 1, k, i)= t_lhs(3, 1, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 1, k, i)= t_lhs(4, 1, k, i) - coeff*t_lhs(4, 2, k, i);
  t_c(0, 1, k, i) = t_c(0, 1, k, i) - coeff*t_c(0, 2, k, i);
  t_c(1, 1, k, i) = t_c(1, 1, k, i) - coeff*t_c(1, 2, k, i);
  t_c(2, 1, k, i) = t_c(2, 1, k, i) - coeff*t_c(2, 2, k, i);
  t_c(3, 1, k, i) = t_c(3, 1, k, i) - coeff*t_c(3, 2, k, i);
  t_c(4, 1, k, i) = t_c(4, 1, k, i) - coeff*t_c(4, 2, k, i);
  r[1]   = r[1]   - coeff*r[2];

  coeff = t_lhs(2, 3, k, i);
  t_lhs(3, 3, k, i)= t_lhs(3, 3, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 3, k, i)= t_lhs(4, 3, k, i) - coeff*t_lhs(4, 2, k, i);
  t_c(0, 3, k, i) = t_c(0, 3, k, i) - coeff*t_c(0, 2, k, i);
  t_c(1, 3, k, i) = t_c(1, 3, k, i) - coeff*t_c(1, 2, k, i);
  t_c(2, 3, k, i) = t_c(2, 3, k, i) - coeff*t_c(2, 2, k, i);
  t_c(3, 3, k, i) = t_c(3, 3, k, i) - coeff*t_c(3, 2, k, i);
  t_c(4, 3, k, i) = t_c(4, 3, k, i) - coeff*t_c(4, 2, k, i);
  r[3]   = r[3]   - coeff*r[2];

  coeff = t_lhs(2, 4, k, i);
  t_lhs(3, 4, k, i)= t_lhs(3, 4, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 2, k, i);
  t_c(0, 4, k, i) = t_c(0, 4, k, i) - coeff*t_c(0, 2, k, i);
  t_c(1, 4, k, i) = t_c(1, 4, k, i) - coeff*t_c(1, 2, k, i);
  t_c(2, 4, k, i) = t_c(2, 4, k, i) - coeff*t_c(2, 2, k, i);
  t_c(3, 4, k, i) = t_c(3, 4, k, i) - coeff*t_c(3, 2, k, i);
  t_c(4, 4, k, i) = t_c(4, 4, k, i) - coeff*t_c(4, 2, k, i);
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/t_lhs(3, 3, k, i);
  t_lhs(4, 3, k, i) = t_lhs(4, 3, k, i)*pivot;
  t_c(0, 3, k, i) = t_c(0, 3, k, i)*pivot;
  t_c(1, 3, k, i) = t_c(1, 3, k, i)*pivot;
  t_c(2, 3, k, i) = t_c(2, 3, k, i)*pivot;
  t_c(3, 3, k, i) = t_c(3, 3, k, i)*pivot;
  t_c(4, 3, k, i) = t_c(4, 3, k, i)*pivot;
  r[3]   = r[3]  *pivot;

  coeff = t_lhs(3, 0, k, i);
  t_lhs(4, 0, k, i)= t_lhs(4, 0, k, i) - coeff*t_lhs(4, 3, k, i);
  t_c(0, 0, k, i) = t_c(0, 0, k, i) - coeff*t_c(0, 3, k, i);
  t_c(1, 0, k, i) = t_c(1, 0, k, i) - coeff*t_c(1, 3, k, i);
  t_c(2, 0, k, i) = t_c(2, 0, k, i) - coeff*t_c(2, 3, k, i);
  t_c(3, 0, k, i) = t_c(3, 0, k, i) - coeff*t_c(3, 3, k, i);
  t_c(4, 0, k, i) = t_c(4, 0, k, i) - coeff*t_c(4, 3, k, i);
  r[0]   = r[0]   - coeff*r[3];

  coeff = t_lhs(3, 1, k, i);
  t_lhs(4, 1, k, i)= t_lhs(4, 1, k, i) - coeff*t_lhs(4, 3, k, i);
  t_c(0, 1, k, i) = t_c(0, 1, k, i) - coeff*t_c(0, 3, k, i);
  t_c(1, 1, k, i) = t_c(1, 1, k, i) - coeff*t_c(1, 3, k, i);
  t_c(2, 1, k, i) = t_c(2, 1, k, i) - coeff*t_c(2, 3, k, i);
  t_c(3, 1, k, i) = t_c(3, 1, k, i) - coeff*t_c(3, 3, k, i);
  t_c(4, 1, k, i) = t_c(4, 1, k, i) - coeff*t_c(4, 3, k, i);
  r[1]   = r[1]   - coeff*r[3];

  coeff = t_lhs(3, 2, k, i);
  t_lhs(4, 2, k, i)= t_lhs(4, 2, k, i) - coeff*t_lhs(4, 3, k, i);
  t_c(0, 2, k, i) = t_c(0, 2, k, i) - coeff*t_c(0, 3, k, i);
  t_c(1, 2, k, i) = t_c(1, 2, k, i) - coeff*t_c(1, 3, k, i);
  t_c(2, 2, k, i) = t_c(2, 2, k, i) - coeff*t_c(2, 3, k, i);
  t_c(3, 2, k, i) = t_c(3, 2, k, i) - coeff*t_c(3, 3, k, i);
  t_c(4, 2, k, i) = t_c(4, 2, k, i) - coeff*t_c(4, 3, k, i);
  r[2]   = r[2]   - coeff*r[3];

  coeff = t_lhs(3, 4, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 3, k, i);
  t_c(0, 4, k, i) = t_c(0, 4, k, i) - coeff*t_c(0, 3, k, i);
  t_c(1, 4, k, i) = t_c(1, 4, k, i) - coeff*t_c(1, 3, k, i);
  t_c(2, 4, k, i) = t_c(2, 4, k, i) - coeff*t_c(2, 3, k, i);
  t_c(3, 4, k, i) = t_c(3, 4, k, i) - coeff*t_c(3, 3, k, i);
  t_c(4, 4, k, i) = t_c(4, 4, k, i) - coeff*t_c(4, 3, k, i);
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/t_lhs(4, 4, k, i);
  t_c(0, 4, k, i) = t_c(0, 4, k, i)*pivot;
  t_c(1, 4, k, i) = t_c(1, 4, k, i)*pivot;
  t_c(2, 4, k, i) = t_c(2, 4, k, i)*pivot;
  t_c(3, 4, k, i) = t_c(3, 4, k, i)*pivot;
  t_c(4, 4, k, i) = t_c(4, 4, k, i)*pivot;
  r[4]   = r[4]  *pivot;

  coeff = t_lhs(4, 0, k, i);
  t_c(0, 0, k, i) = t_c(0, 0, k, i) - coeff*t_c(0, 4, k, i);
  t_c(1, 0, k, i) = t_c(1, 0, k, i) - coeff*t_c(1, 4, k, i);
  t_c(2, 0, k, i) = t_c(2, 0, k, i) - coeff*t_c(2, 4, k, i);
  t_c(3, 0, k, i) = t_c(3, 0, k, i) - coeff*t_c(3, 4, k, i);
  t_c(4, 0, k, i) = t_c(4, 0, k, i) - coeff*t_c(4, 4, k, i);
  r[0]   = r[0]   - coeff*r[4];

  coeff = t_lhs(4, 1, k, i);
  t_c(0, 1, k, i) = t_c(0, 1, k, i) - coeff*t_c(0, 4, k, i);
  t_c(1, 1, k, i) = t_c(1, 1, k, i) - coeff*t_c(1, 4, k, i);
  t_c(2, 1, k, i) = t_c(2, 1, k, i) - coeff*t_c(2, 4, k, i);
  t_c(3, 1, k, i) = t_c(3, 1, k, i) - coeff*t_c(3, 4, k, i);
  t_c(4, 1, k, i) = t_c(4, 1, k, i) - coeff*t_c(4, 4, k, i);
  r[1]   = r[1]   - coeff*r[4];

  coeff = t_lhs(4, 2, k, i);
  t_c(0, 2, k, i) = t_c(0, 2, k, i) - coeff*t_c(0, 4, k, i);
  t_c(1, 2, k, i) = t_c(1, 2, k, i) - coeff*t_c(1, 4, k, i);
  t_c(2, 2, k, i) = t_c(2, 2, k, i) - coeff*t_c(2, 4, k, i);
  t_c(3, 2, k, i) = t_c(3, 2, k, i) - coeff*t_c(3, 4, k, i);
  t_c(4, 2, k, i) = t_c(4, 2, k, i) - coeff*t_c(4, 4, k, i);
  r[2]   = r[2]   - coeff*r[4];

  coeff = t_lhs(4, 3, k, i);
  t_c(0, 3, k, i) = t_c(0, 3, k, i) - coeff*t_c(0, 4, k, i);
  t_c(1, 3, k, i) = t_c(1, 3, k, i) - coeff*t_c(1, 4, k, i);
  t_c(2, 3, k, i) = t_c(2, 3, k, i) - coeff*t_c(2, 4, k, i);
  t_c(3, 3, k, i) = t_c(3, 3, k, i) - coeff*t_c(3, 4, k, i);
  t_c(4, 3, k, i) = t_c(4, 3, k, i) - coeff*t_c(4, 4, k, i);
  r[3]   = r[3]   - coeff*r[4];

#undef t_lhs
#undef t_c
}

__device__
void matvec_sub_y(double *ablock, 
                  double avec[5], 
                  double bvec[5],
                  int k, int i, int WORK_NUM_ITEM_DEFAULT)
{
#define t_ablock(a, b, c, d) ablock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
  //---------------------------------------------------------------------
  // rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] 
  // $                  - lhs[ia][ablock][0][i]*
  //---------------------------------------------------------------------
  bvec[0] = bvec[0] - t_ablock(0, 0, k, i)*avec[0]
                    - t_ablock(1, 0, k, i)*avec[1]
                    - t_ablock(2, 0, k, i)*avec[2]
                    - t_ablock(3, 0, k, i)*avec[3]
                    - t_ablock(4, 0, k, i)*avec[4];
  bvec[1] = bvec[1] - t_ablock(0, 1, k, i)*avec[0]
                    - t_ablock(1, 1, k, i)*avec[1]
                    - t_ablock(2, 1, k, i)*avec[2]
                    - t_ablock(3, 1, k, i)*avec[3]
                    - t_ablock(4, 1, k, i)*avec[4];
  bvec[2] = bvec[2] - t_ablock(0, 2, k, i)*avec[0]
                    - t_ablock(1, 2, k, i)*avec[1]
                    - t_ablock(2, 2, k, i)*avec[2]
                    - t_ablock(3, 2, k, i)*avec[3]
                    - t_ablock(4, 2, k, i)*avec[4];
  bvec[3] = bvec[3] - t_ablock(0, 3, k, i)*avec[0]
                    - t_ablock(1, 3, k, i)*avec[1]
                    - t_ablock(2, 3, k, i)*avec[2]
                    - t_ablock(3, 3, k, i)*avec[3]
                    - t_ablock(4, 3, k, i)*avec[4];
  bvec[4] = bvec[4] - t_ablock(0, 4, k, i)*avec[0]
                    - t_ablock(1, 4, k, i)*avec[1]
                    - t_ablock(2, 4, k, i)*avec[2]
                    - t_ablock(3, 4, k, i)*avec[3]
                    - t_ablock(4, 4, k, i)*avec[4];
#undef t_ablock
}

__device__
void matmul_sub_y(double *ablock, 
                  double *bblock, 
                  double *cblock,
                  int k, int i, int WORK_NUM_ITEM_DEFAULT)
{
#define t_ablock(a, b, c, d) ablock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
#define t_bblock(a, b, c, d) bblock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
#define t_cblock(a, b, c, d) cblock[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]

  t_cblock(0, 0, k, i) = t_cblock(0, 0, k, i) - t_ablock(0, 0, k, i)*t_bblock(0, 0, k, i)
                              - t_ablock(1, 0, k, i)*t_bblock(0, 1, k, i)
                              - t_ablock(2, 0, k, i)*t_bblock(0, 2, k, i)
                              - t_ablock(3, 0, k, i)*t_bblock(0, 3, k, i)
                              - t_ablock(4, 0, k, i)*t_bblock(0, 4, k, i);
  t_cblock(0, 1, k, i) = t_cblock(0, 1, k, i) - t_ablock(0, 1, k, i)*t_bblock(0, 0, k, i)
                              - t_ablock(1, 1, k, i)*t_bblock(0, 1, k, i)
                              - t_ablock(2, 1, k, i)*t_bblock(0, 2, k, i)
                              - t_ablock(3, 1, k, i)*t_bblock(0, 3, k, i)
                              - t_ablock(4, 1, k, i)*t_bblock(0, 4, k, i);
  t_cblock(0, 2, k, i) = t_cblock(0, 2, k, i) - t_ablock(0, 2, k, i)*t_bblock(0, 0, k, i)
                              - t_ablock(1, 2, k, i)*t_bblock(0, 1, k, i)
                              - t_ablock(2, 2, k, i)*t_bblock(0, 2, k, i)
                              - t_ablock(3, 2, k, i)*t_bblock(0, 3, k, i)
                              - t_ablock(4, 2, k, i)*t_bblock(0, 4, k, i);
  t_cblock(0, 3, k, i) = t_cblock(0, 3, k, i) - t_ablock(0, 3, k, i)*t_bblock(0, 0, k, i)
                              - t_ablock(1, 3, k, i)*t_bblock(0, 1, k, i)
                              - t_ablock(2, 3, k, i)*t_bblock(0, 2, k, i)
                              - t_ablock(3, 3, k, i)*t_bblock(0, 3, k, i)
                              - t_ablock(4, 3, k, i)*t_bblock(0, 4, k, i);
  t_cblock(0, 4, k, i) = t_cblock(0, 4, k, i) - t_ablock(0, 4, k, i)*t_bblock(0, 0, k, i)
                              - t_ablock(1, 4, k, i)*t_bblock(0, 1, k, i)
                              - t_ablock(2, 4, k, i)*t_bblock(0, 2, k, i)
                              - t_ablock(3, 4, k, i)*t_bblock(0, 3, k, i)
                              - t_ablock(4, 4, k, i)*t_bblock(0, 4, k, i);
  t_cblock(1, 0, k, i) = t_cblock(1, 0, k, i) - t_ablock(0, 0, k, i)*t_bblock(1, 0, k, i)
                              - t_ablock(1, 0, k, i)*t_bblock(1, 1, k, i)
                              - t_ablock(2, 0, k, i)*t_bblock(1, 2, k, i)
                              - t_ablock(3, 0, k, i)*t_bblock(1, 3, k, i)
                              - t_ablock(4, 0, k, i)*t_bblock(1, 4, k, i);
  t_cblock(1, 1, k, i) = t_cblock(1, 1, k, i) - t_ablock(0, 1, k, i)*t_bblock(1, 0, k, i)
                              - t_ablock(1, 1, k, i)*t_bblock(1, 1, k, i)
                              - t_ablock(2, 1, k, i)*t_bblock(1, 2, k, i)
                              - t_ablock(3, 1, k, i)*t_bblock(1, 3, k, i)
                              - t_ablock(4, 1, k, i)*t_bblock(1, 4, k, i);
  t_cblock(1, 2, k, i) = t_cblock(1, 2, k, i) - t_ablock(0, 2, k, i)*t_bblock(1, 0, k, i)
                              - t_ablock(1, 2, k, i)*t_bblock(1, 1, k, i)
                              - t_ablock(2, 2, k, i)*t_bblock(1, 2, k, i)
                              - t_ablock(3, 2, k, i)*t_bblock(1, 3, k, i)
                              - t_ablock(4, 2, k, i)*t_bblock(1, 4, k, i);
  t_cblock(1, 3, k, i) = t_cblock(1, 3, k, i) - t_ablock(0, 3, k, i)*t_bblock(1, 0, k, i)
                              - t_ablock(1, 3, k, i)*t_bblock(1, 1, k, i)
                              - t_ablock(2, 3, k, i)*t_bblock(1, 2, k, i)
                              - t_ablock(3, 3, k, i)*t_bblock(1, 3, k, i)
                              - t_ablock(4, 3, k, i)*t_bblock(1, 4, k, i);
  t_cblock(1, 4, k, i) = t_cblock(1, 4, k, i) - t_ablock(0, 4, k, i)*t_bblock(1, 0, k, i)
                              - t_ablock(1, 4, k, i)*t_bblock(1, 1, k, i)
                              - t_ablock(2, 4, k, i)*t_bblock(1, 2, k, i)
                              - t_ablock(3, 4, k, i)*t_bblock(1, 3, k, i)
                              - t_ablock(4, 4, k, i)*t_bblock(1, 4, k, i);
  t_cblock(2, 0, k, i) = t_cblock(2, 0, k, i) - t_ablock(0, 0, k, i)*t_bblock(2, 0, k, i)
                              - t_ablock(1, 0, k, i)*t_bblock(2, 1, k, i)
                              - t_ablock(2, 0, k, i)*t_bblock(2, 2, k, i)
                              - t_ablock(3, 0, k, i)*t_bblock(2, 3, k, i)
                              - t_ablock(4, 0, k, i)*t_bblock(2, 4, k, i);
  t_cblock(2, 1, k, i) = t_cblock(2, 1, k, i) - t_ablock(0, 1, k, i)*t_bblock(2, 0, k, i)
                              - t_ablock(1, 1, k, i)*t_bblock(2, 1, k, i)
                              - t_ablock(2, 1, k, i)*t_bblock(2, 2, k, i)
                              - t_ablock(3, 1, k, i)*t_bblock(2, 3, k, i)
                              - t_ablock(4, 1, k, i)*t_bblock(2, 4, k, i);
  t_cblock(2, 2, k, i) = t_cblock(2, 2, k, i) - t_ablock(0, 2, k, i)*t_bblock(2, 0, k, i)
                              - t_ablock(1, 2, k, i)*t_bblock(2, 1, k, i)
                              - t_ablock(2, 2, k, i)*t_bblock(2, 2, k, i)
                              - t_ablock(3, 2, k, i)*t_bblock(2, 3, k, i)
                              - t_ablock(4, 2, k, i)*t_bblock(2, 4, k, i);
  t_cblock(2, 3, k, i) = t_cblock(2, 3, k, i) - t_ablock(0, 3, k, i)*t_bblock(2, 0, k, i)
                              - t_ablock(1, 3, k, i)*t_bblock(2, 1, k, i)
                              - t_ablock(2, 3, k, i)*t_bblock(2, 2, k, i)
                              - t_ablock(3, 3, k, i)*t_bblock(2, 3, k, i)
                              - t_ablock(4, 3, k, i)*t_bblock(2, 4, k, i);
  t_cblock(2, 4, k, i) = t_cblock(2, 4, k, i) - t_ablock(0, 4, k, i)*t_bblock(2, 0, k, i)
                              - t_ablock(1, 4, k, i)*t_bblock(2, 1, k, i)
                              - t_ablock(2, 4, k, i)*t_bblock(2, 2, k, i)
                              - t_ablock(3, 4, k, i)*t_bblock(2, 3, k, i)
                              - t_ablock(4, 4, k, i)*t_bblock(2, 4, k, i);
  t_cblock(3, 0, k, i) = t_cblock(3, 0, k, i) - t_ablock(0, 0, k, i)*t_bblock(3, 0, k, i)
                              - t_ablock(1, 0, k, i)*t_bblock(3, 1, k, i)
                              - t_ablock(2, 0, k, i)*t_bblock(3, 2, k, i)
                              - t_ablock(3, 0, k, i)*t_bblock(3, 3, k, i)
                              - t_ablock(4, 0, k, i)*t_bblock(3, 4, k, i);
  t_cblock(3, 1, k, i) = t_cblock(3, 1, k, i) - t_ablock(0, 1, k, i)*t_bblock(3, 0, k, i)
                              - t_ablock(1, 1, k, i)*t_bblock(3, 1, k, i)
                              - t_ablock(2, 1, k, i)*t_bblock(3, 2, k, i)
                              - t_ablock(3, 1, k, i)*t_bblock(3, 3, k, i)
                              - t_ablock(4, 1, k, i)*t_bblock(3, 4, k, i);
  t_cblock(3, 2, k, i) = t_cblock(3, 2, k, i) - t_ablock(0, 2, k, i)*t_bblock(3, 0, k, i)
                              - t_ablock(1, 2, k, i)*t_bblock(3, 1, k, i)
                              - t_ablock(2, 2, k, i)*t_bblock(3, 2, k, i)
                              - t_ablock(3, 2, k, i)*t_bblock(3, 3, k, i)
                              - t_ablock(4, 2, k, i)*t_bblock(3, 4, k, i);
  t_cblock(3, 3, k, i) = t_cblock(3, 3, k, i) - t_ablock(0, 3, k, i)*t_bblock(3, 0, k, i)
                              - t_ablock(1, 3, k, i)*t_bblock(3, 1, k, i)
                              - t_ablock(2, 3, k, i)*t_bblock(3, 2, k, i)
                              - t_ablock(3, 3, k, i)*t_bblock(3, 3, k, i)
                              - t_ablock(4, 3, k, i)*t_bblock(3, 4, k, i);
  t_cblock(3, 4, k, i) = t_cblock(3, 4, k, i) - t_ablock(0, 4, k, i)*t_bblock(3, 0, k, i)
                              - t_ablock(1, 4, k, i)*t_bblock(3, 1, k, i)
                              - t_ablock(2, 4, k, i)*t_bblock(3, 2, k, i)
                              - t_ablock(3, 4, k, i)*t_bblock(3, 3, k, i)
                              - t_ablock(4, 4, k, i)*t_bblock(3, 4, k, i);
  t_cblock(4, 0, k, i) = t_cblock(4, 0, k, i) - t_ablock(0, 0, k, i)*t_bblock(4, 0, k, i)
                              - t_ablock(1, 0, k, i)*t_bblock(4, 1, k, i)
                              - t_ablock(2, 0, k, i)*t_bblock(4, 2, k, i)
                              - t_ablock(3, 0, k, i)*t_bblock(4, 3, k, i)
                              - t_ablock(4, 0, k, i)*t_bblock(4, 4, k, i);
  t_cblock(4, 1, k, i) = t_cblock(4, 1, k, i) - t_ablock(0, 1, k, i)*t_bblock(4, 0, k, i)
                              - t_ablock(1, 1, k, i)*t_bblock(4, 1, k, i)
                              - t_ablock(2, 1, k, i)*t_bblock(4, 2, k, i)
                              - t_ablock(3, 1, k, i)*t_bblock(4, 3, k, i)
                              - t_ablock(4, 1, k, i)*t_bblock(4, 4, k, i);
  t_cblock(4, 2, k, i) = t_cblock(4, 2, k, i) - t_ablock(0, 2, k, i)*t_bblock(4, 0, k, i)
                              - t_ablock(1, 2, k, i)*t_bblock(4, 1, k, i)
                              - t_ablock(2, 2, k, i)*t_bblock(4, 2, k, i)
                              - t_ablock(3, 2, k, i)*t_bblock(4, 3, k, i)
                              - t_ablock(4, 2, k, i)*t_bblock(4, 4, k, i);
  t_cblock(4, 3, k, i) = t_cblock(4, 3, k, i) - t_ablock(0, 3, k, i)*t_bblock(4, 0, k, i)
                              - t_ablock(1, 3, k, i)*t_bblock(4, 1, k, i)
                              - t_ablock(2, 3, k, i)*t_bblock(4, 2, k, i)
                              - t_ablock(3, 3, k, i)*t_bblock(4, 3, k, i)
                              - t_ablock(4, 3, k, i)*t_bblock(4, 4, k, i);
  t_cblock(4, 4, k, i) = t_cblock(4, 4, k, i) - t_ablock(0, 4, k, i)*t_bblock(4, 0, k, i)
                              - t_ablock(1, 4, k, i)*t_bblock(4, 1, k, i)
                              - t_ablock(2, 4, k, i)*t_bblock(4, 2, k, i)
                              - t_ablock(3, 4, k, i)*t_bblock(4, 3, k, i)
                              - t_ablock(4, 4, k, i)*t_bblock(4, 4, k, i);
#undef t_ablock
#undef t_bblock
#undef t_cblock
}

__device__
void binvrhs(double *lhs, 
             double r[5],
             int k, int i, int WORK_NUM_ITEM_DEFAULT)
{
#define t_lhs(a, b, c, d) lhs[(((a)*5 + (b))*WORK_NUM_ITEM_DEFAULT + (c))*(PROBLEM_SIZE-1) + (d)]
  double pivot, coeff;

  pivot = 1.00/t_lhs(0, 0, k, i);
  t_lhs(1, 0, k, i) = t_lhs(1, 0, k, i)*pivot;
  t_lhs(2, 0, k, i) = t_lhs(2, 0, k, i)*pivot;
  t_lhs(3, 0, k, i) = t_lhs(3, 0, k, i)*pivot;
  t_lhs(4, 0, k, i) = t_lhs(4, 0, k, i)*pivot;
  r[0]   = r[0]  *pivot;

  coeff = t_lhs(0, 1, k, i);
  t_lhs(1, 1, k, i)= t_lhs(1, 1, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 1, k, i)= t_lhs(2, 1, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 1, k, i)= t_lhs(3, 1, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 1, k, i)= t_lhs(4, 1, k, i) - coeff*t_lhs(4, 0, k, i);
  r[1]   = r[1]   - coeff*r[0];

  coeff = t_lhs(0, 2, k, i);
  t_lhs(1, 2, k, i)= t_lhs(1, 2, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 2, k, i)= t_lhs(2, 2, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 2, k, i)= t_lhs(3, 2, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 2, k, i)= t_lhs(4, 2, k, i) - coeff*t_lhs(4, 0, k, i);
  r[2]   = r[2]   - coeff*r[0];

  coeff = t_lhs(0, 3, k, i);
  t_lhs(1, 3, k, i)= t_lhs(1, 3, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 3, k, i)= t_lhs(2, 3, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 3, k, i)= t_lhs(3, 3, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 3, k, i)= t_lhs(4, 3, k, i) - coeff*t_lhs(4, 0, k, i);
  r[3]   = r[3]   - coeff*r[0];

  coeff = t_lhs(0, 4, k, i);
  t_lhs(1, 4, k, i)= t_lhs(1, 4, k, i) - coeff*t_lhs(1, 0, k, i);
  t_lhs(2, 4, k, i)= t_lhs(2, 4, k, i) - coeff*t_lhs(2, 0, k, i);
  t_lhs(3, 4, k, i)= t_lhs(3, 4, k, i) - coeff*t_lhs(3, 0, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 0, k, i);
  r[4]   = r[4]   - coeff*r[0];


  pivot = 1.00/t_lhs(1, 1, k, i);
  t_lhs(2, 1, k, i) = t_lhs(2, 1, k, i)*pivot;
  t_lhs(3, 1, k, i) = t_lhs(3, 1, k, i)*pivot;
  t_lhs(4, 1, k, i) = t_lhs(4, 1, k, i)*pivot;
  r[1]   = r[1]  *pivot;

  coeff = t_lhs(1, 0, k, i);
  t_lhs(2, 0, k, i)= t_lhs(2, 0, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 0, k, i)= t_lhs(3, 0, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 0, k, i)= t_lhs(4, 0, k, i) - coeff*t_lhs(4, 1, k, i);
  r[0]   = r[0]   - coeff*r[1];

  coeff = t_lhs(1, 2, k, i);
  t_lhs(2, 2, k, i)= t_lhs(2, 2, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 2, k, i)= t_lhs(3, 2, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 2, k, i)= t_lhs(4, 2, k, i) - coeff*t_lhs(4, 1, k, i);
  r[2]   = r[2]   - coeff*r[1];

  coeff = t_lhs(1, 3, k, i);
  t_lhs(2, 3, k, i)= t_lhs(2, 3, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 3, k, i)= t_lhs(3, 3, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 3, k, i)= t_lhs(4, 3, k, i) - coeff*t_lhs(4, 1, k, i);
  r[3]   = r[3]   - coeff*r[1];

  coeff = t_lhs(1, 4, k, i);
  t_lhs(2, 4, k, i)= t_lhs(2, 4, k, i) - coeff*t_lhs(2, 1, k, i);
  t_lhs(3, 4, k, i)= t_lhs(3, 4, k, i) - coeff*t_lhs(3, 1, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 1, k, i);
  r[4]   = r[4]   - coeff*r[1];


  pivot = 1.00/t_lhs(2, 2, k, i);
  t_lhs(3, 2, k, i) = t_lhs(3, 2, k, i)*pivot;
  t_lhs(4, 2, k, i) = t_lhs(4, 2, k, i)*pivot;
  r[2]   = r[2]  *pivot;

  coeff = t_lhs(2, 0, k, i);
  t_lhs(3, 0, k, i)= t_lhs(3, 0, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 0, k, i)= t_lhs(4, 0, k, i) - coeff*t_lhs(4, 2, k, i);
  r[0]   = r[0]   - coeff*r[2];

  coeff = t_lhs(2, 1, k, i);
  t_lhs(3, 1, k, i)= t_lhs(3, 1, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 1, k, i)= t_lhs(4, 1, k, i) - coeff*t_lhs(4, 2, k, i);
  r[1]   = r[1]   - coeff*r[2];

  coeff = t_lhs(2, 3, k, i);
  t_lhs(3, 3, k, i)= t_lhs(3, 3, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 3, k, i)= t_lhs(4, 3, k, i) - coeff*t_lhs(4, 2, k, i);
  r[3]   = r[3]   - coeff*r[2];

  coeff = t_lhs(2, 4, k, i);
  t_lhs(3, 4, k, i)= t_lhs(3, 4, k, i) - coeff*t_lhs(3, 2, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 2, k, i);
  r[4]   = r[4]   - coeff*r[2];


  pivot = 1.00/t_lhs(3, 3, k, i);
  t_lhs(4, 3, k, i) = t_lhs(4, 3, k, i)*pivot;
  r[3]   = r[3]  *pivot;

  coeff = t_lhs(3, 0, k, i);
  t_lhs(4, 0, k, i)= t_lhs(4, 0, k, i) - coeff*t_lhs(4, 3, k, i);
  r[0]   = r[0]   - coeff*r[3];

  coeff = t_lhs(3, 1, k, i);
  t_lhs(4, 1, k, i)= t_lhs(4, 1, k, i) - coeff*t_lhs(4, 3, k, i);
  r[1]   = r[1]   - coeff*r[3];

  coeff = t_lhs(3, 2, k, i);
  t_lhs(4, 2, k, i)= t_lhs(4, 2, k, i) - coeff*t_lhs(4, 3, k, i);
  r[2]   = r[2]   - coeff*r[3];

  coeff = t_lhs(3, 4, k, i);
  t_lhs(4, 4, k, i)= t_lhs(4, 4, k, i) - coeff*t_lhs(4, 3, k, i);
  r[4]   = r[4]   - coeff*r[3];


  pivot = 1.00/t_lhs(4, 4, k, i);
  r[4]   = r[4]  *pivot;

  coeff = t_lhs(4, 0, k, i);
  r[0]   = r[0]   - coeff*r[4];

  coeff = t_lhs(4, 1, k, i);
  r[1]   = r[1]   - coeff*r[4];

  coeff = t_lhs(4, 2, k, i);
  r[2]   = r[2]   - coeff*r[4];

  coeff = t_lhs(4, 3, k, i);
  r[3]   = r[3]   - coeff*r[4];

#undef t_lhs
}


__launch_bounds__(min(PROBLEM_SIZE-2, MAX_THREAD_DIM_0))
__global__ 
void k_y_solve_memlayout(double *m_qs,
                         double *m_rho_i,
                         double *m_square,
                         double *m_u,
                         double *m_rhs,
                         double *m_lhs, 
                         double *m_fjac,
                         double *m_njac,
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
                         int split_flag,
                         int WORK_NUM_ITEM_DEFAULT)
{
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

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

  jsize = gp1-1;

  for (j = 0; j <= jsize; j++) {
    tmp1 = rho_i[k][j][i];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    fjac(j, 0, 0, k, i) = 0.0;
    fjac(j, 1, 0, k, i) = 0.0;
    fjac(j, 2, 0, k, i) = 1.0;
    fjac(j, 3, 0, k, i) = 0.0;
    fjac(j, 4, 0, k, i) = 0.0;

    fjac(j, 0, 1, k, i) = - ( u[k][j][i][1]*u[k][j][i][2] ) * tmp2;
    fjac(j, 1, 1, k, i) = u[k][j][i][2] * tmp1;
    fjac(j, 2, 1, k, i) = u[k][j][i][1] * tmp1;
    fjac(j, 3, 1, k, i) = 0.0;
    fjac(j, 4, 1, k, i) = 0.0;

    fjac(j, 0, 2, k, i) = - ( u[k][j][i][2]*u[k][j][i][2]*tmp2)
      + c2 * qs[k][j][i];
    fjac(j, 1, 2, k, i) = - c2 *  u[k][j][i][1] * tmp1;
    fjac(j, 2, 2, k, i) = ( 2.0 - c2 ) *  u[k][j][i][2] * tmp1;
    fjac(j, 3, 2, k, i) = - c2 * u[k][j][i][3] * tmp1;
    fjac(j, 4, 2, k, i) = c2;

    fjac(j, 0, 3, k, i) = - ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac(j, 1, 3, k, i) = 0.0;
    fjac(j, 2, 3, k, i) = u[k][j][i][3] * tmp1;
    fjac(j, 3, 3, k, i) = u[k][j][i][2] * tmp1;
    fjac(j, 4, 3, k, i) = 0.0;

    fjac(j, 0, 4, k, i) = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
      * u[k][j][i][2] * tmp2;
    fjac(j, 1, 4, k, i) = - c2 * u[k][j][i][1]*u[k][j][i][2] * tmp2;
    fjac(j, 2, 4, k, i) = c1 * u[k][j][i][4] * tmp1 
      - c2 * ( qs[k][j][i] + u[k][j][i][2]*u[k][j][i][2] * tmp2 );
    fjac(j, 3, 4, k, i) = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * tmp2;
    fjac(j, 4, 4, k, i) = c1 * u[k][j][i][2] * tmp1;

    njac(j, 0, 0, k, i) = 0.0;
    njac(j, 1, 0, k, i) = 0.0;
    njac(j, 2, 0, k, i) = 0.0;
    njac(j, 3, 0, k, i) = 0.0;
    njac(j, 4, 0, k, i) = 0.0;

    njac(j, 0, 1, k, i) = - c3c4 * tmp2 * u[k][j][i][1];
    njac(j, 1, 1, k, i) =   c3c4 * tmp1;
    njac(j, 2, 1, k, i) =   0.0;
    njac(j, 3, 1, k, i) =   0.0;
    njac(j, 4, 1, k, i) =   0.0;

    njac(j, 0, 2, k, i) = - con43 * c3c4 * tmp2 * u[k][j][i][2];
    njac(j, 1, 2, k, i) =   0.0;
    njac(j, 2, 2, k, i) =   con43 * c3c4 * tmp1;
    njac(j, 3, 2, k, i) =   0.0;
    njac(j, 4, 2, k, i) =   0.0;

    njac(j, 0, 3, k, i) = - c3c4 * tmp2 * u[k][j][i][3];
    njac(j, 1, 3, k, i) =   0.0;
    njac(j, 2, 3, k, i) =   0.0;
    njac(j, 3, 3, k, i) =   c3c4 * tmp1;
    njac(j, 4, 3, k, i) =   0.0;

    njac(j, 0, 4, k, i) = - (  c3c4
        - c1345 ) * tmp3 * (u[k][j][i][1]*u[k][j][i][1])
      - ( con43 * c3c4
          - c1345 ) * tmp3 * (u[k][j][i][2]*u[k][j][i][2])
      - ( c3c4 - c1345 ) * tmp3 * (u[k][j][i][3]*u[k][j][i][3])
      - c1345 * tmp2 * u[k][j][i][4];

    njac(j, 1, 4, k, i) = (  c3c4 - c1345 ) * tmp2 * u[k][j][i][1];
    njac(j, 2, 4, k, i) = ( con43 * c3c4 - c1345 ) * tmp2 * u[k][j][i][2];
    njac(j, 3, 4, k, i) = ( c3c4 - c1345 ) * tmp2 * u[k][j][i][3];
    njac(j, 4, 4, k, i) = ( c1345 ) * tmp1;
  }

  //---------------------------------------------------------------------
  // now joacobians set, so form left hand side in y direction
  //---------------------------------------------------------------------
  lhsinit_y(&lhs(0, 0, 0, 0, 0, 0), jsize, k, i, WORK_NUM_ITEM_DEFAULT);
  for (j = 1; j <= jsize-1; j++) {
    tmp1 = dt * ty1;
    tmp2 = dt * ty2;

    lhs(j, AA, 0, 0, k, i) = - tmp2 * fjac(j-1, 0, 0, k, i)
      - tmp1 * njac(j-1, 0, 0, k, i)
      - tmp1 * dy1; 
    lhs(j, AA, 1, 0, k, i) = - tmp2 * fjac(j-1, 1, 0, k, i)
      - tmp1 * njac(j-1, 1, 0, k, i);
    lhs(j, AA, 2, 0, k, i) = - tmp2 * fjac(j-1, 2, 0, k, i)
      - tmp1 * njac(j-1, 2, 0, k, i);
    lhs(j, AA, 3, 0, k, i) = - tmp2 * fjac(j-1, 3, 0, k, i)
      - tmp1 * njac(j-1, 3, 0, k, i);
    lhs(j, AA, 4, 0, k, i) = - tmp2 * fjac(j-1, 4, 0, k, i)
      - tmp1 * njac(j-1, 4, 0, k, i);

    lhs(j, AA, 0, 1, k, i) = - tmp2 * fjac(j-1, 0, 1, k, i)
      - tmp1 * njac(j-1, 0, 1, k, i);
    lhs(j, AA, 1, 1, k, i) = - tmp2 * fjac(j-1, 1, 1, k, i)
      - tmp1 * njac(j-1, 1, 1, k, i)
      - tmp1 * dy2;
    lhs(j, AA, 2, 1, k, i) = - tmp2 * fjac(j-1, 2, 1, k, i)
      - tmp1 * njac(j-1, 2, 1, k, i);
    lhs(j, AA, 3, 1, k, i) = - tmp2 * fjac(j-1, 3, 1, k, i)
      - tmp1 * njac(j-1, 3, 1, k, i);
    lhs(j, AA, 4, 1, k, i) = - tmp2 * fjac(j-1, 4, 1, k, i)
      - tmp1 * njac(j-1, 4, 1, k, i);

    lhs(j, AA, 0, 2, k, i) = - tmp2 * fjac(j-1, 0, 2, k, i)
      - tmp1 * njac(j-1, 0, 2, k, i);
    lhs(j, AA, 1, 2, k, i) = - tmp2 * fjac(j-1, 1, 2, k, i)
      - tmp1 * njac(j-1, 1, 2, k, i);
    lhs(j, AA, 2, 2, k, i) = - tmp2 * fjac(j-1, 2, 2, k, i)
      - tmp1 * njac(j-1, 2, 2, k, i)
      - tmp1 * dy3;
    lhs(j, AA, 3, 2, k, i) = - tmp2 * fjac(j-1, 3, 2, k, i)
      - tmp1 * njac(j-1, 3, 2, k, i);
    lhs(j, AA, 4, 2, k, i) = - tmp2 * fjac(j-1, 4, 2, k, i)
      - tmp1 * njac(j-1, 4, 2, k, i);

    lhs(j, AA, 0, 3, k, i) = - tmp2 * fjac(j-1, 0, 3, k, i)
      - tmp1 * njac(j-1, 0, 3, k, i);
    lhs(j, AA, 1, 3, k, i) = - tmp2 * fjac(j-1, 1, 3, k, i)
      - tmp1 * njac(j-1, 1, 3, k, i);
    lhs(j, AA, 2, 3, k, i) = - tmp2 * fjac(j-1, 2, 3, k, i)
      - tmp1 * njac(j-1, 2, 3, k, i);
    lhs(j, AA, 3, 3, k, i) = - tmp2 * fjac(j-1, 3, 3, k, i)
      - tmp1 * njac(j-1, 3, 3, k, i)
      - tmp1 * dy4;
    lhs(j, AA, 4, 3, k, i) = - tmp2 * fjac(j-1, 4, 3, k, i)
      - tmp1 * njac(j-1, 4, 3, k, i);

    lhs(j, AA, 0, 4, k, i) = - tmp2 * fjac(j-1, 0, 4, k, i)
      - tmp1 * njac(j-1, 0, 4, k, i);
    lhs(j, AA, 1, 4, k, i) = - tmp2 * fjac(j-1, 1, 4, k, i)
      - tmp1 * njac(j-1, 1, 4, k, i);
    lhs(j, AA, 2, 4, k, i) = - tmp2 * fjac(j-1, 2, 4, k, i)
      - tmp1 * njac(j-1, 2, 4, k, i);
    lhs(j, AA, 3, 4, k, i) = - tmp2 * fjac(j-1, 3, 4, k, i)
      - tmp1 * njac(j-1, 3, 4, k, i);
    lhs(j, AA, 4, 4, k, i) = - tmp2 * fjac(j-1, 4, 4, k, i)
      - tmp1 * njac(j-1, 4, 4, k, i)
      - tmp1 * dy5;

    lhs(j, BB, 0, 0, k, i) = 1.0
      + tmp1 * 2.0 * njac(j, 0, 0, k, i)
      + tmp1 * 2.0 * dy1;
    lhs(j, BB, 1, 0, k, i) = tmp1 * 2.0 * njac(j, 1, 0, k, i);
    lhs(j, BB, 2, 0, k, i) = tmp1 * 2.0 * njac(j, 2, 0, k, i);
    lhs(j, BB, 3, 0, k, i) = tmp1 * 2.0 * njac(j, 3, 0, k, i);
    lhs(j, BB, 4, 0, k, i) = tmp1 * 2.0 * njac(j, 4, 0, k, i);

    lhs(j, BB, 0, 1, k, i) = tmp1 * 2.0 * njac(j, 0, 1, k, i);
    lhs(j, BB, 1, 1, k, i) = 1.0
      + tmp1 * 2.0 * njac(j, 1, 1, k, i)
      + tmp1 * 2.0 * dy2;
    lhs(j, BB, 2, 1, k, i) = tmp1 * 2.0 * njac(j, 2, 1, k, i);
    lhs(j, BB, 3, 1, k, i) = tmp1 * 2.0 * njac(j, 3, 1, k, i);
    lhs(j, BB, 4, 1, k, i) = tmp1 * 2.0 * njac(j, 4, 1, k, i);

    lhs(j, BB, 0, 2, k, i) = tmp1 * 2.0 * njac(j, 0, 2, k, i);
    lhs(j, BB, 1, 2, k, i) = tmp1 * 2.0 * njac(j, 1, 2, k, i);
    lhs(j, BB, 2, 2, k, i) = 1.0
      + tmp1 * 2.0 * njac(j, 2, 2, k, i)
      + tmp1 * 2.0 * dy3;
    lhs(j, BB, 3, 2, k, i) = tmp1 * 2.0 * njac(j, 3, 2, k, i);
    lhs(j, BB, 4, 2, k, i) = tmp1 * 2.0 * njac(j, 4, 2, k, i);

    lhs(j, BB, 0, 3, k, i) = tmp1 * 2.0 * njac(j, 0, 3, k, i);
    lhs(j, BB, 1, 3, k, i) = tmp1 * 2.0 * njac(j, 1, 3, k, i);
    lhs(j, BB, 2, 3, k, i) = tmp1 * 2.0 * njac(j, 2, 3, k, i);
    lhs(j, BB, 3, 3, k, i) = 1.0
      + tmp1 * 2.0 * njac(j, 3, 3, k, i)
      + tmp1 * 2.0 * dy4;
    lhs(j, BB, 4, 3, k, i) = tmp1 * 2.0 * njac(j, 4, 3, k, i);

    lhs(j, BB, 0, 4, k, i) = tmp1 * 2.0 * njac(j, 0, 4, k, i);
    lhs(j, BB, 1, 4, k, i) = tmp1 * 2.0 * njac(j, 1, 4, k, i);
    lhs(j, BB, 2, 4, k, i) = tmp1 * 2.0 * njac(j, 2, 4, k, i);
    lhs(j, BB, 3, 4, k, i) = tmp1 * 2.0 * njac(j, 3, 4, k, i);
    lhs(j, BB, 4, 4, k, i) = 1.0
      + tmp1 * 2.0 * njac(j, 4, 4, k, i) 
      + tmp1 * 2.0 * dy5;

    lhs(j, CC, 0, 0, k, i) =  tmp2 * fjac(j+1, 0, 0, k, i)
      - tmp1 * njac(j+1, 0, 0, k, i)
      - tmp1 * dy1;
    lhs(j, CC, 1, 0, k, i) =  tmp2 * fjac(j+1, 1, 0, k, i)
      - tmp1 * njac(j+1, 1, 0, k, i);
    lhs(j, CC, 2, 0, k, i) =  tmp2 * fjac(j+1, 2, 0, k, i)
      - tmp1 * njac(j+1, 2, 0, k, i);
    lhs(j, CC, 3, 0, k, i) =  tmp2 * fjac(j+1, 3, 0, k, i)
      - tmp1 * njac(j+1, 3, 0, k, i);
    lhs(j, CC, 4, 0, k, i) =  tmp2 * fjac(j+1, 4, 0, k, i)
      - tmp1 * njac(j+1, 4, 0, k, i);

    lhs(j, CC, 0, 1, k, i) =  tmp2 * fjac(j+1, 0, 1, k, i)
      - tmp1 * njac(j+1, 0, 1, k, i);
    lhs(j, CC, 1, 1, k, i) =  tmp2 * fjac(j+1, 1, 1, k, i)
      - tmp1 * njac(j+1, 1, 1, k, i)
      - tmp1 * dy2;
    lhs(j, CC, 2, 1, k, i) =  tmp2 * fjac(j+1, 2, 1, k, i)
      - tmp1 * njac(j+1, 2, 1, k, i);
    lhs(j, CC, 3, 1, k, i) =  tmp2 * fjac(j+1, 3, 1, k, i)
      - tmp1 * njac(j+1, 3, 1, k, i);
    lhs(j, CC, 4, 1, k, i) =  tmp2 * fjac(j+1, 4, 1, k, i)
      - tmp1 * njac(j+1, 4, 1, k, i);

    lhs(j, CC, 0, 2, k, i) =  tmp2 * fjac(j+1, 0, 2, k, i)
      - tmp1 * njac(j+1, 0, 2, k, i);
    lhs(j, CC, 1, 2, k, i) =  tmp2 * fjac(j+1, 1, 2, k, i)
      - tmp1 * njac(j+1, 1, 2, k, i);
    lhs(j, CC, 2, 2, k, i) =  tmp2 * fjac(j+1, 2, 2, k, i)
      - tmp1 * njac(j+1, 2, 2, k, i)
      - tmp1 * dy3;
    lhs(j, CC, 3, 2, k, i) =  tmp2 * fjac(j+1, 3, 2, k, i)
      - tmp1 * njac(j+1, 3, 2, k, i);
    lhs(j, CC, 4, 2, k, i) =  tmp2 * fjac(j+1, 4, 2, k, i)
      - tmp1 * njac(j+1, 4, 2, k, i);

    lhs(j, CC, 0, 3, k, i) =  tmp2 * fjac(j+1, 0, 3, k, i)
      - tmp1 * njac(j+1, 0, 3, k, i);
    lhs(j, CC, 1, 3, k, i) =  tmp2 * fjac(j+1, 1, 3, k, i)
      - tmp1 * njac(j+1, 1, 3, k, i);
    lhs(j, CC, 2, 3, k, i) =  tmp2 * fjac(j+1, 2, 3, k, i)
      - tmp1 * njac(j+1, 2, 3, k, i);
    lhs(j, CC, 3, 3, k, i) =  tmp2 * fjac(j+1, 3, 3, k, i)
      - tmp1 * njac(j+1, 3, 3, k, i)
      - tmp1 * dy4;
    lhs(j, CC, 4, 3, k, i) =  tmp2 * fjac(j+1, 4, 3, k, i)
      - tmp1 * njac(j+1, 4, 3, k, i);

    lhs(j, CC, 0, 4, k, i) =  tmp2 * fjac(j+1, 0, 4, k, i)
      - tmp1 * njac(j+1, 0, 4, k, i);
    lhs(j, CC, 1, 4, k, i) =  tmp2 * fjac(j+1, 1, 4, k, i)
      - tmp1 * njac(j+1, 1, 4, k, i);
    lhs(j, CC, 2, 4, k, i) =  tmp2 * fjac(j+1, 2, 4, k, i)
      - tmp1 * njac(j+1, 2, 4, k, i);
    lhs(j, CC, 3, 4, k, i) =  tmp2 * fjac(j+1, 3, 4, k, i)
      - tmp1 * njac(j+1, 3, 4, k, i);
    lhs(j, CC, 4, 4, k, i) =  tmp2 * fjac(j+1, 4, 4, k, i)
      - tmp1 * njac(j+1, 4, 4, k, i)
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
  binvcrhs_y( &lhs(0, BB, 0, 0, 0, 0), &lhs(0, CC, 0, 0, 0, 0), rhs[k][0][i], k, i, WORK_NUM_ITEM_DEFAULT);

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
    matvec_sub_y(&lhs(j, AA, 0, 0, 0, 0), rhs[k][j-1][i], rhs[k][j][i], k, i, WORK_NUM_ITEM_DEFAULT);

    //-------------------------------------------------------------------
    // B(j) = B(j) - C(j-1)*A(j)
    //-------------------------------------------------------------------
    matmul_sub_y(&lhs(j, AA, 0, 0, 0, 0), &lhs(j-1, CC, 0, 0, 0, 0), &lhs(j, BB, 0, 0, 0, 0), k, i, WORK_NUM_ITEM_DEFAULT);

    //-------------------------------------------------------------------
    // multiply c[k][j][i] by b_inverse and copy back to c
    // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
    //-------------------------------------------------------------------
    binvcrhs_y(&lhs(j, BB, 0, 0, 0, 0), &lhs(j, CC, 0, 0, 0, 0), rhs[k][j][i], k, i, WORK_NUM_ITEM_DEFAULT);
  }

  //---------------------------------------------------------------------
  // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
  //---------------------------------------------------------------------
  matvec_sub_y(&lhs(jsize, AA, 0, 0, 0, 0), rhs[k][jsize-1][i], rhs[k][jsize][i], k, i, WORK_NUM_ITEM_DEFAULT);

  //---------------------------------------------------------------------
  // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
  // matmul_sub(AA,i,jsize,k,c,
  // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
  //---------------------------------------------------------------------
  matmul_sub_y(&lhs(jsize, AA, 0, 0, 0, 0), &lhs(jsize-1, CC, 0, 0, 0, 0), &lhs(jsize, BB, 0, 0, 0, 0), k, i, WORK_NUM_ITEM_DEFAULT);

  //---------------------------------------------------------------------
  // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
  //---------------------------------------------------------------------
  binvrhs(&lhs(jsize, BB, 0, 0, 0, 0), rhs[k][jsize][i], k, i, WORK_NUM_ITEM_DEFAULT);

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
          - lhs(j, CC, n, m, k, i)*rhs[k][j+1][i][n];
      }
    }
  }

#undef lhs
#undef fjac
#undef njac
}


