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

#include "npbparams.h"
#include "header.h"
__global__ void k_add(double * m_u, double * m_rhs,
                      int gp0, int gp1, int gp2,
                      int work_base, 
                      int work_num_item, 
                      int split_flag, 
                      int WORK_NUM_ITEM_DEFAULT_J)
{
  int k = blockIdx.z * blockDim.z + threadIdx.z+1;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int t_i = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t_i/5 + 1;
  int m = t_i%5;

  if (k > gp2-2 || j+work_base < 1 || j+work_base > gp1-2 || j >= work_num_item || i > gp0-2) return;

  if (!split_flag) j += work_base;

  m_u[((k * WORK_NUM_ITEM_DEFAULT_J + j) * (IMAXP+1) + i) * 5 + m]
    += m_rhs[((k * WORK_NUM_ITEM_DEFAULT_J + j) * (IMAXP+1) + i) * 5 + m];
}
