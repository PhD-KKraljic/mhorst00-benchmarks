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

__kernel void l2norm_baseline(__global double *m_v,
                              __global double *m_sum,
                              __global double *m_tmp_sum,
                              int nz0,
                              int jst, int jend,
                              int ist, int iend,
                              int work_base,
                              int work_num_item, 
                              int split_flag,
                              int buffer_base)
{
  int m;
  int temp = get_global_id(0);
  int i = temp % (iend - ist) + ist;
  temp = temp / (iend-ist);
  int j = temp % (jend-jst) + jst;
  int k = temp / (jend-jst);

  int l_id = get_local_id(0);
  int l_size = get_local_size(0);
  int wg_id = get_group_id(0);
  int step;
  int dummy = 0;

  if (k + work_base >= nz0-1 || k+work_base < 1 || k >= work_num_item || j >= jend || i >= iend)
    dummy = 1;

  if (split_flag) 
    k += buffer_base;
  else
    k += work_base;

  __global double (* v)[ISIZ2/2*2+1][ISIZ1/2*2+1][5]
    = (__global double (*)[ISIZ2/2*2+1][ISIZ1/2*2+1][5])m_v;
  __global double (* tmp_sum)[5]
    = (__global double (*)[5])m_tmp_sum;
  __global double (* sum)[5]
    = (__global double (*)[5])m_sum;

  __global double (* t_sum)[5] = &tmp_sum[wg_id * l_size];

  for (m = 0; m < 5; m++)
    t_sum[l_id][m] = 0.0;

  if (!dummy) {
    for (m = 0; m < 5; m++)
      t_sum[l_id][m] = v[k][j][i][m] * v[k][j][i][m];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  for (step = l_size/2; step > 0; step = step >> 1) {
    if (l_id < step) {
      for (m = 0; m < 5; m++)
        t_sum[l_id][m] += t_sum[l_id+step][m];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  if (l_id == 0) {
    for (m = 0; m < 5; m++)
      sum[wg_id][m] += t_sum[l_id][m];
  }
}
