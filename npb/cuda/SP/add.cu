//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB SP code. This CUDA® C  //
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
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
    int i, j, k, m;

    if (timeron) timer_start(t_add);
    for (k = 1; k <= nz2; k++) {
        for (j = 1; j <= ny2; j++) {
            for (i = 1; i <= nx2; i++) {
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
                }
            }
        }
    }
    if (timeron) timer_stop(t_add);
}

void add_gpu(int t, int st, int ed, int st2, int ed2)
{
    //---------------------------------------------------------------------
    // add kernel
    //---------------------------------------------------------------------
    int add_base_j = st2;
    int add_offset_j = st - st2;
    int add_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
        int add_gws[] = {nz2};
        int add_lws[] = {16};
        add_gws[0] = RoundWorkSize(add_gws[0], add_lws[0]);
        dim3 blockSize(add_gws[0]/add_lws[0], 1, 1);
        dim3 threadSize(add_lws[0], 1, 1);

        cuda_ProfilerStartEventRecord("k_add_base", cmd_queue[0]);
        k_add_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
             buf_u0, buf_u1, buf_u2, buf_u3, buf_u4,
             add_base_j, add_offset_j, add_gws_j, nx2, ny2, nz2,
             WORK_NUM_ITEM_J
            );
        cuda_ProfilerEndEventRecord("k_add_base", cmd_queue[0]);
    }
    else {
        int add_gws[] = {nx2, add_gws_j, nz2};
        int add_lws[] = {16, 16, 1};
        add_gws[0] = RoundWorkSize(add_gws[0], add_lws[0]);
        add_gws[1] = RoundWorkSize(add_gws[1], add_lws[1]);
        add_gws[2] = RoundWorkSize(add_gws[2], add_lws[2]);
        dim3 blockSize(add_gws[0]/add_lws[0],
                       add_gws[1]/add_lws[1],
                       add_gws[2]/add_lws[2]);
        dim3 threadSize(add_lws[0], add_lws[1], add_lws[2]);

        cuda_ProfilerStartEventRecord("k_add_opt", cmd_queue[0]);
        k_add_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
             buf_u0, buf_u1, buf_u2, buf_u3, buf_u4,
             add_base_j, add_offset_j, add_gws_j, nx2, ny2, nz2,
             WORK_NUM_ITEM_J
            );
        cuda_ProfilerEndEventRecord("k_add_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    if (timeron) timer_stop(t_add);
}
