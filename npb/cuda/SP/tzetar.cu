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
// block-diagonal matrix-vector multiplication                       
//---------------------------------------------------------------------
void tzetar()
{
    int i, j, k;
    double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
    double btuz, ac2u, uzik1;

    if (timeron) timer_start(t_tzetar);
    for (k = 1; k <= nz2; k++) {
        for (j = 1; j <= ny2; j++) {
            for (i = 1; i <= nx2; i++) {
                xvel = us[k][j][i];
                yvel = vs[k][j][i];
                zvel = ws[k][j][i];
                ac   = speed[k][j][i];

                ac2u = ac*ac;

                r1 = rhs[k][j][i][0];
                r2 = rhs[k][j][i][1];
                r3 = rhs[k][j][i][2];
                r4 = rhs[k][j][i][3];
                r5 = rhs[k][j][i][4];     

                uzik1 = u[k][j][i][0];
                btuz  = bt * uzik1;

                t1 = btuz/ac * (r4 + r5);
                t2 = r3 + t1;
                t3 = btuz * (r4 - r5);

                rhs[k][j][i][0] = t2;
                rhs[k][j][i][1] = -uzik1*r2 + xvel*t2;
                rhs[k][j][i][2] =  uzik1*r1 + yvel*t2;
                rhs[k][j][i][3] =  zvel*t2  + t3;
                rhs[k][j][i][4] =  uzik1*(-xvel*r2 + yvel*r1) + 
                    qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;
            }
        }
    }
    if (timeron) timer_stop(t_tzetar);
}

void tzetar_gpu(int t, int st, int ed, int st2, int ed2)
{
    if (timeron) timer_start(t_tzetar);

    //---------------------------------------------------------------------
    // tzetar kernel
    //---------------------------------------------------------------------
    int tzetar_base_j = st2;
    int tzetar_offset_j = st - st2;
    int tzetar_gws_j = ed - st + 1;

    if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
        int tzetar_gws[] = {nz2};
        int tzetar_lws[] = {16};

        tzetar_gws[0] = RoundWorkSize(tzetar_gws[0], tzetar_lws[0]);

        dim3 blockSize(tzetar_gws[0]/tzetar_lws[0], 1, 1);
        dim3 threadSize(tzetar_lws[0], 1, 1);
        
        cuda_ProfilerStartEventRecord("k_tzetar_base", cmd_queue[0]);
        k_tzetar_base<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
             buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, buf_us,
             buf_vs, buf_ws, buf_qs, buf_speed,
             tzetar_base_j, tzetar_offset_j, tzetar_gws_j,
             nx2, ny2, nz2, WORK_NUM_ITEM_J
            );
        cuda_ProfilerEndEventRecord("k_tzetar_base", cmd_queue[0]);
    }
    else {
        int tzetar_gws[] = {nx2, tzetar_gws_j, nz2};
        int tzetar_lws[] = {16, 16, 1};

        tzetar_gws[0] = RoundWorkSize(tzetar_gws[0], tzetar_lws[0]);
        tzetar_gws[1] = RoundWorkSize(tzetar_gws[1], tzetar_lws[1]);
        tzetar_gws[2] = RoundWorkSize(tzetar_gws[2], tzetar_lws[2]);

        dim3 blockSize(tzetar_gws[0]/tzetar_lws[0],
                       tzetar_gws[1]/tzetar_lws[1],
                       tzetar_gws[2]/tzetar_lws[2]);
        dim3 threadSize(tzetar_lws[0], tzetar_lws[1], tzetar_lws[2]);

        cuda_ProfilerStartEventRecord("k_tzetar_opt", cmd_queue[0]);
        k_tzetar_opt<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4,
             buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, buf_us,
             buf_vs, buf_ws, buf_qs, buf_speed,
             tzetar_base_j, tzetar_offset_j, tzetar_gws_j,
             nx2, ny2, nz2, WORK_NUM_ITEM_J
            );
        cuda_ProfilerEndEventRecord("k_tzetar_opt", cmd_queue[0]);
    }

    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------

    if (timeron) timer_stop(t_tzetar);
}
