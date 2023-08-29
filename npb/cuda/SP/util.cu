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

void transpose_x_gpu() {
    const int P = KMAX * (JMAXP + 1), Q = IMAXP + 1;
    int gws[] = {Q, P}, lws[] = {16, 16};
    gws[0] = RoundWorkSize(gws[0], lws[0]);
    gws[1] = RoundWorkSize(gws[1], lws[1]);

    dim3 blockSize(gws[0]/lws[0]);
    dim3 threadSize(gws[1]/lws[1]);

    // rhs0
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>>(buf_rhs0, buf_temp, P, Q);
    CUCHK(cudaGetLastError());
    CUCHK(cudaMemcpyAsync(buf_rhs0, buf_temp, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));

    // rhs1
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>>(buf_rhs1, buf_temp, P, Q);
    CUCHK(cudaGetLastError());
    CUCHK(cudaMemcpyAsync(buf_rhs1, buf_temp, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));

    // rhs2
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>>(buf_rhs2, buf_temp, P, Q);
    CUCHK(cudaGetLastError());
    CUCHK(cudaMemcpyAsync(buf_rhs2, buf_temp, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));

    // rhs3
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>>(buf_rhs3, buf_temp, P, Q);
    CUCHK(cudaGetLastError());
    CUCHK(cudaMemcpyAsync(buf_rhs3, buf_temp, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));

    // rhs4
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>>(buf_rhs4, buf_temp, P, Q);
    CUCHK(cudaGetLastError());
    CUCHK(cudaMemcpyAsync(buf_rhs4, buf_temp, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));
}

void detranspose_x_gpu() {
    const int P = IMAXP + 1, Q = KMAX * (JMAXP + 1);
    size_t gws[] = {Q, P}, lws[] = {16, 16};
    gws[0] = RoundWorkSize(gws[0], lws[0]);
    gws[1] = RoundWorkSize(gws[1], lws[1]);

    dim3 blockSize(gws[0]/lws[0]);
    dim3 threadSize(gws[1]/lws[1]);


    // rhs0
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs0, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>> (buf_temp, buf_rhs0, P, Q);
    CUCHK(cudaGetLastError());

    // rhs1
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs1, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>> (buf_temp, buf_rhs1, P, Q);
    CUCHK(cudaGetLastError());

    // rhs2
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs2, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>> (buf_temp, buf_rhs2, P, Q);
    CUCHK(cudaGetLastError());

    // rhs3
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs3, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>> (buf_temp, buf_rhs3, P, Q);
    CUCHK(cudaGetLastError());

    // rhs4
    CUCHK(cudaMemcpyAsync(buf_temp, buf_rhs4, P * Q * sizeof(double), cudaMemcpyDeviceToDevice, cmd_queue));
    k_transpose<<< blockSize, threadSize, 0, cmd_queue >>> (buf_temp, buf_rhs4, P, Q);
    CUCHK(cudaGetLastError());
}

void scatter_gpu() {
    int nx = IMAXP + 1, ny = JMAXP + 1, nz = KMAX;
    int scatter_gws[] = {nx, ny, nz};
    int scatter_lws[] = {16, 16, 1};
    scatter_gws[0] = RoundWorkSize(scatter_gws[0], scatter_lws[0]);
    scatter_gws[1] = RoundWorkSize(scatter_gws[1], scatter_lws[1]);
    scatter_gws[2] = RoundWorkSize(scatter_gws[2], scatter_lws[2]);

    dim3 blockSize(scatter_gws[0]/scatter_lws[0],
                   scatter_gws[1]/scatter_lws[1],
                   scatter_gws[2]/scatter_lws[2]);
    dim3 threadSize(scatter_lws[0], scatter_lws[1], scatter_lws[2]);

    //---------------------------------------------------------------------
    // scatter buffer u -> u0, u1, u2, u3, u4
    //---------------------------------------------------------------------
    k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_u[0], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz
        );
    CUCHK(cudaGetLastError());
    //---------------------------------------------------------------------
    // scatter buffer forcing -> forcing0, forcing1, forcing2, forcing3, forcing4
    //---------------------------------------------------------------------
    k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_forcing[0], buf_forcing0, buf_forcing1, buf_forcing2,
         buf_forcing3, buf_forcing4, nx, ny, nz
        );
    CUCHK(cudaGetLastError());

    //---------------------------------------------------------------------
    // scatter buffer rhs -> rhs0, rhs1, rhs2, rhs3, rhs4
    //---------------------------------------------------------------------
    k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs[0], buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, nx, ny, nz
        );
    CUCHK(cudaGetLastError());

}

void gather_gpu() {
    int nx = IMAXP + 1, ny = JMAXP + 1, nz = KMAX;
    int gather_gws[] = {nx, ny, nz};
    int gather_lws[] = {16, 16, 1};
    gather_gws[0] = RoundWorkSize(gather_gws[0], gather_lws[0]);
    gather_gws[1] = RoundWorkSize(gather_gws[1], gather_lws[1]);
    gather_gws[2] = RoundWorkSize(gather_gws[2], gather_lws[2]);

    dim3 blockSize(gather_gws[0]/gather_lws[0],
                   gather_gws[1]/gather_lws[1],
                   gather_gws[2]/gather_lws[2]);
    dim3 threadSize(gather_lws[0], gather_lws[1], gather_lws[2]);
    

    //---------------------------------------------------------------------
    // gather buffer u -> u0, u1, u2, u3, u4
    //---------------------------------------------------------------------

    k_gather<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_u[0], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz
        );
    CUCHK(cudaGetLastError());

    //---------------------------------------------------------------------
    // gather buffer rhs0, rhs1, rhs2, rhs3, rhs4 -> rhs
    //---------------------------------------------------------------------

    k_gather<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_rhs[0], buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, nx, ny, nz
        );
    CUCHK(cudaGetLastError());

    //---------------------------------------------------------------------
    // gather buffer forcing0, forcing1, forcing2, forcing3, forcing4 -> forcing
    //---------------------------------------------------------------------

    k_gather<<< blockSize, threadSize, 0, cmd_queue[0] >>>
        (
         buf_forcing[0], buf_forcing0, buf_forcing1, buf_forcing2, buf_forcing3, buf_forcing4, nx, ny, nz
        );
    CUCHK(cudaGetLastError());

}
