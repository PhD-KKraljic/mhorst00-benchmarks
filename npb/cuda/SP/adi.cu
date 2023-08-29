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
#include "cuda_util.h"
#include <stdio.h>

void adi()
{
    compute_rhs();

    txinvr();

    x_solve();

    y_solve();

    z_solve();

    add();
}

void adi_gpu()
{
    int partition;

    for (partition = 0; partition < NUM_PARTITIONS; partition++) {
        int i = partition;
        int i1 = partition + 1;

        int st = 1 + i * nz2 / NUM_PARTITIONS;
        int ed = (i + 1) * nz2 / NUM_PARTITIONS;

        int st2 = (i == 0) ? 0 : st - 2;
        int ed2 = (i == NUM_PARTITIONS - 1) ? nz2 + 1 : ed + 2;

        //---------------------------------------------------------------------
        // write i=0 buffer when NUM_PARTITIONS > 1
        //---------------------------------------------------------------------
        if (NUM_PARTITIONS > 1 && i == 0) {
            size_t write_base = st2;
            size_t write_k = ed2 - write_base + 1;
            CUCHK(cudaMemcpyAsync(buf_u[0], &u[write_base], 
                        write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[1]));
            CUCHK(cudaMemcpyAsync(buf_forcing[0], &forcing[write_base],
                        write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                        cudaMemcpyHostToDevice, cmd_queue[1]));
            CUCHK(cudaEventRecord(write_event[0], cmd_queue[1]));

        }

        if (NUM_PARTITIONS > 1) {
            //---------------------------------------------------------------------
            // scatter gpu
            //---------------------------------------------------------------------
            int nx = IMAXP + 1, ny = JMAXP + 1, nz = ed2 - st2 + 1;
            int scatter_gws[] = {nx, ny, nz};
            int scatter_lws[] = {16, 16, 1};
            scatter_gws[0] = RoundWorkSize(scatter_gws[0], scatter_lws[0]);
            scatter_gws[1] = RoundWorkSize(scatter_gws[1], scatter_lws[1]);
            scatter_gws[2] = RoundWorkSize(scatter_gws[2], scatter_lws[2]);
            dim3 blockSize(scatter_gws[0]/scatter_lws[0],
                           scatter_gws[1]/scatter_lws[1],
                           scatter_gws[2]/scatter_lws[2]);
            dim3 threadSize(scatter_lws[0], scatter_lws[1], scatter_lws[2]);

            CUCHK(cudaStreamWaitEvent(cmd_queue[0], write_event[i], 0));
            //---------------------------------------------------------------------
            // scatter buffer u -> u0, u1, u2, u3, u4
            //---------------------------------------------------------------------
            cuda_ProfilerStartEventRecord("k_scatter", cmd_queue[0]);
            k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (
                 buf_u[i%2], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz
                );
            CUCHK(cudaGetLastError());
            cuda_ProfilerEndEventRecord("k_scatter", cmd_queue[0]);

            //---------------------------------------------------------------------
            // scatter buffer forcing -> forcing0, forcing1, forcing2, forcing3, forcing4
            //---------------------------------------------------------------------
            cuda_ProfilerStartEventRecord("k_scatter", cmd_queue[0]);
            k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (
                 buf_forcing[i%2], buf_forcing0, buf_forcing1, buf_forcing2,
                 buf_forcing3, buf_forcing4, nx, ny, nz
                );
            CUCHK(cudaGetLastError());
            cuda_ProfilerEndEventRecord("k_scatter", cmd_queue[0]);
            //---------------------------------------------------------------------
        }
        //---------------------------------------------------------------------

        compute_rhs_gpu(i, st, ed, st2, ed2);

        txinvr_gpu(i, st, ed, st2, ed2);

        x_solve_gpu(i, st, ed, st2, ed2);

        y_solve_gpu(i, st, ed, st2, ed2);

        if (i1 < NUM_PARTITIONS) {
            int stn = 1 + i1 * nz2 / NUM_PARTITIONS;
            int edn = (i1 + 1) * nz2 / NUM_PARTITIONS;

            int st2n = (i1 == 0) ? 0 : stn - 2;
            int ed2n = (i1 == NUM_PARTITIONS - 1) ? nz2 + 1 : edn + 2;

            size_t write_base = st2n;
            size_t write_k = ed2n - write_base + 1;

            CUCHK(cudaMemcpyAsync(buf_u[i1%2], &u[write_base], 
                    write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                    cudaMemcpyHostToDevice, cmd_queue[1]));

            CUCHK(cudaMemcpyAsync(buf_forcing[i1%2], &forcing[write_base],
                    write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                    cudaMemcpyHostToDevice, cmd_queue[1]));
            CUCHK(cudaEventRecord(write_event[i1], cmd_queue[1]));
            CUCHK(cudaStreamSynchronize(cmd_queue[1]));
        }

        //---------------------------------------------------------------------
        // read i buffer when NUM_PARTITIONS > 1
        //---------------------------------------------------------------------
        size_t write_base = st2;
        size_t read_base = (i == 0) ? 0 : st;
        size_t read_offset = read_base - write_base;
        size_t read_k = (i == NUM_PARTITIONS-1) ? nz2+1-read_base+1 : ed-read_base+1;
        if (NUM_PARTITIONS > 1) {
            //---------------------------------------------------------------------
            // gather gpu
            //---------------------------------------------------------------------
            int nx = IMAXP + 1, ny = JMAXP + 1, nz = ed2 - st2 + 1;
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
            // gather buffer rhs0, rhs1, rhs2, rhs3, rhs4 -> rhs
            //---------------------------------------------------------------------
            cuda_ProfilerStartEventRecord("k_gather", cmd_queue[0]);
            k_gather<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (
                 buf_rhs[i%2], buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, nx, ny, nz
                );
            CUCHK(cudaGetLastError());
            cuda_ProfilerEndEventRecord("k_gather", cmd_queue[0]);
            //---------------------------------------------------------------------

            CUCHK(cudaMemcpyAsync(&rhs[read_base],
                    ((unsigned char*)buf_rhs[i%2]) + read_offset * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                        read_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                        cudaMemcpyDeviceToHost, cmd_queue[0]));
            CUCHK(cudaStreamSynchronize(cmd_queue[0]));
        }
        //---------------------------------------------------------------------
    }

    for (partition = 0; partition < NUM_PARTITIONS; partition++) {
        int i = partition;
        int i1 = partition + 1;

        int st = 1 + i * ny2 / NUM_PARTITIONS;
        int ed = (i + 1) * ny2 / NUM_PARTITIONS;

        int st2 = (i == 0) ? 0 : st - 2;
        int ed2 = (i == NUM_PARTITIONS - 1) ? ny2 + 1 : ed + 2;

        //---------------------------------------------------------------------
        // write buffer when NUM_PARTITIONS > 1
        //---------------------------------------------------------------------
        if (NUM_PARTITIONS > 1 && i == 0) {
            size_t write_base = st2;
            size_t write_offset = 0;
            size_t write_j = ed2 - write_base + 1;

            size_t buffer_origin[3];
            size_t host_origin[3];
            size_t region[3];
            size_t buffer_row_pitch;
            size_t buffer_slice_pitch;
            size_t host_row_pitch;
            size_t host_slice_pitch;

            buffer_origin[2] = 0;
            buffer_origin[1] = write_offset;
            buffer_origin[0] = 0;

            host_origin[2] = 0;
            host_origin[1] = write_base;
            host_origin[0] = 0;

            region[2] = KMAX;
            region[1] = write_j;
            region[0] = (IMAXP+1) * 5 * sizeof(double);

            buffer_row_pitch = region[0];
            buffer_slice_pitch = WORK_NUM_ITEM_J * buffer_row_pitch;

            host_row_pitch = region[0];
            host_slice_pitch = (JMAXP+1) * host_row_pitch;

            struct cudaMemcpy3DParms p = { 0 };
            p.srcPtr = make_cudaPitchedPtr(rhs, host_row_pitch, host_row_pitch / sizeof(double), JMAXP+1);
            p.srcPos = make_cudaPos(host_origin[0], host_origin[1], host_origin[2]);
            p.dstPtr = make_cudaPitchedPtr(buf_rhs[0], buffer_row_pitch, buffer_row_pitch / sizeof(double), WORK_NUM_ITEM_J);
            p.dstPos = make_cudaPos(buffer_origin[0], buffer_origin[1], buffer_origin[2]);
            p.extent = make_cudaExtent(region[0], region[1], region[2]);
            p.kind = cudaMemcpyHostToDevice;
            CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[1]));

            p.srcPtr = make_cudaPitchedPtr(u, host_row_pitch, host_row_pitch / sizeof(double), JMAXP+1);
            p.srcPos = make_cudaPos(host_origin[0], host_origin[1], host_origin[2]);
            p.dstPtr = make_cudaPitchedPtr(buf_u[0], buffer_row_pitch, buffer_row_pitch / sizeof(double), WORK_NUM_ITEM_J);
            p.dstPos = make_cudaPos(buffer_origin[0], buffer_origin[1], buffer_origin[2]);
            p.extent = make_cudaExtent(region[0], region[1], region[2]);
            p.kind = cudaMemcpyHostToDevice;
            CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[1]));
            CUCHK(cudaEventRecord(write_event[0], cmd_queue[1]));
        }

        if (NUM_PARTITIONS > 1) {
            //---------------------------------------------------------------------
            // scatter gpu
            //---------------------------------------------------------------------
            int nx = IMAXP + 1, ny = ed2 - st2 + 1, nz = KMAX;
            int scatter_gws[] = {nx, ny, nz};
            int scatter_lws[] = {16, 16, 1};
            scatter_gws[0] = RoundWorkSize(scatter_gws[0], scatter_lws[0]);
            scatter_gws[1] = RoundWorkSize(scatter_gws[1], scatter_lws[1]);
            scatter_gws[2] = RoundWorkSize(scatter_gws[2], scatter_lws[2]);
            dim3 blockSize(scatter_gws[0]/scatter_lws[0],
                            scatter_gws[1]/scatter_lws[1],
                            scatter_gws[2]/scatter_lws[2]);
            dim3 threadSize(scatter_lws[0], scatter_lws[1], scatter_lws[2]);

            CUCHK(cudaStreamWaitEvent(cmd_queue[0], write_event[i], 0));
            //---------------------------------------------------------------------
            // scatter buffer rhs -> rhs0, rhs1, rhs2, rhs3, rhs4
            //---------------------------------------------------------------------a
            cuda_ProfilerStartEventRecord("k_scatter_j", cmd_queue[0]);
            k_scatter_j<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (
                 buf_rhs[i%2], buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4, nx, ny, nz, WORK_NUM_ITEM_J
                );
            CUCHK(cudaGetLastError());
            cuda_ProfilerEndEventRecord("k_scatter_j", cmd_queue[0]);
            //---------------------------------------------------------------------

            //---------------------------------------------------------------------
            // scatter buffer u -> u0, u1, u2, u3, u4
            //---------------------------------------------------------------------a
            cuda_ProfilerStartEventRecord("k_scatter_j", cmd_queue[0]);
            k_scatter_j<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (
                 buf_u[i%2], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz, WORK_NUM_ITEM_J
                );
            CUCHK(cudaGetLastError());
            cuda_ProfilerEndEventRecord("k_scatter_j", cmd_queue[0]);
            //---------------------------------------------------------------------
        }
        //---------------------------------------------------------------------

        z_solve_gpu(i, st, ed, st2, ed2);
        add_gpu(i, st, ed, st2, ed2);

        //---------------------------------------------------------------------
        // read buffer when NUM_PARTITIONS > 1
        // write (i+1) buffer when NUM_PARTITONS > 1
        //---------------------------------------------------------------------
        if (i1 < NUM_PARTITIONS) {
            int stn = 1 + i1 * ny2 / NUM_PARTITIONS;
            int edn = (i1 + 1) * ny2 / NUM_PARTITIONS;

            int st2n = (i1 == 0) ? 0 : stn - 2;
            int ed2n = (i1 == NUM_PARTITIONS - 1) ? ny2 + 1 : edn + 2;

            size_t write_base = st2n;
            size_t write_offset = 0;
            size_t write_j = ed2n - write_base + 1;

            size_t buffer_origin[3];
            size_t host_origin[3];
            size_t region[3];
            size_t buffer_row_pitch;
            size_t buffer_slice_pitch;
            size_t host_row_pitch;
            size_t host_slice_pitch;

            buffer_origin[2] = 0;
            buffer_origin[1] = write_offset;
            buffer_origin[0] = 0;

            host_origin[2] = 0;
            host_origin[1] = write_base;
            host_origin[0] = 0;

            region[2] = KMAX;
            region[1] = write_j;
            region[0] = (IMAXP+1) * 5 * sizeof(double);

            buffer_row_pitch = region[0];
            buffer_slice_pitch = WORK_NUM_ITEM_J * buffer_row_pitch;

            host_row_pitch = region[0];
            host_slice_pitch = (JMAXP+1) * host_row_pitch;

            struct cudaMemcpy3DParms p = { 0 };
            p.srcPtr = make_cudaPitchedPtr(rhs, host_row_pitch, host_row_pitch / sizeof(double), JMAXP+1);
            p.srcPos = make_cudaPos(host_origin[0], host_origin[1], host_origin[2]);
            p.dstPtr = make_cudaPitchedPtr(buf_rhs[i1%2], buffer_row_pitch, buffer_row_pitch / sizeof(double), WORK_NUM_ITEM_J);
            p.dstPos = make_cudaPos(buffer_origin[0], buffer_origin[1], buffer_origin[2]);
            p.extent = make_cudaExtent(region[0], region[1], region[2]);
            p.kind = cudaMemcpyHostToDevice;
            CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[1]));

            p.srcPtr = make_cudaPitchedPtr(u, host_row_pitch, host_row_pitch / sizeof(double), JMAXP+1);
            p.srcPos = make_cudaPos(host_origin[0], host_origin[1], host_origin[2]);
            p.dstPtr = make_cudaPitchedPtr(buf_u[i1%2], buffer_row_pitch, buffer_row_pitch / sizeof(double), WORK_NUM_ITEM_J);
            p.dstPos = make_cudaPos(buffer_origin[0], buffer_origin[1], buffer_origin[2]);
            p.extent = make_cudaExtent(region[0], region[1], region[2]);
            p.kind = cudaMemcpyHostToDevice;
            CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[1]));
            CUCHK(cudaEventRecord(write_event[i1], cmd_queue[1]));
            CUCHK(cudaStreamSynchronize(cmd_queue[1]));
        }

        size_t write_base = st2;
        size_t read_base = (i == 0) ? 0 : st;
        size_t read_offset = read_base - write_base;
        size_t read_j = (i == NUM_PARTITIONS-1) ? nz2+1-read_base+1 : ed-read_base+1;
        if (NUM_PARTITIONS > 1) {
            //---------------------------------------------------------------------
            // gather gpu
            //---------------------------------------------------------------------
            int nx = IMAXP + 1, ny = ed2 - st2 + 1, nz = KMAX;
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
            // gather buffer u0, u1, u2, u3, u4 -> u
            //---------------------------------------------------------------------
            cuda_ProfilerStartEventRecord("k_gather_j", cmd_queue[0]);
            k_gather_j<<< blockSize, threadSize, 0, cmd_queue[0] >>>
                (
                 buf_u[i%2], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz, WORK_NUM_ITEM_J
                );
            CUCHK(cudaGetLastError());
            cuda_ProfilerEndEventRecord("k_gather_j", cmd_queue[0]);
            //---------------------------------------------------------------------

            size_t buffer_origin[3];
            size_t host_origin[3];
            size_t region[3];
            size_t buffer_row_pitch;
            size_t buffer_slice_pitch;
            size_t host_row_pitch;
            size_t host_slice_pitch;

            buffer_origin[2] = 0;
            buffer_origin[1] = read_offset;
            buffer_origin[0] = 0;

            host_origin[2] = 0;
            host_origin[1] = read_base;
            host_origin[0] = 0;

            region[2] = KMAX;
            region[1] = read_j;
            region[0] = (IMAXP+1) * 5 * sizeof(double);

            buffer_row_pitch = region[0];
            buffer_slice_pitch = WORK_NUM_ITEM_J * buffer_row_pitch;

            host_row_pitch = region[0];
            host_slice_pitch = (JMAXP+1) * host_row_pitch;

            struct cudaMemcpy3DParms p = { 0 };
            p.srcPtr = make_cudaPitchedPtr(buf_u[i%2], buffer_row_pitch, buffer_row_pitch / sizeof(double), WORK_NUM_ITEM_J);
            p.srcPos = make_cudaPos(buffer_origin[0], buffer_origin[1], buffer_origin[2]);
            p.dstPtr = make_cudaPitchedPtr(u, host_row_pitch, host_row_pitch / sizeof(double), JMAXP+1);
            p.dstPos = make_cudaPos(host_origin[0], host_origin[1], host_origin[2]);
            p.extent = make_cudaExtent(region[0], region[1], region[2]);
            p.kind = cudaMemcpyDeviceToHost;

            CUCHK(cudaMemcpy3DAsync(&p, cmd_queue[0]));
            CUCHK(cudaStreamSynchronize(cmd_queue[0]));
        }
        //---------------------------------------------------------------------
    }
}
