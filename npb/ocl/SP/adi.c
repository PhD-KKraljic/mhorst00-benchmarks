//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB SP code. This OpenCL C  //
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

#include "header.h"

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

      err_code = clEnqueueWriteBuffer(cmd_queue[1],
                                      buf_u[0],
                                      CL_FALSE, 0,
                                      write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                                      &u[write_base],
                                      0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
      err_code = clEnqueueWriteBuffer(cmd_queue[1],
                                      buf_forcing[0],
                                      CL_FALSE, 0,
                                      write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                                      &forcing[write_base],
                                      0, NULL, &write_event[0]);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
    }
    //---------------------------------------------------------------------
    
    if (NUM_PARTITIONS > 1) {
      //---------------------------------------------------------------------
      // scatter gpu
      //---------------------------------------------------------------------
      int nx = IMAXP + 1, ny = JMAXP + 1, nz = ed2 - st2 + 1;
      size_t scatter_gws[] = {nx, ny, nz};
      size_t scatter_lws[] = {16, 16, 1};
      scatter_gws[0] = clu_RoundWorkSize(scatter_gws[0], scatter_lws[0]);
      scatter_gws[1] = clu_RoundWorkSize(scatter_gws[1], scatter_lws[1]);
      scatter_gws[2] = clu_RoundWorkSize(scatter_gws[2], scatter_lws[2]);

      //---------------------------------------------------------------------
      // scatter buffer u -> u0, u1, u2, u3, u4
      //---------------------------------------------------------------------
      err_code  = clSetKernelArg(k_scatter, 0, sizeof(cl_mem), &buf_u[i%2]);
      err_code |= clSetKernelArg(k_scatter, 1, sizeof(cl_mem), &buf_u0);
      err_code |= clSetKernelArg(k_scatter, 2, sizeof(cl_mem), &buf_u1);
      err_code |= clSetKernelArg(k_scatter, 3, sizeof(cl_mem), &buf_u2);
      err_code |= clSetKernelArg(k_scatter, 4, sizeof(cl_mem), &buf_u3);
      err_code |= clSetKernelArg(k_scatter, 5, sizeof(cl_mem), &buf_u4);
      err_code |= clSetKernelArg(k_scatter, 6, sizeof(int), &nx);
      err_code |= clSetKernelArg(k_scatter, 7, sizeof(int), &ny);
      err_code |= clSetKernelArg(k_scatter, 8, sizeof(int), &nz);
      clu_CheckError(err_code, "clSetKernelArg()");

      err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_scatter, 3, NULL, scatter_gws, scatter_lws, 1, &write_event[i], NULL);
      clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
    
      //---------------------------------------------------------------------
      // scatter buffer forcing -> forcing0, forcing1, forcing2, forcing3, forcing4
      //---------------------------------------------------------------------
      err_code  = clSetKernelArg(k_scatter, 0, sizeof(cl_mem), &buf_forcing[i%2]);
      err_code |= clSetKernelArg(k_scatter, 1, sizeof(cl_mem), &buf_forcing0);
      err_code |= clSetKernelArg(k_scatter, 2, sizeof(cl_mem), &buf_forcing1);
      err_code |= clSetKernelArg(k_scatter, 3, sizeof(cl_mem), &buf_forcing2);
      err_code |= clSetKernelArg(k_scatter, 4, sizeof(cl_mem), &buf_forcing3);
      err_code |= clSetKernelArg(k_scatter, 5, sizeof(cl_mem), &buf_forcing4);
      err_code |= clSetKernelArg(k_scatter, 6, sizeof(int), &nx);
      err_code |= clSetKernelArg(k_scatter, 7, sizeof(int), &ny);
      err_code |= clSetKernelArg(k_scatter, 8, sizeof(int), &nz);
      clu_CheckError(err_code, "clSetKernelArg()");

      err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_scatter, 3, NULL, scatter_gws, scatter_lws, 0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
      //---------------------------------------------------------------------
    }
    //---------------------------------------------------------------------

    compute_rhs_gpu(i, st, ed, st2, ed2);

    txinvr_gpu(i, st, ed, st2, ed2);

    x_solve_gpu(i, st, ed, st2, ed2);

    y_solve_gpu(i, st, ed, st2, ed2);

    //---------------------------------------------------------------------
    // write (i+1) buffer when NUM_PARTITIONS > 1 
    //---------------------------------------------------------------------
    if (i1 < NUM_PARTITIONS) {
      int stn = 1 + i1 * nz2 / NUM_PARTITIONS;
      int edn = (i1 + 1) * nz2 / NUM_PARTITIONS;
      
      int st2n = (i1 == 0) ? 0 : stn - 2;
      int ed2n = (i1 == NUM_PARTITIONS - 1) ? nz2 + 1 : edn + 2;

      size_t write_base = st2n;
      size_t write_k = ed2n - write_base + 1;

      err_code = clEnqueueWriteBuffer(cmd_queue[1],
                                      buf_u[i1%2],
                                      CL_TRUE, 0,
                                      write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                                      &u[write_base],
                                      0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
      err_code = clEnqueueWriteBuffer(cmd_queue[1],
                                      buf_forcing[i1%2],
                                      CL_TRUE, 0,
                                      write_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                                      &forcing[write_base],
                                      0, NULL, &write_event[i1]);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
    }
    //---------------------------------------------------------------------

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
      size_t gather_gws[] = {nx, ny, nz};
      size_t gather_lws[] = {16, 16, 1};
      gather_gws[0] = clu_RoundWorkSize(gather_gws[0], gather_lws[0]);
      gather_gws[1] = clu_RoundWorkSize(gather_gws[1], gather_lws[1]);
      gather_gws[2] = clu_RoundWorkSize(gather_gws[2], gather_lws[2]);

      //---------------------------------------------------------------------
      // gather buffer rhs0, rhs1, rhs2, rhs3, rhs4 -> rhs
      //---------------------------------------------------------------------
      err_code  = clSetKernelArg(k_gather, 0, sizeof(cl_mem), &buf_rhs[i%2]);
      err_code |= clSetKernelArg(k_gather, 1, sizeof(cl_mem), &buf_rhs0);
      err_code |= clSetKernelArg(k_gather, 2, sizeof(cl_mem), &buf_rhs1);
      err_code |= clSetKernelArg(k_gather, 3, sizeof(cl_mem), &buf_rhs2);
      err_code |= clSetKernelArg(k_gather, 4, sizeof(cl_mem), &buf_rhs3);
      err_code |= clSetKernelArg(k_gather, 5, sizeof(cl_mem), &buf_rhs4);
      err_code |= clSetKernelArg(k_gather, 6, sizeof(int), &nx);
      err_code |= clSetKernelArg(k_gather, 7, sizeof(int), &ny);
      err_code |= clSetKernelArg(k_gather, 8, sizeof(int), &nz);
      clu_CheckError(err_code, "clSetKernelArg()");

      err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_gather, 3, NULL, gather_gws, gather_lws, 0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
      //---------------------------------------------------------------------

      err_code = clEnqueueReadBuffer(cmd_queue[0],
                                      buf_rhs[i%2],
                                      CL_TRUE,
                                      read_offset * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                                      read_k * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                                      &rhs[read_base],
                                      0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueReadBuffer()");
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
    // write i=0 buffer when NUM_PARTITIONS > 1 
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

      err_code = clEnqueueWriteBufferRect(cmd_queue[1],
                                          buf_rhs[0],
                                          CL_FALSE,
                                          buffer_origin,
                                          host_origin,
                                          region,
                                          buffer_row_pitch,
                                          buffer_slice_pitch,
                                          host_row_pitch,
                                          host_slice_pitch,
                                          (void*)rhs,
                                          0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
      err_code = clEnqueueWriteBufferRect(cmd_queue[1],
                                          buf_u[0],
                                          CL_FALSE,
                                          buffer_origin,
                                          host_origin,
                                          region,
                                          buffer_row_pitch,
                                          buffer_slice_pitch,
                                          host_row_pitch,
                                          host_slice_pitch,
                                          (void*)u,
                                          0, NULL, &write_event[0]);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
    }
    //---------------------------------------------------------------------

    if (NUM_PARTITIONS > 1) {
      //---------------------------------------------------------------------
      // scatter gpu
      //---------------------------------------------------------------------
      int nx = IMAXP + 1, ny = ed2 - st2 + 1, nz = KMAX;
      size_t scatter_gws[] = {nx, ny, nz};
      size_t scatter_lws[] = {16, 16, 1};
      scatter_gws[0] = clu_RoundWorkSize(scatter_gws[0], scatter_lws[0]);
      scatter_gws[1] = clu_RoundWorkSize(scatter_gws[1], scatter_lws[1]);
      scatter_gws[2] = clu_RoundWorkSize(scatter_gws[2], scatter_lws[2]);

      //---------------------------------------------------------------------
      // scatter buffer rhs -> rhs0, rhs1, rhs2, rhs3, rhs4
      //---------------------------------------------------------------------
      err_code  = clSetKernelArg(k_scatter_j, 0, sizeof(cl_mem), &buf_rhs[i%2]);
      err_code |= clSetKernelArg(k_scatter_j, 1, sizeof(cl_mem), &buf_rhs0);
      err_code |= clSetKernelArg(k_scatter_j, 2, sizeof(cl_mem), &buf_rhs1);
      err_code |= clSetKernelArg(k_scatter_j, 3, sizeof(cl_mem), &buf_rhs2);
      err_code |= clSetKernelArg(k_scatter_j, 4, sizeof(cl_mem), &buf_rhs3);
      err_code |= clSetKernelArg(k_scatter_j, 5, sizeof(cl_mem), &buf_rhs4);
      err_code |= clSetKernelArg(k_scatter_j, 6, sizeof(int), &nx);
      err_code |= clSetKernelArg(k_scatter_j, 7, sizeof(int), &ny);
      err_code |= clSetKernelArg(k_scatter_j, 8, sizeof(int), &nz);
      clu_CheckError(err_code, "clSetKernelArg()");

      err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_scatter_j, 3, NULL, scatter_gws, scatter_lws, 1, &write_event[i], NULL);
      clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // scatter buffer u -> u0, u1, u2, u3, u4
      //---------------------------------------------------------------------
      err_code  = clSetKernelArg(k_scatter_j, 0, sizeof(cl_mem), &buf_u[i%2]);
      err_code |= clSetKernelArg(k_scatter_j, 1, sizeof(cl_mem), &buf_u0);
      err_code |= clSetKernelArg(k_scatter_j, 2, sizeof(cl_mem), &buf_u1);
      err_code |= clSetKernelArg(k_scatter_j, 3, sizeof(cl_mem), &buf_u2);
      err_code |= clSetKernelArg(k_scatter_j, 4, sizeof(cl_mem), &buf_u3);
      err_code |= clSetKernelArg(k_scatter_j, 5, sizeof(cl_mem), &buf_u4);
      err_code |= clSetKernelArg(k_scatter_j, 6, sizeof(int), &nx);
      err_code |= clSetKernelArg(k_scatter_j, 7, sizeof(int), &ny);
      err_code |= clSetKernelArg(k_scatter_j, 8, sizeof(int), &nz);
      clu_CheckError(err_code, "clSetKernelArg()");

      err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_scatter_j, 3, NULL, scatter_gws, scatter_lws, 0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
      //---------------------------------------------------------------------
    }
    //---------------------------------------------------------------------

    z_solve_gpu(i, st, ed, st2, ed2);

    add_gpu(i, st, ed, st2, ed2);
    
    //---------------------------------------------------------------------
    // write (i+1) buffer when NUM_PARTITIONS > 1 
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

      err_code = clEnqueueWriteBufferRect(cmd_queue[1],
                                          buf_rhs[i1%2],
                                          CL_TRUE,
                                          buffer_origin,
                                          host_origin,
                                          region,
                                          buffer_row_pitch,
                                          buffer_slice_pitch,
                                          host_row_pitch,
                                          host_slice_pitch,
                                          (void*)rhs,
                                          0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
      err_code = clEnqueueWriteBufferRect(cmd_queue[1],
                                          buf_u[i1%2],
                                          CL_TRUE,
                                          buffer_origin,
                                          host_origin,
                                          region,
                                          buffer_row_pitch,
                                          buffer_slice_pitch,
                                          host_row_pitch,
                                          host_slice_pitch,
                                          (void*)u,
                                          0, NULL, &write_event[i1]);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
    }
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // read i buffer when NUM_PARTITIONS > 1
    //---------------------------------------------------------------------
    size_t write_base = st2;
    size_t read_base = (i == 0) ? 0 : st;
    size_t read_offset = read_base - write_base;
    size_t read_j = (i == NUM_PARTITIONS-1) ? nz2+1-read_base+1 : ed-read_base+1;

    if (NUM_PARTITIONS > 1) {
      //---------------------------------------------------------------------
      // gather gpu
      //---------------------------------------------------------------------
      int nx = IMAXP + 1, ny = ed2 - st2 + 1, nz = KMAX;
      size_t gather_gws[] = {nx, ny, nz};
      size_t gather_lws[] = {16, 16, 1};
      gather_gws[0] = clu_RoundWorkSize(gather_gws[0], gather_lws[0]);
      gather_gws[1] = clu_RoundWorkSize(gather_gws[1], gather_lws[1]);
      gather_gws[2] = clu_RoundWorkSize(gather_gws[2], gather_lws[2]);

      //---------------------------------------------------------------------
      // gather buffer u0, u1, u2, u3, u4 -> u
      //---------------------------------------------------------------------
      err_code  = clSetKernelArg(k_gather_j, 0, sizeof(cl_mem), &buf_u[i%2]);
      err_code |= clSetKernelArg(k_gather_j, 1, sizeof(cl_mem), &buf_u0);
      err_code |= clSetKernelArg(k_gather_j, 2, sizeof(cl_mem), &buf_u1);
      err_code |= clSetKernelArg(k_gather_j, 3, sizeof(cl_mem), &buf_u2);
      err_code |= clSetKernelArg(k_gather_j, 4, sizeof(cl_mem), &buf_u3);
      err_code |= clSetKernelArg(k_gather_j, 5, sizeof(cl_mem), &buf_u4);
      err_code |= clSetKernelArg(k_gather_j, 6, sizeof(int), &nx);
      err_code |= clSetKernelArg(k_gather_j, 7, sizeof(int), &ny);
      err_code |= clSetKernelArg(k_gather_j, 8, sizeof(int), &nz);
      clu_CheckError(err_code, "clSetKernelArg()");

      err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_gather_j, 3, NULL, gather_gws, gather_lws, 0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
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

      err_code = clEnqueueReadBufferRect(cmd_queue[0],
                                          buf_u[i%2],
                                          CL_TRUE,
                                          buffer_origin,
                                          host_origin,
                                          region,
                                          buffer_row_pitch,
                                          buffer_slice_pitch,
                                          host_row_pitch,
                                          host_slice_pitch,
                                          (void*)u,
                                          0, NULL, NULL);
      clu_CheckError(err_code, "clEnqueueWriteBuffer()");
    }
    //---------------------------------------------------------------------
  }
}
