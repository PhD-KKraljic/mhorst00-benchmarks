//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB LU code. This CUDA® C  //
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

#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "applu.incl"
extern "C" {
#include "timers.h"
}
//#include "kernel_l2norm.cu"
//---------------------------------------------------------------------
// to compute the l2-norm of vector v.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------

cudaEvent_t     ev_k_l2norm_head1[2],
                ev_k_l2norm_head2[2],
                (*ev_k_l2norm_body1)[2],
                (*ev_k_l2norm_body2)[2],
                ev_d_l2norm_tail1[2],
                ev_d_l2norm_tail2[2];

void l2norm_head(double sum[5], 
                 double (* g_sum)[5], 
                 double *m_sum, 
                 cudaEvent_t *ev_k_start_ptr,
                 cudaEvent_t *ev_k_end_ptr);

void l2norm_body(int step, int iter,
                 int base, int item,
                 cudaEvent_t *ev_wb_ptr, 
                 cudaEvent_t *ev_kernel_start_ptr,
                 cudaEvent_t *ev_kernel_end_ptr,
                 double* m_sum, int nz0, 
                 int jst, int jend, 
                 int ist, int iend);

void l2norm_tail(double sum[5], 
                 double (* g_sum)[5], 
                 double* m_sum, 
                 int nx0, 
                 int ny0, 
                 int nz0, 
                 cudaEvent_t *ev_data_ptr);

/* l2norm baseline functions */
void l2norm_body_baseline(int step, int iter,
                          int base, int item,
                          cudaEvent_t *ev_wb_ptr,
                          cudaEvent_t *ev_kernel_start_ptr,
                          cudaEvent_t *ev_kernel_end_ptr,
                          double *m_sum, int nz0,
                          int jst, int jend,
                          int ist, int iend);

/* l2norm global memory acccess opt functions */
void l2norm_body_gmem(int step, int iter,
                      int base, int item,
                      cudaEvent_t *ev_wb_ptr,
                      cudaEvent_t *ev_kernel_start_ptr,
                      cudaEvent_t *ev_kernel_end_ptr,
                      double *m_sum, int nz0,
                      int jst, int jend,
                      int ist, int iend);

void l2norm_init(int iter)
{
  int i;

  ev_k_l2norm_body1 = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);
  ev_k_l2norm_body2 = (cudaEvent_t(*)[2])malloc(sizeof(cudaEvent_t)*2*iter);

  cudaEventCreate(&ev_k_l2norm_head1[0]);
  cudaEventCreate(&ev_k_l2norm_head1[1]);
  cudaEventCreate(&ev_k_l2norm_head2[0]);
  cudaEventCreate(&ev_k_l2norm_head2[1]);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&ev_k_l2norm_body1[i][0]);
    cudaEventCreate(&ev_k_l2norm_body2[i][0]);

    cudaEventCreate(&ev_k_l2norm_body1[i][1]);
    cudaEventCreate(&ev_k_l2norm_body2[i][1]);
  } 

  cudaEventCreate(&ev_d_l2norm_tail1[0]);
  cudaEventCreate(&ev_d_l2norm_tail1[1]);
  cudaEventCreate(&ev_d_l2norm_tail2[0]);
  cudaEventCreate(&ev_d_l2norm_tail2[1]);
}

void l2norm_release(int iter)
{
  int i;

  cudaEventDestroy(ev_k_l2norm_head1[0]);
  cudaEventDestroy(ev_k_l2norm_head1[1]);
  cudaEventDestroy(ev_k_l2norm_head2[0]);
  cudaEventDestroy(ev_k_l2norm_head2[1]);

  for (i = 0; i < iter; i++) {
    cudaEventDestroy(ev_k_l2norm_body1[i][0]);
    cudaEventDestroy(ev_k_l2norm_body2[i][0]);

    cudaEventDestroy(ev_k_l2norm_body1[i][1]);
    cudaEventDestroy(ev_k_l2norm_body2[i][1]);
  }

  cudaEventDestroy(ev_d_l2norm_tail1[0]);
  cudaEventDestroy(ev_d_l2norm_tail1[1]);
  cudaEventDestroy(ev_d_l2norm_tail2[0]);
  cudaEventDestroy(ev_d_l2norm_tail2[1]);
}

void l2norm_body(int step, int iter,
                 int base, int item,
                 cudaEvent_t *ev_wb_ptr,
                 cudaEvent_t *ev_k_start_ptr,
                 cudaEvent_t *ev_k_end_ptr,
                 double *m_sum, int nz0,
                 int jst, int jend,
                 int ist, int iend)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      l2norm_body_baseline(step, iter, 
                           base, item,
                           ev_wb_ptr,
                           ev_k_start_ptr,
                           ev_k_end_ptr,
                           m_sum, nz0,
                           jst, jend,
                           ist, iend);
      break;
    case OPT_GLOBALMEM:
    case OPT_FULL:
      l2norm_body_gmem(step, iter, 
                       base, item,
                       ev_wb_ptr,
                       ev_k_start_ptr,
                       ev_k_end_ptr,
                       m_sum, nz0,
                       jst, jend,
                       ist, iend);
      break;
    case OPT_PARALLEL:
    case OPT_MEMLAYOUT:
    case OPT_SYNC:
    default:
      l2norm_body_baseline(step, iter, 
                           base, item,
                           ev_wb_ptr,
                           ev_k_start_ptr,
                           ev_k_end_ptr,
                           m_sum, nz0,
                           jst, jend,
                           ist, iend);
      break;
  }
}

void l2norm(int ldx, int ldy, int ldz, int nx0, int ny0, int nz0,
    int ist, int iend, int jst, int jend,
    double sum[5], double * m_v)
{


  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int m;
  double (* g_sum)[5];

  int work_step, work_max_iter, work_base, work_num_item;
  int next_work_base, next_work_num_item;
  cudaEvent_t (*ev_kernel)[2], 
              *ev_wb;

  int next_buf_idx;

  for (m = 0; m < 5; m++) {
    sum[m] = 0.0;
  }

  g_sum = (double (*)[5])malloc(sizeof(double) * 5 * l2norm_wg_num ) ;

  work_max_iter = (nz0-2 - 1) / loop2_work_num_item_default + 1;

  ev_kernel = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*work_max_iter*2);
  ev_wb = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*work_max_iter);

  l2norm_head(sum, g_sum, m_sum1, NULL, NULL);

  for (int i = 0; i < work_max_iter; ++i) {
    CUCHK(cudaEventCreate(&ev_kernel[i][0]));
    CUCHK(cudaEventCreate(&ev_kernel[i][1]));
    CUCHK(cudaEventCreate(&ev_wb[i]));
  }

  //###################
  // Write First buffer
  //###################
  if (split_flag && buffering_flag) {
    work_base = 1;
    work_num_item = min(loop2_work_num_item_default, nz0 - 1 - work_base);

    CUCHK(cudaMemcpyAsync(m_rsd[0],
          &(rsd[work_base][0][0][0]),
          sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
    CUCHK(cudaEventRecord(ev_wb[0], cmd_q[DATA_Q]));
  }

  for (work_step = 0; work_step < work_max_iter; work_step++) {

    work_base = work_step*loop2_work_num_item_default + 1;
    work_num_item = min(loop2_work_num_item_default, nz0 - 1 - work_base);
    next_buf_idx = ((work_step+1)%2)*buffering_flag;

    if (split_flag) {
      if (!buffering_flag) {
        //#################
        // Write buffer
        //#################
        CUCHK(cudaMemcpyAsync(m_rsd[0], &(rsd[work_base][0][0][0]),
              sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

      }
      else if (work_step < work_max_iter - 1) {
        //#################
        // Write Next buffer
        //#################
        next_work_base = (work_step+1)*loop2_work_num_item_default + 1;
        next_work_num_item = min(loop2_work_num_item_default, nz0-1 - next_work_base);

        if (work_step == 0) {
          CUCHK(cudaMemcpyAsync(m_rsd[next_buf_idx],
                &(rsd[next_work_base][0][0][0]),
                sizeof(double)*next_work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
          CUCHK(cudaEventRecord(ev_wb[work_step+1], cmd_q[DATA_Q]));
        }
        else {
          CUCHK(cudaStreamWaitEvent(cmd_q[DATA_Q], ev_kernel[work_step-1][1], 0));
          CUCHK(cudaMemcpyAsync( m_rsd[next_buf_idx],
                &(rsd[next_work_base][0][0][0]),
                sizeof(double)*next_work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
          CUCHK(cudaEventRecord(ev_wb[work_step+1], cmd_q[DATA_Q]));
        }
      }
    }


    //#################
    // Kernel execution
    //#################
    l2norm_body(work_step, work_max_iter,
                work_base, work_num_item,
                &ev_wb[work_step], 
                &ev_kernel[work_step][0],
                &ev_kernel[work_step][1],
                m_sum1, nz0, jst, jend, ist, iend);
  }

  if (buffering_flag) {
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));

  }


  l2norm_tail(sum, g_sum, m_sum1, nx0, ny0, nz0, NULL, NULL);


  for (int i = 0; i < work_max_iter; ++i) {
    CUCHK(cudaEventDestroy(ev_kernel[i][0]));
    CUCHK(cudaEventDestroy(ev_kernel[i][1]));
    CUCHK(cudaEventDestroy(ev_wb[i]));
  }
  free(ev_kernel);
  free(ev_wb);
  free(g_sum);
}

void l2norm_head(double sum[5], 
                 double (* g_sum)[5], 
                 double *m_sum, 
                 cudaEvent_t *ev_k_start_ptr,
                 cudaEvent_t *ev_k_end_ptr)
{
  size_t lws, gws;
  int m;

  for (m = 0; m < 5; m++) {
    sum[m] = 0.0;
  }

  lws = max_work_group_size;
  gws = l2norm_wg_num*5;
  gws = RoundWorkSize(gws, lws);

  if (ev_k_start_ptr)
    CUCHK(cudaEventRecord(*ev_k_start_ptr, cmd_q[KERNEL_Q]));

  cuda_ProfilerStartEventRecord("k_l2norm_head",  cmd_q[KERNEL_Q]);
  k_l2norm_head<<< gws / lws, lws, 0, cmd_q[KERNEL_Q]>>> 
    (m_sum, l2norm_wg_num);
  CUCHK(cudaGetLastError());
  cuda_ProfilerEndEventRecord("k_l2norm_head",  cmd_q[KERNEL_Q]);

  if (ev_k_end_ptr)
    CUCHK(cudaEventRecord(*ev_k_end_ptr, cmd_q[KERNEL_Q]));

  CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
}



void l2norm_tail(double sum[5], 
                 double (* g_sum)[5], 
                 double* m_sum, int nx0, 
                 int ny0, int nz0, 
                 cudaEvent_t *ev_d_start_ptr,
                 cudaEvent_t *ev_d_end_ptr)
{
  int i, m;
  //#################
  // Read buffer
  //#################
  if (!buffering_flag) {

    if (ev_d_start_ptr)
      CUCHK(cudaEventRecord(*ev_d_start_ptr, cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(g_sum, m_sum, 
                          sizeof(double)*5*l2norm_wg_num, 
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    if (ev_d_end_ptr)
      CUCHK(cudaEventRecord(*ev_d_end_ptr, cmd_q[KERNEL_Q]));

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }
  else {

    if (ev_d_start_ptr)
      CUCHK(cudaEventRecord(*ev_d_start_ptr, cmd_q[DATA_Q]));

    CUCHK(cudaMemcpyAsync(g_sum, m_sum, 
                          sizeof(double)*5*l2norm_wg_num, 
                          cudaMemcpyDeviceToHost, 
                          cmd_q[DATA_Q]));

    if (ev_d_end_ptr)
      CUCHK(cudaEventRecord(*ev_d_end_ptr, cmd_q[DATA_Q]));

    CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
  }

  //reduction
  for (i = 0; i < (int)l2norm_wg_num; i++) {
    for (m = 0; m < 5; m++) {
      sum[m] += g_sum[i][m];
    }
  }

  for (m = 0; m < 5; m++) {
    sum[m] = sqrt ( sum[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }
}
