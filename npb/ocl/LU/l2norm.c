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

#include <math.h>
#include <stdio.h>
#include "applu.incl"
#include "timers.h"
//---------------------------------------------------------------------
// to compute the l2-norm of vector v.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------

cl_kernel k_l2norm_head;

cl_event  ev_kernel_l2norm_head1, 
          ev_kernel_l2norm_head2,
          *ev_k_l2norm_body1,
          *ev_k_l2norm_body2,
          ev_data_l2norm_tail1, 
          ev_data_l2norm_tail2;

void l2norm_head(double sum[5], 
                 double (* g_sum)[5], 
                 cl_mem * m_sum, 
                 cl_event *ev_kernel_ptr);

void l2norm_body(int work_step, 
                 int work_max_iter, 
                 int work_base, 
                 int work_num_item,
                 cl_event* ev_wb_ptr, 
                 cl_event* ev_kernel_ptr, 
                 cl_mem* m_sum,
                 int nz0, 
                 int jst, int jend, 
                 int ist, int iend);


void l2norm_tail(double sum[5], 
                 double (* g_sum)[5], 
                 cl_mem* m_sum, 
                 int nx0, int ny0, int nz0, 
                 cl_event *ev_data_ptr);

/* l2norm baseline functions */
void l2norm_init_baseline();
void l2norm_release_baseline();
void l2norm_body_baseline(int work_step,
                          int work_max_iter,
                          int work_base,
                          int work_num_item,
                          cl_event* ev_wb_ptr,
                          cl_event* ev_kernel_ptr,
                          cl_mem* m_sum,
                          int nz0,
                          int jst, int jend,
                          int ist, int iend);


/* l2norm global memory access opt functions */
void l2norm_init_gmem();
void l2norm_release_gmem();
void l2norm_body_gmem(int work_step,
                      int work_max_iter,
                      int work_base,
                      int work_num_item,
                      cl_event* ev_wb_ptr,
                      cl_event* ev_kernel_ptr,
                      cl_mem* m_sum,
                      int nz0,
                      int jst, int jend,
                      int ist, int iend);

void l2norm_init(int iter)
{
  cl_int ecode; 

  k_l2norm_head = clCreateKernel(p_l2norm, 
                                 "l2norm_head", 
                                 &ecode);
  clu_CheckError(ecode, "clCreateKernel()");

  ev_k_l2norm_body1 = (cl_event*)malloc(sizeof(cl_event)*iter);
  ev_k_l2norm_body2 = (cl_event*)malloc(sizeof(cl_event)*iter);

  switch (g_opt_level) {
    case OPT_BASELINE:
      l2norm_init_baseline();
      break;
    case OPT_GLOBALMEM:
    case OPT_FULL:
      l2norm_init_gmem();
      break;
    default:
      l2norm_init_baseline();
      break;
  }
}

void l2norm_release()
{
  clReleaseKernel(k_l2norm_head);

  free(ev_k_l2norm_body1);
  free(ev_k_l2norm_body2);

  switch (g_opt_level) {
    case OPT_BASELINE:
      l2norm_release_baseline();
      break;
    case OPT_GLOBALMEM:
    case OPT_FULL:
      l2norm_release_gmem();
      break;
    default:
      l2norm_release_baseline();
      break;
  }
}

void l2norm_release_ev(int iter, int step, int norm, int itmax)
{
  int i;

  if ((step%norm) == 0) {
    clReleaseEvent(ev_kernel_l2norm_head1);
    clReleaseEvent(ev_data_l2norm_tail1);
  }

  if ((step%norm) == 0 || step == itmax) {
    clReleaseEvent(ev_kernel_l2norm_head2);
    clReleaseEvent(ev_data_l2norm_tail2);
  }

  if ((step%norm) == 0) {
    for (i = 0; i < iter; i++)
      clReleaseEvent(ev_k_l2norm_body1[i]);
  }

  if ((step%norm) == 0 || step == itmax) {
    for (i = 0; i < iter; i++)
      clReleaseEvent(ev_k_l2norm_body2[i]);
  }

}

void l2norm_body(int work_step, int work_max_iter, 
                 int work_base, int work_num_item,
                 cl_event* ev_wb_ptr, 
                 cl_event* ev_kernel_ptr, 
                 cl_mem* m_sum,
                 int nz0, int jst, int jend, 
                 int ist, int iend)
{

  switch (g_opt_level) {
    case OPT_BASELINE:
      l2norm_body_baseline(work_step, work_max_iter, 
                           work_base, work_num_item,
                           ev_wb_ptr, ev_kernel_ptr, 
                           m_sum, nz0, jst, jend,
                           ist, iend);
      break;
    case OPT_GLOBALMEM:
    case OPT_FULL:
      l2norm_body_gmem(work_step, work_max_iter, 
                       work_base, work_num_item,
                       ev_wb_ptr, ev_kernel_ptr, 
                       m_sum, nz0, jst, jend,
                       ist, iend);
      break;
    default:
      l2norm_body_baseline(work_step, work_max_iter, 
                           work_base, work_num_item,
                           ev_wb_ptr, ev_kernel_ptr, 
                           m_sum, nz0, jst, jend,
                           ist, iend);
      break;
  }
}

void l2norm(int ldx, int ldy, int ldz, 
            int nx0, int ny0, int nz0,
            int ist, int iend, 
            int jst, int jend,
            double v[][ldy/2*2+1][ldx/2*2+1][5], 
            double sum[5], cl_mem * m_v)
{


  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  double (* g_sum)[5];
  cl_int ecode;
  int work_step, work_max_iter, 
      work_base, work_num_item,
      next_work_base, next_work_num_item;
  cl_event *ev_kernel, *ev_wb;
  cl_event *wait_ev;
  int num_wait;
  int next_buf_idx;

  g_sum = (double (*)[5])malloc(sizeof(double) * 5 * l2norm_wg_num ) ;

  work_max_iter = (nz0-2 - 1) / loop2_work_num_item_default + 1;

  ev_kernel = (cl_event*)malloc(sizeof(cl_event)*work_max_iter);
  ev_wb = (cl_event*)malloc(sizeof(cl_event)*work_max_iter);



  l2norm_head(sum, g_sum, &m_sum1, NULL);


  //#################
  // Write First buffer
  //#################
  if (split_flag && buffering_flag) {
    work_base = 1;
    work_num_item = min(loop2_work_num_item_default, nz0 - 1 - work_base);

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q],
                                 m_rsd[0], 
                                 CL_FALSE,
                                 0, 
                                 rsd_slice_size*work_num_item,
                                 &(rsd[work_base][0][0][0]), 
                                 0, NULL, &ev_wb[0]);
    clu_CheckError(ecode, "l2norm loop 1 write buffer - m_v");
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
        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[0], 
                                     CL_TRUE,
                                     0, 
                                     rsd_slice_size*work_num_item,
                                     &(rsd[work_base][0][0][0]), 
                                     0, NULL, NULL);
        clu_CheckError(ecode, "l2norm loop 1 write buffer - m_v");

      }
      else if (work_step < work_max_iter - 1) {
        //#################
        // Write Next buffer
        //#################
        next_work_base = (work_step+1)*loop2_work_num_item_default + 1;
        next_work_num_item = min(loop2_work_num_item_default, nz0-1 - next_work_base);

        num_wait = (work_step > 0) ? 1 : 0;
        wait_ev = (work_step > 0) ? &ev_kernel[work_step-1] : NULL;

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[next_buf_idx], 
                                     CL_FALSE,
                                     0, 
                                     rsd_slice_size*next_work_num_item,
                                     &(rsd[next_work_base][0][0][0]), 
                                     num_wait, wait_ev,
                                     &ev_wb[work_step+1]);
        clu_CheckError(ecode, "l2norm loop 1 write buffer - m_v");

        clFlush(cmd_q[DATA_Q]);
      }
    }


    //#################
    // Kernel execution
    //#################
    l2norm_body(work_step, work_max_iter, work_base, work_num_item,
                &ev_wb[work_step], &ev_kernel[work_step],
                &m_sum1, nz0, jst, jend, ist, iend);


  }

  if (buffering_flag) {
    clFinish(cmd_q[KERNEL_Q]);
    clFinish(cmd_q[DATA_Q]);
  }


  l2norm_tail(sum, g_sum, &m_sum1, nx0, ny0, nz0, NULL);


  if (split_flag && buffering_flag) {
    for (work_step = 0; work_step < work_max_iter; work_step++) {
      clReleaseEvent(ev_kernel[work_step]);
      clReleaseEvent(ev_wb[work_step]);
    }
  }


  free(ev_kernel);
  free(ev_wb);
  free(g_sum);
}


void l2norm_head(double *sum, 
                 double (* g_sum)[5], 
                 cl_mem * m_sum, 
                 cl_event *ev_kernel_ptr)
{

  cl_int ecode;
  size_t lws, gws;
  int m;

  for (m = 0; m < 5; m++) {
    sum[m] = 0.0;
  }

  ecode  = clSetKernelArg(k_l2norm_head, 0, sizeof(cl_mem), m_sum);
  ecode |= clSetKernelArg(k_l2norm_head, 1, sizeof(int), &l2norm_wg_num);
  clu_CheckError(ecode, "clSetKernelArg()");

  lws = max_work_group_size;
  gws = l2norm_wg_num*5;
  gws = clu_RoundWorkSize(gws, lws);

  ecode = clEnqueueNDRangeKernel(cmd_q[KERNEL_Q], 
                                 k_l2norm_head,  
                                 1, NULL, 
                                 &gws, &lws, 
                                 0, NULL, 
                                 ev_kernel_ptr);
  clu_CheckError(ecode , "clEnqueueNDRangeKernel()");

  clFinish(cmd_q[KERNEL_Q]);
}


void l2norm_tail(double sum[5], 
                 double (* g_sum)[5], 
                 cl_mem* m_sum, 
                 int nx0, int ny0, int nz0, 
                 cl_event *ev_data_ptr)
{

  cl_int ecode;
  int i, m;

  //#################
  // Read buffer
  //#################
  ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                              *m_sum, 
                              CL_TRUE,
                              0, 
                              sizeof(double)*5*l2norm_wg_num, 
                              g_sum, 
                              0, NULL, 
                              ev_data_ptr);
  clu_CheckError(ecode, "clEnqueueWriteBuffer for k_l2norm - m_sum");


  //reduction
  for (i = 0; i < l2norm_wg_num; i++) {
    for (m = 0; m < 5; m++) {
      sum[m] += g_sum[i][m];
    }
  }

  for (m = 0; m < 5; m++) {
    sum[m] = sqrt ( sum[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }
}
