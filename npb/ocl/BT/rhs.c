//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB BT code. This OpenCL C  //
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
#include "timers.h"
#include <math.h>
#include <stdio.h>

void compute_rhs_init_baseline(int iter);
void compute_rhs_release_baseline();
void compute_rhs_release_ev_baseline(int iter);
void compute_rhs_body_baseline(int work_step, 
                               int work_base, 
                               int work_num_item, 
                               int copy_buffer_base, 
                               int copy_num_item, 
                               int buf_idx,
                               int wait,
                               cl_event* ev_wb_end_ptr);

void compute_rhs_init_parallel(int iter);
void compute_rhs_release_parallel();
void compute_rhs_release_ev_parallel(int iter);
void compute_rhs_body_parallel(int work_step, 
                               int work_base, 
                               int work_num_item, 
                               int copy_buffer_base, 
                               int copy_num_item, 
                               int buf_idx,
                               int wait,
                               cl_event* ev_wb_end_ptr);

void compute_rhs_init(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      compute_rhs_init_baseline(iter);
      break;
    case OPT_PARALLEL:
    case OPT_FULL:
      compute_rhs_init_parallel(iter);
      break;
    default :
      compute_rhs_init_baseline(iter);
  }
}

void compute_rhs_release()
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      compute_rhs_release_baseline();
      break;
    case OPT_PARALLEL:
    case OPT_FULL:
      compute_rhs_release_parallel();
      break;
    default :
      compute_rhs_release_baseline();
  }
}

void compute_rhs_release_ev(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      compute_rhs_release_ev_baseline(iter);
      break;
    case OPT_PARALLEL:
    case OPT_FULL:
      compute_rhs_release_ev_parallel(iter);
      break;
    default :
      compute_rhs_release_ev_baseline(iter);
  }
}

void compute_rhs_body(int work_step, int work_base, int work_num_item, 
                      int copy_buffer_base, int copy_num_item, 
                      int buf_idx,
                      int wait,
                      cl_event * ev_wb_end_ptr)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      compute_rhs_body_baseline(work_step, work_base, work_num_item,
                                copy_buffer_base, copy_num_item,
                                buf_idx, wait, ev_wb_end_ptr);
      break;
    case OPT_PARALLEL:
    case OPT_FULL:
      compute_rhs_body_parallel(work_step, work_base, work_num_item,
                                copy_buffer_base, copy_num_item,
                                buf_idx, wait, ev_wb_end_ptr);
      break;
    default :
      compute_rhs_body_baseline(work_step, work_base, work_num_item,
                                copy_buffer_base, copy_num_item,
                                buf_idx, wait, ev_wb_end_ptr);
      break;
  }
}

void compute_rhs()
{
  cl_int ecode;
  int work_step, work_max_iter, work_base;
  int work_num_item;
  int temp_work_num_item_default;
  int copy_buffer_base, copy_num_item, copy_host_base;

  if (timeron) timer_start(t_rhs);

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound.
  //---------------------------------------------------------------------
  temp_work_num_item_default = (split_flag) ? (work_num_item_default-4) : work_num_item_default;

  // the number of whole items to be processed
  work_max_iter = ( grid_points[2] - 1 ) / temp_work_num_item_default + 1;

  for (work_step = 0; work_step < work_max_iter; work_step++) {

    //work_base = work_step*work_num_item_default;
    work_base = work_step*temp_work_num_item_default;

    // the end index of items + 1
    work_num_item = min(temp_work_num_item_default, grid_points[2] - work_base);

    copy_num_item = get_loop1_copy_num_item(work_base, work_num_item);
    copy_buffer_base = get_loop1_copy_buffer_base(work_base);
    copy_host_base = get_loop1_copy_host_base(work_base);

    if (split_flag) {
      ecode = clEnqueueWriteBuffer(cmd_q[KERNEL_Q], 
                                   m_u[0], 
                                   CL_TRUE,
                                   sizeof(double)*copy_buffer_base*(JMAXP+1)*(IMAXP+1)*5,
                                   sizeof(double)*copy_num_item*(JMAXP+1)*(IMAXP+1)*5,
                                   &(u[copy_host_base][0][0][0]), 
                                   0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");

      ecode = clEnqueueWriteBuffer(cmd_q[KERNEL_Q], 
                                   m_forcing[0], 
                                   CL_TRUE,
                                   sizeof(double)*copy_buffer_base*(JMAXP+1)*(IMAXP+1)*5,
                                   sizeof(double)*copy_num_item*(JMAXP+1)*(IMAXP+1)*5,
                                   &(forcing[copy_host_base][0][0][0]), 
                                   0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueWriteBuffer()");
    }

    compute_rhs_body(work_step, work_base, work_num_item,
                     copy_buffer_base, copy_num_item,
                     0, 0, NULL);

    clFinish(cmd_q[KERNEL_Q]);

    if (split_flag) {
      ecode = clEnqueueReadBuffer(cmd_q[KERNEL_Q], 
                                  m_rhs[0], 
                                  CL_TRUE,
                                  sizeof(double)*2*(JMAXP+1)*(IMAXP+1)*5,
                                  sizeof(double)*work_num_item*(JMAXP+1)*(IMAXP+1)*5,
                                  &(rhs[work_base][0][0][0]), 0, NULL, NULL);
      clu_CheckError(ecode, "clEnqueueReadBuffer()");
    }
  }

  compute_rhs_release_ev(work_max_iter);

  if (timeron) timer_stop(t_rhs);
}
