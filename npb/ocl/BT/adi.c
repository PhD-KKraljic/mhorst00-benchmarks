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
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <stdio.h>

enum DataTransferType dt_type = DT_CONT;

double **temp_u, **temp_rhs;

void loop2_memcpy_pre();
void loop2_memcpy_post();
void loop2_write_cont(int ws, int work_num_item, int work_base);
void loop2_write(int ws, int work_num_item, int work_base);
void loop2_read_cont(int ws, int work_num_item, int work_base, int n_wait, cl_event *ev_wait);
void loop2_read(int ws, int work_num_item, int work_base, int n_wait, cl_event *ev_wait);

void adi_init()
{
  if (!split_flag)
    return; 

  int ws;
  temp_u = (double **)malloc(sizeof(double*)*loop2_work_max_iter);
  temp_rhs = (double **)malloc(sizeof(double*)*loop2_work_max_iter);
    
  for (ws = 0; ws < loop2_work_max_iter; ws++) {
    temp_u[ws] = (double *)malloc(sizeof(double*)*KMAX*work_num_item_default*(IMAXP+1)*5);
    temp_rhs[ws] = (double *)malloc(sizeof(double*)*KMAX*work_num_item_default*(IMAXP+1)*5);
  }

  // profiling to choose data transfer strategy
  double start_t, end_t;
  double cont_time, rect_time;
  int work_step, work_base, work_num_item;

  DETAIL_LOG(" Choosing Data Transfer Strategy ... ");

  // profiling data transfer using contiguous memory
  timer_clear(t_prof_cont);
  start_t = timer_read(t_prof_cont);
  timer_start(t_prof_cont);

  loop2_memcpy_pre();

  for (work_step = 0; work_step < loop2_work_max_iter; work_step++) {
    work_base = work_step*work_num_item_default + 1;
    work_num_item = min(work_num_item_default, grid_points[1]-1 - work_base);

    loop2_write_cont(work_step, work_num_item, work_base);

    loop2_read_cont(work_step, work_num_item, work_base, 0, NULL);

    clFinish(cmd_q[KERNEL_Q]);
    clFinish(cmd_q[DATA_Q]);
  }

  loop2_memcpy_post();

  timer_stop(t_prof_cont);
  end_t = timer_read(t_prof_cont);
  cont_time = end_t - start_t;   

  // profiling data transfer using rect API
  timer_clear(t_prof_rect);
  start_t = timer_read(t_prof_rect);
  timer_start(t_prof_rect);

  for (work_step = 0; work_step < loop2_work_max_iter; work_step++) {
    work_base = work_step*work_num_item_default + 1;
    work_num_item = min(work_num_item_default, grid_points[1]-1 - work_base);

    loop2_write(work_step, work_num_item, work_base);

    loop2_read(work_step, work_num_item, work_base, 0, NULL);

    clFinish(cmd_q[KERNEL_Q]);
    clFinish(cmd_q[DATA_Q]);
  }

  timer_stop(t_prof_rect);
  end_t = timer_read(t_prof_rect);
  rect_time = end_t - start_t;

  if (cont_time < rect_time)
    dt_type = DT_CONT;
  else
    dt_type = DT_RECT;

  DETAIL_LOG(" Selected Data Transfer Strategy : %s",
      (dt_type == DT_CONT) ? "Cont" : "Rect");

  if (dt_type == DT_RECT) {
    for (ws = 0; ws < loop2_work_max_iter; ws++) {
      free(temp_u[ws]);
      free(temp_rhs[ws]);
    }
    free(temp_u);
    free(temp_rhs);
  }
}

void adi_free()
{
  if (!split_flag)
    return; 

  int ws;
  if (dt_type == DT_CONT) {
    for (ws = 0; ws < loop2_work_max_iter; ws++) {
      free(temp_u[ws]);
      free(temp_rhs[ws]);
    }
    free(temp_u);
    free(temp_rhs);
  }
}

int get_loop1_copy_num_item(int work_base, int work_num_item)
{
  int ret;

  ret = work_num_item;

  if (split_flag) {
    // front alignment
    ret += (work_base < 2) ? work_base : 2;

    // end alignment
    if (grid_points[2] - (work_base + work_num_item) < 2)
      ret += (grid_points[2] - (work_base + work_num_item));
    else
      ret += 2;
  }

  return ret;
}

int get_loop1_copy_buffer_base(int work_base)
{
  int ret;
  if (split_flag) {
    ret = 2;
    ret -= (work_base < 2) ? work_base : 2;
  }
  else {
    ret = work_base;
  }

  return ret;
}

int get_loop1_copy_host_base(int work_base)
{
  int ret;
  ret = work_base;

  if (split_flag) {
    ret -= (work_base < 2) ? work_base : 2;
  }

  return ret;
}

void loop1_write(int ws, int copy_bbase, int copy_hbase, int copy_item)
{
  cl_int ecode;
  int q_type, b_idx;
  cl_bool u_blk, forcing_blk;

  if (!buffering_flag) {
    q_type = KERNEL_Q;
    b_idx = 0;
    u_blk = CL_TRUE;
    forcing_blk = CL_TRUE;
  }
  else {
    q_type = DATA_Q;
    b_idx = ws%2;
    u_blk = CL_FALSE;
    forcing_blk = CL_TRUE;
  }

  ecode = clEnqueueWriteBuffer(cmd_q[q_type],
                               m_u[b_idx],
                               u_blk,
                               sizeof(double)*copy_bbase*(JMAXP+1)*(IMAXP+1)*5,
                               sizeof(double)*copy_item*(JMAXP+1)*(IMAXP+1)*5,
                               &(u[copy_hbase][0][0][0]),
                               0, NULL, &loop1_ev_wb_start[ws]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer");

  if (u_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);

  ecode = clEnqueueWriteBuffer(cmd_q[q_type],
                               m_forcing[b_idx],
                               forcing_blk,
                               sizeof(double)*copy_bbase*(JMAXP+1)*(IMAXP+1)*5,
                               sizeof(double)*copy_item*(JMAXP+1)*(IMAXP+1)*5,
                               &(forcing[copy_hbase][0][0][0]),
                               0, NULL, &loop1_ev_wb_end[ws]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer");
}

void loop1_read(int ws, int work_num_item, int work_base, cl_event *ev_k_end_ptr)
{
  cl_int ecode;
  int q_type, b_idx;
  cl_bool rhs_blk;
  int n_wait;
  cl_event *ev_wait;

  if (!buffering_flag) {
    q_type = KERNEL_Q;
    b_idx = 0;
    rhs_blk = CL_TRUE;
    n_wait = 0;
    ev_wait = NULL;
  }
  else {
    q_type = DATA_Q;
    b_idx = ws%2;
    rhs_blk = CL_FALSE;
    n_wait = 1;
    ev_wait = ev_k_end_ptr;
  }

  ecode = clEnqueueReadBuffer(cmd_q[q_type],
                              m_rhs[b_idx],
                              rhs_blk,
                              sizeof(double)*2*(JMAXP+1)*(IMAXP+1)*5,
                              sizeof(double)*work_num_item*(JMAXP+1)*(IMAXP+1)*5,
                              &(rhs[work_base][0][0][0]),
                              n_wait, ev_wait,
                              &loop1_ev_rb_end[ws]);
  clu_CheckError(ecode, "clEnqueueReadBuffer");

  if (rhs_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);
}

void loop2_write(int ws, int work_num_item, int work_base)
{
  cl_int ecode;
  int q_type, b_idx;
  cl_bool u_blk, rhs_blk;
  size_t buffer_origin[3], host_origin[3], region[3];
  size_t buffer_row_pitch, buffer_slice_pitch;
  size_t host_row_pitch, host_slice_pitch;

  if (!buffering_flag) {
    q_type = KERNEL_Q;
    b_idx = 0;
    u_blk = CL_TRUE;
    rhs_blk = CL_TRUE;
  }
  else {
    q_type = DATA_Q;
    b_idx = ws%2;
    u_blk = CL_FALSE;
    rhs_blk = CL_TRUE;
  }

  buffer_origin[2] = 0;
  buffer_origin[1] = 0;
  buffer_origin[0] = 0;

  host_origin[2] = 0;
  host_origin[1] = work_base;
  host_origin[0] = 0;

  region[2] = KMAX;
  region[1] = work_num_item;
  region[0] = (IMAXP+1)*5*sizeof(double);

  buffer_row_pitch = region[0];
  buffer_slice_pitch = work_num_item_default*buffer_row_pitch;

  host_row_pitch = region[0];
  host_slice_pitch = (JMAXP+1)*host_row_pitch;

  ecode = clEnqueueWriteBufferRect(cmd_q[q_type],
                                   m_u[b_idx],
                                   u_blk,
                                   buffer_origin,
                                   host_origin,
                                   region,
                                   buffer_row_pitch,
                                   buffer_slice_pitch,
                                   host_row_pitch,
                                   host_slice_pitch,
                                   u,
                                   0, NULL, &loop2_ev_wb_start[ws]);
  clu_CheckError(ecode, "clEnqueueWriteBufferRect");

  if (u_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);

  ecode = clEnqueueWriteBufferRect(cmd_q[q_type],
                                   m_rhs[b_idx],
                                   rhs_blk,
                                   buffer_origin,
                                   host_origin,
                                   region,
                                   buffer_row_pitch,
                                   buffer_slice_pitch,
                                   host_row_pitch,
                                   host_slice_pitch,
                                   rhs,
                                   0, NULL, &loop2_ev_wb_end[ws]);
  clu_CheckError(ecode, "clEnqueueWriteBufferRect");
}

void loop2_memcpy_pre()
{
  int ws, witem, wbase, k;

  if (timeron)
    timer_start(t_memcpy_pre);

  for (ws = 0; ws < loop2_work_max_iter; ws++) {

    wbase = ws*work_num_item_default + 1;
    witem = min(work_num_item_default, grid_points[1]-1 - wbase);

    #pragma omp parallel for
    for (k = 0; k < KMAX; k++) {
      memcpy(&temp_u[ws][k*work_num_item_default*(IMAXP+1)*5],
             &u[k][wbase][0][0],
             sizeof(double)*witem*(IMAXP+1)*5);

      memcpy(&temp_rhs[ws][k*work_num_item_default*(IMAXP+1)*5],
             &rhs[k][wbase][0][0],
             sizeof(double)*witem*(IMAXP+1)*5);
    }
  }

  if (timeron)
    timer_stop(t_memcpy_pre);
}

void loop2_memcpy_post()
{
  int ws, witem, wbase, k;

  if (timeron)
    timer_start(t_memcpy_post);

  for (ws = 0; ws < loop2_work_max_iter; ws++) {
    wbase = ws*work_num_item_default + 1;
    witem = min(work_num_item_default, grid_points[1]-1 - wbase);

    #pragma omp parallel for
    for (k = 0; k < KMAX; k++) {
      memcpy(&u[k][wbase][0][0],
             &temp_u[ws][k*work_num_item_default*(IMAXP+1)*5],
             sizeof(double)*witem*(IMAXP+1)*5);

      memcpy(&rhs[k][wbase][0][0],
             &temp_rhs[ws][k*work_num_item_default*(IMAXP+1)*5],
             sizeof(double)*witem*(IMAXP+1)*5);
    }
  }

  if (timeron)
    timer_stop(t_memcpy_post);
}

void loop2_write_cont(int ws, int work_num_item, int work_base)
{
  cl_int ecode;
  int q_type, b_idx;
  cl_bool u_blk, rhs_blk;

  if (!buffering_flag) {
    q_type = KERNEL_Q;
    b_idx = 0;
    u_blk = CL_TRUE;
    rhs_blk = CL_TRUE;
  }
  else {
    q_type = DATA_Q;
    b_idx = ws%2;
    u_blk = CL_FALSE;
    rhs_blk = CL_TRUE;
  }
  
  ecode = clEnqueueWriteBuffer(cmd_q[q_type],
                               m_u[b_idx],
                               u_blk,
                               0,
                               sizeof(double)*KMAX*work_num_item_default*(IMAXP+1)*5,
                               &temp_u[ws][0],
                               0, NULL, &loop2_ev_wb_start[ws]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer");
                               
  if (u_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);
  
  ecode = clEnqueueWriteBuffer(cmd_q[q_type],
                               m_rhs[b_idx],
                               rhs_blk,
                               0,
                               sizeof(double)*KMAX*work_num_item_default*(IMAXP+1)*5,
                               &temp_rhs[ws][0],
                               0, NULL, &loop2_ev_wb_end[ws]);
  clu_CheckError(ecode, "clEnqueueWriteBuffer");
}

cl_int event_status(cl_event ev) {
  cl_int ecode, ret;;
  ecode = clGetEventInfo(ev,
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &ret,
                         NULL);
  clu_CheckError(ecode, "");

  return ret;
}

void loop2_read(int ws, int work_num_item, int work_base, int n_wait, cl_event *ev_wait)
{
  cl_int ecode;
  int q_type, b_idx;
  cl_bool u_blk, rhs_blk;
  size_t buffer_origin[3], host_origin[3], region[3];
  size_t buffer_row_pitch, buffer_slice_pitch;
  size_t host_row_pitch, host_slice_pitch;

  if (!buffering_flag) {
    q_type = KERNEL_Q;
    b_idx = 0;
    u_blk = CL_TRUE;
    rhs_blk = CL_TRUE;
  }
  else {
    q_type = DATA_Q;
    b_idx = ws%2;
    u_blk = CL_FALSE;
    rhs_blk = CL_FALSE;
  }

  buffer_origin[2] = 0;
  buffer_origin[1] = 0;
  buffer_origin[0] = 0;

  host_origin[2] = 0;
  host_origin[1] = work_base;
  host_origin[0] = 0;

  region[2] = KMAX;
  region[1] = work_num_item;
  region[0] = (IMAXP+1)*5*sizeof(double);

  buffer_row_pitch = region[0];
  buffer_slice_pitch = work_num_item_default*buffer_row_pitch;

  host_row_pitch = region[0];
  host_slice_pitch = (JMAXP+1)*host_row_pitch;

  ecode = clEnqueueReadBufferRect(cmd_q[q_type],
                                  m_u[b_idx],
                                  u_blk,
                                  buffer_origin,
                                  host_origin,
                                  region,
                                  buffer_row_pitch,
                                  buffer_slice_pitch,
                                  host_row_pitch,
                                  host_slice_pitch,
                                  u,
                                  n_wait, ev_wait, 
                                  &loop2_ev_rb_start[ws]);
  clu_CheckError(ecode, "clEnqueueReadBufferRect");

  // this is for 7970 ReadRect Bug
  if (u_blk == CL_FALSE) {
    while (event_status(loop2_ev_rb_start[ws]) == CL_SUBMITTED
        || event_status(loop2_ev_rb_start[ws]) == CL_QUEUED) {
      clFlush(cmd_q[q_type]);
    };
  }

  ecode = clEnqueueReadBufferRect(cmd_q[q_type],
                                  m_rhs[b_idx],
                                  rhs_blk,
                                  buffer_origin,
                                  host_origin,
                                  region,
                                  buffer_row_pitch,
                                  buffer_slice_pitch,
                                  host_row_pitch,
                                  host_slice_pitch,
                                  rhs,
                                  0, NULL, &loop2_ev_rb_end[ws]);
  clu_CheckError(ecode, "clEnqueueReadBufferRect");

  if (rhs_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);

}

void loop2_read_cont(int ws, int work_num_item, int work_base, int n_wait, cl_event *ev_wait) 
{
  cl_int ecode;
  int q_type, b_idx;
  cl_bool u_blk, rhs_blk;

  if (!buffering_flag) {
    q_type = KERNEL_Q;
    b_idx = 0;
    u_blk = CL_TRUE;
    rhs_blk = CL_TRUE;
  }
  else {
    q_type = DATA_Q;
    b_idx = ws%2;
    u_blk = CL_FALSE;
    rhs_blk = CL_FALSE;
  }

  ecode = clEnqueueReadBuffer(cmd_q[q_type],
                              m_u[b_idx],
                              u_blk,
                              0,
                              sizeof(double)*KMAX*work_num_item_default*(IMAXP+1)*5,
                              &temp_u[ws][0],
                              n_wait, ev_wait, &loop2_ev_rb_start[ws]);
  clu_CheckError(ecode, "clEnqueueReadBuffer");

  if (u_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);

  ecode = clEnqueueReadBuffer(cmd_q[q_type],
                              m_rhs[b_idx],
                              rhs_blk,
                              0,
                              sizeof(double)*KMAX*work_num_item_default*(IMAXP+1)*5,
                              &temp_rhs[ws][0],
                              0, NULL, &loop2_ev_rb_end[ws]);
  clu_CheckError(ecode, "clEnqueueReadBuffer");

  if (rhs_blk == CL_FALSE)
    clFlush(cmd_q[q_type]);
}

void adi()
{
  int work_step;
  int work_base, work_num_item;
  int copy_num_item, copy_buffer_base, copy_host_base;
  int next_work_base, next_work_num_item;
  int next_copy_num_item, next_copy_buffer_base, next_copy_host_base;
  int i;

  cl_event *loop1_ev_kernel_end_ptr;

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound.
  //---------------------------------------------------------------------

  // #################################
  //  Write First Buffer
  // #################################
  if (buffering_flag) {
    work_step = 0;
    work_base = work_step*loop1_work_num_item_default;

    // the end index of items + 1
    work_num_item = min(loop1_work_num_item_default, grid_points[2] - work_base);

    copy_num_item = get_loop1_copy_num_item(work_base, work_num_item);
    copy_buffer_base = get_loop1_copy_buffer_base(work_base);
    copy_host_base = get_loop1_copy_host_base(work_base);

    if (split_flag) {
      loop1_write(work_step, copy_buffer_base, copy_host_base, copy_num_item);
    }
  }

  for (work_step = 0; work_step < loop1_work_max_iter; work_step++) {

    work_base = work_step*loop1_work_num_item_default;

    // the end index of items + 1
    work_num_item = min(loop1_work_num_item_default, grid_points[2] - work_base);

    copy_num_item = get_loop1_copy_num_item(work_base, work_num_item);
    copy_buffer_base = get_loop1_copy_buffer_base(work_base);
    copy_host_base = get_loop1_copy_host_base(work_base);


    if (split_flag) {
      if (!buffering_flag) {
        // #################################
        //  Write buffer
        // #################################
        loop1_write(work_step, copy_buffer_base, copy_host_base, copy_num_item);
      }
    }

    compute_rhs_body(work_step, work_base, work_num_item,
                     copy_buffer_base, copy_num_item,
                     (work_step%2)*buffering_flag,
                     buffering_flag,
                     &loop1_ev_wb_end[work_step]);

    x_solve(work_step, 
            work_base, 
            work_num_item, 
            (work_step%2)*buffering_flag);

    loop1_ev_kernel_end_ptr = y_solve(work_step, 
                                      work_base, 
                                      work_num_item, 
                                      (work_step%2)*buffering_flag);

    if (split_flag) {
      if (buffering_flag && work_step < loop1_work_max_iter-1) {
        // #################################
        //  Write Next buffer
        // #################################
        next_work_base = (work_step+1)*loop1_work_num_item_default;

        // the end index of items + 1
        next_work_num_item = min(loop1_work_num_item_default, grid_points[2] - next_work_base);

        next_copy_num_item = get_loop1_copy_num_item(next_work_base, next_work_num_item);
        next_copy_buffer_base = get_loop1_copy_buffer_base(next_work_base);
        next_copy_host_base = get_loop1_copy_host_base(next_work_base);

        loop1_write(work_step+1, next_copy_buffer_base, next_copy_host_base, next_copy_num_item);
      }
    }

    // ############################################
    //  Read Buffer
    // ############################################
    if (split_flag) {
      loop1_read(work_step, work_num_item, work_base, loop1_ev_kernel_end_ptr);
    }
  }

  if (buffering_flag) {
    clFinish(cmd_q[KERNEL_Q]);
    clFinish(cmd_q[DATA_Q]);
  }

  for (i = 0; i < loop1_work_max_iter; i++) {
    if (split_flag) {
      clReleaseEvent(loop1_ev_wb_start[i]);
      clReleaseEvent(loop1_ev_wb_end[i]);
      clReleaseEvent(loop1_ev_rb_end[i]);
    }
  }

  x_solve_release_ev(loop1_work_max_iter);

  y_solve_release_ev(loop1_work_max_iter);

  compute_rhs_release_ev(loop1_work_max_iter);


  if (split_flag) {
    if (dt_type == DT_CONT)
      loop2_memcpy_pre();
  }

  // #################################
  //  Write First Buffer
  // #################################
  if (buffering_flag) {
    work_step = 0;
    work_base = work_step*work_num_item_default + 1;
    // the end index of items + 1
    work_num_item = min(work_num_item_default, grid_points[1]-1 - work_base);

    if (dt_type == DT_CONT) {
      loop2_write_cont(work_step, work_num_item, work_base);
    }
    else {
      loop2_write(work_step, work_num_item, work_base);
    }
  }


  for (work_step = 0; work_step < loop2_work_max_iter; work_step++) {
    // BE careful
    work_base = work_step*work_num_item_default + 1;
    // the end index of items + 1
    work_num_item = min(work_num_item_default, grid_points[1]-1 - work_base);

    if (split_flag) {
      if (!buffering_flag) {
        // #################################
        //  Write Buffer
        // #################################
        if (dt_type == DT_CONT) {
          loop2_write_cont(work_step, work_num_item, work_base);
        }
        else {
          loop2_write(work_step, work_num_item, work_base);
        }
      }
    }

    z_solve(work_step, 
            work_base, 
            work_num_item, 
            (work_step%2)*buffering_flag, 
            &loop2_ev_wb_end[work_step]);

    add(work_step, 
        work_base, 
        work_num_item,
        (work_step%2)*buffering_flag); 

    if (split_flag && buffering_flag && work_step < loop2_work_max_iter-1) {
      // #################################
      //  Write Next Buffer
      // #################################
      next_work_base = (work_step+1)*work_num_item_default + 1;
      // the end index of items + 1
      next_work_num_item = min(work_num_item_default, grid_points[1]-1 - next_work_base);

      if (dt_type == DT_CONT) {
        loop2_write_cont(work_step+1, next_work_num_item, next_work_base);
      }
      else {
        loop2_write(work_step+1, next_work_num_item, next_work_base);
      }
    }

    // #################################
    //  Read Buffer
    // #################################
    if (split_flag) {
      if (dt_type == DT_CONT) {
        loop2_read_cont(work_step, work_num_item, work_base, 1, &loop2_ev_kernel_add[work_step]);
      }
      else {
        loop2_read(work_step, work_num_item, work_base, 1, &loop2_ev_kernel_add[work_step]);
      }
    }

  }

  if (buffering_flag) {
    clFinish(cmd_q[KERNEL_Q]);
    clFinish(cmd_q[DATA_Q]);
  }

  if (split_flag)
    if (dt_type == DT_CONT)
      loop2_memcpy_post();

  for (i = 0; i < loop2_work_max_iter; i++) {
    clReleaseEvent(loop2_ev_kernel_add[i]);
    if (split_flag) {
      clReleaseEvent(loop2_ev_wb_start[i]);
      clReleaseEvent(loop2_ev_wb_end[i]);
      clReleaseEvent(loop2_ev_rb_start[i]);
      clReleaseEvent(loop2_ev_rb_end[i]);
    }
  }

  z_solve_release_ev(loop2_work_max_iter);

}
