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

#include "applu.incl"
#include "timers.h"
#include <math.h>
#include <stdio.h>



/* function declarations for data transfer variables */
int get_u_copy_item(int item, int base);
int get_u_copy_bbase(int base);
int get_u_copy_hbase(int base);

/* rhs baseline functions */
void rhs_init_baseline(int iter);
void rhs_release_baseline();
void rhs_release_ev_baseline(int iter);
cl_event* rhs_body_baseline(int work_step, 
                            int work_base, 
                            int work_num_item, 
                            int copy_buffer_base, 
                            int copy_num_item, 
                            cl_event* ev_wb_ptr);

/* rhs parallelized opt functions */
void rhs_init_parallel(int iter);
void rhs_release_parallel();
void rhs_release_ev_parallel(int iter);
cl_event* rhs_body_parallel(int work_step,
                            int work_base,
                            int work_num_item,
                            int copy_buffer_base,
                            int copy_num_item,
                            cl_event *ev_wb_ptr);

/* rhs global memory access opt functions */
void rhs_init_gmem(int iter);
void rhs_release_gmem();
void rhs_release_ev_gmem(int iter);
cl_event* rhs_body_gmem(int work_step,
                        int work_base,
                        int work_num_item,
                        int copy_buffer_base,
                        int copy_num_item,
                        cl_event *ev_wb_ptr);

/* rhs fullopt functions */
void rhs_init_fullopt(int iter);
void rhs_release_fullopt();
void rhs_release_ev_fullopt(int iter);
cl_event* rhs_body_fullopt(int work_step, 
                           int work_base, 
                           int work_num_item, 
                           int copy_buffer_base, 
                           int copy_num_item, 
                           cl_event* ev_wb_ptr);

void rhs_init(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      rhs_init_baseline(iter);
      break;
    case OPT_GLOBALMEM:
      rhs_init_gmem(iter);
      break;
    case OPT_PARALLEL:
      rhs_init_parallel(iter);
      break;
    case OPT_FULL:
      rhs_init_fullopt(iter);
      break;
    default :
      rhs_init_baseline(iter);
      break;
  }
}

void rhs_release()
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      rhs_release_baseline();
      break;
    case OPT_GLOBALMEM:
      rhs_release_gmem();
      break;
    case OPT_PARALLEL:
      rhs_release_parallel();
      break;
    case OPT_FULL:
      rhs_release_fullopt();
      break;
    default :
      rhs_release_baseline();
      break;
  }
}

void rhs_release_ev(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      rhs_release_ev_baseline(iter);
      break;
    case OPT_GLOBALMEM:
      rhs_release_ev_gmem(iter);
      break;
    case OPT_PARALLEL:
      rhs_release_ev_parallel(iter);
      break;
    case OPT_FULL:
      rhs_release_ev_fullopt(iter);
      break;
    default :
      rhs_release_ev_baseline(iter);
      break;
  }
}

cl_event* rhs_body(int work_step,
                   int work_base, int work_num_item, 
                   int copy_buffer_base, int copy_num_item, 
                   cl_event* ev_wb_ptr)
{

  switch (g_opt_level) {
    case OPT_BASELINE:
      return rhs_body_baseline(work_step, 
                               work_base,
                               work_num_item,
                               copy_buffer_base,
                               copy_num_item,
                               ev_wb_ptr);
      break;
    case OPT_GLOBALMEM:
      return rhs_body_gmem(work_step, 
                           work_base,
                           work_num_item,
                           copy_buffer_base,
                           copy_num_item,
                           ev_wb_ptr);
      break;
    case OPT_PARALLEL:
      return rhs_body_parallel(work_step, 
                               work_base,
                               work_num_item,
                               copy_buffer_base,
                               copy_num_item,
                               ev_wb_ptr);
      break;
    case OPT_FULL:
      return rhs_body_fullopt(work_step,
                              work_base,
                              work_num_item,
                              copy_buffer_base,
                              copy_num_item,
                              ev_wb_ptr);

      break;
    default :
      return rhs_body_baseline(work_step, 
                               work_base,
                               work_num_item,
                               copy_buffer_base,
                               copy_num_item,
                               ev_wb_ptr);
      break;
  }
}

//---------------------------------------------------------------------
// compute the right hand sides
//---------------------------------------------------------------------
void rhs()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  cl_int ecode;
  int work_step, work_max_iter, 
      work_num_item, work_base,
      temp_work_num_item_default,
      u_copy_buffer_base, u_copy_num_item, 
      u_copy_host_base,
      next_work_base, next_work_num_item,
      next_u_copy_host_base, next_u_copy_num_item, 
      next_u_copy_buffer_base;

  int buf_idx, next_buf_idx;

  cl_event *ev_wb, *ev_rb, *ev_k_end;

  if (timeron) timer_start(t_rhs);

  temp_work_num_item_default = (split_flag) ? (work_num_item_default - 4) : work_num_item_default;
  work_max_iter = (nz - 1)/temp_work_num_item_default + 1;

  ev_wb = (cl_event*)malloc(sizeof(cl_event)*work_max_iter);
  ev_rb = (cl_event*)malloc(sizeof(cl_event)*work_max_iter);

  // #####################
  //  Write First Buffer
  // #####################
  if (split_flag && buffering_flag) {

    work_base = 0;
    work_num_item = min(temp_work_num_item_default, nz - work_base);

    u_copy_num_item = get_u_copy_item(work_num_item, work_base);
    u_copy_buffer_base = get_u_copy_bbase(work_base);
    u_copy_host_base = get_u_copy_hbase(work_base);

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_u[0], 
                                 CL_TRUE,
                                 u_slice_size*u_copy_buffer_base,
                                 u_slice_size*u_copy_num_item,
                                 &(u[u_copy_host_base][0][0][0]), 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "rhs loop 1 write buffer m_u");

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_frct[0], 
                                 CL_TRUE,
                                 frct_slice_size*2,
                                 frct_slice_size*work_num_item,
                                 &(frct[work_base][0][0][0]), 
                                 0, NULL, &ev_wb[0]);
    clu_CheckError(ecode, "rhs loop 1 write buffer m_frct");
  }



  for (work_step = 0; work_step < work_max_iter; work_step++) {

    work_base = work_step*temp_work_num_item_default;
    work_num_item = min(temp_work_num_item_default, nz - work_base);

    u_copy_num_item = get_u_copy_item(work_num_item, work_base);
    u_copy_buffer_base = get_u_copy_bbase(work_base);
    u_copy_host_base = get_u_copy_hbase(work_base);

    buf_idx = (work_step%2)*buffering_flag;
    next_buf_idx = ((work_step+1)%2)*buffering_flag;

    if (split_flag) {
      if (!buffering_flag) {

        // ################
        // write buffer
        // ################
        ecode = clEnqueueWriteBuffer(cmd_q[KERNEL_Q], 
                                     m_u[0], 
                                     CL_TRUE,
                                     u_slice_size*u_copy_buffer_base,
                                     u_slice_size*u_copy_num_item,
                                     &(u[u_copy_host_base][0][0][0]), 
                                     0, NULL, NULL);
        clu_CheckError(ecode, "rhs loop 1 write buffer m_u");

        ecode = clEnqueueWriteBuffer(cmd_q[KERNEL_Q], 
                                     m_frct[0], 
                                     CL_TRUE,
                                     frct_slice_size*2,
                                     frct_slice_size*work_num_item,
                                     &(frct[work_base][0][0][0]), 0, NULL, NULL);
        clu_CheckError(ecode, "rhs loop 1 write buffer m_frct");


      }
      else if (work_step < work_max_iter - 1) {

        // ######################
        // Write Next Buffer
        // ######################
        next_work_base = (work_step+1)*temp_work_num_item_default;
        next_work_num_item = min(temp_work_num_item_default, nz - next_work_base);

        next_u_copy_num_item = get_u_copy_item(next_work_num_item, next_work_base);
        next_u_copy_buffer_base = get_u_copy_bbase(next_work_base);
        next_u_copy_host_base = get_u_copy_hbase(next_work_base);

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_u[next_buf_idx], 
                                     CL_TRUE,
                                     u_slice_size*next_u_copy_buffer_base,
                                     u_slice_size*next_u_copy_num_item,
                                     &(u[next_u_copy_host_base][0][0][0]), 
                                     0, NULL, NULL);
        clu_CheckError(ecode, "rhs loop 1 write buffer m_u");

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_frct[next_buf_idx], 
                                     CL_TRUE,
                                     frct_slice_size*2,
                                     frct_slice_size*next_work_num_item,
                                     &(frct[next_work_base][0][0][0]), 
                                     0, NULL, &ev_wb[work_step+1]);
        clu_CheckError(ecode, "rhs loop 1 write buffer m_frct");

      }
    }


    // ################
    // kernel execution
    // ################
    ev_k_end = rhs_body(work_step, 
                        work_base, 
                        work_num_item, 
                        u_copy_buffer_base, 
                        u_copy_num_item,
                        &ev_wb[work_step]);

    // ################
    // read buffer
    // ################
    if (split_flag) {
      if(!buffering_flag){

        ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                    m_rsd[0], 
                                    CL_TRUE,
                                    rsd_slice_size*2,
                                    rsd_slice_size*work_num_item,
                                    &(rsd[work_base][0][0][0]), 
                                    0, NULL, NULL);
        clu_CheckError(ecode, "rhs loop 1 write buffer m_rsd");
      }
      else {
        ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                    m_rsd[buf_idx], 
                                    CL_TRUE,
                                    rsd_slice_size*2,
                                    rsd_slice_size*work_num_item,
                                    &(rsd[work_base][0][0][0]), 
                                    1, ev_k_end, 
                                    &ev_rb[work_step]);
        clu_CheckError(ecode, "rhs loop 1 write buffer m_rsd");
      }
    }

  }

  if (buffering_flag) {
    clFinish(cmd_q[KERNEL_Q]);
    clFinish(cmd_q[DATA_Q]);

  }

  free(ev_wb);
  free(ev_rb);

  rhs_release_ev(work_max_iter);

  if (timeron) timer_stop(t_rhs);
}




int get_u_copy_item(int item, int base)
{
  int ret;

  if (split_flag) {
    ret = item;
    ret += min(base, 2);
    ret += min (nz - (base+item), 2);
  }
  else {
    ret = item;
  }

  return ret;
}

int get_u_copy_bbase(int base)
{
  int ret;

  if (split_flag) {
    ret = 2;
    ret -= min(base, 2);
  }
  else
    ret = base;

  return ret;
}

int get_u_copy_hbase(int base)
{
  int ret;

  ret = base;

  if (split_flag)
    ret -= min(base, 2);

  return ret;
}

