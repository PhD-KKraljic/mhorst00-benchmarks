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

/* function declarations for data transfer variables */
int get_u_copy_item(int item, int base);
int get_u_copy_bbase(int base);
int get_u_copy_hbase(int base);

/* rhs baseline functions */
void rhs_init_baseline(int iter);
void rhs_release_baseline(int iter);
cudaEvent_t* rhs_body_baseline(int work_step, 
                               int work_base, 
                               int work_num_item, 
                               int copy_buffer_base, 
                               int copy_num_item, 
                               cudaEvent_t* ev_wb_ptr);
/* rhs global memory access opt functions */
void rhs_init_gmem(int iter);
void rhs_release_gmem(int iter);
cudaEvent_t* rhs_body_gmem(int work_step,
                           int work_base,
                           int work_num_item,
                           int copy_buffer_base,
                           int copy_num_item,
                           cudaEvent_t *ev_wb_ptr);

/* rhs parallelized opt functions */
void rhs_init_parallel(int iter);
void rhs_release_parallel(int iter);
cudaEvent_t* rhs_body_parallel(int work_step,
                               int work_base,
                               int work_num_item,
                               int copy_buffer_base,
                               int copy_num_item,
                               cudaEvent_t *ev_wb_ptr);

/* rhs fullopt functions */
void rhs_init_fullopt(int iter);
void rhs_release_fullopt(int iter);
cudaEvent_t* rhs_body_fullopt(int work_step, 
                              int work_base, 
                              int work_num_item, 
                              int copy_buffer_base, 
                              int copy_num_item, 
                              cudaEvent_t* ev_wb_ptr);

void rhs_init(int iter)
{  
  switch (g_opt_level) {
    case OPT_BASELINE:
      rhs_init_baseline(iter);
      break;
    case OPT_PARALLEL:
      rhs_init_parallel(iter);
      break;
    case OPT_GLOBALMEM:
      rhs_init_gmem(iter);
      break;
    case OPT_FULL:
      rhs_init_fullopt(iter);
      break;
    case OPT_MEMLAYOUT:
    case OPT_SYNC:
    default :
      rhs_init_baseline(iter);
      break;
  }
}

void rhs_release(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      rhs_release_baseline(iter);
      break;
    case OPT_PARALLEL:
      rhs_release_parallel(iter);
      break;
    case OPT_GLOBALMEM:
      rhs_release_gmem(iter);
      break;
    case OPT_FULL:
      rhs_release_fullopt(iter);
      break;
    case OPT_MEMLAYOUT:
    case OPT_SYNC:
    default :
      rhs_release_baseline(iter);
      break;
  }
}

cudaEvent_t* rhs_body(int work_step, 
                      int work_base, 
                      int work_num_item, 
                      int copy_buffer_base, 
                      int copy_num_item, 
                      cudaEvent_t* ev_wb_ptr)
{

  switch (g_opt_level) {
    case OPT_BASELINE:
      return rhs_body_baseline(work_step, 
                               work_base,
                               work_num_item,
                               copy_buffer_base,
                               copy_num_item,
                               ev_wb_ptr);
    case OPT_PARALLEL:
      return rhs_body_parallel(work_step, 
                               work_base,
                               work_num_item,
                               copy_buffer_base,
                               copy_num_item,
                               ev_wb_ptr);
    case OPT_GLOBALMEM:
      return rhs_body_gmem(work_step, 
                           work_base,
                           work_num_item,
                           copy_buffer_base,
                           copy_num_item,
                           ev_wb_ptr);
    case OPT_FULL:
      return rhs_body_fullopt(work_step,
                              work_base,
                              work_num_item,
                              copy_buffer_base,
                              copy_num_item,
                              ev_wb_ptr);

    case OPT_MEMLAYOUT:
    case OPT_SYNC:
    default :
      return rhs_body_baseline(work_step, 
                               work_base,
                               work_num_item,
                               copy_buffer_base,
                               copy_num_item,
                               ev_wb_ptr);
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
  int work_step, work_max_iter, work_num_item, work_base;
  int temp_work_num_item_default;
  int u_copy_buffer_base, u_copy_num_item, u_copy_host_base;

  int next_work_base, next_work_num_item;
  int next_u_copy_host_base, next_u_copy_num_item, next_u_copy_buffer_base;

  int buf_idx, next_buf_idx;

  cudaEvent_t *ev_wb, *ev_rb, *ev_k_end;


  if (timeron) timer_start(t_rhs);

  temp_work_num_item_default = (split_flag) ? (work_num_item_default - 4) : work_num_item_default;
  work_max_iter = (nz - 1)/temp_work_num_item_default + 1;

  ev_wb = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*work_max_iter);
  ev_rb = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*work_max_iter);

  for (int i = 0; i < work_max_iter; ++i) {
    CUCHK(cudaEventCreate(&ev_wb[i]));
    CUCHK(cudaEventCreate(&ev_rb[i]));
  }

  // #####################
  //  Write First Buffer
  // #####################
  if (split_flag && buffering_flag) {

    work_base = 0;
    work_num_item = min(temp_work_num_item_default, nz - work_base);

    u_copy_num_item = get_u_copy_item(work_num_item, work_base);
    u_copy_buffer_base = get_u_copy_bbase(work_base);
    u_copy_host_base = get_u_copy_hbase(work_base);

    CUCHK(cudaMemcpyAsync(
          ((char*)m_u[0]) + sizeof(double)*u_copy_buffer_base*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          &(u[u_copy_host_base][0][0][0]),
          sizeof(double)*u_copy_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));

    CUCHK(cudaMemcpyAsync(((char*)m_frct[0]) + sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5, 
          &(frct[work_base][0][0][0]),
        sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
        cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
    CUCHK(cudaEventRecord(ev_wb[0], cmd_q[DATA_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
  }



  for (work_step = 0; work_step < work_max_iter; work_step++) {

    work_base = work_step*temp_work_num_item_default;
    work_num_item = min(temp_work_num_item_default, nz - work_base);
    buf_idx = (work_step%2)*buffering_flag;
    next_buf_idx = ((work_step+1)%2)*buffering_flag;

    u_copy_num_item = get_u_copy_item(work_num_item, work_base);
    u_copy_buffer_base = get_u_copy_bbase(work_base);
    u_copy_host_base = get_u_copy_hbase(work_base);


    if (split_flag) {
      if (!buffering_flag) {

        // ################
        // write buffer
        // ################
        CUCHK(cudaMemcpyAsync(((char*)m_u[0]) + sizeof(double)*u_copy_buffer_base*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              &(u[u_copy_host_base][0][0][0]),
              sizeof(double)*u_copy_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_frct[0]) + sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
            &(frct[work_base][0][0][0]),
            sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
            cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
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

        // Double buffering does not need to wait on write buffer ( read(n) -> write(n+2) is guarunteed )
        // but not in Triple buffering
        CUCHK(cudaMemcpyAsync(
              ((char*)m_u[next_buf_idx]) + sizeof(double)*next_u_copy_buffer_base*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              &(u[next_u_copy_host_base][0][0][0]),
              sizeof(double)*next_u_copy_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
        CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_frct[next_buf_idx]) + sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              &(frct[next_work_base][0][0][0]),
              sizeof(double)*next_work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              cudaMemcpyHostToDevice, cmd_q[DATA_Q]));
        CUCHK(cudaEventRecord(ev_wb[work_step+1], cmd_q[DATA_Q]));
        CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
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

        CUCHK(cudaMemcpyAsync(&(rsd[work_base][0][0][0]),
              ((char*)m_rsd[0]) 
                + sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              cudaMemcpyDeviceToHost, cmd_q[KERNEL_Q]));
        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

      }
      else {
        // Double buffering dose not need event object on read buffer 
        // ( read(n) -> write(n+2) is guarunteed )
        // but not in Triple buffering
        CUCHK(cudaStreamWaitEvent(cmd_q[DATA_Q], *ev_k_end, 0));

        CUCHK(cudaMemcpyAsync(&(rsd[work_base][0][0][0]),
              ((char*)m_rsd[buf_idx]) 
                + sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
              cudaMemcpyDeviceToHost, cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(ev_rb[work_step], cmd_q[DATA_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));

      }
    }
  }

  if (buffering_flag) {
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
  }

  cudaEventDestroy(*ev_wb);
  cudaEventDestroy(*ev_rb);
  free(ev_wb);
  free(ev_rb);


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

