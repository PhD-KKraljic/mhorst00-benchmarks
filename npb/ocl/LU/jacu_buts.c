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

#include <stdio.h>
#include "applu.incl"
#include "timers.h"
#include <math.h>

//---------------------------------------------------------------------
// compute the upper triangular part of the jacobian matrix
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block upper triangular solution:
// 
// v <-- ( U-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
cl_event *loop2_ev_copy_u_prev,
         *loop2_ev_copy_r_prev,
         *loop2_ev_copy_u,
         *loop2_ev_copy_rsd;

/* jacu buts baseline functions */
void jacu_buts_init_baseline(int iter, 
                             int item_default, 
                             int blk_size_k, 
                             int blk_size);
void jacu_buts_release_baseline(int iter);
void jacu_buts_release_ev_baseline(int iter);
void jacu_buts_body_baseline(int work_step,
                             int work_max_iter,
                             int work_num_item,
                             int next_work_num_item,
                             int temp_kst,
                             int tmep_kend,
                             cl_event *ev_wb_ptr);

/* jacu buts global memory access opt functions */
void jacu_buts_init_gmem(int iter,
                         int item_default,
                         int blk_size_k,
                         int blk_size);
void jacu_buts_release_gmem(int iter);
void jacu_buts_release_ev_gmem(int iter);
void jacu_buts_body_gmem(int work_step,
                         int work_max_iter,
                         int work_num_item,
                         int next_work_num_item,
                         int temp_kst,
                         int temp_kend,
                         cl_event *ev_wb_ptr);

/* jacu buts synchronization opt functions */
void jacu_buts_init_sync(int iter,
                         int item_default,
                         int blk_size_k,
                         int blk_size);
void jacu_buts_release_sync(int iter);
void jacu_buts_release_ev_sync(int iter);
void jacu_buts_body_sync(int work_step,
                         int work_max_iter,
                         int work_num_item,
                         int next_work_num_item,
                         int temp_kst,
                         int temp_kend,
                         cl_event *ev_wb_ptr);


/* jacu buts fullopt fucntions */
void jacu_buts_init_fullopt(int iter, 
                            int item_defualt, 
                            int blk_size_k, 
                            int blk_size);
void jacu_buts_release_fullopt(int iter);
void jacu_buts_release_ev_fullopt(int iter);
void jacu_buts_body_fullopt(int work_step,
                            int work_max_iter,
                            int work_num_item,
                            int next_work_num_item,
                            int temp_kst,
                            int temp_kend,
                            cl_event *ev_wb_ptr);

void jacu_buts_init(int iter, int item_default, int blk_size_k, int blk_size)
{

  loop2_ev_copy_u_prev  = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_copy_r_prev  = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_copy_u       = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_copy_rsd     = (cl_event*)malloc(sizeof(cl_event)*iter);

  switch (g_opt_level) {
    case OPT_BASELINE:
      jacu_buts_init_baseline(iter, item_default, blk_size_k, blk_size);
      break;
    case OPT_GLOBALMEM:
      jacu_buts_init_gmem(iter, item_default, blk_size_k, blk_size);
      break;
    case OPT_SYNC:
      jacu_buts_init_sync(iter, item_default, blk_size_k, blk_size);
      break;
    case OPT_FULL:
      jacu_buts_init_fullopt(iter, item_default, blk_size_k, blk_size);
      break;
    default:
      jacu_buts_init_baseline(iter, item_default, blk_size_k, blk_size);
      break;
  }
}

void jacu_buts_release(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      jacu_buts_release_baseline(iter);
      break;
    case OPT_GLOBALMEM:
      jacu_buts_release_gmem(iter);
      break;
    case OPT_SYNC:
      jacu_buts_release_sync(iter);
      break;
    case OPT_FULL:
      jacu_buts_release_fullopt(iter);
      break;
    default:
      jacu_buts_release_baseline(iter);
      break;
  }

  free(loop2_ev_copy_u_prev);
  free(loop2_ev_copy_r_prev);
  free(loop2_ev_copy_u);
  free(loop2_ev_copy_rsd);
}

void jacu_buts_release_ev(int iter)
{
  int i;
  if (split_flag) {
    for (i = 0; i < iter; i++) {
      if (i > 0) {
        clReleaseEvent(loop2_ev_copy_u_prev[i]);
        clReleaseEvent(loop2_ev_copy_r_prev[i]);
      }

      if (i < iter-1) {
        clReleaseEvent(loop2_ev_copy_u[i]);
        clReleaseEvent(loop2_ev_copy_rsd[i]);
      }
    }
  }

  switch (g_opt_level) {
    case OPT_BASELINE:
      jacu_buts_release_ev_baseline(iter);
      break;
    case OPT_GLOBALMEM:
      jacu_buts_release_ev_gmem(iter);
      break;
    case OPT_SYNC:
      jacu_buts_release_ev_sync(iter);
      break;
    case OPT_FULL:
      jacu_buts_release_ev_fullopt(iter);
      break;
    default:
      jacu_buts_release_ev_baseline(iter);
      break;
  }
}

void jacu_buts_body(int work_step, 
                    int work_max_iter, 
                    int work_num_item, 
                    int next_work_num_item, 
                    int temp_kst, 
                    int temp_kend,
                    cl_event * ev_wb_ptr)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      jacu_buts_body_baseline(work_step, work_max_iter,
                              work_num_item, next_work_num_item,
                              temp_kst, temp_kend,
                              ev_wb_ptr);
      break;
    case OPT_GLOBALMEM:
      jacu_buts_body_gmem(work_step, work_max_iter,
                          work_num_item, next_work_num_item,
                          temp_kst, temp_kend,
                          ev_wb_ptr);
      break;
    case OPT_SYNC:
      jacu_buts_body_sync(work_step, work_max_iter,
                          work_num_item, next_work_num_item,
                          temp_kst, temp_kend,
                          ev_wb_ptr);
      break;
    case OPT_FULL:
      jacu_buts_body_fullopt(work_step, work_max_iter,
                             work_num_item, next_work_num_item,
                             temp_kst, temp_kend,
                             ev_wb_ptr);
      break;
    default:
      jacu_buts_body_baseline(work_step, work_max_iter,
                              work_num_item, next_work_num_item,
                              temp_kst, temp_kend,
                              ev_wb_ptr);
      break;
  }
}
