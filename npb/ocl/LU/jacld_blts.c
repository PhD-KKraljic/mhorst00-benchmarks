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
#include <stdio.h>
#include <math.h>

//---------------------------------------------------------------------
// compute the lower triangular part of the jacobian matrix
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block lower triangular solution:
// 
// v <-- ( L-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------

/* jacld blts baseline functions */
void jacld_blts_init_baseline(int iter,
                              int item_default,
                              int blk_size_k,
                              int blk_size);
void jacld_blts_release_baseline(int iter);
void jacld_blts_release_ev_baseline(int iter);
cl_event* jacld_blts_body_baseline(int work_step,
                                   int work_max_iter,
                                   int work_base,
                                   int work_num_item);

/* jacld blts gmem access opt functions */
void jacld_blts_init_gmem(int iter, 
                          int item_default, 
                          int blk_size_k, 
                          int blk_size);
void jacld_blts_release_gmem(int iter);
void jacld_blts_release_ev_gmem(int iter);
cl_event* jacld_blts_body_gmem(int work_step,
                               int work_max_iter,
                               int work_base,
                               int work_num_item);

/* jacld blts synchronization opt functions */
void jacld_blts_init_sync(int iter,
                          int item_default,
                          int blk_size_k,
                          int blk_size);
void jacld_blts_release_sync(int iter);
void jacld_blts_release_ev_sync(int iter);
cl_event* jacld_blts_body_sync(int work_step,
                               int work_max_iter,
                               int work_base,
                               int work_num_item);

/* jacld blts fullopt functions */
void jacld_blts_init_fullopt(int iter, 
                             int item_default, 
                             int blk_size_k, 
                             int blk_size);
void jacld_blts_release_fullopt(int iter);
void jacld_blts_release_ev_fullopt(int iter);
cl_event* jacld_blts_body_fullopt(int work_step,
                                  int work_max_iter,
                                  int work_base,
                                  int work_num_item);


void jacld_blts_init(int iter, int item_default, int blk_size_k, int blk_size)
{

  switch (g_opt_level) {
    case OPT_BASELINE:
      jacld_blts_init_baseline(iter, item_default, blk_size_k, blk_size);
      break;
    case OPT_GLOBALMEM:
      jacld_blts_init_gmem(iter, item_default, blk_size_k, blk_size);
      break;
    case OPT_SYNC:
      jacld_blts_init_sync(iter, item_default, blk_size_k, blk_size);
      break;
    case OPT_FULL:
      jacld_blts_init_fullopt(iter, item_default, blk_size_k, blk_size);
      break;
    default:
      jacld_blts_init_baseline(iter, item_default, blk_size_k, blk_size);
      break;
  }
}

void jacld_blts_release(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      jacld_blts_release_baseline(iter);
      break;
    case OPT_GLOBALMEM:
      jacld_blts_release_gmem(iter);
      break;
    case OPT_SYNC:
      jacld_blts_release_sync(iter);
      break;
    case OPT_FULL:
      jacld_blts_release_fullopt(iter);
      break;
    default:
      jacld_blts_release_baseline(iter);
      break;
  }
}


void jacld_blts_release_ev(int iter)
{

  switch (g_opt_level) {
    case OPT_BASELINE:
      jacld_blts_release_ev_baseline(iter);
      break;
    case OPT_GLOBALMEM:
      jacld_blts_release_ev_gmem(iter);
      break;
    case OPT_SYNC:
      jacld_blts_release_ev_sync(iter);
      break;
    case OPT_FULL:
      jacld_blts_release_ev_fullopt(iter);
      break;
    default:
      jacld_blts_release_ev_baseline(iter);
      break;
  }
}

cl_event* jacld_blts_body(int work_step, 
                          int work_max_iter, 
                          int work_base, 
                          int work_num_item)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      return jacld_blts_body_baseline(work_step, work_max_iter,
                                      work_base, work_num_item);
      break;
    case OPT_GLOBALMEM:
      return jacld_blts_body_gmem(work_step, work_max_iter,
                                  work_base, work_num_item);
      break;
    case OPT_SYNC:
      return jacld_blts_body_sync(work_step, work_max_iter,
                                  work_base, work_num_item);
      break;
    case OPT_FULL:
      return jacld_blts_body_fullopt(work_step, work_max_iter,
                                     work_base, work_num_item);
      break;
    default:
      return jacld_blts_body_baseline(work_step, work_max_iter,
                                      work_base, work_num_item);
      break;
  }
}

