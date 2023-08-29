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

//---------------------------------------------------------------------
// Performs line solves in Y direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------

/* y solve baseline functions */
void y_solve_init_baseline(int iter);
void y_solve_release_baseline();
void y_solve_release_ev_baseline(int iter);
cl_event* y_solve_baseline(int work_step, 
                           int work_base, 
                           int work_num_item, 
                           int buf_idx);

/* y solve memory layout opt functions */
void y_solve_init_memlayout(int iter);
void y_solve_release_memlayout();
void y_solve_release_ev_memlayout(int iter);
cl_event* y_solve_memlayout(int work_step, 
                            int work_base, 
                            int work_num_item, 
                            int buf_idx);

/* y solve parallel functions */
void y_solve_init_parallel(int iter);
void y_solve_release_parallel();
void y_solve_release_ev_parallel(int iter);
cl_event* y_solve_parallel(int work_step, 
                           int work_base, 
                           int work_num_item, 
                           int buf_idx);

/* y solve fullopt functions */
void y_solve_init_fullopt(int iter);
void y_solve_release_fullopt();
void y_solve_release_ev_fullopt(int iter);
cl_event* y_solve_fullopt(int work_step, 
                          int work_base, 
                          int work_num_item, 
                          int buf_idx);

void y_solve_init(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      y_solve_init_baseline(iter);
      break;
    case OPT_PARALLEL:
      y_solve_init_parallel(iter);
      break;
    case OPT_MEMLAYOUT:
      y_solve_init_memlayout(iter);
      break;
    case OPT_FULL:
      y_solve_init_fullopt(iter);
      break;
    default :
      y_solve_init_baseline(iter);
      break;
  }
}

void y_solve_release()
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      y_solve_release_baseline();
      break;
    case OPT_PARALLEL:
      y_solve_release_parallel();
      break;
    case OPT_MEMLAYOUT:
      y_solve_release_memlayout();
      break;
    case OPT_FULL:
      y_solve_release_fullopt();
      break;
    default :
      y_solve_release_baseline();
      break; 
  }
}

void y_solve_release_ev(int iter)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      y_solve_release_ev_baseline(iter);
      break;
    case OPT_PARALLEL:
      y_solve_release_ev_parallel(iter);
      break;
    case OPT_MEMLAYOUT:
      y_solve_release_ev_memlayout(iter);
      break;
    case OPT_FULL:
      y_solve_release_ev_fullopt(iter);
      break;
    default :
      y_solve_release_ev_baseline(iter);
      break; 
  }
}

cl_event* y_solve(int work_step, 
                  int work_base, 
                  int work_num_item, 
                  int buf_idx)
{
  switch (g_opt_level) {
    case OPT_BASELINE:
      return y_solve_baseline(work_step, 
                              work_base, 
                              work_num_item, 
                              buf_idx);
      break;
    case OPT_PARALLEL:
      return y_solve_parallel(work_step, 
                              work_base, 
                              work_num_item, 
                              buf_idx);
      break;
    case OPT_MEMLAYOUT:
      return y_solve_memlayout(work_step, 
                               work_base, 
                               work_num_item, 
                               buf_idx);
      break;
    case OPT_FULL:
      return y_solve_fullopt(work_step, 
                             work_base, 
                             work_num_item, 
                             buf_idx);
      break;
    default :
      return y_solve_baseline(work_step, 
                              work_base, 
                              work_num_item, 
                              buf_idx);
      break;
  }
}

