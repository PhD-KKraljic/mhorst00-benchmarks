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
#ifdef _OPENMP
#include <omp.h>
#endif
#include "applu.incl"
#include "timers.h"
#include <math.h>

cl_event  *loop1_ev_wb_rsd,
          *loop1_ev_wb_u,
          *loop1_ev_rb_rsd,
          loop1_ev_pre_wb_rsd,
          loop1_ev_pre_wb_u,
          *loop2_ev_wb_rsd,
          *loop2_ev_wb_u,
          *loop2_ev_wb_frct,
          *loop2_ev_rb_rsd,
          *loop2_ev_rb_u;

cl_event  **loop1_ev_kernel_end_ptr,
          **loop2_ev_kernel_end_ptr;


/* functions declaration for data transfer variables 
   & other variables for kernel execution */
int get_rhs_item(int item_default, int work_end);
int get_rhs_base(int rhs_item, int work_end);
int get_jbu_item(int rhs_base, int rhs_item, int kst, int kend);
int get_jbu_base(int rhs_base, int kst);
int get_temp_kst(int work_base, int kst);
int get_temp_kend(int jbu_item, int temp_kst, int kend);
int get_ssor2_base(int jb_base);
int get_ssor2_item(int jb_num_item, int ssor2_base);
int get_loop2_copy_item(int rhs_item, int rhs_base, int kst, int kend);
int get_loop2_copy_bbase(int rhs_base, int kst);
int get_loop2_copy_hbase(int rhs_base);

/* ssor functions for OpenCL */
void ssor1_init(int iter);
void ssor1_release();
void ssor1_release_ev();
void ssor2_init(int iter);
void ssor2_release();
void ssor2_release_ev();

/* ssor baseline functions */
void ssor1_init_baseline(int iter);
void ssor1_release_baseline();
void ssor1_release_ev_baseline(int iter);
void ssor1_baseline(int item, int base, int step, 
                    int buf_idx, cl_event *ev_wb_ptr);

void ssor2_init_baseline(int iter);
void ssor2_release_baseline();
void ssor2_release_ev_baseline(int iter);
void ssor2_baseline(int item, int base, int step, 
                    int buf_idx, int temp_kst, double tmp2);

void ssor_init(int loop1_iter, int loop2_iter)
{
  ssor1_init(loop1_iter);

  ssor2_init(loop2_iter);

  ssor_alloc_ev1(loop1_iter);

  ssor_alloc_ev2(loop2_iter);
}

void ssor_release()
{
  ssor1_release();

  ssor2_release();

  free(loop1_ev_kernel_end_ptr);
  free(loop2_ev_kernel_end_ptr);

  free(loop1_ev_wb_rsd);
  free(loop1_ev_wb_u);
  free(loop1_ev_rb_rsd);

  free(loop2_ev_wb_rsd);
  free(loop2_ev_wb_u);
  free(loop2_ev_wb_frct);
  free(loop2_ev_rb_rsd);
  free(loop2_ev_rb_u);
}

void ssor_alloc_ev1(int iter)
{
  loop1_ev_kernel_end_ptr = (cl_event**)malloc(sizeof(cl_event*)*iter);

  loop1_ev_wb_rsd = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop1_ev_wb_u = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop1_ev_rb_rsd = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void ssor_release_ev1(int iter)
{
  int i;
  ssor1_release_ev(iter);

  if (split_flag) {
    for (i = 0; i < iter; i++) {
      clReleaseEvent(loop1_ev_wb_rsd[i]);
      clReleaseEvent(loop1_ev_wb_u[i]);
      clReleaseEvent(loop1_ev_rb_rsd[i]);
    }

    if (!buffering_flag) {
      clReleaseEvent(loop1_ev_pre_wb_rsd);
      clReleaseEvent(loop1_ev_pre_wb_u);
    }
  }
}

void ssor_alloc_ev2(int iter)
{
  loop2_ev_kernel_end_ptr = (cl_event**)malloc(sizeof(cl_event*)*iter);

  loop2_ev_wb_rsd = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_wb_u = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_wb_frct = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_rb_rsd = (cl_event*)malloc(sizeof(cl_event)*iter);
  loop2_ev_rb_u = (cl_event*)malloc(sizeof(cl_event)*iter);
}

void ssor_release_ev2(int iter)
{
  int i;
  ssor2_release_ev(iter);

  if (split_flag) {
    for (i = 0; i < iter; i++) {
      clReleaseEvent(loop2_ev_wb_rsd[i]);
      clReleaseEvent(loop2_ev_wb_u[i]);
      clReleaseEvent(loop2_ev_wb_frct[i]);
      clReleaseEvent(loop2_ev_rb_rsd[i]);
      clReleaseEvent(loop2_ev_rb_u[i]);
    } 
  }
}

void ssor1_init(int iter)
{
  ssor1_init_baseline(iter);
}

void ssor1_release()
{
  ssor1_release_baseline();
}

void ssor1_release_ev(int iter)
{
  ssor1_release_ev_baseline(iter);
}

void ssor1(int item, int base, int step, int buf_idx, cl_event *ev_wb_ptr)
{
  ssor1_baseline(item, base, step, buf_idx, ev_wb_ptr);
}


void ssor2_init(int iter)
{
  ssor2_init_baseline(iter);
}

void ssor2_release()
{
  ssor2_release_baseline();
}

void ssor2_release_ev(int iter)
{
  ssor2_release_ev_baseline(iter);
}

void ssor2(int item, int base, int step, int buf_idx, int temp_kst, double tmp2)
{
  ssor2_baseline(item, base, step, buf_idx, temp_kst, tmp2);
}

//---------------------------------------------------------------------
// to perform pseudo-time stepping SSOR iterations
// for five nonlinear pde's.
//---------------------------------------------------------------------
void ssor(int niter)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, m, n;
  int istep;
  double tmp, tmp2;
  // modified to use exact array index for gpu data copy
  //double tv[ISIZ2][ISIZ1][5];
  //double tv[ISIZ2][ISIZ1/2*2+1][5];

  double delunm[5];

  // have to be int type
  int work_step, work_base, 
      work_num_item, work_end,
      next_work_base, next_work_num_item, 
      next_work_end,
      rhs_item, rhs_base,
      next_rhs_base = 0, next_rhs_item = 0,
      jbu_item, jbu_base,
      next_jbu_item,
      temp_kst, temp_kend, 
      next_temp_kst, next_temp_kend = 0,
      ssor2_item, ssor2_base,
      loop2_copy_item, loop2_copy_bbase, 
      loop2_copy_hbase,
      next_loop2_copy_item = 0, 
      next_loop2_copy_bbase = 0, 
      next_loop2_copy_hbase = 0;

  int kst = 1, kend = nz-1;

  int buf_idx, next_buf_idx;

  cl_int  ecode;

  //---------------------------------------------------------------------
  // begin pseudo-time stepping iterations
  //---------------------------------------------------------------------
  tmp = 1.0 / ( omega * ( 2.0 - omega ) );

  //---------------------------------------------------------------------
  // initialize a,b,c,d to zero (guarantees that page tables have been
  // formed, if applicable on given architecture, before timestepping).
  //---------------------------------------------------------------------
  for (j = jst; j < jend; j++) {
    for (i = ist; i < iend; i++) {
      for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
          a[j][i][n][m] = 0.0;
          b[j][i][n][m] = 0.0;
          c[j][i][n][m] = 0.0;
          d[j][i][n][m] = 0.0;
        }
      }
    }
  }

  for (j = jend - 1; j >= jst; j--) {
    for (i = iend - 1; i >= ist; i--) {
      for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
          au[j][i][n][m] = 0.0;
          bu[j][i][n][m] = 0.0;
          cu[j][i][n][m] = 0.0;
          du[j][i][n][m] = 0.0;
        }
      }
    }
  }

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  clu_ProfilerClear();

  //---------------------------------------------------------------------
  // compute the steady-state residuals
  //---------------------------------------------------------------------



  // !SPLIT FLAG
  if (!split_flag) {
    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_frct[0], 
                                 CL_TRUE,
                                 0, 
                                 frct_buf_size,
                                 frct, 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer");

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_u[0], 
                                 CL_TRUE,
                                 0, 
                                 u_buf_size,
                                 u, 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer");
  }

  rhs();
  clFinish(cmd_q[KERNEL_Q]);

  if (!split_flag) {
    ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                m_rsd[0], 
                                CL_TRUE,
                                0, 
                                rsd_buf_size,
                                rsd, 
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer");

    ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                m_rho_i[0], 
                                CL_TRUE,
                                0, 
                                rho_i_buf_size,
                                rho_i, 
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer");

    ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                m_qs[0], 
                                CL_TRUE,
                                0, 
                                qs_buf_size,
                                qs, 
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer");
  }


  //---------------------------------------------------------------------
  // compute the L2 norms of newton iteration residuals
  //---------------------------------------------------------------------
  l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
      ist, iend, jst, jend, rsd, rsdnm, &(m_rsd[0]));

  clFinish(cmd_q[KERNEL_Q]);


  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  clu_ProfilerClear();


  if (!split_flag) {
    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_rsd[0], 
                                 CL_TRUE,
                                 0, 
                                 rsd_buf_size,
                                 rsd, 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer");

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_u[0], 
                                 CL_TRUE,
                                 0, 
                                 u_buf_size,
                                 u, 
                                 0, NULL, NULL); 
    clu_CheckError(ecode, "clEnqueueWriteBuffer"); 

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_qs[0], 
                                 CL_TRUE,
                                 0, 
                                 qs_buf_size,
                                 qs, 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer");

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_rho_i[0], 
                                 CL_TRUE,
                                 0, 
                                 rho_i_buf_size,
                                 rho_i, 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer");

    ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                 m_frct[0], 
                                 CL_TRUE,
                                 0, 
                                 frct_buf_size,
                                 frct, 
                                 0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer");
  }


  timer_start(1);
  clu_ProfilerStart();



  //---------------------------------------------------------------------
  // the timestep loop
  //---------------------------------------------------------------------
  for (istep = 1; istep <= niter; istep++) {
    if ((istep % 20) == 0 || istep == itmax || istep == 1) {
      if (niter > 1) printf(" Time step %4d\n", istep);
    }

    //---------------------------------------------------------------------
    // perform SSOR iteration
    //---------------------------------------------------------------------
    if (timeron) timer_start(t_rhs);


    // ###########################
    //    Loop 1 Start 
    // ###########################  



    // #################
    //  Write First Buffer
    // #################
    if (split_flag) {
      if (!buffering_flag) {
        // Write First slice
        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[0], 
                                     CL_TRUE,
                                     0, 
                                     rsd_slice_size,
                                     &(rsd[kst-1][0][0][0]), 
                                     0, NULL, &loop1_ev_pre_wb_rsd);
        clu_CheckError(ecode, "clEnqueueWriteBuffer()");  

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_u[0], 
                                     CL_TRUE,
                                     0, 
                                     u_slice_size,
                                     &(u[kst-1][0][0][0]), 
                                     0, NULL, &loop1_ev_pre_wb_u);
        clu_CheckError(ecode, "clEnqueueWriteBuffer()");  
      }
      else {
        work_base = kst;
        work_num_item = min(loop1_work_num_item_default, kend - work_base);

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[0], 
                                     CL_TRUE,
                                     0, 
                                     rsd_slice_size*(work_num_item+1),
                                     &(rsd[kst-1][0][0][0]), 
                                     0, NULL, &loop1_ev_wb_rsd[0]);
        clu_CheckError(ecode, "clEnqueueWriteBuffer()"); 

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_u[0], 
                                     CL_TRUE,
                                     0, 
                                     u_slice_size*(work_num_item+1),
                                     &(u[kst-1][0][0][0]), 
                                     0, NULL, &loop1_ev_wb_u[0]);
        clu_CheckError(ecode, "clEnqueueWriteBuffer()");
      }
    }


    for (work_step = 0; work_step < loop1_work_max_iter; work_step++) {

      work_base = work_step*loop1_work_num_item_default+1;
      work_num_item = min(loop1_work_num_item_default, nz-1 - work_base); 
      buf_idx = (work_step%2)*buffering_flag;
      next_buf_idx = ((work_step+1)%2)*buffering_flag;


      // #################
      //  Write Buffer
      // #################
      if (split_flag && !buffering_flag) {
        // First slice(first k) is already written
        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[0], 
                                     CL_TRUE,
                                     rsd_slice_size,
                                     rsd_slice_size*work_num_item,
                                     &(rsd[work_base][0][0][0]), 
                                     0, NULL, &loop1_ev_wb_rsd[work_step]);
        clu_CheckError(ecode, "clEnqueueWriteBuffer()");

        // First slice(first k) is already written
        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_u[0], 
                                     CL_TRUE,
                                     u_slice_size,
                                     u_slice_size*work_num_item,
                                     &(u[work_base][0][0][0]), 
                                     0, NULL, &loop1_ev_wb_u[work_step]);
        clu_CheckError(ecode, "clEnqueueWriteBuffer()");
      }


      // #################
      //  Kernel Execution
      // #################


      ssor1(work_num_item, work_base, work_step, buf_idx, &loop1_ev_wb_u[work_step]);

      //---------------------------------------------------------------------
      // form the lower triangular part of the jacobian matrix
      //---------------------------------------------------------------------
      if (timeron) timer_start(t_blts);


      loop1_ev_kernel_end_ptr[work_step] 
        = jacld_blts_body(work_step, 
                          loop1_work_max_iter, 
                          work_base, 
                          work_num_item);

      if (timeron) timer_stop(t_blts);

      if (split_flag && buffering_flag && work_step < loop1_work_max_iter - 1) {
        // #################
        //  Write Next Buffer
        // #################


        next_work_base = (work_step+1)*loop1_work_num_item_default + 1;

        next_work_num_item = min(loop1_work_num_item_default, nz-1 - next_work_base); 
        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[next_buf_idx], 
                                     CL_FALSE,
                                     rsd_slice_size,
                                     rsd_slice_size*next_work_num_item,
                                     &(rsd[next_work_base][0][0][0]), 
                                     0, NULL, 
                                     &loop1_ev_wb_rsd[work_step+1]);
        clu_CheckError(ecode, "ssor1 loop write buffer m_rsd");

        clFlush (cmd_q[DATA_Q]);

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_u[next_buf_idx], 
                                     CL_TRUE,
                                     u_slice_size,
                                     u_slice_size*next_work_num_item,
                                     &(u[next_work_base][0][0][0]), 
                                     0, NULL, 
                                     &loop1_ev_wb_u[work_step+1]);
        clu_CheckError(ecode, "ssor1 loop write buffer m_u");


      }



      // #################
      //  Read Buffer
      // #################
      if (split_flag) {

        cl_bool blocking = (buffering_flag) ? CL_FALSE : CL_TRUE;
        cl_event *wait_ev = (buffering_flag) ? loop1_ev_kernel_end_ptr[work_step] : NULL;
        int num_wait = (buffering_flag) ? 1 : 0;

        ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                    m_rsd[buf_idx], 
                                    blocking,
                                    rsd_slice_size,
                                    rsd_slice_size*work_num_item,
                                    &(rsd[work_base][0][0][0]), 
                                    num_wait, wait_ev, 
                                    &loop1_ev_rb_rsd[work_step]);
        clu_CheckError(ecode, "clEnqueueReadBuffer()");


        if (buffering_flag)
          clFlush(cmd_q[DATA_Q]);


      }
    }

    if (buffering_flag) {
      clFinish(cmd_q[KERNEL_Q]);
      clFinish(cmd_q[DATA_Q]);
    }

    if (timeron) timer_stop(t_rhs);

    /* release event objects */
    ssor_release_ev1(loop1_work_max_iter);
    jacld_blts_release_ev(loop1_work_max_iter);

    // ###########################
    //    Loop 1 Finished
    // ###########################



    // ###########################
    //    Loop 2  Start
    // ###########################

    if ( (istep % inorm) == 0 ) {
      l2norm_head(delunm, g_sum1, &m_sum1, &ev_kernel_l2norm_head1);
    }

    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      l2norm_head(rsdnm, g_sum2, &m_sum2, &ev_kernel_l2norm_head2);
    }


    tmp2 = tmp;

    // SET loop parameters 
    loop2_work_num_item_default = (split_flag) ? (work_num_item_default - 4) : work_num_item_default;
    loop2_work_max_iter = (nz-1)/loop2_work_num_item_default + 1;

    //###################
    // Write First Buffer
    //###################

    if (split_flag && buffering_flag) {

      // RHS Loop Fusion
      work_end = nz;
      rhs_item = min(loop2_work_num_item_default, work_end);
      rhs_base = work_end - rhs_item;

      loop2_copy_item = rhs_item;
      loop2_copy_bbase = 2;
      loop2_copy_hbase = rhs_base;

      // front alignment calculation
      loop2_copy_item += min(2, rhs_base);
      loop2_copy_bbase -= min(2, rhs_base);
      loop2_copy_hbase -= min(2, rhs_base);

      ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                   m_rsd[0], 
                                   CL_TRUE,
                                   rsd_slice_size*loop2_copy_bbase,
                                   rsd_slice_size*loop2_copy_item,
                                   &rsd[loop2_copy_hbase][0][0][0], 
                                   0, NULL, 
                                   &loop2_ev_wb_rsd[0]);
      clu_CheckError(ecode, "ssor2 loop2 write buffer - m_rsd");

      ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                   m_u[0], 
                                   CL_TRUE,
                                   u_slice_size*loop2_copy_bbase,
                                   u_slice_size*loop2_copy_item,
                                   &u[loop2_copy_hbase][0][0][0], 
                                   0, NULL, 
                                   &loop2_ev_wb_u[0]);
      clu_CheckError(ecode, "ssor2 loop2 write buffer - m_frct");

      ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                   m_frct[0], 
                                   CL_TRUE,
                                   frct_slice_size*2,
                                   frct_slice_size*rhs_item,
                                   &frct[rhs_base][0][0][0], 
                                   0, NULL, 
                                   &loop2_ev_wb_frct[0]);
      clu_CheckError(ecode, "ssor2 loop2 write buffer - m_frct");
    }




    for (work_step = 0; work_step < loop2_work_max_iter; work_step++) {

      work_end = nz - work_step*loop2_work_num_item_default;
      buf_idx = (work_step%2)*buffering_flag;
      next_buf_idx = ((work_step+1)%2)*buffering_flag;

      rhs_item = get_rhs_item(loop2_work_num_item_default, work_end);
      rhs_base = get_rhs_base(rhs_item, work_end);

      jbu_item = get_jbu_item(rhs_base, rhs_item, kst, kend);
      jbu_base = get_jbu_base(rhs_base, kst);

      temp_kst = get_temp_kst(rhs_base, kst);
      temp_kend = get_temp_kend(jbu_item, temp_kst, kend);

      ssor2_base = get_ssor2_base(jbu_base);
      ssor2_item = get_ssor2_item(jbu_item, ssor2_base);

      loop2_copy_item = get_loop2_copy_item(rhs_item, rhs_base, kst, kend);
      loop2_copy_bbase = get_loop2_copy_bbase(rhs_base, kst);
      loop2_copy_hbase = get_loop2_copy_hbase(rhs_base);


      if (split_flag) {
        if (!buffering_flag) {
          //###################
          // Write Buffer
          //###################
          ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                       m_u[0], 
                                       CL_TRUE,
                                       u_slice_size*loop2_copy_bbase,
                                       u_slice_size*loop2_copy_item,
                                       &(u[loop2_copy_hbase][0][0][0]), 
                                       0, NULL, 
                                       &loop2_ev_wb_u[work_step]);
          clu_CheckError(ecode, "ssor2 loop write buffer - m_u");

          ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                       m_rsd[0], 
                                       CL_TRUE,
                                       rsd_slice_size*loop2_copy_bbase,
                                       rsd_slice_size*loop2_copy_item,
                                       &(rsd[loop2_copy_hbase][0][0][0]), 
                                       0, NULL, 
                                       &loop2_ev_wb_rsd[work_step]);
          clu_CheckError(ecode, "ssor2 loop write buffer - m_rsd");

          ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                       m_frct[0], 
                                       CL_TRUE,
                                       frct_slice_size*2,
                                       frct_slice_size*rhs_item,
                                       &(frct[rhs_base][0][0][0]), 
                                       0, NULL, 
                                       &loop2_ev_wb_frct[work_step]);
          clu_CheckError(ecode, "ssor2 loop write buffer - m_frct");
        }
        else if (work_step < loop2_work_max_iter - 1) {
          //###################
          // Variables to Write Next Buffer 
          //################### 

          next_work_end = nz - (work_step+1)*loop2_work_num_item_default;

          next_rhs_item = get_rhs_item(loop2_work_num_item_default, next_work_end);
          next_rhs_base = get_rhs_base(next_rhs_item, next_work_end);

          next_jbu_item = get_jbu_item(next_rhs_base, next_rhs_item, kst, kend);

          next_temp_kst = get_temp_kst(next_rhs_base, kst);
          next_temp_kend = get_temp_kend(next_jbu_item, next_temp_kst, kend);

          next_loop2_copy_item = get_loop2_copy_item(next_rhs_item, next_rhs_base, kst, kend);
          next_loop2_copy_bbase = get_loop2_copy_bbase(next_rhs_base, kst);
          next_loop2_copy_hbase = get_loop2_copy_hbase(next_rhs_base);
        }

        // end alignment calculation -- this is for rhs data gen
        loop2_copy_item += min(2, nz-(rhs_base + rhs_item));
      }




      //###################
      // Kernel Execution
      //###################

      //---------------------------------------------------------------------
      // form the strictly upper triangular part of the jacobian matrix
      //---------------------------------------------------------------------
      if (timeron) timer_start(t_buts);

      jacu_buts_body(work_step, 
                     loop2_work_max_iter, 
                     jbu_item, 
                     next_temp_kend,
                     temp_kst, temp_kend, 
                     &loop2_ev_wb_frct[work_step]); 

      if (timeron) timer_stop(t_buts);


      if (timeron) timer_start(t_add);

      ssor2(ssor2_item, ssor2_base, work_step, 
            buf_idx, temp_kst, tmp2);


      if (timeron) timer_stop(t_add);


      if ( (istep % inorm) == 0 ) {

        if (timeron) timer_start(t_l2norm);
        //---------------------------------------------------------------------
        // compute the max-norms of newton iteration corrections
        //---------------------------------------------------------------------

        l2norm_body(work_step, loop2_work_max_iter, 
                    rhs_base, rhs_item,
                    &loop2_ev_wb_frct[work_step], 
                    &ev_k_l2norm_body1[work_step], 
                    &m_sum1, nz0, 
                    jst, jend, ist, iend);


        if (timeron) timer_stop(t_l2norm);
      }

      //---------------------------------------------------------------------
      // compute the steady-state residuals
      //---------------------------------------------------------------------


      loop2_ev_kernel_end_ptr[work_step] = rhs_body(work_step, 
                                                    rhs_base, rhs_item,
                                                    loop2_copy_bbase, 
                                                    loop2_copy_item, 
                                                    &loop2_ev_wb_frct[work_step]);


      //---------------------------------------------------------------------
      // compute the max-norms of newton iteration residuals
      //---------------------------------------------------------------------
      if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {

        if (timeron) timer_start(t_l2norm);

        l2norm_body(work_step, loop2_work_max_iter, 
                    rhs_base, rhs_item,
                    &loop2_ev_wb_frct[work_step], 
                    &ev_k_l2norm_body2[work_step], 
                    &m_sum2, nz0, 
                    jst, jend, ist, iend);

        if (timeron) timer_stop(t_l2norm);
        /*
           if ( ipr == 1 ) {
           printf(" \n RMS-norm of steady-state residual for "
           "first pde  = %12.5E\n"
           " RMS-norm of steady-state residual for "
           "second pde = %12.5E\n"
           " RMS-norm of steady-state residual for "
           "third pde  = %12.5E\n"
           " RMS-norm of steady-state residual for "
           "fourth pde = %12.5E\n"
           " RMS-norm of steady-state residual for "
           "fifth pde  = %12.5E\n", 
           rsdnm[0], rsdnm[1], rsdnm[2], rsdnm[3], rsdnm[4]);
           }
         */
        loop2_ev_kernel_end_ptr[work_step] = &ev_k_l2norm_body2[work_step];
      }


      if (split_flag && buffering_flag && work_step < loop2_work_max_iter - 1) {

        //###################
        // Write Next Buffer
        //################### 

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_rsd[next_buf_idx], 
                                     CL_FALSE,
                                     rsd_slice_size*next_loop2_copy_bbase,
                                     rsd_slice_size*next_loop2_copy_item,
                                     &(rsd[next_loop2_copy_hbase][0][0][0]), 
                                     0, NULL, 
                                     &loop2_ev_wb_rsd[work_step+1]);
        clu_CheckError(ecode, "ssor2 loop write buffer - m_rsd");

        clFlush (cmd_q[DATA_Q]);

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_u[next_buf_idx], 
                                     CL_FALSE,
                                     u_slice_size*next_loop2_copy_bbase,
                                     u_slice_size*next_loop2_copy_item,
                                     &(u[next_loop2_copy_hbase][0][0][0]), 
                                     0, NULL, 
                                     &loop2_ev_wb_u[work_step+1]);
        clu_CheckError(ecode, "ssor2 loop write buffer - m_u");

        clFlush (cmd_q[DATA_Q]);

        ecode = clEnqueueWriteBuffer(cmd_q[DATA_Q], 
                                     m_frct[next_buf_idx], 
                                     CL_TRUE,
                                     frct_slice_size*2,
                                     frct_slice_size*next_rhs_item,
                                     &(frct[next_rhs_base][0][0][0]), 
                                     0, NULL, 
                                     &loop2_ev_wb_frct[work_step+1]);
        clu_CheckError(ecode, "ssor2 loop write buffer - m_frct");

      }

      //###################
      // Read Buffer
      //###################
      if (split_flag) {

        int num_wait = (buffering_flag) ? 1 : 0;
        cl_event *wait_ev = (buffering_flag) ? loop2_ev_kernel_end_ptr[work_step] : NULL;
        cl_bool blocking = (buffering_flag) ? CL_FALSE : CL_TRUE;

        ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                    m_rsd[buf_idx], 
                                    blocking,
                                    rsd_slice_size*2,
                                    rsd_slice_size*rhs_item,
                                    &(rsd[rhs_base][0][0][0]), 
                                    num_wait, wait_ev,
                                    &loop2_ev_rb_rsd[work_step]);
        clu_CheckError(ecode, "clEnqueueReadBuffer()");

        if (buffering_flag) 
          clFlush(cmd_q[DATA_Q]);

        ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                    m_u[buf_idx], 
                                    blocking,
                                    u_slice_size*2,
                                    u_slice_size*rhs_item,
                                    &(u[rhs_base][0][0][0]), 
                                    0, NULL, 
                                    &loop2_ev_rb_u[work_step]);
        clu_CheckError(ecode, "clEnqueueReadBuffer()");

        if (buffering_flag)
          clFlush(cmd_q[DATA_Q]);
      }

    }

    if (buffering_flag) {
      clFinish(cmd_q[KERNEL_Q]);
      clFinish(cmd_q[DATA_Q]);
    }

    if ( (istep % inorm) == 0 ) {
      l2norm_tail(delunm, g_sum1, &m_sum1, nx0, ny0, nz0, &ev_data_l2norm_tail1);
    }

    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      l2norm_tail(rsdnm, g_sum2, &m_sum2, nx0, ny0, nz0, &ev_data_l2norm_tail2);
    }

    // ###########################
    //    Loop 2 Finished 
    // ###########################

    /* Release event objects */
    ssor_release_ev2(loop2_work_max_iter);
    jacu_buts_release_ev(loop2_work_max_iter);
    l2norm_release_ev(loop2_work_max_iter, istep, inorm, itmax);
    rhs_release_ev(loop2_work_max_iter);

    //---------------------------------------------------------------------
    // check the newton-iteration residuals against the tolerance levels
    //---------------------------------------------------------------------
    if ( ( rsdnm[0] < tolrsd[0] ) && ( rsdnm[1] < tolrsd[1] ) &&
        ( rsdnm[2] < tolrsd[2] ) && ( rsdnm[3] < tolrsd[3] ) &&
        ( rsdnm[4] < tolrsd[4] ) ) {
      //if (ipr == 1 ) {
      printf(" \n convergence was achieved after %4d pseudo-time steps\n",
          istep);
      //}
      break;
    }

  }

  clu_ProfilerStop();
  timer_stop(1);
  maxtime = timer_read(1);


  if (!split_flag) {
    ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                m_rsd[0], 
                                CL_TRUE,
                                0, 
                                rsd_slice_size*ISIZ3,
                                rsd, 
                                0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueReadBuffer");

    ecode = clEnqueueReadBuffer(cmd_q[DATA_Q], 
                                m_u[0], 
                                CL_TRUE,
                                0, 
                                u_slice_size*ISIZ3,
                                u, 
                                0, NULL, NULL);
    clu_CheckError(ecode , "clEnqueueReadBuffer");
  }
}

/* functions for data transfer variables & other variables for kernel execution */
int get_rhs_item(int item_default, int work_end)
{
  return (split_flag) ? min(item_default, work_end) : nz; 
}

int get_rhs_base(int rhs_item, int work_end)
{
  return (split_flag) ? work_end - rhs_item : 0; 
}

int get_jbu_item(int rhs_base, int rhs_item, int kst, int kend)
{
  int ret; 

  if (split_flag) {
    ret  = min(rhs_item, kend - rhs_base);
    ret += min(2, rhs_base - 1);
  }
  else
    ret = kend - kst;

  return ret;
}

int get_jbu_base(int rhs_base, int kst)
{
  int ret;

  if (split_flag) {
    ret  = rhs_base;
    ret -= min(2, rhs_base - 1);
  }
  else
    ret = kst;

  return ret;
}

int get_temp_kst(int work_base, int kst)
{
  int ret;

  if (split_flag) {
    ret = 2;
    ret -= min(2, work_base-1);
  }
  else
    ret = kst;

  return ret;
}

int get_temp_kend(int jbu_item, int temp_kst, int kend)
{
  return (split_flag) ? temp_kst+jbu_item : kend;
}

int get_ssor2_base(int jb_base)
{
  return (split_flag) ? jb_base : 1; 
}

int get_ssor2_item(int jb_num_item, int ssor2_base)
{
  int ret;

  if (split_flag) {
    ret  = jb_num_item;
    ret += min(2, nz-1 - (ssor2_base + ret));
  }
  else
    ret = nz-2;

  return ret;
}

int get_loop2_copy_item(int rhs_item, int rhs_base, int kst, int kend)
{
  int ret;

  if (split_flag) {
    ret  = rhs_item;
    ret += min(2, rhs_base);
  }
  else
    ret = kend - kst;

  return ret;
}

int get_loop2_copy_bbase(int rhs_base, int kst)
{
  int ret;
  
  if (split_flag) {
    ret  = 2;
    ret -= min(2, rhs_base);
  }
  else
    ret = kst;

  return ret;
}

int get_loop2_copy_hbase(int rhs_base)
{
  int ret;

  if (split_flag) {
    ret  = rhs_base;
    ret -= min(2, rhs_base);
  }
  else 
    ret = 0;

  return ret;
}

