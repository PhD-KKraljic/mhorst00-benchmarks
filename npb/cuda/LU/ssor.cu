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

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "applu.incl"
extern "C" {
#include "timers.h"
#include "npbparams.h"
}

cudaEvent_t   (*loop1_ev_wb_rsd)[2],
              (*loop1_ev_wb_u)[2],
              (*loop1_ev_rb_rsd)[2],
              loop1_ev_pre_wb_rsd[2],
              loop1_ev_pre_wb_u[2],
              (*loop2_ev_wb_rsd)[2],
              (*loop2_ev_wb_u)[2],
              (*loop2_ev_wb_frct)[2],
              (*loop2_ev_rb_rsd)[2],
              (*loop2_ev_rb_u)[2];

cudaEvent_t   **loop1_ev_kernel_end_ptr,
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

/* ssor functions for CUDA */
void ssor1_init(int iter);
void ssor1_release(int iter);
void ssor2_init(int iter);
void ssor2_release(int iter);

/* ssor baseline = fullopt functions */
void ssor1_init_baseline(int iter);
void ssor1_release_baseline(int iter);
void ssor1_baseline(int item, int base, 
                    int step, int buf_idx, 
                    cudaEvent_t *ev_wb_ptr);
void ssor2_init_baseline(int iter);
void ssor2_release_baseline(int iter);
void ssor2_baseline(int item, int base,
                    int step, int buf_idx,
                    int temp_kst, double tmp2);

void ssor_init(int loop1_iter, int loop2_iter)
{
  ssor1_init(loop1_iter);

  ssor2_init(loop2_iter);

  ssor_alloc_ev1(loop1_iter);

  ssor_alloc_ev2(loop2_iter);
}

void ssor_release(int loop1_iter, int loop2_iter)
{
  int i;

  ssor1_release(loop1_iter);

  ssor2_release(loop2_iter);

  free(loop1_ev_kernel_end_ptr);
  free(loop2_ev_kernel_end_ptr);

  cudaEventDestroy(loop1_ev_pre_wb_rsd[0]);
  cudaEventDestroy(loop1_ev_pre_wb_u[0]);
  cudaEventDestroy(loop1_ev_pre_wb_rsd[1]);
  cudaEventDestroy(loop1_ev_pre_wb_u[1]);

  for (i = 0; i < loop1_iter; i++) {
    cudaEventDestroy(loop1_ev_wb_rsd[i][0]);
    cudaEventDestroy(loop1_ev_wb_u[i][0]);
    cudaEventDestroy(loop1_ev_rb_rsd[i][0]);

    cudaEventDestroy(loop1_ev_wb_rsd[i][1]);
    cudaEventDestroy(loop1_ev_wb_u[i][1]);
    cudaEventDestroy(loop1_ev_rb_rsd[i][1]);
  }

  free(loop1_ev_wb_rsd);
  free(loop1_ev_wb_u);
  free(loop1_ev_rb_rsd);


  for (i = 0; i < loop2_iter; i++) {
    cudaEventDestroy(loop2_ev_wb_rsd[i][0]);
    cudaEventDestroy(loop2_ev_wb_u[i][0]);
    cudaEventDestroy(loop2_ev_wb_frct[i][0]);
    cudaEventDestroy(loop2_ev_rb_rsd[i][0]);
    cudaEventDestroy(loop2_ev_rb_u[i][0]);

    cudaEventDestroy(loop2_ev_wb_rsd[i][1]);
    cudaEventDestroy(loop2_ev_wb_u[i][1]);
    cudaEventDestroy(loop2_ev_wb_frct[i][1]);
    cudaEventDestroy(loop2_ev_rb_rsd[i][1]);
    cudaEventDestroy(loop2_ev_rb_u[i][1]);
  }

  free(loop2_ev_wb_rsd);
  free(loop2_ev_wb_u);
  free(loop2_ev_wb_frct);
  free(loop2_ev_rb_rsd);
  free(loop2_ev_rb_u);
}

void ssor_alloc_ev1(int iter)
{
  int i;

  loop1_ev_kernel_end_ptr = (cudaEvent_t**)malloc(sizeof(cudaEvent_t*)*iter);

  loop1_ev_wb_rsd = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);
  loop1_ev_wb_u = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);
  loop1_ev_rb_rsd = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);

  cudaEventCreate(&loop1_ev_pre_wb_rsd[0]);
  cudaEventCreate(&loop1_ev_pre_wb_u[0]);
  cudaEventCreate(&loop1_ev_pre_wb_rsd[1]);
  cudaEventCreate(&loop1_ev_pre_wb_u[1]);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&loop1_ev_wb_rsd[i][0]);
    cudaEventCreate(&loop1_ev_wb_u[i][0]);
    cudaEventCreate(&loop1_ev_rb_rsd[i][0]);

    cudaEventCreate(&loop1_ev_wb_rsd[i][1]);
    cudaEventCreate(&loop1_ev_wb_u[i][1]);
    cudaEventCreate(&loop1_ev_rb_rsd[i][1]);
  }

}

void ssor_alloc_ev2(int iter)
{
  int i;

  loop2_ev_kernel_end_ptr = (cudaEvent_t**)malloc(sizeof(cudaEvent_t*)*iter);

  loop2_ev_wb_rsd = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);
  loop2_ev_wb_u = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);
  loop2_ev_wb_frct = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);
  loop2_ev_rb_rsd = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);
  loop2_ev_rb_u = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*iter*2);

  for (i = 0; i < iter; i++) {
    cudaEventCreate(&loop2_ev_wb_rsd[i][0]);
    cudaEventCreate(&loop2_ev_wb_u[i][0]);
    cudaEventCreate(&loop2_ev_wb_frct[i][0]);
    cudaEventCreate(&loop2_ev_rb_rsd[i][0]);
    cudaEventCreate(&loop2_ev_rb_u[i][0]);

    cudaEventCreate(&loop2_ev_wb_rsd[i][1]);
    cudaEventCreate(&loop2_ev_wb_u[i][1]);
    cudaEventCreate(&loop2_ev_wb_frct[i][1]);
    cudaEventCreate(&loop2_ev_rb_rsd[i][1]);
    cudaEventCreate(&loop2_ev_rb_u[i][1]);
  }

}

void ssor1_init(int iter)
{
  ssor1_init_baseline(iter);
}

void ssor1_release(int iter)
{
  ssor1_release_baseline(iter);
}

void ssor1(int item, int base,
           int step, int buf_idx,
           cudaEvent_t *ev_wb_end_ptr)
{
  ssor1_baseline(item, base,
                 step, buf_idx,
                 ev_wb_end_ptr);
}

void ssor2_init(int iter)
{
  ssor2_init_baseline(iter);
}

void ssor2_release(int iter)
{
  ssor2_release_baseline(iter);
}

void ssor2(int item, int base,
           int step, int buf_idx,
           int temp_kst, double tmp2)
{
  ssor2_baseline(item, base,
                 step, buf_idx,
                 temp_kst, tmp2);
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

  double delunm[5];


  int work_step, work_base, work_num_item, work_end;
  int next_work_base, next_work_num_item, next_work_end;

  int rhs_item, rhs_base;
  int next_rhs_base = 0, next_rhs_item = 0;

  int jbu_item, jbu_base;
  int next_jbu_item;
  int temp_kst, temp_kend, next_temp_kst, next_temp_kend = 0;

  int ssor2_item, ssor2_base;

  int loop2_copy_item, loop2_copy_bbase, loop2_copy_hbase;
  int next_loop2_copy_item = 0, next_loop2_copy_bbase = 0, next_loop2_copy_hbase = 0;

  int kst = 1, kend = nz-1;


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
  cuda_ProfilerClear();

  //---------------------------------------------------------------------
  // compute the steady-state residuals
  //---------------------------------------------------------------------



  // !SPLIT FLAG
  if (!split_flag) {
    CUCHK(cudaMemcpyAsync(m_frct[0], frct, 
          sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_u[0], u, 
          sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  }

  rhs();


  // !SPLIT FLAG
  if (!split_flag) {
    CUCHK(cudaMemcpyAsync(rsd, m_rsd[0], sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyDeviceToHost, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(rho_i, m_rho_i[0], sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1),
          cudaMemcpyDeviceToHost, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(qs, m_qs[0], sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1),
          cudaMemcpyDeviceToHost, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }


  //---------------------------------------------------------------------
  // compute the L2 norms of newton iteration residuals
  //---------------------------------------------------------------------

  l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
      ist, iend, jst, jend, rsdnm, m_rsd[0]);

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  cuda_ProfilerClear();

  if (!split_flag) {
    CUCHK(cudaMemcpyAsync(m_rsd[0], rsd,
          sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_u[0], u, sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5, 
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_qs[0], qs, sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1), 
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_rho_i[0], rho_i, sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1), 
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_frct[0], frct, sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }


  timer_start(1);
  cuda_ProfilerStart();


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
        CUCHK(cudaEventRecord(loop1_ev_pre_wb_rsd[0], cmd_q[KERNEL_Q]));

        CUCHK(cudaMemcpyAsync(m_rsd[0], 
                              &(rsd[kst-1][0][0][0]),
                              sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_pre_wb_rsd[1], cmd_q[KERNEL_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_pre_wb_u[0], cmd_q[KERNEL_Q]));

        CUCHK(cudaMemcpyAsync(m_u[0], 
                              &(u[kst-1][0][0][0]),
                              sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_pre_wb_u[1], cmd_q[KERNEL_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
      }
      else {

        work_base = kst;
        work_num_item = min((int)loop1_work_num_item_default, kend - work_base);

        CUCHK(cudaEventRecord(loop1_ev_wb_rsd[0][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(m_rsd[0], 
                              &(rsd[kst-1][0][0][0]),
                              sizeof(double)*(work_num_item+1)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_rsd[0][1], cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_u[0][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(m_u[0], 
                              &(u[kst-1][0][0][0]),
                              sizeof(double)*(work_num_item+1)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_u[0][1], cmd_q[DATA_Q]));
      }
    }

    for (work_step = 0; work_step < loop1_work_max_iter; work_step++) {

      work_base = work_step*loop1_work_num_item_default+1;
      work_num_item = min((int)loop1_work_num_item_default, nz-1 - work_base); 

      // #################
      //  Write Buffer
      // #################

      if (split_flag && !buffering_flag) {

        CUCHK(cudaEventRecord(loop1_ev_wb_rsd[work_step][0], cmd_q[KERNEL_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_rsd[0])+sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5, 
                              &(rsd[work_base][0][0][0]),
                              sizeof(double)*(work_num_item)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_rsd[work_step][1], cmd_q[KERNEL_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_u[work_step][0], cmd_q[KERNEL_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_u[0])+sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5, 
                              &(u[work_base][0][0][0]),
                              sizeof(double)*(work_num_item)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_u[work_step][1], cmd_q[KERNEL_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
      }



      // #################
      //  Kernel Execution
      // #################
      ssor1(work_num_item, work_base,
            work_step, (work_step%2)*buffering_flag,
            &loop1_ev_wb_u[work_step][1]);

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

        next_work_num_item = min((int)loop1_work_num_item_default, nz-1 - next_work_base); 

        // Double buffering does not need to wait on write buffer ( read(n) -> write(n+2) is guarunteed )
        // but not in Triple buffering

        CUCHK(cudaEventRecord(loop1_ev_wb_rsd[work_step+1][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_rsd[(work_step+1)%2])+sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              &(rsd[next_work_base][0][0][0]),
                              sizeof(double)*next_work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_rsd[work_step+1][1], cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_u[work_step+1][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_u[(work_step+1)%2])+sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              &(u[next_work_base][0][0][0]),
                              sizeof(double)*next_work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_wb_u[work_step+1][1], cmd_q[DATA_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
      }


      // #################
      //  Read Buffer
      // #################

      if (split_flag && !buffering_flag) {

        CUCHK(cudaEventRecord(loop1_ev_rb_rsd[work_step][0], cmd_q[KERNEL_Q]));

        CUCHK(cudaMemcpyAsync(&(rsd[work_base][0][0][0]), 
                              ((char*)m_rsd[0])+sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyDeviceToHost, 
                              cmd_q[KERNEL_Q]));

        CUCHK(cudaEventRecord(loop1_ev_rb_rsd[work_step][1], cmd_q[KERNEL_Q]));

        CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
      }
      else if (split_flag && buffering_flag) {
        // Double buffering dose not need event object on read buffer 
        // ( read(n) -> write(n+2) is guarunteed )
        // but not in Triple buffering  
        CUCHK(cudaStreamWaitEvent(cmd_q[DATA_Q],*(loop1_ev_kernel_end_ptr[work_step]), 0));

        CUCHK(cudaEventRecord(loop1_ev_rb_rsd[work_step][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(&(rsd[work_base][0][0][0]), 
                              ((char*)m_rsd[work_step%2])+sizeof(double)*1*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              sizeof(double)*work_num_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyDeviceToHost, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop1_ev_rb_rsd[work_step][1], cmd_q[DATA_Q]));
      }
    }


    if (buffering_flag) {
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
      CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
    }


    if (timeron) timer_stop(t_rhs);



    // ###########################
    //    Loop 1 Finished
    // ###########################





    // ###########################
    //    Loop 2  Start
    // ###########################

    if ( (istep % inorm) == 0 ) {
      l2norm_head(delunm, g_sum1, m_sum1, &ev_k_l2norm_head1[0], &ev_k_l2norm_head1[1]);
    }

    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      l2norm_head(rsdnm, g_sum2, m_sum2, &ev_k_l2norm_head2[0], &ev_k_l2norm_head2[1]);
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
      rhs_item = min((int)loop2_work_num_item_default, work_end);
      rhs_base = work_end - rhs_item;

      loop2_copy_item = rhs_item;
      loop2_copy_bbase = 2;
      loop2_copy_hbase = rhs_base;

      // front alignment calculation
      loop2_copy_item += min(2, rhs_base);
      loop2_copy_bbase -= min(2, rhs_base);
      loop2_copy_hbase -= min(2, rhs_base);

      CUCHK(cudaEventRecord(loop2_ev_wb_rsd[0][0], cmd_q[DATA_Q]));

      CUCHK(cudaMemcpyAsync(((char*)m_rsd[0])+sizeof(double)*loop2_copy_bbase*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                            &rsd[loop2_copy_hbase][0][0][0],
                            sizeof(double)*loop2_copy_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                            cudaMemcpyHostToDevice, 
                            cmd_q[DATA_Q]));

      CUCHK(cudaEventRecord(loop2_ev_wb_rsd[0][1], cmd_q[DATA_Q]));

      CUCHK(cudaEventRecord(loop2_ev_wb_u[0][0], cmd_q[DATA_Q]));

      CUCHK(cudaMemcpyAsync(((char*)m_u[0])+sizeof(double)*loop2_copy_bbase*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                            &u[loop2_copy_hbase][0][0][0],
                            sizeof(double)*loop2_copy_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                            cudaMemcpyHostToDevice, 
                            cmd_q[DATA_Q]));

      CUCHK(cudaEventRecord(loop2_ev_wb_u[0][1], cmd_q[DATA_Q]));

      CUCHK(cudaEventRecord(loop2_ev_wb_frct[0][0], cmd_q[DATA_Q]));

      CUCHK(cudaMemcpyAsync(((char*)m_frct[0])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                            &frct[rhs_base][0][0][0],
                            sizeof(double)*rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                            cudaMemcpyHostToDevice, 
                            cmd_q[DATA_Q]));

      CUCHK(cudaEventRecord(loop2_ev_wb_frct[0][1], cmd_q[DATA_Q]));

      CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
    }



    for (work_step = 0; work_step < loop2_work_max_iter; work_step++) {

      work_end = nz - work_step*loop2_work_num_item_default;

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

          CUCHK(cudaEventRecord(loop2_ev_wb_u[work_step][0], cmd_q[KERNEL_Q]));

          CUCHK(cudaMemcpyAsync(((char*)m_u[0])+sizeof(double)*loop2_copy_bbase*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                &(u[loop2_copy_hbase][0][0][0]),
                                sizeof(double)*loop2_copy_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyHostToDevice, 
                                cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_wb_u[work_step][1], cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_wb_rsd[work_step][0], cmd_q[KERNEL_Q]));

          CUCHK(cudaMemcpyAsync(((char*)m_rsd[0])+sizeof(double)*loop2_copy_bbase*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                &(rsd[loop2_copy_hbase][0][0][0]),
                                sizeof(double)*loop2_copy_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyHostToDevice, 
                                cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_wb_rsd[work_step][1], cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_wb_frct[work_step][0], cmd_q[KERNEL_Q]));

          CUCHK(cudaMemcpyAsync(((char*)m_frct[0])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                &(frct[rhs_base][0][0][0]),
                                sizeof(double)*rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyHostToDevice, 
                                cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_wb_frct[work_step][1], cmd_q[KERNEL_Q]));

          CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

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
                     &loop2_ev_wb_frct[work_step][1]);

      if (timeron) timer_stop(t_buts);

      if (timeron) timer_start(t_add);

      ssor2(ssor2_item, ssor2_base,
            work_step, (work_step%2)*buffering_flag,
            temp_kst, tmp2);

      if (timeron) timer_stop(t_add);

      if ( (istep % inorm) == 0 ) {

        if (timeron) timer_start(t_l2norm);
        //---------------------------------------------------------------------
        // compute the max-norms of newton iteration corrections
        //---------------------------------------------------------------------

        l2norm_body(work_step, loop2_work_max_iter, 
                    rhs_base, rhs_item, 
                    &loop2_ev_wb_frct[work_step][1], 
                    &ev_k_l2norm_body1[work_step][0],
                    &ev_k_l2norm_body1[work_step][1],
                    m_sum1, nz0, 
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
                                                    &loop2_ev_wb_frct[work_step][1]);

      //---------------------------------------------------------------------
      // compute the max-norms of newton iteration residuals
      //---------------------------------------------------------------------
      if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {

        if (timeron) timer_start(t_l2norm);

        l2norm_body(work_step, loop2_work_max_iter, 
                    rhs_base, rhs_item, 
                    &loop2_ev_wb_frct[work_step][1], 
                    &ev_k_l2norm_body2[work_step][0],
                    &ev_k_l2norm_body2[work_step][1],
                    m_sum2, nz0, 
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
        loop2_ev_kernel_end_ptr[work_step] = &ev_k_l2norm_body2[work_step][1];
      }


      if (split_flag && buffering_flag && work_step < loop2_work_max_iter - 1) {

        //###################
        // Write Next Buffer
        //###################

        CUCHK(cudaEventRecord(loop2_ev_wb_rsd[work_step+1][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_rsd[(work_step+1)%2])+sizeof(double)*next_loop2_copy_bbase*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              &(rsd[next_loop2_copy_hbase][0][0][0]),
                              sizeof(double)*next_loop2_copy_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop2_ev_wb_rsd[work_step+1][1], cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop2_ev_wb_u[work_step+1][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_u[(work_step+1)%2])+sizeof(double)*next_loop2_copy_bbase*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              &(u[next_loop2_copy_hbase][0][0][0]),
                              sizeof(double)*next_loop2_copy_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop2_ev_wb_u[work_step+1][1], cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop2_ev_wb_frct[work_step+1][0], cmd_q[DATA_Q]));

        CUCHK(cudaMemcpyAsync(((char*)m_frct[(work_step+1)%2])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              &(frct[next_rhs_base][0][0][0]),
                              sizeof(double)*next_rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                              cudaMemcpyHostToDevice, 
                              cmd_q[DATA_Q]));

        CUCHK(cudaEventRecord(loop2_ev_wb_frct[work_step+1][1], cmd_q[DATA_Q]));
      }

      //###################
      // Read Buffer
      //###################
      if (split_flag) {
        if (!buffering_flag) {

          CUCHK(cudaEventRecord(loop2_ev_rb_rsd[work_step][0], cmd_q[KERNEL_Q]));

          CUCHK(cudaMemcpyAsync(&(rsd[rhs_base][0][0][0]),
                                ((char*)m_rsd[0])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                sizeof(double)*rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyDeviceToHost, 
                                cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_rb_rsd[work_step][1], cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_rb_u[work_step][0], cmd_q[KERNEL_Q]));

          CUCHK(cudaMemcpyAsync(&(u[rhs_base][0][0][0]),
                                ((char*)m_u[0])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                sizeof(double)*rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyDeviceToHost, 
                                cmd_q[KERNEL_Q]));

          CUCHK(cudaEventRecord(loop2_ev_rb_u[work_step][1], cmd_q[KERNEL_Q]));

          CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

        }
        else {
          // Double buffering dose not need event object on read buffer 
          // ( read(n) -> write(n+2) is guarunteed )
          // but not in Triple buffering
          CUCHK(cudaStreamWaitEvent(cmd_q[DATA_Q], *(loop2_ev_kernel_end_ptr[work_step]), 0));

          CUCHK(cudaEventRecord(loop2_ev_rb_rsd[work_step][0], cmd_q[DATA_Q]));

          CUCHK(cudaMemcpyAsync(&(rsd[rhs_base][0][0][0]), 
                                ((char*)m_rsd[work_step%2])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                sizeof(double)*rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyDeviceToHost, 
                                cmd_q[DATA_Q]));

          CUCHK(cudaEventRecord(loop2_ev_rb_rsd[work_step][1], cmd_q[DATA_Q]));

          CUCHK(cudaEventRecord(loop2_ev_rb_u[work_step][0], cmd_q[DATA_Q]));

          CUCHK(cudaMemcpyAsync(&(u[rhs_base][0][0][0]),
                                ((char*)m_u[work_step%2])+sizeof(double)*2*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                sizeof(double)*rhs_item*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                                cudaMemcpyDeviceToHost, 
                                cmd_q[DATA_Q]));

          CUCHK(cudaEventRecord(loop2_ev_rb_u[work_step][1], cmd_q[DATA_Q]));
        }
      }
    }

    if (buffering_flag) {
      CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
      CUCHK(cudaStreamSynchronize(cmd_q[DATA_Q]));
    }

    if ( (istep % inorm) == 0 ) {
      l2norm_tail(delunm, g_sum1, m_sum1, nx0, ny0, nz0, &ev_d_l2norm_tail1[0], &ev_d_l2norm_tail1[1]);
    }

    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      l2norm_tail(rsdnm, g_sum2, m_sum2, nx0, ny0, nz0, &ev_d_l2norm_tail2[0], &ev_d_l2norm_tail2[1]);
    }

    // ###########################
    //    Loop 2 Finished 
    // ###########################


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

  cuda_ProfilerStop();
  timer_stop(1);
  maxtime = timer_read(1);

  // ! SPLIT FLAG
  if (!split_flag) {

    CUCHK(cudaMemcpyAsync(rsd, m_rsd[0],
          sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyDeviceToHost, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(u, m_u[0],
          sizeof(double)*ISIZ3*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
          cudaMemcpyDeviceToHost, cmd_q[KERNEL_Q]));
    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));


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


