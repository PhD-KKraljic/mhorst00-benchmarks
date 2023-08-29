//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB IS code. This OpenCL C  //
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

#include "npbparams.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <getopt.h>
#include <assert.h>

/******************/
/* default values */
/******************/
#ifndef CLASS
#define CLASS 'S'
#endif


/*************/
/*  CLASS S  */
/*************/
#if CLASS == 'S'
#define  TOTAL_KEYS_LOG_2    16
#define  MAX_KEY_LOG_2       11
#define  CLASS_ID            0
#endif


/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define  TOTAL_KEYS_LOG_2    20
#define  MAX_KEY_LOG_2       16
#define  CLASS_ID            1
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define  TOTAL_KEYS_LOG_2    23
#define  MAX_KEY_LOG_2       19
#define  CLASS_ID            2
#endif


/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define  TOTAL_KEYS_LOG_2    25
#define  MAX_KEY_LOG_2       21
#define  CLASS_ID            3
#endif


/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define  TOTAL_KEYS_LOG_2    27
#define  MAX_KEY_LOG_2       23
#define  CLASS_ID            4
#endif


/*************/
/*  CLASS D  */
/*************/
#if CLASS == 'D'
#define  TOTAL_KEYS_LOG_2    31
#define  MAX_KEY_LOG_2       27
#define  CLASS_ID            5
#endif


#if CLASS == 'D'
#define  TOTAL_KEYS          (1L << TOTAL_KEYS_LOG_2)
#else
#define  TOTAL_KEYS          (1 << TOTAL_KEYS_LOG_2)
#endif
#define  MAX_KEY             (1 << MAX_KEY_LOG_2)
#define  NUM_KEYS            TOTAL_KEYS
#define  SIZE_OF_BUFFERS     NUM_KEYS  


#define  MAX_ITERATIONS      10
#define  TEST_ARRAY_SIZE     5

// the number of keys that one work item covers
// cycle shoule be two to the power of n
#define CYCLE 4 


/*************************************/
/* Typedef: if necessary, change the */
/* size of int here by changing the  */
/* int type to, say, long            */
/*************************************/
#if CLASS == 'D'
typedef  long INT_TYPE;
#else
typedef  int  INT_TYPE;
#endif


/********************/
/* Some global info */
/********************/
INT_TYPE *key_buff_ptr_global;         /* used by full_verify to get */
/* copies of rank info        */

int      passed_verification;
int      g_passed_verification[TEST_ARRAY_SIZE*MAX_ITERATIONS];
int      g_failed[TEST_ARRAY_SIZE*MAX_ITERATIONS];


/************************************/
/* These are the three main arrays. */
/* See SIZE_OF_BUFFERS def above    */
/************************************/
INT_TYPE key_array[SIZE_OF_BUFFERS],    
         key_buff1[MAX_KEY],    
         key_buff2[SIZE_OF_BUFFERS],
         partial_verify_vals[TEST_ARRAY_SIZE];

INT_TYPE    log_local_size;
INT_TYPE    k_is3_wg;
INT_TYPE    *wg_key_buff_ptr;



/**********************/
/* Partial verif info */
/**********************/
INT_TYPE test_index_array[TEST_ARRAY_SIZE],
         test_rank_array[TEST_ARRAY_SIZE],

         S_test_index_array[TEST_ARRAY_SIZE] = 
                             {48427,17148,23627,62548,4431},
         S_test_rank_array[TEST_ARRAY_SIZE] = 
                             {0,18,346,64917,65463},

         W_test_index_array[TEST_ARRAY_SIZE] = 
                             {357773,934767,875723,898999,404505},
         W_test_rank_array[TEST_ARRAY_SIZE] = 
                             {1249,11698,1039987,1043896,1048018},

         A_test_index_array[TEST_ARRAY_SIZE] = 
                             {2112377,662041,5336171,3642833,4250760},
         A_test_rank_array[TEST_ARRAY_SIZE] = 
                             {104,17523,123928,8288932,8388264},

         B_test_index_array[TEST_ARRAY_SIZE] = 
                             {41869,812306,5102857,18232239,26860214},
         B_test_rank_array[TEST_ARRAY_SIZE] = 
                             {33422937,10244,59149,33135281,99}, 

         C_test_index_array[TEST_ARRAY_SIZE] = 
                             {44172927,72999161,74326391,129606274,21736814},
         C_test_rank_array[TEST_ARRAY_SIZE] = 
                             {61147,882988,266290,133997595,133525895},

         D_test_index_array[TEST_ARRAY_SIZE] = 
                             {1317351170,995930646,1157283250,1503301535,1453734525},
         D_test_rank_array[TEST_ARRAY_SIZE] = 
                             {1,36538729,1978098519,2145192618,2147425337};


/**********************/
/* OpenCL variables   */
/**********************/

#include <CL/cl.h>
#include "cl_util.h" 

#define KERNEL_Q  0
#define DATA_Q    1
#define NUM_Q     2

//#define DETAIL_INFO

#ifdef DETAIL_INFO
#define DETAIL_LOG(fmt, ...) fprintf(stdout, " [OpenCL Detailed Info] " fmt "\n", ## __VA_ARGS__)
#else
#define DETAIL_LOG(fmt, ...) 
#endif

#define min(a,b) ((a<b)?a:b)

cl_int                  ecode;
cl_device_type          device_type;
cl_device_id            device;
char                    * device_name;
cl_context              context;
cl_command_queue        cmd_queue_kernel;
cl_command_queue        cmd_queue_data;

cl_program              p_is;

cl_uint                 max_compute_units;
cl_ulong                max_mem_alloc_size;
cl_ulong                gmem_size;
cl_ulong                avail_gmem;
INT_TYPE                work_num_keys_default;
INT_TYPE                work_num_keys, prev_work_num_keys;
INT_TYPE                work_max_iter, work_step;
INT_TYPE                work_base, prev_work_base;
int                     split_flag=0;
int                     buffering_flag;
int                     cycle = CYCLE;


size_t                  max_work_item_sizes[3];
size_t                  max_work_group_size;

cl_kernel               k_pv_set,
                        k_is1, 
                        k_is2, 
                        k_is3_baseline,
                        k_is3_gmem,
                        k_is4_baseline,
                        k_is4_gmem,
                        k_is5,
                        k_pv;

cl_mem                  m_wg_key_buff_ptr;
cl_mem                  *m_key_array;
cl_mem                  *m_key_buff2;
cl_mem                  m_key_buff1;

cl_mem                  m_partial_verify_vals;
cl_mem                  m_test_rank_array;
cl_mem                  m_test_index_array;
cl_mem                  m_passed_verification;
cl_mem                  m_failed;

cl_event                ev_wb_test_rank,
                        ev_wb_test_index,
                        ev_rb_key_buff1,
                        *ev_wb_key_array,
                        ev_rb_pass,
                        ev_rb_fail,
                        ev_k_pv_set,
                        ev_wb_pv,
                        ev_k_is1,
                        *ev_k_is2,
                        ev_k_is3,
                        ev_k_is4,
                        ev_k_is5,
                        ev_k_pv;

/* OpenCL optimization levels enumerate */
enum OptLevel {
  OPT_BASELINE=0,
  OPT_PARALLEL,
  OPT_GLOBALMEM,
  OPT_MEMLAYOUT,
  OPT_SYNC,
  OPT_FULL,
};

enum OptLevel g_opt_level;





/***********************/
/* function prototypes */
/***********************/
double    randlc( double *X, double *A );

void full_verify( void );

void c_print_results( char  *name,
                      char   _class,
                      int    n1, 
                      int    n2,
                      int    n3,
                      int    niter,
                      double t,
                      double mops,
                      char  *optype,
                      int    passed_verification,
                      char  *npbversion,
                      char  *compiletime,
                      char  *cc,
                      char  *clink,
                      char  *c_lib,
                      char  *c_inc,
                      char  *cflags,
                      char  *clinkflags,
                      char  *crand,
                const char  *ocl_dev_type,
                      char  *ocl_dev_name );


void    timer_clear( int n );
void    timer_start( int n );
void    timer_stop( int n );
double  timer_read( int n );


static void setup_opencl(int argc, char *argv[]);
static void release_opencl();
void get_pv_result(int iterations);
static void print_opt_level(enum OptLevel ol);

/*
 *    FUNCTION RANDLC (X, A)
 *
 *  This routine returns a uniform pseudorandom double precision number in the
 *  range (0, 1) by using the linear congruential generator
 *
 *  x_{k+1} = a x_k  (mod 2^46)
 *
 *  where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
 *  before repeating.  The argument A is the same as 'a' in the above formula,
 *  and X is the same as x_0.  A and X must be odd double precision integers
 *  in the range (1, 2^46).  The returned value RANDLC is normalized to be
 *  between 0 and 1, i.e. RANDLC = 2^(-46) * x_1.  X is updated to contain
 *  the new seed x_1, so that subsequent calls to RANDLC using the same
 *  arguments will generate a continuous sequence.
 *
 *  This routine should produce the same results on any computer with at least
 *  48 mantissa bits in double precision floating point data.  On Cray systems,
 *  double precision should be disabled.
 *
 *  David H. Bailey     October 26, 1990
 *
 *     IMPLICIT DOUBLE PRECISION (A-H, O-Z)
 *     SAVE KS, R23, R46, T23, T46
 *     DATA KS/0/
 *
 *  If this is the first call to RANDLC, compute R23 = 2 ^ -23, R46 = 2 ^ -46,
 *  T23 = 2 ^ 23, and T46 = 2 ^ 46.  These are computed in loops, rather than
 *  by merely using the ** operator, in order to insure that the results are
 *  exact on all systems.  This code assumes that 0.5D0 is represented exactly.
 */


/*****************************************************************/
/*************           R  A  N  D  L  C             ************/
/*************                                        ************/
/*************    portable random number generator    ************/
/*****************************************************************/

double    randlc( double *X, double *A )
{
      static int        KS=0;
      static double R23, R46, T23, T46;
      double        T1, T2, T3, T4;
      double        A1;
      double        A2;
      double        X1;
      double        X2;
      double        Z;
      int           i, j;

      if (KS == 0) 
      {
        R23 = 1.0;
        R46 = 1.0;
        T23 = 1.0;
        T46 = 1.0;
    
        for (i=1; i<=23; i++)
        {
          R23 = 0.50 * R23;
          T23 = 2.0 * T23;
        }
        for (i=1; i<=46; i++)
        {
          R46 = 0.50 * R46;
          T46 = 2.0 * T46;
        }
        KS = 1;
      }

/*  Break A into two parts such that A = 2^23 * A1 + A2 and set X = N.  */

      T1 = R23 * *A;
      j  = T1;
      A1 = j;
      A2 = *A - T23 * A1;

/*  Break X into two parts such that X = 2^23 * X1 + X2, compute
    Z = A1 * X2 + A2 * X1  (mod 2^23), and then
    X = 2^23 * Z + A2 * X2  (mod 2^46).                            */

      T1 = R23 * *X;
      j  = T1;
      X1 = j;
      X2 = *X - T23 * X1;
      T1 = A1 * X2 + A2 * X1;
      
      j  = R23 * T1;
      T2 = j;
      Z = T1 - T23 * T2;
      T3 = T23 * Z + A2 * X2;
      j  = R46 * T3;
      T4 = j;
      *X = T3 - T46 * T4;
      return(R46 * *X);
} 




/*****************************************************************/
/*************      C  R  E  A  T  E  _  S  E  Q      ************/
/*****************************************************************/

void    create_seq( double seed, double a )
{
    double x;
    INT_TYPE i, k;

    k = MAX_KEY/4;

    for (i=0; i<NUM_KEYS; i++)
    {
        x = randlc(&seed, &a);
        x += randlc(&seed, &a);
        x += randlc(&seed, &a);
        x += randlc(&seed, &a);  

        key_array[i] = k*x;
    }
}




/*****************************************************************/
/*************    F  U  L  L  _  V  E  R  I  F  Y     ************/
/*****************************************************************/


void full_verify( void )
{
    INT_TYPE    i, j;


    
/*  Now, finally, sort the keys:  */


/*  Copy keys into work array; keys in key_array will be reassigned. */
    for( i=0; i<NUM_KEYS; i++ )
        key_buff2[i] = key_array[i];


    for( i=0; i<NUM_KEYS; i++ )
        key_array[--key_buff_ptr_global[key_buff2[i]]] = key_buff2[i];


/*  Confirm keys correctly sorted: count incorrectly sorted keys, if any */

    j = 0;
    for( i=1; i<NUM_KEYS; i++ )
        if( key_array[i-1] > key_array[i] )
            j++;


    if( j != 0 )
    {
        printf( "Full_verify: number of keys out of sort: %ld\n",
                (long)j );
    }
    else
        passed_verification++;
           

}




/*****************************************************************/
/*************             R  A  N  K             ****************/
/*****************************************************************/
void rank( int iteration )
{
  INT_TYPE    i;
  INT_TYPE    *key_buff_ptr, *key_buff_ptr2;
  size_t      lws[2], gws[2];

  key_array[iteration] = iteration;
  key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;

  if (!split_flag) {
    lws[0] = 1;
    gws[0] = 1;

    ecode =  clSetKernelArg(k_pv_set, 0, sizeof(cl_mem), &m_test_index_array);
    ecode |= clSetKernelArg(k_pv_set, 1, sizeof(cl_mem), &m_key_array[0]);
    ecode |= clSetKernelArg(k_pv_set, 2, sizeof(cl_mem), &m_partial_verify_vals);
    ecode |= clSetKernelArg(k_pv_set, 3, sizeof(int), &iteration);
    clu_CheckError(ecode, "clSetKernelArg for k_pv_set");

    ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                   k_pv_set,
                                   1, NULL, 
                                   gws, lws,
                                   0, NULL, &ev_k_pv_set);
    clu_CheckError(ecode , "clEnqueueNDRangeKerenel for k_pv_set ");
  }
  else {
    for( i=0; i<TEST_ARRAY_SIZE; i++ )        
      partial_verify_vals[i] = key_array[test_index_array[i]];

    ecode = clEnqueueWriteBuffer(cmd_queue_kernel, 
                                 m_partial_verify_vals, 
                                 CL_TRUE,
                                 0,
                                 sizeof(INT_TYPE)*TEST_ARRAY_SIZE, 
                                 partial_verify_vals, 
                                 0, NULL, &ev_wb_pv);
    clu_CheckError(ecode, "clEnqueueWriteBuffer for m_test_rank_array");

    key_buff_ptr2 = key_array;
  }

  /*  Clear the work array */
  /*
     for( i=0; i<MAX_KEY; i++ )
     key_buff1[i] = 0;
   */
  lws[0] = max_work_group_size;
  gws[0] = MAX_KEY;

  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode = clSetKernelArg( k_is1, 0, sizeof(cl_mem), &m_key_buff1);
  clu_CheckError(ecode, "clSetKernelArg for k_is1");


  ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                 k_is1,
                                 1, NULL, 
                                 gws, lws,
                                 0, NULL, &ev_k_is1);
  clu_CheckError(ecode , "clEnqueueNDRangeKerenel for k_is1");

  /*  In this section, the keys themselves are used as their 
      own indexes to determine how many of each there are: their
      individual population                                       */

  /*
     for( i=0; i<NUM_KEYS; i++ )
     key_buff_ptr[key_buff_ptr2[i]]++;  
   */
  /* Now they have individual key   */
  /* population                     */

  work_base = 0;
  work_num_keys = min(NUM_KEYS - work_base, work_num_keys_default);

  if (split_flag) {
    // ####################
    // Write first buffer
    // ####################
    ecode = clEnqueueWriteBuffer(cmd_queue_data, 
                                 m_key_array[0], 
                                 CL_FALSE,
                                 0, 
                                 sizeof(INT_TYPE)*work_num_keys, 
                                 &(key_buff_ptr2[work_base]), 
                                 0, NULL, &ev_wb_key_array[0]);
    clu_CheckError(ecode, "clEnqueueWriteBuffer for m_key_array");
  }

  for (work_step = 0; work_step < work_max_iter; work_step++)
  {
    lws[0] = max_work_group_size;
    gws[0] = work_num_keys;
    gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

    ecode  = clSetKernelArg(k_is2, 0, sizeof(cl_mem), &m_key_buff1);// key_buff1 == key_buff_ptr
    ecode |= clSetKernelArg(k_is2, 1, sizeof(cl_mem), &m_key_array[work_step%2]);
    ecode |= clSetKernelArg(k_is2, 2, sizeof(INT_TYPE), &work_num_keys);
    clu_CheckError(ecode, "clSetKernelArg for k_is2");

    cl_event *k_wait_ev;
    cl_event *w_wait_ev, *w_my_ev;
    int k_wait_num;
    int w_wait_num;

    if (!split_flag) {
      k_wait_ev = NULL;
      k_wait_num = 0;
    }
    else {
      k_wait_ev = &ev_wb_key_array[work_step];
      k_wait_num =  1;
    }

    ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                   k_is2,
                                   1, NULL, 
                                   gws, lws,
                                   k_wait_num, k_wait_ev, 
                                   &ev_k_is2[work_step]);
    clu_CheckError(ecode, "clEnqueueNDRangeKerenel for k_is2");

    prev_work_num_keys = work_num_keys;
    prev_work_base = work_base;

    work_base += work_num_keys;
    work_num_keys = min(NUM_KEYS - work_base, work_num_keys_default);

    // ##############
    // Write buffer
    // ##############

    if (split_flag && work_step < work_max_iter - 1)
    {
      if (work_step < 1) {
        w_wait_ev = NULL;
        w_wait_num = 0;
      }
      else {
        w_wait_ev = &ev_k_is2[work_step-1];
        w_wait_num = 1;
      }
      w_my_ev = &ev_wb_key_array[work_step+1];

      ecode = clEnqueueWriteBuffer(cmd_queue_data, 
                                   m_key_array[(work_step+1)%2], 
                                   CL_FALSE,
                                   0, 
                                   sizeof(INT_TYPE)*work_num_keys, 
                                   &(key_buff_ptr2[work_base]), 
                                   w_wait_num, w_wait_ev, w_my_ev);
      clu_CheckError(ecode, "clEnqueueWriteBuffer for m_key_array");
    }
  }

  if (split_flag)
    clFinish(cmd_queue_data);

  /*  To obtain ranks of each key, successively add the individual key
      population                                                  */
  /*
     for( i=0; i<MAX_KEY-1; i++ )   
     key_buff_ptr[i+1] += key_buff_ptr[i];  
   */

  lws[0] = (1 << log_local_size);
  gws[0] = k_is3_wg*lws[0];

  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  switch (g_opt_level) {
    case OPT_BASELINE:
    case OPT_PARALLEL:
    case OPT_MEMLAYOUT:
    case OPT_SYNC:
      ecode  = clSetKernelArg(k_is3_baseline, 0, sizeof(cl_mem), &m_key_buff1);// key_buff1 == key_buff_ptr
      ecode |= clSetKernelArg(k_is3_baseline, 1, sizeof(cl_mem), &m_wg_key_buff_ptr);
      ecode |= clSetKernelArg(k_is3_baseline, 2, sizeof(int), &log_local_size);
      clu_CheckError(ecode, "clSetKernelArg for k_is3_baseline");

      ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                     k_is3_baseline,
                                     1, NULL, 
                                     gws, lws,
                                     0, NULL, &ev_k_is3);
      break;

    case OPT_GLOBALMEM:
    case OPT_FULL:
      ecode  = clSetKernelArg(k_is3_gmem, 0, sizeof(cl_mem), &m_key_buff1);// key_buff1 == key_buff_ptr
      ecode |= clSetKernelArg(k_is3_gmem, 1, sizeof(cl_mem), &m_wg_key_buff_ptr);
      ecode |= clSetKernelArg(k_is3_gmem, 2, sizeof(INT_TYPE)*lws[0], NULL);
      ecode |= clSetKernelArg(k_is3_gmem, 3, sizeof(int), &log_local_size);
      clu_CheckError(ecode, "clSetKernelArg for k_is3_gmem");

      ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                     k_is3_gmem,
                                     1, NULL, 
                                     gws, lws,
                                     0, NULL, &ev_k_is3);
      break;

    default:
      assert(0);
  }
  clu_CheckError(ecode , "clEnqueueNDRangeKerenel for k_is3 ");


  lws[0] = (1 << log_local_size);
  gws[0] = lws[0];
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  switch (g_opt_level) {
    case OPT_BASELINE:
    case OPT_PARALLEL:
    case OPT_MEMLAYOUT:
    case OPT_SYNC:
      ecode =  clSetKernelArg(k_is4_baseline, 0, sizeof(cl_mem), &m_wg_key_buff_ptr); // key_buff1 == key_buff_ptr
      ecode |= clSetKernelArg(k_is4_baseline, 1, sizeof(int), &log_local_size);
      ecode |= clSetKernelArg(k_is4_baseline, 2, sizeof(INT_TYPE), &k_is3_wg);
      clu_CheckError(ecode, "clSetKernelArg for k_is4_baseline");

      ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                     k_is4_baseline,
                                     1, NULL, 
                                     gws, lws,
                                     0, NULL, &ev_k_is4);
      break;

    case OPT_GLOBALMEM:
    case OPT_FULL:
      ecode =  clSetKernelArg(k_is4_gmem, 0, sizeof(cl_mem), &m_wg_key_buff_ptr); // key_buff1 == key_buff_ptr
      ecode |= clSetKernelArg(k_is4_gmem, 1, sizeof(INT_TYPE)*lws[0], NULL);
      ecode |= clSetKernelArg(k_is4_gmem, 2, sizeof(int), &log_local_size);
      ecode |= clSetKernelArg(k_is4_gmem, 3, sizeof(INT_TYPE), &k_is3_wg);
      clu_CheckError(ecode, "clSetKernelArg for k_is4_gmem");

      ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                     k_is4_gmem,
                                     1, NULL, 
                                     gws, lws,
                                     0, NULL, &ev_k_is4);
      break;

    default:
      assert(0);
  }
  clu_CheckError(ecode , "clEnqueueNDRangeKerenel for k_is4 ");


  lws[0] = (1 << log_local_size);
  gws[0] = (k_is3_wg-1)*lws[0];
  gws[0] = clu_RoundWorkSize(gws[0], lws[0]);

  ecode =  clSetKernelArg(k_is5, 0, sizeof(cl_mem), &m_wg_key_buff_ptr);
  ecode |= clSetKernelArg(k_is5, 1, sizeof(cl_mem), &m_key_buff1);
  clu_CheckError(ecode, "clSetKernelArg for k_is5");

  ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                 k_is5,
                                 1, NULL, 
                                 gws, lws,
                                 0, NULL, &ev_k_is5);
  clu_CheckError(ecode , "clEnqueueNDRangeKerenel for k_is9 ");
 
  /*  Ranking of all keys occurs in this section:                 */
  key_buff_ptr = key_buff1;

  lws[0] = TEST_ARRAY_SIZE;
  gws[0] = TEST_ARRAY_SIZE; 

  ecode =  clSetKernelArg(k_pv, 0, sizeof(cl_mem), &m_partial_verify_vals);
  ecode |= clSetKernelArg(k_pv, 1, sizeof(cl_mem), &m_key_buff1);
  ecode |= clSetKernelArg(k_pv, 2, sizeof(cl_mem), &m_test_rank_array);
  ecode |= clSetKernelArg(k_pv, 3, sizeof(cl_mem), &m_passed_verification);
  ecode |= clSetKernelArg(k_pv, 4, sizeof(cl_mem), &m_failed);
  ecode |= clSetKernelArg(k_pv, 5, sizeof(int), &iteration);

  clu_CheckError(ecode, "clSetKernelArg for k_pv");

  ecode = clEnqueueNDRangeKernel(cmd_queue_kernel, 
                                 k_pv,
                                 1, NULL, 
                                 gws, lws,
                                 0, NULL, &ev_k_pv);
  clu_CheckError(ecode , "clEnqueueNDRangeKerenel for k_pv ");

  /*  Make copies of rank info for use by full_verify: these variables
      in rank are local; making them global slows down the code, probably
      since they cannot be made register by compiler                        */

  if( iteration == MAX_ITERATIONS ) 
    key_buff_ptr_global = key_buff_ptr;

  if (split_flag)
    clReleaseEvent(ev_k_pv_set);
  else
    clReleaseEvent(ev_wb_pv);

  clReleaseEvent(ev_k_is1);
  clReleaseEvent(ev_k_is3);
  clReleaseEvent(ev_k_is4);
  clReleaseEvent(ev_k_is5);
  clReleaseEvent(ev_k_pv);

  for (work_step = 0; work_step < work_max_iter; work_step++) {
    if (split_flag)
      clReleaseEvent(ev_wb_key_array[work_step]);
    clReleaseEvent(ev_k_is2[work_step]);
  }
}

/*****************************************************************/
/*************             M  A  I  N             ****************/
/*****************************************************************/

int main( int argc, char **argv )
{

  int             i, iteration, timer_on;

  double          timecounter;

  FILE            *fp;


  /*  Initialize timers  */
  timer_on = 0;            
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    fclose(fp);
    timer_on = 1;
  }
  timer_clear( 0 );
  if (timer_on) {
    timer_clear( 1 );
    timer_clear( 2 );
    timer_clear( 3 );
  }

  if (timer_on) timer_start( 3 );


  /*  Initialize the verification arrays if a valid class */
  for( i=0; i<TEST_ARRAY_SIZE; i++ )
    switch( CLASS )
    {
      case 'S':
        test_index_array[i] = S_test_index_array[i];
        test_rank_array[i]  = S_test_rank_array[i];
        break;
      case 'A':
        test_index_array[i] = A_test_index_array[i];
        test_rank_array[i]  = A_test_rank_array[i];
        break;
      case 'W':
        test_index_array[i] = W_test_index_array[i];
        test_rank_array[i]  = W_test_rank_array[i];
        break;
      case 'B':
        test_index_array[i] = B_test_index_array[i];
        test_rank_array[i]  = B_test_rank_array[i];
        break;
      case 'C':
        test_index_array[i] = C_test_index_array[i];
        test_rank_array[i]  = C_test_rank_array[i];
        break;
      case 'D':
        test_index_array[i] = D_test_index_array[i];
        test_rank_array[i]  = D_test_rank_array[i];
        break;
    };



  /*  Printout initial NPB info */
  printf
    ( "\n\n NAS Parallel Benchmarks (NPB3.3.1-OCL) - IS Benchmark\n\n" );
  printf( " Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS );
  printf( " Iterations:   %d\n", MAX_ITERATIONS );

  /*    Setup OpenCL */
  setup_opencl(argc, argv);

  if (timer_on) timer_start( 1 );

  /*  Generate random number sequence and subsequent keys on all procs */
  create_seq( 314159265.00,                    /* Random number gen seed */
      1220703125.00 );                 /* Random number gen mult */
  if (timer_on) timer_stop( 1 );


  if (!split_flag) {
    ecode = clEnqueueWriteBuffer(cmd_queue_kernel, 
        m_key_array[0], 
        CL_TRUE, 0, 
        sizeof(INT_TYPE)*NUM_KEYS, key_array, 
        0, NULL, &ev_wb_key_array[0]);
    clu_CheckError(ecode, "clEnqueueWriteBuffer for m_key_array");

    clReleaseEvent(ev_wb_key_array[0]);
  }
  /*  Do one interation for free (i.e., untimed) to guarantee initialization of  
      all data and code pages and respective tables */
  rank( 1 );

  get_pv_result(1);

  ecode = clEnqueueReadBuffer(cmd_queue_kernel, 
      m_key_buff1, 
      CL_TRUE,
      0, 
      sizeof(INT_TYPE)*MAX_KEY, 
      key_buff1, 
      0, NULL, &ev_rb_key_buff1);
  clu_CheckError(ecode, "clEnqueueWriteBuffe- m_key_buff1");

  clReleaseEvent(ev_rb_key_buff1);

  /*  Start verification counter */
  passed_verification = 0;

  if( CLASS != 'S' ) printf( "\n   iteration\n" );

  if (!split_flag) {
    ecode = clEnqueueWriteBuffer(cmd_queue_kernel, 
        m_key_array[0], 
        CL_TRUE, 0, 
        sizeof(INT_TYPE)*NUM_KEYS, key_array, 
        0, NULL, NULL);
    clu_CheckError(ecode, "clEnqueueWriteBuffer for m_key_array");
  }

  /*  Start timer  */             
  timer_start( 0 );
  clu_ProfilerStart();

  /*  This is the main iteration */
  for( iteration=1; iteration<=MAX_ITERATIONS; iteration++ )
  {
    if( CLASS != 'S' ) printf( "        %d\n", iteration );
    rank( iteration );
  }

  if (!split_flag)
    clFinish(cmd_queue_kernel);

  get_pv_result(MAX_ITERATIONS);

  /*  End of timing, obtain maximum time of all processors */
  clu_ProfilerStop();
  timer_stop( 0 );
  timecounter = timer_read( 0 );

  ecode = clEnqueueReadBuffer(cmd_queue_kernel, 
      m_key_buff1, 
      CL_TRUE, 0, 
      sizeof(INT_TYPE)*MAX_KEY, 
      key_buff1, 
      0, NULL, &ev_rb_key_buff1);

  clu_CheckError(ecode, "clEnqueueWriteBuffer for k_is4 - m_key_buff1");

  clReleaseEvent(ev_rb_key_buff1);

  /*  This tests that keys are in sequence: sorting of last ranked key seq
      occurs here, but is an untimed operation                             */
  if (timer_on) timer_start( 2 );
  full_verify();
  if (timer_on) timer_stop( 2 );

  if (timer_on) timer_stop( 3 );


  /*  The final printout  */
  if( passed_verification != 5*MAX_ITERATIONS + 1 )
    passed_verification = 0;
  c_print_results( "IS",
      CLASS,
      (int)(TOTAL_KEYS/64),
      64,
      0,
      MAX_ITERATIONS,
      timecounter,
      ((double) (MAX_ITERATIONS*TOTAL_KEYS))
      /timecounter/1000000.,
      "keys ranked", 
      passed_verification,
      NPBVERSION,
      COMPILETIME,
      CC,
      CLINK,
      C_LIB,
      C_INC,
      CFLAGS,
      CLINKFLAGS,
      "",
      clu_GetDeviceTypeName(device_type),
      device_name);


  /*  Print additional timers  */
  if (timer_on) {
    double t_total, t_percent;

    t_total = timer_read( 3 );
    printf("\nAdditional timers -\n");
    printf(" Total execution: %8.3f\n", t_total);
    if (t_total == 0.0) t_total = 1.0;
    timecounter = timer_read(1);
    t_percent = timecounter/t_total * 100.;
    printf(" Initialization : %8.3f (%5.2f%%)\n", timecounter, t_percent);
    timecounter = timer_read(0);
    t_percent = timecounter/t_total * 100.;
    printf(" Benchmarking   : %8.3f (%5.2f%%)\n", timecounter, t_percent);
    timecounter = timer_read(2);
    t_percent = timecounter/t_total * 100.;
    printf(" Sorting        : %8.3f (%5.2f%%)\n", timecounter, t_percent);

    clu_ProfilerPrintResult();
  }


  /*    Release OpenCL objects */

  release_opencl();


  return 0;
  /**************************/
}        /*  E N D  P R O G R A M  */
         /**************************/




// OpenCL setup function
static void setup_opencl(int argc, char *argv[])
{

  cl_int ecode;
  char * source_dir = "../IS" ;
  char build_option[200];
  int temp;

  clu_ProfilerSetup();

  int c;
  char optimization_flag[1024];
  char opt_source_dir[1024];
  int opt_level_i = -1;
  while ((c = getopt(argc, argv, "o:s:")) != -1) {
    switch (c) {
      case 'o':
        memcpy(optimization_flag, optarg, 1024);
        opt_level_i = atoi(optimization_flag);
        break;
      case 's':
        memcpy(opt_source_dir, optarg, 1024);
        DETAIL_LOG("opt source dir : %s\n", opt_source_dir);
        source_dir = opt_source_dir;
        break;
    } 
  }

  // set optimization level 
  switch (opt_level_i) {
    case 0:
      g_opt_level = OPT_BASELINE;
      break;
    case 1:
      g_opt_level = OPT_PARALLEL;
      fprintf(stderr, " ERROR : This App does NOT support partially optimized version for Opt. group 1\n");
      exit(EXIT_FAILURE);
      break;
    case 2:
      g_opt_level = OPT_GLOBALMEM;
      break;
    case 3:
      g_opt_level = OPT_MEMLAYOUT;
      fprintf(stderr, " ERROR : This App does NOT support partially optimized version for Opt. group 3\n");
      exit(EXIT_FAILURE);
      break;
    case 4:
      g_opt_level = OPT_SYNC;
      fprintf(stderr, " ERROR : This App does NOT support partially optimized version for Opt. group 4\n");
      exit(EXIT_FAILURE);
      break;
    default:
      g_opt_level = OPT_FULL;
      break;
  }

  print_opt_level(g_opt_level);

  //-----------------------------------------------------------------------
  // 1. Find the default device type and get a device for the device type
  //-----------------------------------------------------------------------
  device_type = clu_GetDefaultDeviceType();
  device      = clu_GetAvailableDevice(device_type);
  device_name = clu_GetDeviceName(device);

  // Device information
  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(max_work_item_sizes),
                          &max_work_item_sizes,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(size_t),
                          &max_work_group_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(cl_uint),
                          &max_compute_units,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(cl_ulong),
                          &max_mem_alloc_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");

  ecode = clGetDeviceInfo(device,
                          CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(cl_ulong),
                          &gmem_size,
                          NULL);
  clu_CheckError(ecode, "clGetDiviceInfo()");



  //-----------------------------------------------------------------------
  // 2. Create a context for the specified device
  //-----------------------------------------------------------------------
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &ecode);
  clu_CheckError(ecode, "clCreateContext()");

  //-----------------------------------------------------------------------
  // 3. Create a command queue
  //-----------------------------------------------------------------------
  cmd_queue_kernel = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");
  cmd_queue_data = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ecode);
  clu_CheckError(ecode, "clCreateCommandQueue()");


  //-----------------------------------------------------------------------
  // 4. Build programs 
  //-----------------------------------------------------------------------
#if CLASS == 'D'
  sprintf(build_option, 
      " -DCLASS_ID=%d \
        -DNUM_KEYS=%ld \
        -DMAX_KEY=%d \
        -DTEST_ARRAY_SIZE=%d \
        -DMAX_ITERATIONS=%d ", 
        CLASS_ID, 
        NUM_KEYS, 
        MAX_KEY, 
        TEST_ARRAY_SIZE, 
        MAX_ITERATIONS);
#else
  sprintf(build_option, 
      " -DCLASS_ID=%d \
        -DNUM_KEYS=%d \
        -DMAX_KEY=%d \
        -DTEST_ARRAY_SIZE=%d \
        -DMAX_ITERATIONS=%d ", 
        CLASS_ID, 
        NUM_KEYS, 
        MAX_KEY, 
        TEST_ARRAY_SIZE, 
        MAX_ITERATIONS);
#endif

  p_is = clu_MakeProgram(context, 
                         device, 
                         source_dir,
                         "kernel_is.cl",
                         build_option);

  //-----------------------------------------------------------------------
  // 5. Create buffers 
  //-----------------------------------------------------------------------

  log_local_size = 0;
  temp = max_work_group_size;
  while( (temp=temp >> 1) > 0 ) log_local_size++;

  buffering_flag = 1;

  k_is3_wg = min(1<<log_local_size, MAX_KEY/(1<<log_local_size));

  // compute tight memory bound
  avail_gmem = gmem_size*0.8;

  avail_gmem -= sizeof(INT_TYPE)*k_is3_wg; // m_wg_key_buff_ptr
  avail_gmem -= sizeof(INT_TYPE)*MAX_KEY; // m_key_buff1
  avail_gmem -= sizeof(INT_TYPE)*TEST_ARRAY_SIZE; //m_partial_verify_vals
  avail_gmem -= sizeof(INT_TYPE)*TEST_ARRAY_SIZE; //m_test_rank_array
  avail_gmem -= sizeof(INT_TYPE)*TEST_ARRAY_SIZE; //m_test_index_array
  avail_gmem -= sizeof(INT_TYPE)*TEST_ARRAY_SIZE*MAX_ITERATIONS; //m_passed_verification
  avail_gmem -= sizeof(INT_TYPE)*TEST_ARRAY_SIZE*MAX_ITERATIONS; //m_failed

  cl_ulong tmp;

  tmp = avail_gmem/sizeof(INT_TYPE) ; 
  tmp = min(tmp, max_mem_alloc_size/sizeof(INT_TYPE));
  tmp = min(tmp, NUM_KEYS);
  work_num_keys_default = (INT_TYPE)tmp;

  if (work_num_keys_default != NUM_KEYS) 
    split_flag = 1;

  if (!split_flag) {
    work_max_iter = 1;
    work_num_keys_default = NUM_KEYS;
  }
  else {
    work_num_keys_default /= 2;
    work_max_iter = (NUM_KEYS - 1) / work_num_keys_default + 1;
  }

  // memory allocation and compute each work group size

  wg_key_buff_ptr = (INT_TYPE *) malloc (sizeof(INT_TYPE)*k_is3_wg);

  m_wg_key_buff_ptr = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     sizeof(INT_TYPE)*k_is3_wg, 
                                     NULL, 
                                     &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_wg_key_buff_ptr");

  m_partial_verify_vals = clCreateBuffer(context, 
                                         CL_MEM_READ_WRITE, 
                                         sizeof(INT_TYPE)*TEST_ARRAY_SIZE, 
                                         NULL, 
                                         &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_partial_verify_vals");

  m_test_rank_array = clCreateBuffer(context, 
                                     CL_MEM_READ_ONLY, 
                                     sizeof(INT_TYPE)*TEST_ARRAY_SIZE, 
                                     NULL, 
                                     &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_test_rank_array");

  m_test_index_array = clCreateBuffer(context, 
                                      CL_MEM_READ_ONLY, 
                                      sizeof(INT_TYPE)*TEST_ARRAY_SIZE, 
                                      NULL, 
                                      &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_test_index_array");

  m_passed_verification = clCreateBuffer(context, 
                                         CL_MEM_READ_WRITE, 
                                         sizeof(int)*TEST_ARRAY_SIZE*MAX_ITERATIONS,
                                         NULL, 
                                         &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_passed_verification");

  m_failed = clCreateBuffer(context, 
                            CL_MEM_READ_WRITE, 
                            sizeof(int)*TEST_ARRAY_SIZE*MAX_ITERATIONS,
                            NULL,
                            &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_failed");

  if (!split_flag) {
    m_key_array = (cl_mem*)malloc(sizeof(cl_mem));

    m_key_array[0] = clCreateBuffer(context,
                                    CL_MEM_READ_WRITE, 
                                    sizeof(INT_TYPE)*SIZE_OF_BUFFERS,
                                    NULL,
                                    &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_key_array");

    m_key_buff2 = (cl_mem*)malloc(sizeof(cl_mem));

    m_key_buff2[0] = clCreateBuffer(context, 
                                    CL_MEM_READ_WRITE, 
                                    sizeof(INT_TYPE)*SIZE_OF_BUFFERS, 
                                    NULL, 
                                    &ecode);
    clu_CheckError(ecode, "clCreateBuffer() for m_key_buff2");
  }
  else {
    if (!buffering_flag) {
      m_key_array = (cl_mem*)malloc(sizeof(cl_mem));

      m_key_array[0] = clCreateBuffer(context, 
                                      CL_MEM_READ_WRITE, 
                                      sizeof(INT_TYPE)*work_num_keys_default,
                                      NULL,
                                      &ecode);
      clu_CheckError(ecode, "clCreateBuffer() for m_key_array");

      m_key_buff2 = (cl_mem*)malloc(sizeof(cl_mem));

      m_key_buff2[0] = clCreateBuffer(context, 
                                      CL_MEM_READ_WRITE, 
                                      sizeof(INT_TYPE)*work_num_keys_default,
                                      NULL,
                                      &ecode);
      clu_CheckError(ecode, "clCreateBuffer() for m_key_buff2");

    }
    else {
      m_key_array = (cl_mem*)malloc(sizeof(cl_mem)*2);

      m_key_array[0] = clCreateBuffer(context, 
                                      CL_MEM_READ_WRITE, 
                                      sizeof(INT_TYPE)*work_num_keys_default, 
                                      NULL, 
                                      &ecode);
      clu_CheckError(ecode, "clCreateBuffer() for m_key_array");

      m_key_array[1] = clCreateBuffer(context, 
                                      CL_MEM_READ_WRITE, 
                                      sizeof(INT_TYPE)*work_num_keys_default,
                                      NULL,
                                      &ecode);
      clu_CheckError(ecode, "clCreateBuffer() for m_key_array");

      m_key_buff2 = (cl_mem*)malloc(sizeof(cl_mem)*2);

      m_key_buff2[0] = clCreateBuffer(context, 
                                      CL_MEM_READ_WRITE, 
                                      sizeof(INT_TYPE)*work_num_keys_default, 
                                      NULL, 
                                      &ecode);
      clu_CheckError(ecode, "clCreateBuffer() for m_key_buff2");

      m_key_buff2[1] = clCreateBuffer(context, 
                                      CL_MEM_READ_WRITE, 
                                      sizeof(INT_TYPE)*work_num_keys_default, 
                                      NULL, 
                                      &ecode);
      clu_CheckError(ecode, "clCreateBuffer() for m_key_buff2");
    }
  }

  m_key_buff1 = clCreateBuffer(context, 
                               CL_MEM_READ_WRITE, 
                               sizeof(INT_TYPE)*MAX_KEY, 
                               NULL, 
                               &ecode);
  clu_CheckError(ecode, "clCreateBuffer() for m_key_buff1");


  //-----------------------------------------------------------------------
  // 6. Create kernels 
  //-----------------------------------------------------------------------
  k_pv_set = clCreateKernel(p_is, "k_pv_set", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for partial_verification");

  k_is1 = clCreateKernel(p_is, "k_is1", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for is6");

  k_is2 = clCreateKernel(p_is, "k_is2", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for is7");

  k_is3_baseline = clCreateKernel(p_is, "k_is3_baseline", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for k_is3_baseline");

  k_is3_gmem = clCreateKernel(p_is, "k_is3_gmem", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for k_is3_gmem");

  k_is4_baseline = clCreateKernel(p_is, "k_is4_baseline", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for k_is4_baseline");
  k_is4_gmem = clCreateKernel(p_is, "k_is4_gmem", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for k_is4");

  k_is5 = clCreateKernel(p_is, "k_is5", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for k_is5");

  k_pv = clCreateKernel(p_is, "k_pv", &ecode);
  clu_CheckError(ecode, "clCreateKernel() for partial_verification");



  //-----------------------------------------------------------------------
  // 6. Create events
  //-----------------------------------------------------------------------

  ev_wb_key_array = (cl_event*)malloc(sizeof(cl_event)*work_max_iter);
  ev_k_is2 = (cl_event*)malloc(sizeof(cl_event)*work_max_iter);




  if (!buffering_flag) 
    clReleaseCommandQueue(cmd_queue_data);

  // set test_rank_array
  ecode = clEnqueueWriteBuffer(cmd_queue_kernel, 
                               m_test_rank_array, 
                               CL_TRUE,
                               0,
                               sizeof(INT_TYPE)*TEST_ARRAY_SIZE, 
                               test_rank_array, 
                               0, NULL, &ev_wb_test_rank);
  clu_CheckError(ecode, "clEnqueueWriteBuffer for m_test_rank_array");

  ecode = clEnqueueWriteBuffer(cmd_queue_kernel, 
                               m_test_index_array, 
                               CL_TRUE,
                               0,
                               sizeof(INT_TYPE)*TEST_ARRAY_SIZE, 
                               test_index_array, 
                               0, NULL, &ev_wb_test_index);
  clu_CheckError(ecode, "clEnqueueWriteBuffer for m_test_index_array");

  clReleaseEvent(ev_wb_test_rank);
  clReleaseEvent(ev_wb_test_index);

  DETAIL_LOG("Tiling flag : %d", split_flag);
}

static void release_opencl()
{
  free(ev_wb_key_array);
  free(ev_k_is2);

  clReleaseKernel(k_pv_set);
  clReleaseKernel(k_is1);
  clReleaseKernel(k_is2);
  clReleaseKernel(k_is3_baseline);
  clReleaseKernel(k_is3_gmem);

  clReleaseKernel(k_is4_baseline);
  clReleaseKernel(k_is4_gmem);

  clReleaseKernel(k_is5);
  clReleaseKernel(k_pv);

  clReleaseMemObject(m_wg_key_buff_ptr);

  if(!split_flag || !buffering_flag){
    clReleaseMemObject(m_key_array[0]);

    clReleaseMemObject(m_key_buff2[0]);
  }
  else {
    clReleaseMemObject(m_key_array[0]);
    clReleaseMemObject(m_key_array[1]);

    clReleaseMemObject(m_key_buff2[0]);
    clReleaseMemObject(m_key_buff2[1]);

  }
  free(m_key_array);
  free(m_key_buff2);

  clReleaseMemObject(m_key_buff1);

  clReleaseMemObject(m_partial_verify_vals);
  clReleaseMemObject(m_test_rank_array);
  clReleaseMemObject(m_test_index_array);
  clReleaseMemObject(m_passed_verification);
  clReleaseMemObject(m_failed);

  free(wg_key_buff_ptr);


  clReleaseProgram(p_is);

  clReleaseCommandQueue(cmd_queue_kernel);
  if(buffering_flag) clReleaseCommandQueue(cmd_queue_data);

  clReleaseContext(context);

  clu_ProfilerRelease();
}

void get_pv_result(int iterations)
{

  cl_int ecode;
  int i, t;

  ecode = clEnqueueReadBuffer(cmd_queue_kernel, 
                              m_passed_verification, 
                              CL_FALSE,
                              0, 
                              sizeof(int)*TEST_ARRAY_SIZE*MAX_ITERATIONS, 
                              g_passed_verification, 
                              0, NULL, &ev_rb_pass);
  clu_CheckError(ecode, "clEnqueueReadBuffer for m_passed_verification");

  ecode = clEnqueueReadBuffer(cmd_queue_kernel, 
                              m_failed, 
                              CL_TRUE,
                              0, 
                              sizeof(int)*TEST_ARRAY_SIZE*MAX_ITERATIONS, 
                              g_failed, 
                              0, NULL, &ev_rb_fail);
  clu_CheckError(ecode, "clEnqueueReadBuffer for m_failed");

  for (i = 1; i <= iterations; i++) {
    for (t = 0; t < TEST_ARRAY_SIZE; t++) {
      passed_verification += g_passed_verification[(i-1)*TEST_ARRAY_SIZE + t];

      if (g_failed[(i-1)*TEST_ARRAY_SIZE + t]) {
        printf( "Failed partial verification: "
            "iteration %d, test key %d\n", 
            i, (int)t);
      }
    }
  }

  clReleaseEvent(ev_rb_pass);
  clReleaseEvent(ev_rb_fail);
}

static void print_opt_level(enum OptLevel ol)
{
  switch (ol) {
    case OPT_BASELINE:
      DETAIL_LOG("Optimization level 0 (base line)");
      break;
    case OPT_PARALLEL:
      DETAIL_LOG("Optimization level 1 (parallelization)");
      break;
    case OPT_GLOBALMEM:
      DETAIL_LOG("Optimization level 2 (global mem opt)");
      break;
    case OPT_MEMLAYOUT:
      DETAIL_LOG("Optimization level 3 (mem layout opt)");
      break;
    case OPT_SYNC:
      DETAIL_LOG("Optimization level 4 (synch opt)");
      break;
    case OPT_FULL:
      DETAIL_LOG("Optimization level 5 (full opt)");
      break;
  }
}
