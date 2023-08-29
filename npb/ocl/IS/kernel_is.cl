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

#if CLASS_ID == 0
#define CLASS 'S'
#elif CLASS_ID == 1
#define CLASS 'W'
#elif CLASS_ID == 2
#define CLASS 'A'
#elif CLASS_ID == 3
#define CLASS 'B'
#elif CLASS_ID == 4
#define CLASS 'C'
#elif CLASS_ID == 5
#define CLASS 'D'
#endif

#if CLASS == 'D'
typedef  long INT_TYPE;
#ifndef cl_khr_int64_base_atomics
#error "int64 base atomic operations are not supported by OpenCL implementation"
#else
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define atomic_inc(x) atom_inc(x)
#define atomic_add(x,y) atom_add(x,y)
#endif

#else
typedef  int  INT_TYPE;
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#endif

#define min(a,b) ((a<b)?a:b)

/*  Determine where the partial verify test keys are, load into  */
/*  top of array bucket_size                                     */
__kernel 
void k_pv_set(__global const INT_TYPE *test_index_array,
              __global INT_TYPE *key_array,
              __global INT_TYPE *partial_verify_vals,
              int iteration)
{
  int i;

  key_array[iteration] = iteration;
  key_array[iteration + MAX_ITERATIONS] = MAX_KEY - iteration;
  for (i = 0; i < TEST_ARRAY_SIZE; i++) 
    partial_verify_vals[i] = key_array[test_index_array[i]];
}

__kernel 
void k_is1(__global INT_TYPE * key_buff1)
{

    size_t i = get_global_id(0);
    if(i >= MAX_KEY) return;

/*  Clear the work array */
    key_buff1[i] = 0;
}

__kernel 
void k_is2(__global INT_TYPE * key_buff_ptr,
           __global INT_TYPE * key_buff_ptr2,
           INT_TYPE work_num_keys)
{

  int i = get_global_id(0);

  if (i >= work_num_keys)
    return;
  INT_TYPE key = key_buff_ptr2[i];
  atomic_inc(&key_buff_ptr[key]);

}



//prefix sum implementation
__kernel 
void k_is3_baseline(__global INT_TYPE *key_buff_ptr,
                    __global INT_TYPE *wg_key_buff_ptr,
                    int log_local_size)
{

  /*  To obtain ranks of each key, 
      successively add the individual key
      population
   */
  /*
     for( i=0; i<MAX_KEY-1; i++ )   
     key_buff_ptr[i+1] += key_buff_ptr[i];  
   */
  
  size_t i;
  size_t l_i = get_local_id(0);
  int l_size = get_local_size(0);
  int wg_id = get_group_id(0);
  size_t step;
  INT_TYPE temp;
  INT_TYPE prev_sum;

  size_t chunk = MAX_KEY / get_num_groups(0);
  size_t wg_start = wg_id * chunk;
  size_t wg_end = wg_start + chunk;
  size_t c;

  prev_sum = 0;

  for (c = wg_start; c < wg_end; c += l_size) {
    i = c + l_i;

    for (step = 0; step < log_local_size; step++) {
      temp = key_buff_ptr[i];

      if (l_i >= (1<<step))
        temp += key_buff_ptr[i - (1<<step)];

      barrier(CLK_GLOBAL_MEM_FENCE);
      key_buff_ptr[i] = temp;
      barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // temp for prev_sum
    temp = key_buff_ptr[c+l_size-1];
    barrier(CLK_GLOBAL_MEM_FENCE);

    key_buff_ptr[i] = key_buff_ptr[i] + prev_sum;

    barrier(CLK_GLOBAL_MEM_FENCE);
    prev_sum += temp;

  }

  if (l_i == 0)
    wg_key_buff_ptr[wg_id] = prev_sum;

}

__kernel 
void k_is3_gmem(__global INT_TYPE *key_buff_ptr,
                __global INT_TYPE *wg_key_buff_ptr,
                __local INT_TYPE *l_key_buff_ptr,
                int log_local_size)
{

  /*  To obtain ranks of each key, 
      successively add the individual key
      population
   */
  /*
     for( i=0; i<MAX_KEY-1; i++ )   
     key_buff_ptr[i+1] += key_buff_ptr[i];  
   */
  
  size_t i;
  size_t l_i = get_local_id(0);
  int l_size = get_local_size(0);
  int wg_id = get_group_id(0);
  size_t step;
  INT_TYPE temp;
  INT_TYPE prev_sum;

  size_t chunk = MAX_KEY / get_num_groups(0);
  size_t wg_start = wg_id * chunk;
  size_t wg_end = wg_start + chunk;
  size_t c;

  prev_sum = 0;

  for (c = wg_start; c < wg_end; c += l_size) {
    i = c + l_i;

    l_key_buff_ptr[l_i] = key_buff_ptr[i];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (step = 0; step < log_local_size; step++) {
      temp = l_key_buff_ptr[l_i];

      if (l_i >= (1<<step))
        temp += l_key_buff_ptr[l_i - (1<<step)];

      barrier(CLK_LOCAL_MEM_FENCE);
      l_key_buff_ptr[l_i] = temp;
      barrier(CLK_LOCAL_MEM_FENCE); 
    }

    key_buff_ptr[i] = l_key_buff_ptr[l_i] + prev_sum;
    prev_sum += l_key_buff_ptr[l_size-1];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (l_i == 0)
    wg_key_buff_ptr[wg_id] = prev_sum;

}

// this kernel is executed with only one work group
__kernel 
void k_is4_baseline(__global INT_TYPE *wg_key_buff_ptr,
                    int log_local_size,
                    INT_TYPE k_is3_wg)
{
  size_t l_i = get_local_id(0);
  size_t step;
  INT_TYPE temp;

  for (step = 0; step < log_local_size; step++) {
    if (l_i < k_is3_wg)
      temp = wg_key_buff_ptr[l_i];
    else
      temp = 0;

    if (l_i >= (1<<step) && (l_i - (1<<step) < k_is3_wg))
      temp += wg_key_buff_ptr[l_i - (1<<step)];

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (l_i < k_is3_wg)
      wg_key_buff_ptr[l_i] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel 
void k_is4_gmem(__global INT_TYPE *wg_key_buff_ptr,
                __local INT_TYPE *l_wg_key_buff_ptr,
                int log_local_size,
                INT_TYPE k_is3_wg)
{
  size_t l_i = get_local_id(0);
  size_t step;
  INT_TYPE temp;

  if (l_i < k_is3_wg)
    l_wg_key_buff_ptr[l_i] = wg_key_buff_ptr[l_i];
  else 
    l_wg_key_buff_ptr[l_i] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (step = 0; step < log_local_size; step++) {
    temp = l_wg_key_buff_ptr[l_i];

    if (l_i >= (1<<step))
      temp += l_wg_key_buff_ptr[l_i - (1<<step)];

    barrier(CLK_LOCAL_MEM_FENCE);
    l_wg_key_buff_ptr[l_i] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (l_i < k_is3_wg)
    wg_key_buff_ptr[l_i] = l_wg_key_buff_ptr[l_i];
}

__kernel 
void k_is5(__global INT_TYPE *wg_key_buff_ptr,
           __global INT_TYPE *key_buff_ptr)
{
  size_t wg_id = get_group_id(0);
  size_t chunk = MAX_KEY / (get_num_groups(0) + 1);
  size_t wg_start = (wg_id + 1)*chunk;
  size_t wg_end = wg_start + chunk;
  size_t l_i = get_local_id(0);
  size_t l_size = get_local_size(0);
  size_t c;
  size_t i;

  for (c = wg_start; c < wg_end; c += l_size) {
    i = c + l_i;

    if (i >= MAX_KEY)
      return;

    key_buff_ptr[i] += wg_key_buff_ptr[wg_id];
  }
}


/* This is the partial verify test section */
/* Observe that test_rank_array vals are   */
/* shifted differently for different cases */
__kernel 
void k_pv(__global INT_TYPE *partial_verify_vals,
          __global INT_TYPE *key_buff_ptr,
          __global const INT_TYPE *test_rank_array,
          __global int *m_passed_verification,
          __global int *m_failed,
          int iteration)
{
  int i = get_global_id(0);
  INT_TYPE k;
  int passed_verification = 0;
  int failed = 0;

  /* This is the partial verify test section */
  /* Observe that test_rank_array vals are   */
  /* shifted differently for different cases */
  k = partial_verify_vals[i];          /* test vals were put here */
  if( 0 < k  &&  k <= NUM_KEYS-1 )
  {
    INT_TYPE key_rank = key_buff_ptr[k-1];

    switch( CLASS )
    {
      case 'S':
        {
          if( i <= 2 )
          {
            if( key_rank != test_rank_array[i]+iteration )
              failed = 1;
            else
              passed_verification++;
          }
          else
          {
            if( key_rank != test_rank_array[i]-iteration )
              failed = 1;
            else
              passed_verification++;
          }
          break;
        }
      case 'W':
        if( i < 2 )
        {
          if( key_rank != test_rank_array[i]+(iteration-2) )
            failed = 1;
          else
            passed_verification++;
        }
        else
        {
          if( key_rank != test_rank_array[i]-iteration )
            failed = 1;
          else
            passed_verification++;
        }
        break;
      case 'A':
        if( i <= 2 )
        {
          if( key_rank != test_rank_array[i]+(iteration-1) )
            failed = 1;
          else
            passed_verification++;
        }
        else
        {
          if( key_rank != test_rank_array[i]-(iteration-1) )
            failed = 1;
          else
            passed_verification++;
        }
        break;
      case 'B':
        if( i == 1 || i == 2 || i == 4 )
        {
          if( key_rank != test_rank_array[i]+iteration )
            failed = 1;
          else
            passed_verification++;
        }
        else
        {
          if( key_rank != test_rank_array[i]-iteration )
            failed = 1;
          else
            passed_verification++;
        }
        break;
      case 'C':
        if( i <= 2 )
        {
          if( key_rank != test_rank_array[i]+iteration )
            failed = 1;
          else
            passed_verification++;
        }
        else
        {
          if( key_rank != test_rank_array[i]-iteration )
            failed = 1;
          else
            passed_verification++;
        }
        break;
      case 'D':
        if( i < 2 )
        {
          if( key_rank != test_rank_array[i]+iteration )
            failed = 1;
          else
            passed_verification++;
        }
        else
        {
          if( key_rank != test_rank_array[i]-iteration )
            failed = 1;
          else
            passed_verification++;
        }
        break;
    }
  }

  m_passed_verification[i + (iteration-1)*TEST_ARRAY_SIZE] = passed_verification;
  m_failed[i + (iteration-1)*TEST_ARRAY_SIZE] = failed;

}
