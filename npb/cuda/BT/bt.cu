//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB BT code. This CUDA® C  //
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

//---------------------------------------------------------------------
// program BT
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <assert.h>

extern "C" {
#include "print_results.h"
}

#include "header.h"
#include "cuda_util.h"

//---------------------------------------------------------------------
// CUDA Variables
//---------------------------------------------------------------------

/* CUDA environment variables */
int             device;
cudaDeviceProp  deviceProp;
cudaStream_t    cmd_q[NUM_Q];

double        *m_us, 
              *m_vs, 
              *m_ws, 
              *m_qs, 
              *m_rho_i, 
              *m_square,
              *m_forcing[2], 
              *m_u[2], 
              *m_rhs[2],
              *m_lhsA, 
              *m_lhsB, 
              *m_lhsC;

double        *m_fjac,
              *m_njac,
              *m_lhs;

/* CUDA events */
cudaEvent_t   *loop1_ev_wb_start, 
              *loop1_ev_wb_end,
              *loop1_ev_rb_start, 
              *loop1_ev_rb_end,
              *loop2_ev_wb_start, 
              *loop2_ev_wb_end,
              *loop2_ev_rb_start, 
              *loop2_ev_rb_end,
              (*loop2_ev_kernel_add)[2];

/* CUDA dynamic configuration flags */
int           split_flag,
              buffering_flag;

/* CUDA optimization level */
OptLevel      g_opt_level;

/* CUDA device dependent variables */
size_t        max_work_item_sizes[3],
              max_work_group_size;
unsigned long local_mem_size, 
              max_mem_alloc_size, 
              gmem_size, 
              temp;
int           work_num_item_default,
              work_num_item_default_j;
int           loop1_work_max_iter, 
              loop1_work_num_item_default,
              loop2_work_max_iter, 
              loop2_work_num_item_default;

static void setup_cuda(int argc, char *argv[]);
static void release_cuda();
static void print_opt_level(enum OptLevel ol);

//---------------------------------------------------------------------


/* common /global/ */
double elapsed_time;
int grid_points[3];
logical timeron;

/* common /constants/ */
double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
     dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
     dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
     ce[5][13], dxmax, dymax, dzmax, xxcon1, xxcon2, 
     xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
     dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
     yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
     zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
     dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
     dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
     c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1,
     dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
     c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
     c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

// to improve cache performance, grid dimensions padded by 1 
// for even number sizes only.
/* common /fields/ */
double us     [KMAX][JMAXP+1][IMAXP+1];
double vs     [KMAX][JMAXP+1][IMAXP+1];
double ws     [KMAX][JMAXP+1][IMAXP+1];
double qs     [KMAX][JMAXP+1][IMAXP+1];
double rho_i  [KMAX][JMAXP+1][IMAXP+1];
double square [KMAX][JMAXP+1][IMAXP+1];
double forcing[KMAX][JMAXP+1][IMAXP+1][5];
double u      [KMAX][JMAXP+1][IMAXP+1][5];
double rhs    [KMAX][JMAXP+1][IMAXP+1][5];

/* common /work_1d/ */
double cuf[PROBLEM_SIZE+1];
double q  [PROBLEM_SIZE+1];
double ue [PROBLEM_SIZE+1][5];
double buf[PROBLEM_SIZE+1][5];


/* common /work_lhs/ */
double fjac[PROBLEM_SIZE+1][5][5];
double njac[PROBLEM_SIZE+1][5][5];
double lhs [PROBLEM_SIZE+1][3][5][5];
double tmp1, tmp2, tmp3;

int main(int argc, char *argv[])
{
  int i, niter, step;
  double navg, mflops, n3;

  double tmax, t, trecs[t_last+1];
  logical verified;
  char Class;
  const char *t_names[t_last+1];

  size_t u_size       = sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5;
  size_t forcing_size = sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5;
  size_t rhs_size     = sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1)*5;
  size_t qs_size      = sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1);
  size_t square_size  = sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1);
  size_t rho_i_size   = sizeof(double)*KMAX*(JMAXP+1)*(IMAXP+1);

  //---------------------------------------------------------------------
  // Root node reads input file (if it exists) else takes
  // defaults from parameters
  //---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_xsolve] = "xsolve";
    t_names[t_ysolve] = "ysolve";
    t_names[t_zsolve] = "zsolve";
    t_names[t_rdis1] = "redist1";
    t_names[t_rdis2] = "redist2";
    t_names[t_add] = "add";
    fclose(fp);
  } else {
    timeron = false;
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-CUDA) - BT Benchmark\n\n");

  if ((fp = fopen("inputbt.data", "r")) != NULL) {
    int result;
    printf(" Reading from input file inputbt.data\n");
    result = fscanf(fp, "%d", &niter);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &dt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d\n", 
        &grid_points[0], &grid_points[1], &grid_points[2]);
    fclose(fp);
  } else {
    printf(" No input file inputbt.data. Using compiled defaults\n");
    niter = NITER_DEFAULT;
    dt    = DT_DEFAULT;
    grid_points[0] = PROBLEM_SIZE;
    grid_points[1] = PROBLEM_SIZE;
    grid_points[2] = PROBLEM_SIZE;
  }

  printf(" Size: %4dx%4dx%4d\n",
      grid_points[0], grid_points[1], grid_points[2]);
  printf(" Iterations: %4d       dt: %11.7f\n", niter, dt);
  printf("\n");

  if ( (grid_points[0] > IMAX) ||
      (grid_points[1] > JMAX) ||
      (grid_points[2] > KMAX) ) {
    printf(" %d, %d, %d\n", grid_points[0], grid_points[1], grid_points[2]);
    printf(" Problem size too big for compiled array sizes\n");
    return 0;
  }
  set_constants();

  setup_cuda(argc, argv);

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_clear(t_memcpy_pre);
  timer_clear(t_memcpy_post);

  initialize();

  exact_rhs();


  //---------------------------------------------------------------------
  // do one time step to touch all code, and reinitialize
  //---------------------------------------------------------------------

  if (!split_flag) {
    CUCHK(cudaMemcpyAsync(m_u[0], u, u_size,
                          cudaMemcpyHostToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_forcing[0], forcing, forcing_size,
                          cudaMemcpyHostToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  adi();

  CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));

  if (!split_flag) {
    CUCHK(cudaMemcpyAsync(qs, m_qs, qs_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(square, m_square, square_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(rho_i, m_rho_i, rho_i_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(rhs, m_rhs[0], rhs_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(u, m_u[0], u_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  initialize();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_clear(t_memcpy_pre);
  timer_clear(t_memcpy_post);


  if (!split_flag) {  
    CUCHK(cudaMemcpyAsync(m_u[0], u, u_size,
                          cudaMemcpyHostToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(m_forcing[0], forcing, forcing_size,
                          cudaMemcpyHostToDevice, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  timer_start(1);
  cuda_ProfilerStart();

  for (step = 1; step <= niter; step++) {
    if ((step % 20) == 0 || step == 1)
      printf(" Time step %4d\n", step);

    adi();
  }

  cuda_ProfilerStop();
  timer_stop(1);
  tmax = timer_read(1);

  if (!split_flag) {
    CUCHK(cudaMemcpyAsync(rhs, m_rhs[0], rhs_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaMemcpyAsync(u, m_u[0], u_size,
                          cudaMemcpyDeviceToHost, 
                          cmd_q[KERNEL_Q]));

    CUCHK(cudaStreamSynchronize(cmd_q[KERNEL_Q]));
  }

  verify(niter, &Class, &verified);

  n3 = 1.0*grid_points[0]*grid_points[1]*grid_points[2];
  navg = (grid_points[0]+grid_points[1]+grid_points[2])/3.0;
  if(tmax != 0.0) {
    mflops = 1.0e-6 * (double)niter *
      (3478.8 * n3 - 17655.7 * (navg*navg) + 28023.7 * navg)
      / tmax;
  } else {
    mflops = 0.0;
  }
  c_print_results((char *)"BT", 
                  Class, 
                  grid_points[0], 
                  grid_points[1], 
                  grid_points[2], 
                  niter,
                  tmax, 
                  mflops, 
                  (char *)"          floating point", 
                  verified, 
                  (char *)NPBVERSION,
                  (char *)COMPILETIME, 
                  (char *)CS1, 
                  (char *)CS2, 
                  (char *)CS3, 
                  (char *)CS4, 
                  (char *)CS5, 
                  (char *)CS6, 
                  (char *)"(none)", 
                  deviceProp.name);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    for (i = 1; i <= t_last; i++) {
      trecs[i] = timer_read(i);
    }
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION   Time (secs)\n");
    for (i = 1; i <= t_last; i++) {
      printf("  %-8s:%9.3f  (%6.2f%%)\n", 
          t_names[i], trecs[i], trecs[i]*100./tmax);
      if (i == t_rhs) {
        t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
        t = trecs[t_rhs] - t;
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
      } else if (i==t_zsolve) {
        t = trecs[t_zsolve] - trecs[t_rdis1] - trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "sub-zsol", t, t*100./tmax);
      } else if (i==t_rdis2) {
        t = trecs[t_rdis1] + trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "redist", t, t*100./tmax);
      }
    }

    cuda_ProfilerPrintResult();
    cuda_ProfilerPrintElapsedTime("Memcpy for Data Transfer",
                                  timer_read(t_memcpy_pre) + timer_read(t_memcpy_post));
  }

  release_cuda();
  fflush(stdout);

  return 0;
}

// CUDA setup function
static void setup_cuda(int argc, char *argv[]) 
{
  cuda_ProfilerSetup();

  size_t forcing_buf_size, u_buf_size, rhs_buf_size,
         us_buf_size, vs_buf_size, ws_buf_size, qs_buf_size,
         rho_i_buf_size, square_buf_size,
         lhsA_buf_size, lhsB_buf_size, lhsC_buf_size,
         lhs_buf_size, njac_buf_size, fjac_buf_size;

  size_t forcing_slice_size = sizeof(double)*(JMAXP+1)*(IMAXP+1)*5;
  size_t u_slice_size       = sizeof(double)*(JMAXP+1)*(IMAXP+1)*5;
  size_t rhs_slice_size     = sizeof(double)*(JMAXP+1)*(IMAXP+1)*5;
  size_t us_slice_size      = sizeof(double)*(JMAXP+1)*(IMAXP+1);
  size_t vs_slice_size      = sizeof(double)*(JMAXP+1)*(IMAXP+1);
  size_t ws_slice_size      = sizeof(double)*(JMAXP+1)*(IMAXP+1);
  size_t qs_slice_size      = sizeof(double)*(JMAXP+1)*(IMAXP+1);
  size_t rho_i_slice_size   = sizeof(double)*(JMAXP+1)*(IMAXP+1);
  size_t square_slice_size  = sizeof(double)*(JMAXP+1)*(IMAXP+1);

  size_t lhsA_slice_size    = sizeof(double)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5;
  size_t lhsB_slice_size    = sizeof(double)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5;
  size_t lhsC_slice_size    = sizeof(double)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5;

  size_t lhs_slice_size     = sizeof(double)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*3*5*5;
  size_t njac_slice_size    = sizeof(double)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5;
  size_t fjac_slice_size    = sizeof(double)*(PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5;

  int c;
  char optimization_flag[1024];
  int opt_level_i = -1;

  while ((c = getopt(argc, argv, "o:")) != -1) {
    switch (c) {
      case 'o':
        memcpy(optimization_flag, optarg, 1024);
        opt_level_i = atoi(optimization_flag);
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
      break;
    case 2:
      g_opt_level = OPT_GLOBALMEM;
      fprintf(stderr, " ERROR : This App does NOT support partially optimized version for Opt. group 2\n");
      exit(EXIT_FAILURE);
      break;
    case 3:
      g_opt_level = OPT_MEMLAYOUT;
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
  CUCHK(cudaGetDevice(&device));
  CUCHK(cudaGetDeviceProperties(&deviceProp, device));

  max_work_item_sizes[0] = deviceProp.maxThreadsDim[0];
  max_work_item_sizes[1] = deviceProp.maxThreadsDim[1];
  max_work_item_sizes[2] = deviceProp.maxThreadsDim[2];

  max_work_group_size = deviceProp.maxThreadsPerBlock;

  local_mem_size = deviceProp.sharedMemPerBlock;

  gmem_size = deviceProp.totalGlobalMem;
  max_mem_alloc_size = deviceProp.totalGlobalMem / 4;

  //-----------------------------------------------------------------------
  // 3. Create a command queue
  //-----------------------------------------------------------------------
  CUCHK(cudaStreamCreate(&cmd_q[KERNEL_Q]));
  CUCHK(cudaStreamCreate(&cmd_q[DATA_Q]));

   //-----------------------------------------------------------------------
  // 4. Create buffers 
  //-----------------------------------------------------------------------
 
  // conservatively calculate available work item
  gmem_size *= 0.8;

  temp = gmem_size;
  temp /= sizeof(double);
  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_PARALLEL
      || g_opt_level == OPT_GLOBALMEM
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC)
  {
    // 15 + 6, 5*5*3 lhs, 5*5 njac, 5*5, fjac
    temp /= (21*(JMAXP+1)*(IMAXP+1) + (PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5*5);
  }
  else { 
    // only OPT_FULL
    // 15 + 6, 5*5*3(lhs)
    temp /= (21*(JMAXP+1)*(IMAXP+1) + (PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5*3); 
  }

  work_num_item_default = (int)temp;

  temp = max_mem_alloc_size / sizeof(double);
  temp /= ((PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5);

  work_num_item_default = min(work_num_item_default, (int)temp);
  work_num_item_default = min(work_num_item_default, JMAXP+1);

  // to fit the lhs size(z solve) 
  if (work_num_item_default == (JMAXP+1)) {
    work_num_item_default = KMAX;     
    split_flag = 0;
  }
  else 
    split_flag = 1;

  if (!split_flag) 
    buffering_flag = 0;
  else 
    buffering_flag = 1;

  if (buffering_flag) {
    // recalculate work_num_item_default for double buffering
    temp = gmem_size;
    temp /= sizeof(double);

    if (g_opt_level == OPT_BASELINE
        || g_opt_level == OPT_PARALLEL
        || g_opt_level == OPT_GLOBALMEM
        || g_opt_level == OPT_MEMLAYOUT
        || g_opt_level == OPT_SYNC)
    {
      // 15*2 + 6, 5*5*3(lhs), 5*5(njac), 5*5(fjac)
      temp /= (36*(JMAXP+1)*(IMAXP+1) + (PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5*5);
    }
    else {
      // 36 =  15*2 + 6, 5*5*3(lhs)
      temp /= (36*(JMAXP+1)*(IMAXP+1) + (PROBLEM_SIZE-1)*(PROBLEM_SIZE+1)*5*5*3);
    }

    work_num_item_default = (int)temp;

    if (work_num_item_default < 5) {
      printf(" GPU memory is too small \n");
      exit(EXIT_FAILURE);
    }
  }

  if (!split_flag) 
    work_num_item_default_j = JMAXP+1;
  else 
    work_num_item_default_j = work_num_item_default;
  // split : work_num_item_default_j = work_num_item default;
  // !split : work_num_item_default_j = JMAXP + 1; work_num_item_default = KMAX;
  // Thus, always work_num_item_default_j >= work_num_item_default

  forcing_buf_size  = forcing_slice_size;
  rhs_buf_size      = rhs_slice_size;
  u_buf_size        = u_slice_size;
  us_buf_size       = us_slice_size;
  vs_buf_size       = vs_slice_size;
  ws_buf_size       = ws_slice_size;
  qs_buf_size       = qs_slice_size;
  rho_i_buf_size    = rho_i_slice_size;
  square_buf_size   = square_slice_size;

  lhs_buf_size      = lhs_slice_size;
  njac_buf_size     = njac_slice_size;
  fjac_buf_size     = fjac_slice_size;

  lhsA_buf_size     = lhsA_slice_size;
  lhsB_buf_size     = lhsB_slice_size;
  lhsC_buf_size     = lhsC_slice_size;

  forcing_buf_size *= work_num_item_default;
  rhs_buf_size     *= work_num_item_default;
  u_buf_size       *= work_num_item_default;
  us_buf_size      *= work_num_item_default;
  vs_buf_size      *= work_num_item_default;
  ws_buf_size      *= work_num_item_default;
  qs_buf_size      *= work_num_item_default;
  rho_i_buf_size   *= work_num_item_default;
  square_buf_size  *= work_num_item_default;

  lhs_buf_size     *= work_num_item_default_j;
  njac_buf_size    *= work_num_item_default_j;
  fjac_buf_size    *= work_num_item_default_j;

  lhsA_buf_size    *= work_num_item_default_j;
  lhsB_buf_size    *= work_num_item_default_j;
  lhsC_buf_size    *= work_num_item_default_j;

  CUCHK(cudaMalloc(&m_forcing[0], forcing_buf_size));
  CUCHK(cudaMalloc(&m_u[0], u_buf_size));
  CUCHK(cudaMalloc(&m_rhs[0], rhs_buf_size));

  if (buffering_flag) {
    CUCHK(cudaMalloc(&m_forcing[1], forcing_buf_size));
    CUCHK(cudaMalloc(&m_u[1], u_buf_size));
    CUCHK(cudaMalloc(&m_rhs[1], rhs_buf_size));
  }

  CUCHK(cudaMalloc(&m_us, us_buf_size));
  CUCHK(cudaMalloc(&m_vs, vs_buf_size));
  CUCHK(cudaMalloc(&m_ws, ws_buf_size));
  CUCHK(cudaMalloc(&m_qs, qs_buf_size));
  CUCHK(cudaMalloc(&m_rho_i, rho_i_buf_size));
  CUCHK(cudaMalloc(&m_square, square_buf_size));

  // lhs Allocation
  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_GLOBALMEM
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC)
  {
    CUCHK(cudaMalloc(&m_lhs, lhs_buf_size));
  }
  else { // OPT_PARALLEL, OPT_FULL
    CUCHK(cudaMalloc(&m_lhsA, lhsA_buf_size));
    CUCHK(cudaMalloc(&m_lhsB, lhsB_buf_size));
    CUCHK(cudaMalloc(&m_lhsC, lhsC_buf_size));
  }

  // njac fjac allocation
  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_PARALLEL
      || g_opt_level == OPT_GLOBALMEM
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC)
  {
    CUCHK(cudaMalloc(&m_njac, njac_buf_size));
    CUCHK(cudaMalloc(&m_fjac, fjac_buf_size));
  }

  //-----------------------------------------------------------------------
  // 5. Build programs 
  //-----------------------------------------------------------------------

  loop1_work_num_item_default = (split_flag) ? (work_num_item_default-4) : work_num_item_default;
  loop1_work_max_iter = ( grid_points[2] - 1 ) / loop1_work_num_item_default + 1;

  loop2_work_num_item_default = work_num_item_default;
  loop2_work_max_iter = ( grid_points[1]-2 - 1 ) / loop2_work_num_item_default + 1;

  loop1_ev_wb_start = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop1_work_max_iter);
  loop1_ev_wb_end = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop1_work_max_iter);
  loop1_ev_rb_start = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop1_work_max_iter);
  loop1_ev_rb_end = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop1_work_max_iter);

  loop2_ev_wb_start = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop2_work_max_iter);
  loop2_ev_wb_end = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop2_work_max_iter);
  loop2_ev_rb_start = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop2_work_max_iter);
  loop2_ev_rb_end = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*loop2_work_max_iter);

  loop2_ev_kernel_add = (cudaEvent_t (*)[2])malloc(sizeof(cudaEvent_t)*2*loop2_work_max_iter);

  for (int i = 0; i < loop1_work_max_iter; ++i) {
    CUCHK(cudaEventCreate(&loop1_ev_wb_start[i]));
    CUCHK(cudaEventCreate(&loop1_ev_wb_end[i]));
    CUCHK(cudaEventCreate(&loop1_ev_rb_start[i]));
    CUCHK(cudaEventCreate(&loop1_ev_rb_end[i]));
  }

  for (int i = 0; i < loop2_work_max_iter; ++i) {
    CUCHK(cudaEventCreate(&loop2_ev_wb_start[i]));
    CUCHK(cudaEventCreate(&loop2_ev_wb_end[i]));
    CUCHK(cudaEventCreate(&loop2_ev_rb_start[i]));
    CUCHK(cudaEventCreate(&loop2_ev_rb_end[i]));

    CUCHK(cudaEventCreate(&loop2_ev_kernel_add[i][0]));
    CUCHK(cudaEventCreate(&loop2_ev_kernel_add[i][1]));
  }

  DETAIL_LOG("Tiling flag : %d", split_flag);

  adi_init();

  compute_rhs_init(loop1_work_max_iter);

  x_solve_init(loop1_work_max_iter);

  y_solve_init(loop1_work_max_iter);

  z_solve_init(loop2_work_max_iter);

}

static void release_cuda() 
{
  adi_free();

  compute_rhs_release(loop1_work_max_iter);

  x_solve_release(loop1_work_max_iter);

  y_solve_release(loop1_work_max_iter);

  z_solve_release(loop2_work_max_iter);

  if (buffering_flag) {
    cudaFree(m_forcing[1]);
    cudaFree(m_u[1]);
    cudaFree(m_rhs[1]);
  }
  cudaFree(m_forcing[0]);
  cudaFree(m_u[0]);
  cudaFree(m_rhs[0]);
  cudaFree(m_us);
  cudaFree(m_vs);
  cudaFree(m_ws);
  cudaFree(m_qs);
  cudaFree(m_rho_i);
  cudaFree(m_square);

  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_GLOBALMEM
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC)
  {
    cudaFree(m_lhs);
  }
  else {
    cudaFree(m_lhsA);
    cudaFree(m_lhsB);
    cudaFree(m_lhsC);
  }

  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_PARALLEL
      || g_opt_level == OPT_GLOBALMEM
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC)
  {
    cudaFree(m_njac);
    cudaFree(m_fjac);
  }

  cudaStreamDestroy(cmd_q[KERNEL_Q]);
  cudaStreamDestroy(cmd_q[DATA_Q]);

  for (int i = 0; i < loop1_work_max_iter; ++i) {
    CUCHK(cudaEventDestroy(loop1_ev_wb_start[i]));
    CUCHK(cudaEventDestroy(loop1_ev_wb_end[i]));
    CUCHK(cudaEventDestroy(loop1_ev_rb_start[i]));
    CUCHK(cudaEventDestroy(loop1_ev_rb_end[i]));
  }

  for (int i = 0; i < loop2_work_max_iter; ++i) {
    CUCHK(cudaEventDestroy(loop2_ev_wb_start[i]));
    CUCHK(cudaEventDestroy(loop2_ev_wb_end[i]));
    CUCHK(cudaEventDestroy(loop2_ev_rb_start[i]));
    CUCHK(cudaEventDestroy(loop2_ev_rb_end[i]));

    CUCHK(cudaEventDestroy(loop2_ev_kernel_add[i][0]));
    CUCHK(cudaEventDestroy(loop2_ev_kernel_add[i][1]));
  }

  free(loop1_ev_wb_start);
  free(loop1_ev_wb_end);
  free(loop1_ev_rb_start);
  free(loop1_ev_rb_end);

  free(loop2_ev_wb_start);
  free(loop2_ev_wb_end);
  free(loop2_ev_rb_start);
  free(loop2_ev_rb_end);

  free(loop2_ev_kernel_add);

  cuda_ProfilerRelease();
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
