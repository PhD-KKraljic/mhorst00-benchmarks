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

//---------------------------------------------------------------------
//   program applu
//---------------------------------------------------------------------

//---------------------------------------------------------------------
//
//   driver for the performance evaluation of the solver for
//   five coupled parabolic/elliptic partial differential equations.
//
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <getopt.h>
#include <assert.h>

#include "applu.incl"
extern "C"{
#include "timers.h"
#include "print_results.h"
}

//---------------------------------------------------------------------
// grid
//---------------------------------------------------------------------
/* common/cgcon/ */
double dxi, deta, dzeta;
double tx1, tx2, tx3;
double ty1, ty2, ty3;
double tz1, tz2, tz3;
int nx, ny, nz;
int nx0, ny0, nz0;
int ist, iend;
int jst, jend;
int ii1, ii2;
int ji1, ji2;
int ki1, ki2;

//---------------------------------------------------------------------
// dissipation
//---------------------------------------------------------------------
/* common/disp/ */
double dx1, dx2, dx3, dx4, dx5;
double dy1, dy2, dy3, dy4, dy5;
double dz1, dz2, dz3, dz4, dz5;
double dssp;

//---------------------------------------------------------------------
// field variables and residuals
// to improve cache performance, second two dimensions padded by 1 
// for even number sizes only.
// Note: corresponding array (called "v") in routines blts, buts, 
// and l2norm are similarly padded
//---------------------------------------------------------------------
/* common/cvar/ */
double u    [ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1][5];
double rsd  [ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1][5];
double frct [ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1][5];
double flux [ISIZ1][5];
double qs   [ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double rho_i[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];

//---------------------------------------------------------------------
// output control parameters
//---------------------------------------------------------------------
/* common/cprcon/ */
int ipr, inorm;

//---------------------------------------------------------------------
// newton-raphson iteration control parameters
//---------------------------------------------------------------------
/* common/ctscon/ */
double dt, omega, tolrsd[5], rsdnm[5], errnm[5], frc, ttotal;
int itmax, invert;

/* common/cjac/ */
double a[ISIZ2][ISIZ1/2*2+1][5][5];
double b[ISIZ2][ISIZ1/2*2+1][5][5];
double c[ISIZ2][ISIZ1/2*2+1][5][5];
double d[ISIZ2][ISIZ1/2*2+1][5][5];

/* common/cjacu/ */
double au[ISIZ2][ISIZ1/2*2+1][5][5];
double bu[ISIZ2][ISIZ1/2*2+1][5][5];
double cu[ISIZ2][ISIZ1/2*2+1][5][5];
double du[ISIZ2][ISIZ1/2*2+1][5][5];


//---------------------------------------------------------------------
// coefficients of the exact solution
//---------------------------------------------------------------------
/* common/cexact/ */
double ce[5][13];


//---------------------------------------------------------------------
// pintgr() - segmentation fault
//---------------------------------------------------------------------
double phi1[ISIZ3+2][ISIZ2+2];
double phi2[ISIZ3+2][ISIZ2+2];


//---------------------------------------------------------------------
// timers
//---------------------------------------------------------------------
/* common/timer/ */
double maxtime;
logical timeron;




//---------------------------------------------------------------------
// CUDA variables
//---------------------------------------------------------------------

/* CUDA environment variables */
int                 device;
cudaDeviceProp      deviceProp;
cudaStream_t        cmd_q[NUM_Q];

/* CUDA memory objects and sizes of them */
double              *m_sum1, 
                    *m_sum2, 
                    *m_u_prev,
                    *m_r_prev,
                    *m_rsd[2], 
                    *m_u[2], 
                    *m_frct[2], 
                    *m_qs[2], 
                    *m_rho_i[2];

/* CUDA memory objects for baseline optimization */ 
double              *m_flux,
                    *m_utmp,
                    *m_rtmp,
                    *m_tmp_sum,
                    *m_a,
                    *m_b,
                    *m_c,
                    *m_d;

size_t              u_buf_size,
                    rsd_buf_size,
                    frct_buf_size,
                    qs_buf_size,
                    rho_i_buf_size,
                    u_prev_buf_size,
                    r_prev_buf_size;

const size_t        u_slice_size        = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                    rsd_slice_size      = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                    frct_slice_size     = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                    qs_slice_size       = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1),
                    rho_i_slice_size    = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1),
                    u_prev_slice_size   = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*6,
                    r_prev_slice_size   = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5;

/* baseline memory objects sizes */
const size_t        flux_slice_size     = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                    utmp_slice_size     = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*6,
                    rtmp_slice_size     = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                    tmp_sum_slice_size  = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5,
                    a_slice_size        = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5*5,
                    b_slice_size        = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5*5,
                    c_slice_size        = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5*5,
                    d_slice_size        = sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5*5;
size_t
                    flux_buf_size,
                    utmp_buf_size,
                    rtmp_buf_size,
                    tmp_sum_buf_size,
                    a_buf_size,
                    b_buf_size,
                    c_buf_size,
                    d_buf_size;

/* CUDA dynamic configuration flags */
int                 split_flag = 0;
int                 buffering_flag;

/* CUDA device dependent variables */
int                 max_compute_units;
int                 max_work_item_sizes[3];
size_t              max_work_group_size;
size_t              local_mem_size;
size_t              gmem_size;
size_t              max_mem_alloc_size;
size_t              l2norm_lws, 
                    l2norm_gws, 
                    l2norm_wg_num,
                    rhsx_lws[2],
                    rhsy_lws[2], 
                    rhsz_lws[2],
                    jacld_blts_lws, 
                    jacu_buts_lws;
int                 work_num_item_default; 
int                 loop1_work_max_iter, loop2_work_max_iter;
size_t              loop1_work_num_item_default, loop2_work_num_item_default;

/* Wave propagation variables */
int                 block_size, block_size_k;

/* Reduction variables */
double              (* g_sum1)[5], (* g_sum2)[5];

/* CUDA optimization level */
enum OptLevel       g_opt_level;

static void setup_cuda(int argc, char *argv[]);
static void release_cuda();
void print_opt_level(enum OptLevel ol);

int main(int argc, char *argv[])
{
  char Class;
  logical verified;
  double mflops;

  double t, tmax, trecs[t_last+1];
  int i;
  const char *t_names[t_last+1];

  //---------------------------------------------------------------------
  // Setup info for timers
  //---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_jacld] = "jacld";
    t_names[t_blts] = "blts";
    t_names[t_jacu] = "jacu";
    t_names[t_buts] = "buts";
    t_names[t_add] = "add";
    t_names[t_l2norm] = "l2norm";
    fclose(fp);
  } else {
    timeron = false;
  }



  //---------------------------------------------------------------------
  // read input data
  //---------------------------------------------------------------------
  read_input();

  //---------------------------------------------------------------------
  // set up domain sizes
  //---------------------------------------------------------------------
  domain();

  //---------------------------------------------------------------------
  // set up coefficients
  //---------------------------------------------------------------------
  setcoeff();

  //---------------------------------------------------------------------
  // set the boundary values for dependent variables
  //---------------------------------------------------------------------
  setbv();

  //---------------------------------------------------------------------
  // set the initial values for dependent variables
  //---------------------------------------------------------------------
  setiv();

  //---------------------------------------------------------------------
  // compute the forcing term based on prescribed exact solution
  //---------------------------------------------------------------------
  erhs();


  //---------------------------------------------------------------------
  // setup cuda environments
  //---------------------------------------------------------------------
  setup_cuda(argc, argv);

  //---------------------------------------------------------------------
  // perform one SSOR iteration to touch all data pages
  //---------------------------------------------------------------------
  ssor(1);

  //---------------------------------------------------------------------
  // reset the boundary and initial values
  //---------------------------------------------------------------------
  setbv();
  setiv();

  //---------------------------------------------------------------------
  // perform the SSOR iterations
  //---------------------------------------------------------------------
  ssor(itmax);

  //---------------------------------------------------------------------
  // compute the solution error
  //---------------------------------------------------------------------
  error();

  //---------------------------------------------------------------------
  // compute the surface integral
  //---------------------------------------------------------------------
  pintgr();

  //---------------------------------------------------------------------
  // verification test
  //---------------------------------------------------------------------
  verify ( rsdnm, errnm, frc, &Class, &verified );
  mflops = (double)itmax * (1984.77 * (double)nx0
      * (double)ny0
      * (double)nz0
      - 10923.3 * pow(((double)(nx0+ny0+nz0)/3.0), 2.0) 
      + 27770.9 * (double)(nx0+ny0+nz0)/3.0
      - 144010.0)
    / (maxtime*1000000.0);

  c_print_results((char *)"LU", 
                  Class, 
                  nx0,
                  ny0, 
                  nz0, 
                  itmax,
                  maxtime, 
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
    tmax = maxtime;
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION     Time (secs)\n");
    for (i = 1; i <= t_last; i++) {
      printf("  %-8s:%9.3f  (%6.2f%%)\n",
          t_names[i], trecs[i], trecs[i]*100./tmax);
      if (i == t_rhs) {
        t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
        t = trecs[i] - t;
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
      }
    }

    cuda_ProfilerPrintResult();
  }

  release_cuda();

  return 0;
}

static void setup_cuda(int argc, char *argv[])
{
  cuda_ProfilerSetup();

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
      break;
    case 3:
      g_opt_level = OPT_MEMLAYOUT;
      fprintf(stderr, " ERROR : This App does NOT support partially optimized version for Opt. group 3\n");
      exit(EXIT_FAILURE);
      break;
    case 4:
      g_opt_level = OPT_SYNC;
      break;
    default:
      g_opt_level = OPT_FULL;
      break;
  }

  print_opt_level(g_opt_level);

  //-----------------------------------------------------------------------
  // 1. Find the default device type and get a device for the device type
  //-----------------------------------------------------------------------
  int devCount;
  CUCHK(cudaGetDeviceCount(&devCount));
  CUCHK(cudaGetDevice(&device));

  CUCHK(cudaGetDeviceProperties(&deviceProp, device));

  max_work_group_size = deviceProp.maxThreadsPerBlock;
  max_mem_alloc_size = deviceProp.totalGlobalMem / 4;
  gmem_size = deviceProp.totalGlobalMem;

  max_work_item_sizes[0] = deviceProp.maxThreadsDim[0];
  max_work_item_sizes[1] = deviceProp.maxThreadsDim[1];
  max_work_item_sizes[2] = deviceProp.maxThreadsDim[2];

  max_compute_units = deviceProp.multiProcessorCount;
  local_mem_size = deviceProp.sharedMemPerBlock;

  //-----------------------------------------------------------------------
  // 2. Create a context for the specified device
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // 3. Create a command queue
  //-----------------------------------------------------------------------
  CUCHK(cudaStreamCreate(&cmd_q[KERNEL_Q]));
  CUCHK(cudaStreamCreate(&cmd_q[DATA_Q]));

  //-----------------------------------------------------------------------
  // 4. Create buffers 
  //-----------------------------------------------------------------------

  l2norm_lws = max_work_group_size;
  l2norm_lws = min(max_work_group_size, local_mem_size/(sizeof(double)*5));
  size_t cnt = 0;
  size_t temp = l2norm_lws;
  while(temp > 1){
    temp = temp >> 1;
    cnt++;
  }
  l2norm_lws = 1 << cnt;

  l2norm_gws = nz*(jend - jst)*(iend - ist);
  l2norm_gws = (((l2norm_gws - 1)/l2norm_lws) + 1)*l2norm_lws;
  l2norm_wg_num = l2norm_gws / l2norm_lws;

  // the size of m_sum is hard coded with l2norm function
  CUCHK(cudaMalloc(&m_sum1, sizeof(double)*5*l2norm_wg_num));
  CUCHK(cudaMalloc(&m_sum2, sizeof(double)*5*l2norm_wg_num));

  g_sum1 = (double (*)[5])malloc(sizeof(double)*5*l2norm_wg_num);
  g_sum2 = (double (*)[5])malloc(sizeof(double)*5*l2norm_wg_num);

  // Only Use 80% of GPU memory 
  gmem_size *= 0.8;

  unsigned long long int avail_gmem;
  unsigned long long int max_alloc_item;


  max_alloc_item = max_mem_alloc_size;
  max_alloc_item /= sizeof(double);
  // conservatively calculate the number of maximum work items
  max_alloc_item /= (ISIZ2/2*2+1)*(ISIZ1/2*2+1)*5;



  // check if splitting needs
  avail_gmem = gmem_size - sizeof(double)*5*l2norm_wg_num*2;
  if (g_opt_level == OPT_BASELINE 
      || g_opt_level == OPT_PARALLEL
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC) {  
    // flux, utmp, rtmp, tmp_sum, a, b, c, d
    avail_gmem /= sizeof(double)*((ISIZ2/2*2+1)*(ISIZ1/2*2+1)*138);
  }
  else
    avail_gmem /= sizeof(double)*((ISIZ2/2*2+1)*(ISIZ1/2*2+1)*17); 

  work_num_item_default = (int)avail_gmem;
  work_num_item_default = min(work_num_item_default, (int)max_alloc_item);
  work_num_item_default = min(work_num_item_default, ISIZ3);

  if(work_num_item_default == ISIZ3) 
    split_flag = 0;
  else 
    split_flag = 1;

  if (!split_flag) 
    buffering_flag = 0;
  else 
    buffering_flag = 1;

  if (split_flag) {
    avail_gmem = gmem_size - sizeof(double)*5*l2norm_wg_num*2;
    avail_gmem -= sizeof(double)*(ISIZ2/2*2+1)*(ISIZ1/2*2+1)*2*11; // m_u_prev, m_r_prev;
    if (g_opt_level == OPT_BASELINE
        || g_opt_level == OPT_PARALLEL
        || g_opt_level == OPT_MEMLAYOUT
        || g_opt_level == OPT_SYNC) { 
      // flux, utmp, rtmp, tmp_sum, a, b, c, d
      avail_gmem /= sizeof(double)*((ISIZ2/2*2+1)*(ISIZ1/2*2+1)*138);
    }
    else
      avail_gmem /= sizeof(double)*((ISIZ2/2*2+1)*(ISIZ1/2*2+1)*17); 

    work_num_item_default = (int)avail_gmem;
    work_num_item_default = min(work_num_item_default, (int)max_alloc_item);

    // for double buffering
    if (buffering_flag)
      work_num_item_default /= 2;
  }

  if (work_num_item_default <= 5) {
    fprintf(stderr, "GPU Memory is too small\n");
  }
  assert(work_num_item_default > 5);

  rsd_buf_size      = rsd_slice_size;
  u_buf_size        = u_slice_size;
  frct_buf_size     = frct_slice_size;
  qs_buf_size       = qs_slice_size;
  rho_i_buf_size    = rho_i_slice_size;
  u_prev_buf_size   = u_prev_slice_size;
  r_prev_buf_size   = r_prev_slice_size;

  rsd_buf_size      *= work_num_item_default;
  u_buf_size        *= work_num_item_default;
  frct_buf_size     *= work_num_item_default;
  qs_buf_size       *= work_num_item_default;
  rho_i_buf_size    *= work_num_item_default;
  u_prev_buf_size   *= 2;
  r_prev_buf_size   *= 2;

  flux_buf_size     = flux_slice_size * work_num_item_default;
  utmp_buf_size     = utmp_slice_size * work_num_item_default;
  rtmp_buf_size     = rtmp_slice_size * work_num_item_default;
  tmp_sum_buf_size  = tmp_sum_slice_size * work_num_item_default;
  a_buf_size        = a_slice_size * work_num_item_default;
  b_buf_size        = b_slice_size * work_num_item_default;
  c_buf_size        = c_slice_size * work_num_item_default;
  d_buf_size        = d_slice_size * work_num_item_default;

  /* Memory allocation */

  // for double buffering
  if (buffering_flag) {
    CUCHK(cudaMalloc(&m_rsd[1], rsd_buf_size));
    CUCHK(cudaMalloc(&m_u[1], u_buf_size));
    CUCHK(cudaMalloc(&m_frct[1], frct_buf_size));
    CUCHK(cudaMalloc(&m_qs[1], qs_buf_size));
    CUCHK(cudaMalloc(&m_rho_i[1], rho_i_buf_size));
  }

  CUCHK(cudaMalloc(&m_rsd[0], rsd_buf_size));
  CUCHK(cudaMalloc(&m_u[0], u_buf_size));
  CUCHK(cudaMalloc(&m_frct[0], frct_buf_size));
  CUCHK(cudaMalloc(&m_qs[0], qs_buf_size));
  CUCHK(cudaMalloc(&m_rho_i[0], rho_i_buf_size));

  if (split_flag) {
    CUCHK(cudaMalloc(&m_u_prev, sizeof(double)*2*(ISIZ1/2*2+1)*(ISIZ2/2*2+1)*6));
    CUCHK(cudaMalloc(&m_r_prev, sizeof(double)*2*(ISIZ1/2*2+1)*(ISIZ2/2*2+1)*5));
  }

  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_PARALLEL
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC) 
  {
    CUCHK(cudaMalloc(&m_flux, flux_buf_size));
    CUCHK(cudaMalloc(&m_utmp, utmp_buf_size));
    CUCHK(cudaMalloc(&m_rtmp, rtmp_buf_size));
    CUCHK(cudaMalloc(&m_tmp_sum, tmp_sum_buf_size));
    CUCHK(cudaMalloc(&m_a, a_buf_size));
    CUCHK(cudaMalloc(&m_b, b_buf_size));
    CUCHK(cudaMalloc(&m_c, c_buf_size));
    CUCHK(cudaMalloc(&m_d, d_buf_size));
  }


  //-----------------------------------------------------------------------
  // 5. Build programs 
  //-----------------------------------------------------------------------
  rhsx_lws[1] = 1;
  rhsx_lws[0] = min(jend-jst, max_work_item_sizes[0]);
  rhsx_lws[0] = min(rhsx_lws[0], max_work_group_size);

  rhsy_lws[1] = 1;
  rhsy_lws[0] = min(iend-ist, max_work_item_sizes[0]);
  rhsy_lws[0] = min(rhsy_lws[0], max_work_group_size);

  rhsz_lws[1] = 1;
  rhsz_lws[0] = min(iend-ist, max_work_item_sizes[0]);
  rhsz_lws[0] = min(rhsz_lws[0], max_work_group_size);


  // set wave propagation parameters
  if (ISIZ1 <= (int)sqrt(max_work_group_size))
    block_size = ISIZ1;
  else
    block_size = 8;

  block_size_k = block_size;
  jacld_blts_lws = block_size * block_size;
  jacu_buts_lws = jacld_blts_lws;

  loop1_work_num_item_default = (split_flag) ? (work_num_item_default - 1) : work_num_item_default;
  loop1_work_max_iter = (nz-2 - 1)/loop1_work_num_item_default + 1;

  loop2_work_num_item_default = (split_flag) ? (work_num_item_default - 4) : work_num_item_default; 
  loop2_work_max_iter = (nz-1)/loop2_work_num_item_default + 1;

  //-----------------------------------------------------------------------
  // 6. Create kernels 
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // 7. Create Events 
  //----------------------------------------------------------------------- 

  ssor_init(loop1_work_max_iter, loop2_work_max_iter);

  jacld_blts_init(loop1_work_max_iter, 
                  loop1_work_num_item_default,
                  block_size_k, 
                  block_size);

  l2norm_init(loop2_work_max_iter);

  rhs_init(loop2_work_max_iter);

  jacu_buts_init(loop2_work_max_iter, 
                 loop2_work_num_item_default,
                 block_size_k, 
                 block_size);

  DETAIL_LOG("Tiling flag : %d", split_flag);
}

static void release_cuda()
{

  cudaFree(m_sum1);
  cudaFree(m_sum2);

  free(g_sum1);
  free(g_sum2);

  // release
  ssor_release(loop1_work_max_iter, loop2_work_max_iter);
  jacld_blts_release(loop1_work_max_iter);

  l2norm_release(loop2_work_max_iter);
  rhs_release(loop2_work_max_iter);
  jacu_buts_release(loop2_work_max_iter);

  if (buffering_flag) {
    cudaFree(m_rsd[1]);
    cudaFree(m_u[1]);
    cudaFree(m_frct[1]);
    cudaFree(m_qs[1]);
    cudaFree(m_rho_i[1]);
  }

  cudaFree(m_rsd[0]);
  cudaFree(m_u[0]);
  cudaFree(m_frct[0]);
  cudaFree(m_qs[0]);
  cudaFree(m_rho_i[0]);

  if (split_flag) {
    cudaFree(m_u_prev);
    cudaFree(m_r_prev);
  }

  if (g_opt_level == OPT_BASELINE
      || g_opt_level == OPT_PARALLEL
      || g_opt_level == OPT_MEMLAYOUT
      || g_opt_level == OPT_SYNC) 
  {
    cudaFree(m_flux);
    cudaFree(m_utmp);
    cudaFree(m_rtmp);
    cudaFree(m_tmp_sum);
    cudaFree(m_a);
    cudaFree(m_b);
    cudaFree(m_c);
    cudaFree(m_d);
  }

  cudaStreamDestroy(cmd_q[KERNEL_Q]);
  cudaStreamDestroy(cmd_q[DATA_Q]);

  cuda_ProfilerRelease();
}

void print_opt_level(enum OptLevel ol)
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

