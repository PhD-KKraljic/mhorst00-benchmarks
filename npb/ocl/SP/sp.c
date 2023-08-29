//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB SP code. This OpenCL C  //
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

//---------------------------------------------------------------------
// program SP
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "header.h"
#include "print_results.h"

//--------------------------------------------------------------------
// OpenCL part
//--------------------------------------------------------------------
cl_device_type   device_type;
cl_device_id     device;
char            *device_name;
cl_context       context;
cl_command_queue cmd_queue[2];
cl_program       p_compute_rhs, p_txinvr, p_x_solve, p_y_solve, p_z_solve, p_add, p_util;
cl_int           err_code;

cl_kernel        k_compute_rhs[10];
cl_kernel        k_txinvr;
cl_kernel        k_x_solve[10];
cl_kernel        k_ninvr;
cl_kernel        k_y_solve[10];
cl_kernel        k_pinvr;
cl_kernel        k_z_solve[10];
cl_kernel        k_tzetar;
cl_kernel        k_add;
cl_kernel        k_transpose;
cl_kernel        k_scatter;
cl_kernel        k_gather;
cl_kernel        k_scatter_j;
cl_kernel        k_gather_j;

cl_mem           buf_u[2];
cl_mem           buf_us;
cl_mem           buf_vs;
cl_mem           buf_ws;
cl_mem           buf_qs;
cl_mem           buf_rho_i;
cl_mem           buf_speed;
cl_mem           buf_square;
cl_mem           buf_rhs[2];
cl_mem           buf_forcing[2];

cl_mem           buf_lhs;
cl_mem           buf_lhsp;
cl_mem           buf_lhsm;

cl_mem           buf_u0, buf_u1, buf_u2, buf_u3, buf_u4;
cl_mem           buf_rhs0, buf_rhs1, buf_rhs2, buf_rhs3, buf_rhs4;
cl_mem           buf_forcing0, buf_forcing1, buf_forcing2, buf_forcing3, buf_forcing4;

cl_mem           buf_lhs0, buf_lhs1, buf_lhs2, buf_lhs3, buf_lhs4;
cl_mem           buf_lhsp0, buf_lhsp1, buf_lhsp2, buf_lhsp3, buf_lhsp4;
cl_mem           buf_lhsm0, buf_lhsm1, buf_lhsm2, buf_lhsm3, buf_lhsm4;
cl_mem           buf_temp;

cl_event write_event[MAX_PARTITIONS];

int KMAXP_D, JMAXP_D, WORK_NUM_ITEM_K, WORK_NUM_ITEM_J;
int NUM_PARTITIONS;

char *source_dir = "../SP";
int opt_level_t = 5;

void setup(int argc, char *argv[]);
void setup_opencl(int argc, char *argv[]);
void release_opencl();
//--------------------------------------------------------------------

/* common /global/ */
int grid_points[3], nx2, ny2, nz2;
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
       c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
       dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
       c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
       c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

/* common /fields/ */
double u      [KMAX][JMAXP+1][IMAXP+1][5];
double us     [KMAX][JMAXP+1][IMAXP+1];
double vs     [KMAX][JMAXP+1][IMAXP+1];
double ws     [KMAX][JMAXP+1][IMAXP+1];
double qs     [KMAX][JMAXP+1][IMAXP+1];
double rho_i  [KMAX][JMAXP+1][IMAXP+1];
double speed  [KMAX][JMAXP+1][IMAXP+1];
double square [KMAX][JMAXP+1][IMAXP+1];
double rhs    [KMAX][JMAXP+1][IMAXP+1][5];
double forcing[KMAX][JMAXP+1][IMAXP+1][5];

/* common /work_1d/ */
double cv  [PROBLEM_SIZE];
double rhon[PROBLEM_SIZE];
double rhos[PROBLEM_SIZE];
double rhoq[PROBLEM_SIZE];
double cuf [PROBLEM_SIZE];
double q   [PROBLEM_SIZE];
double ue [PROBLEM_SIZE][5];
double buf[PROBLEM_SIZE][5];

/* common /work_lhs/ */
double lhs [IMAXP+1][IMAXP+1][5];
double lhsp[IMAXP+1][IMAXP+1][5];
double lhsm[IMAXP+1][IMAXP+1][5];

int main(int argc, char *argv[])
{
  int i, niter, step, n3;
  double mflops, t, tmax, trecs[t_last+1];
  logical verified;
  char Class;
  char *t_names[t_last+1];

  //---------------------------------------------------------------------
  // Read input file (if it exists), else take
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
    t_names[t_tzetar] = "tzetar";
    t_names[t_ninvr] = "ninvr";
    t_names[t_pinvr] = "pinvr";
    t_names[t_txinvr] = "txinvr";
    t_names[t_add] = "add";
    fclose(fp);
  } else {
    timeron = false;
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-OCL) - SP Benchmark\n\n");

  if ((fp = fopen("inputsp.data", "r")) != NULL) {
    int result;
    printf(" Reading from input file inputsp.data\n");
    result = fscanf(fp, "%d", &niter);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &dt);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d", &grid_points[0], &grid_points[1], &grid_points[2]);
    fclose(fp);
  } else {
    printf(" No input file inputsp.data. Using compiled defaults\n");
    niter = NITER_DEFAULT;
    dt    = DT_DEFAULT;
    grid_points[0] = PROBLEM_SIZE;
    grid_points[1] = PROBLEM_SIZE;
    grid_points[2] = PROBLEM_SIZE;
  }

  printf(" Size: %4dx%4dx%4d\n", 
      grid_points[0], grid_points[1], grid_points[2]);
  printf(" Iterations: %4d    dt: %10.6f\n", niter, dt);
  printf("\n");

  if ((grid_points[0] > IMAX) ||
      (grid_points[1] > JMAX) ||
      (grid_points[2] > KMAX) ) {
    printf(" %d, %d, %d\n", grid_points[0], grid_points[1], grid_points[2]);
    printf(" Problem size too big for compiled array sizes\n");
    return 0;
  }
  nx2 = grid_points[0] - 2;
  ny2 = grid_points[1] - 2;
  nz2 = grid_points[2] - 2;

  setup(argc, argv);
  setup_opencl(argc, argv);

  set_constants();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

  exact_rhs();

  initialize();

  //---------------------------------------------------------------------
  // do one time step to touch all code, and reinitialize
  //---------------------------------------------------------------------
  adi();
  initialize();

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_start(1);
  clu_ProfilerStart();

  //---------------------------------------------------------------------
  // write buffer when NUM_PARTITIONS = 1
  //---------------------------------------------------------------------
  if (NUM_PARTITIONS == 1) {
    err_code = clEnqueueWriteBuffer(cmd_queue[0],
                                    buf_forcing[0],
                                    CL_FALSE, 0,
                                    sizeof(forcing),
                                    forcing,
                                    0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueWriteBuffer()");
    err_code = clEnqueueWriteBuffer(cmd_queue[0],
                                    buf_u[0],
                                    CL_FALSE, 0,
                                    sizeof(u),
                                    u,
                                    0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueWriteBuffer()");

    int nx = IMAXP + 1, ny = JMAXP + 1, nz = KMAX;
    size_t scatter_gws[] = {nx, ny, nz};
    size_t scatter_lws[] = {16, 16, 1};
    scatter_gws[0] = clu_RoundWorkSize(scatter_gws[0], scatter_lws[0]);
    scatter_gws[1] = clu_RoundWorkSize(scatter_gws[1], scatter_lws[1]);
    scatter_gws[2] = clu_RoundWorkSize(scatter_gws[2], scatter_lws[2]);

    //---------------------------------------------------------------------
    // scatter buffer u -> u0, u1, u2, u3, u4
    //---------------------------------------------------------------------
    err_code  = clSetKernelArg(k_scatter, 0, sizeof(cl_mem), &buf_u[0]);
    err_code |= clSetKernelArg(k_scatter, 1, sizeof(cl_mem), &buf_u0);
    err_code |= clSetKernelArg(k_scatter, 2, sizeof(cl_mem), &buf_u1);
    err_code |= clSetKernelArg(k_scatter, 3, sizeof(cl_mem), &buf_u2);
    err_code |= clSetKernelArg(k_scatter, 4, sizeof(cl_mem), &buf_u3);
    err_code |= clSetKernelArg(k_scatter, 5, sizeof(cl_mem), &buf_u4);
    err_code |= clSetKernelArg(k_scatter, 6, sizeof(int), &nx);
    err_code |= clSetKernelArg(k_scatter, 7, sizeof(int), &ny);
    err_code |= clSetKernelArg(k_scatter, 8, sizeof(int), &nz);
    clu_CheckError(err_code, "clSetKernelArg()");

    err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_scatter, 3, NULL, scatter_gws, scatter_lws, 0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueNDRangeKernel()");

    //---------------------------------------------------------------------
    // scatter buffer forcing -> forcing0, forcing1, forcing2, forcing3, forcing4
    //---------------------------------------------------------------------
    err_code  = clSetKernelArg(k_scatter, 0, sizeof(cl_mem), &buf_forcing[0]);
    err_code |= clSetKernelArg(k_scatter, 1, sizeof(cl_mem), &buf_forcing0);
    err_code |= clSetKernelArg(k_scatter, 2, sizeof(cl_mem), &buf_forcing1);
    err_code |= clSetKernelArg(k_scatter, 3, sizeof(cl_mem), &buf_forcing2);
    err_code |= clSetKernelArg(k_scatter, 4, sizeof(cl_mem), &buf_forcing3);
    err_code |= clSetKernelArg(k_scatter, 5, sizeof(cl_mem), &buf_forcing4);
    err_code |= clSetKernelArg(k_scatter, 6, sizeof(int), &nx);
    err_code |= clSetKernelArg(k_scatter, 7, sizeof(int), &ny);
    err_code |= clSetKernelArg(k_scatter, 8, sizeof(int), &nz);
    clu_CheckError(err_code, "clSetKernelArg()");

    err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_scatter, 3, NULL, scatter_gws, scatter_lws, 0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  }
  //---------------------------------------------------------------------

  for (step = 1; step <= niter; step++) {
    if ((step % 20) == 0 || step == 1) {
      printf(" Time step %4d\n", step);
    }

    adi_gpu();
  }

  //---------------------------------------------------------------------
  // read buffer when NUM_PARTITIONS = 1
  //---------------------------------------------------------------------
  if (NUM_PARTITIONS == 1) {
    int nx = IMAXP + 1, ny = JMAXP + 1, nz = KMAX;
    size_t gather_gws[] = {nx, ny, nz};
    size_t gather_lws[] = {16, 16, 1};
    gather_gws[0] = clu_RoundWorkSize(gather_gws[0], gather_lws[0]);
    gather_gws[1] = clu_RoundWorkSize(gather_gws[1], gather_lws[1]);
    gather_gws[2] = clu_RoundWorkSize(gather_gws[2], gather_lws[2]);

    //---------------------------------------------------------------------
    // gather buffer u -> u0, u1, u2, u3, u4
    //---------------------------------------------------------------------
    err_code  = clSetKernelArg(k_gather, 0, sizeof(cl_mem), &buf_u[0]);
    err_code |= clSetKernelArg(k_gather, 1, sizeof(cl_mem), &buf_u0);
    err_code |= clSetKernelArg(k_gather, 2, sizeof(cl_mem), &buf_u1);
    err_code |= clSetKernelArg(k_gather, 3, sizeof(cl_mem), &buf_u2);
    err_code |= clSetKernelArg(k_gather, 4, sizeof(cl_mem), &buf_u3);
    err_code |= clSetKernelArg(k_gather, 5, sizeof(cl_mem), &buf_u4);
    err_code |= clSetKernelArg(k_gather, 6, sizeof(int), &nx);
    err_code |= clSetKernelArg(k_gather, 7, sizeof(int), &ny);
    err_code |= clSetKernelArg(k_gather, 8, sizeof(int), &nz);
    clu_CheckError(err_code, "clSetKernelArg()");

    err_code = clEnqueueNDRangeKernel(cmd_queue[0], k_gather, 3, NULL, gather_gws, gather_lws, 0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueNDRangeKernel()");

    err_code = clEnqueueReadBuffer(cmd_queue[0],
                                    buf_u[0],
                                    CL_TRUE, 0,
                                    sizeof(u),
                                    u,
                                    0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadBuffer()");
  }
  //---------------------------------------------------------------------

  clu_ProfilerStop();
  timer_stop(1);
  tmax = timer_read(1);

  verify(niter, &Class, &verified);

  if (tmax != 0.0) {
    n3 = grid_points[0]*grid_points[1]*grid_points[2];
    t = (grid_points[0]+grid_points[1]+grid_points[2])/3.0;
    mflops = (881.174 * (double)n3
             - 4683.91 * (t * t)
             + 11484.5 * t
             - 19272.4) * (double)niter / (tmax*1000000.0);
  } else {
    mflops = 0.0;
  }

  c_print_results("SP", Class, grid_points[0], 
                  grid_points[1], grid_points[2], niter, 
                  tmax, mflops, "          floating point", 
                  verified, NPBVERSION,COMPILETIME,
                  CS1, CS2, CS3, CS4, CS5, CS6, "(none)",
                  clu_GetDeviceTypeName(device_type), device_name);

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
      } else if (i == t_zsolve) {
        t = trecs[t_zsolve] - trecs[t_rdis1] - trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "sub-zsol", t, t*100./tmax);
      } else if (i == t_rdis2) {
        t = trecs[t_rdis1] + trecs[t_rdis2];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n", "redist", t, t*100./tmax);
      }
    }

    clu_ProfilerPrintResult();
  }

  release_opencl();
  return 0;
}

void setup(int argc, char *argv[])
{
  int c, l;
  char opt_level[100];

  while ((c = getopt(argc, argv, "o:s:")) != -1) {
    switch (c) {
      case 's':
        source_dir = (char*)malloc(strlen(optarg) + 1);
        memcpy(source_dir, optarg, strlen(optarg) + 1);
        break;

      case 'o':
        memcpy(opt_level, optarg, strlen(optarg) + 1);
        l = atoi(opt_level);
        if (l == 4) {
          exit(0);
        }
        else {
          opt_level_t = l;
        }
        break;
      case '?':
        if (optopt == 'o') {
           printf("option -o requires OPTLEVEL\n");
        }
        break;
    }
  }
}

//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
void setup_opencl(int argc, char *argv[])
{
  char kname[64];
  int i;

  clu_ProfilerSetup();

  switch (CLASS) {
    case 'D':
      NUM_PARTITIONS = 20;
      break;
    case 'E':
      NUM_PARTITIONS = 120;
      break;
    default:
      NUM_PARTITIONS = 1;
      break;
  }

  // Find the default device type and get a device for the device type
  device_type = clu_GetDefaultDeviceType();
  device      = clu_GetAvailableDevice(device_type);
  device_name = clu_GetDeviceName(device);

  // Create a context for the specified device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err_code);
  clu_CheckError(err_code, "clCreateContext()");

  // Create a command queue
  cmd_queue[0] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
  clu_CheckError(err_code, "clCreateCommandQueue()");

  cmd_queue[1] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
  clu_CheckError(err_code, "clCreateCommandQueue()");

  // Build the program
  char *source_file;
  char build_option[256];
  KMAXP_D = KMAX / NUM_PARTITIONS;
  JMAXP_D = JMAXP / NUM_PARTITIONS;
  WORK_NUM_ITEM_K = (NUM_PARTITIONS == 1) ? (KMAX) : (KMAXP_D+1+4);
  WORK_NUM_ITEM_J = (NUM_PARTITIONS == 1) ? (JMAXP+1) : (JMAXP_D+1+4);
  sprintf(
    build_option,
    "-I. -DCLASS=\'%c\' -DPROBLEM_SIZE=%d -DDT_DEFAULT=%lf -DWORK_NUM_ITEM_K=%d -DWORK_NUM_ITEM_J=%d",
    CLASS, PROBLEM_SIZE, DT_DEFAULT, WORK_NUM_ITEM_K, WORK_NUM_ITEM_J
  );

  if (opt_level_t == 0 || opt_level_t == 2 || opt_level_t == 3) {
    p_compute_rhs = clu_MakeProgram(context, device, source_dir,
                                    "kernel_compute_rhs_base.cl", build_option);
    p_txinvr      = clu_MakeProgram(context, device, source_dir,
                                    "kernel_txinvr_base.cl", build_option);
    p_y_solve     = clu_MakeProgram(context, device, source_dir,
                                    "kernel_y_solve_base.cl", build_option);
    p_z_solve     = clu_MakeProgram(context, device, source_dir,
                                    "kernel_z_solve_base.cl", build_option);
    p_add         = clu_MakeProgram(context, device, source_dir,
                                    "kernel_add_base.cl", build_option);
    if (opt_level_t == 3) {
      p_x_solve   = clu_MakeProgram(context, device, source_dir,
                                    "kernel_x_solve_layout.cl", build_option);
    }
    else {
      p_x_solve   = clu_MakeProgram(context, device, source_dir,
                                    "kernel_x_solve_base.cl", build_option);
    }
    if (opt_level_t == 2) {
      p_util      = clu_MakeProgram(context, device, source_dir,
                                    "kernel_util_opt.cl", build_option);
    }
    else {
      p_util      = clu_MakeProgram(context, device, source_dir,
                                    "kernel_util_base.cl", build_option);
    }
  }
  else {
    p_compute_rhs = clu_MakeProgram(context, device, source_dir,
                                    "kernel_compute_rhs_opt.cl", build_option);
    p_txinvr      = clu_MakeProgram(context, device, source_dir,
                                    "kernel_txinvr_opt.cl", build_option);
    p_y_solve     = clu_MakeProgram(context, device, source_dir,
                                    "kernel_y_solve_opt.cl", build_option);
    p_z_solve     = clu_MakeProgram(context, device, source_dir,
                                    "kernel_z_solve_opt.cl", build_option);
    p_add         = clu_MakeProgram(context, device, source_dir,
                                    "kernel_add_opt.cl", build_option);
    if (opt_level_t == 1) {
      p_x_solve   = clu_MakeProgram(context, device, source_dir,
                                    "kernel_x_solve_parallel.cl", build_option);
      p_util      = clu_MakeProgram(context, device, source_dir,
                                    "kernel_util_base.cl", build_option);
    }
    else {
      p_x_solve   = clu_MakeProgram(context, device, source_dir,
                                    "kernel_x_solve_opt.cl", build_option);
      p_util      = clu_MakeProgram(context, device, source_dir,
                                    "kernel_util_opt.cl", build_option);
    }
  }

  // Create a kernel
  for (i = 0; i < 9; i++) {
    sprintf(kname, "compute_rhs%d", i);
    k_compute_rhs[i] = clCreateKernel(p_compute_rhs, kname, &err_code);
    clu_CheckError(err_code, "clCreateKernel()");
  }
  k_txinvr = clCreateKernel(p_txinvr, "txinvr", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  for (i = 0; i < 5; i++) {
    sprintf(kname, "x_solve%d", i);
    k_x_solve[i] = clCreateKernel(p_x_solve, kname, &err_code);
    clu_CheckError(err_code, "clCreateKernel()");
  }
  k_ninvr = clCreateKernel(p_x_solve, "ninvr", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  for (i = 0; i < 5; i++) {
    sprintf(kname, "y_solve%d", i);
    k_y_solve[i] = clCreateKernel(p_y_solve, kname, &err_code);
    clu_CheckError(err_code, "clCreateKernel()");
  }
  k_pinvr = clCreateKernel(p_y_solve, "pinvr", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  for (i = 0; i < 6; i++) {
    sprintf(kname, "z_solve%d", i);
    k_z_solve[i] = clCreateKernel(p_z_solve, kname, &err_code);
    clu_CheckError(err_code, "clCreateKernel()");
  }
  k_tzetar = clCreateKernel(p_z_solve, "tzetar", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  k_add = clCreateKernel(p_add, "add", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  k_transpose = clCreateKernel(p_util, "transpose", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  k_scatter = clCreateKernel(p_util, "scatter", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  k_gather = clCreateKernel(p_util, "gather", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  k_scatter_j = clCreateKernel(p_util, "scatter_j", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  k_gather_j = clCreateKernel(p_util, "gather_j", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");

  // Create buffers
  buf_u[0] = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                         NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_u[1] = clCreateBuffer(context,
                         CL_MEM_READ_WRITE,
                         (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                         NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_us = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                          NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_vs = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                          NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_ws = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                          NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_qs = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                          NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rho_i = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                             NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_speed = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                             NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_square = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rhs[0] = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rhs[1] = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_forcing[0] = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                               NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_forcing[1] = clCreateBuffer(context,
                               CL_MEM_READ_WRITE,
                               (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                               NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhs = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsp = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                            NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsm = clCreateBuffer(context,
                            CL_MEM_READ_WRITE,
                            (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double),
                            NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  // u0, u1, u2, u3, u4
  buf_u0 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_u1 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_u2 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_u3 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_u4 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  // rhs0, rhs1, rhs2, rhs3, rhs4
  buf_rhs0 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rhs1 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rhs2 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rhs3 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_rhs4 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  // forcing0, forcing1, forcing2, forcing3, forcing4
  buf_forcing0 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_forcing1 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_forcing2 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_forcing3 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_forcing4 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  // lhs0, lhs1, lhs2, lhs3, lhs4
  buf_lhs0 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhs1 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhs2 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhs3 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhs4 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  // lhsp0, lhsp1, lhsp2, lhsp3, lhsp4
  buf_lhsp0 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsp1 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsp2 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsp3 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsp4 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  // lhsm0, lhsm1, lhsm2, lhsm3, lhsm4
  buf_lhsm0 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsm1 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsm2 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsm3 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_lhsm4 = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");

  buf_temp = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double),
                              NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer()");
}

void release_opencl()
{
  int i;

  // Release the memory objects
  for (i = 0; i < 2; i++) {
    clReleaseMemObject(buf_u[i]);
    clReleaseMemObject(buf_rhs[i]);
    clReleaseMemObject(buf_forcing[i]);
  }
  clReleaseMemObject(buf_us);
  clReleaseMemObject(buf_vs);
  clReleaseMemObject(buf_ws);
  clReleaseMemObject(buf_qs);
  clReleaseMemObject(buf_rho_i);
  clReleaseMemObject(buf_speed);
  clReleaseMemObject(buf_square);
  clReleaseMemObject(buf_u0);
  clReleaseMemObject(buf_u1);
  clReleaseMemObject(buf_u2);
  clReleaseMemObject(buf_u3);
  clReleaseMemObject(buf_u4);
  clReleaseMemObject(buf_rhs0);
  clReleaseMemObject(buf_rhs1);
  clReleaseMemObject(buf_rhs2);
  clReleaseMemObject(buf_rhs3);
  clReleaseMemObject(buf_rhs4);
  clReleaseMemObject(buf_forcing0);
  clReleaseMemObject(buf_forcing1);
  clReleaseMemObject(buf_forcing2);
  clReleaseMemObject(buf_forcing3);
  clReleaseMemObject(buf_forcing4);
  clReleaseMemObject(buf_lhs0);
  clReleaseMemObject(buf_lhs1);
  clReleaseMemObject(buf_lhs2);
  clReleaseMemObject(buf_lhs3);
  clReleaseMemObject(buf_lhs4);
  clReleaseMemObject(buf_lhsp0);
  clReleaseMemObject(buf_lhsp1);
  clReleaseMemObject(buf_lhsp2);
  clReleaseMemObject(buf_lhsp3);
  clReleaseMemObject(buf_lhsp4);
  clReleaseMemObject(buf_lhsm0);
  clReleaseMemObject(buf_lhsm1);
  clReleaseMemObject(buf_lhsm2);
  clReleaseMemObject(buf_lhsm3);
  clReleaseMemObject(buf_lhsm4);

  // Release kernel objects
  for (i = 0; i < 9; i++) {
    clReleaseKernel(k_compute_rhs[i]);
  }
  for (i = 0; i < 5; i++) {
    clReleaseKernel(k_x_solve[i]);
    clReleaseKernel(k_y_solve[i]);
  }
  for (i = 0; i < 6; i++) {
    clReleaseKernel(k_z_solve[i]);
  }
  clReleaseKernel(k_compute_rhs[5]);
  clReleaseKernel(k_z_solve[5]);
  clReleaseKernel(k_txinvr);
  clReleaseKernel(k_ninvr);
  clReleaseKernel(k_pinvr);
  clReleaseKernel(k_tzetar);
  clReleaseKernel(k_add);
  clReleaseKernel(k_transpose);
  clReleaseKernel(k_scatter);
  clReleaseKernel(k_gather);
  clReleaseKernel(k_scatter_j);
  clReleaseKernel(k_gather_j);

  // Release other objects
  clReleaseProgram(p_compute_rhs);
  clReleaseProgram(p_txinvr);
  clReleaseProgram(p_x_solve);
  clReleaseProgram(p_y_solve);
  clReleaseProgram(p_z_solve);
  clReleaseProgram(p_add);
  clReleaseProgram(p_util);
  clReleaseCommandQueue(cmd_queue[0]);
  clReleaseCommandQueue(cmd_queue[1]);
  clReleaseContext(context);

  clu_ProfilerRelease();
}
