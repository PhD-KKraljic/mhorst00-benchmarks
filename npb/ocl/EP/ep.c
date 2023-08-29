//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB EP code. This OpenCL C  //
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

//--------------------------------------------------------------------
//      program EMBAR
//--------------------------------------------------------------------
//
//  M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//  numbers.  MK is the Log_2 of the size of each batch of uniform random
//  numbers.  MK can be set for convenience on a given system, since it does
//  not affect the results.
//--------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "type.h"
#include "npbparams.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include <CL/cl.h>
#include "cl_util.h"
#include "ep.h"

#define GROUP_SIZE 64

//--------------------------------------------------------------------
// OpenCL part
//--------------------------------------------------------------------
static cl_device_type   device_type;
static cl_device_id     device;
static char            *device_name;
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_program       program;
static cl_kernel        kernel;
static cl_int           err_code;
static cl_mem           buf_qq, buf_psx, buf_psy, buf_lq, buf_lsx, buf_lsy;
static double           *qq, *psx, *psy;

int NUM_PARTITIONS;

static char *source_dir = "../EP";
static logical use_local_mem = true;

void setup(int argc, char *argv[]);
void setup_opencl(int argc, char *argv[]);
void release_opencl();
//--------------------------------------------------------------------

static int    np;
static double x[2*NK];
static double q[NQ];
static logical timers_enabled;

int main(int argc, char *argv[]) 
{
  double Mops, t1, t2;
  double sx, sy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int    i, nit;
  int    k_offset, j, p;
  logical verified;

  double dum[3] = {1.0, 1.0, 1.0};
  char   size[16];

  FILE *fp;

  setup(argc, argv);
  setup_opencl(argc, argv);

  if ((fp = fopen("timer.flag", "r")) == NULL) {
    timers_enabled = false;
  } else {
    timers_enabled = true;
    fclose(fp);
  }

  //--------------------------------------------------------------------
  //  Because the size of the problem is too large to store in a 32-bit
  //  integer for some classes, we put it into a string (for printing).
  //  Have to strip off the decimal point put in there by the floating
  //  point print statement (internal file)
  //--------------------------------------------------------------------

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (size[j] == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-OCL) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = false;

  //--------------------------------------------------------------------
  //  Compute the number of "batches" of random number pairs generated 
  //  per processor. Adjust if the number of processors does not evenly 
  //  divide the total number
  //--------------------------------------------------------------------

  np = NN;

  //--------------------------------------------------------------------
  //  Call the random number generator functions and initialize
  //  the x-array to reduce the effects of paging on the timings.
  //  Also, call all mathematical functions that are used. Make
  //  sure these initializations cannot be eliminated as dead code.
  //--------------------------------------------------------------------

  vranlc(0, &dum[0], dum[1], &dum[2]);
  dum[0] = randlc(&dum[1], dum[2]);
  for (i = 0; i < 2 * NK; i++) {
    x[i] = -1.0e99;
  }
  Mops = log(sqrt(fabs(MAX(1.0, 1.0))));

  timer_clear(0);
  timer_start(0);
  clu_ProfilerStart();

  t1 = A;
  vranlc(0, &t1, A, x);

  //--------------------------------------------------------------------
  //  Compute AN = A ^ (2 * NK) (mod 2^46).
  //--------------------------------------------------------------------

  t1 = A;

  for (i = 0; i < MK + 1; i++) {
    t2 = randlc(&t1, t1);
  }

  an = t1;
  tt = S;
  gc = 0.0;
  sx = 0.0;
  sy = 0.0;

  for (i = 0; i < NQ; i++) {
    q[i] = 0.0;
  }

  //--------------------------------------------------------------------
  //  Each instance of this loop may be performed independently. We compute
  //  the k offsets separately to take into account the fact that some nodes
  //  have more numbers to generate than others
  //--------------------------------------------------------------------
  k_offset = -1;
  int k_size = np / NUM_PARTITIONS;

  for (p = 0; p < NUM_PARTITIONS; p++) {
    int k_start = p * k_size;

    // Launch the kernel
    err_code  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_qq);
    err_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_psx);
    err_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_psy);
    if (use_local_mem) {
      err_code |= clSetKernelArg(kernel, 3, sizeof(double) * NQ * GROUP_SIZE, NULL);
      err_code |= clSetKernelArg(kernel, 4, sizeof(double) * GROUP_SIZE, NULL);
      err_code |= clSetKernelArg(kernel, 5, sizeof(double) * GROUP_SIZE, NULL);
    }
    else {
      err_code |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_lq);
      err_code |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_lsx);
      err_code |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_lsy);
    }
    err_code |= clSetKernelArg(kernel, 6, sizeof(int), &k_offset);
    err_code |= clSetKernelArg(kernel, 7, sizeof(double), &an);
    err_code |= clSetKernelArg(kernel, 8, sizeof(int), &k_start);
    err_code |= clSetKernelArg(kernel, 9, sizeof(int), &k_size);
    clu_CheckError(err_code, "clSetKernelArg()");
    
    size_t globalWorkSize = k_size;
    size_t localWorkSize = GROUP_SIZE;
    err_code = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL,
                                      &globalWorkSize, &localWorkSize,
                                      0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueNDRangeKernel()");

    // Read buffers
    err_code = clEnqueueReadBuffer(cmd_queue, buf_qq,
                                   CL_FALSE, 0,
                                   k_size / GROUP_SIZE * NQ * sizeof(double),
                                   qq, 0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadBuffer()");

    err_code = clEnqueueReadBuffer(cmd_queue, buf_psx,
                                   CL_FALSE, 0,
                                   k_size / GROUP_SIZE * sizeof(double),
                                   psx, 0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadBuffer()");

    err_code = clEnqueueReadBuffer(cmd_queue, buf_psy,
                                   CL_TRUE, 0,
                                   k_size / GROUP_SIZE * sizeof(double),
                                   psy, 0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadBuffer()");
    
    for (i = 0; i < k_size / GROUP_SIZE; i++) {
      for (j = 0; j < NQ; j++) {
        q[j] += qq[i * NQ + j];
      }
    }
    for (i = 0; i < k_size / GROUP_SIZE; i++) {
      sx = sx + psx[i];
      sy = sy + psy[i];
    }

    for (i = 0; i < NQ; i++) {
      gc = gc + q[i];
    }
  }

  clu_ProfilerStop();
  timer_stop(0);
  tm = timer_read(0);

  nit = 0;
  verified = true;
  if (M == 24) {
    sx_verify_value = -3.247834652034740e+3;
    sy_verify_value = -6.958407078382297e+3;
  } else if (M == 25) {
    sx_verify_value = -2.863319731645753e+3;
    sy_verify_value = -6.320053679109499e+3;
  } else if (M == 28) {
    sx_verify_value = -4.295875165629892e+3;
    sy_verify_value = -1.580732573678431e+4;
  } else if (M == 30) {
    sx_verify_value =  4.033815542441498e+4;
    sy_verify_value = -2.660669192809235e+4;
  } else if (M == 32) {
    sx_verify_value =  4.764367927995374e+4;
    sy_verify_value = -8.084072988043731e+4;
  } else if (M == 36) {
    sx_verify_value =  1.982481200946593e+5;
    sy_verify_value = -1.020596636361769e+5;
  } else if (M == 40) {
    sx_verify_value = -5.319717441530e+05;
    sy_verify_value = -3.688834557731e+05;
  } else {
    verified = false;
  }

  if (verified) {
    sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
    sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
    verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
  }

  Mops = pow(2.0, M+1) / tm / 1000000.0;

  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n", tm);
  printf("N = 2^%5d\n", M);
  printf("No. Gaussian Pairs = %15.0lf\n", gc);
  printf("Sums = %25.15lE %25.15lE\n", sx, sy);
  printf("Counts: \n");
  for (i = 0; i < NQ; i++) {
    printf("%3d%15.0lf\n", i, q[i]);
  }

  c_print_results("EP", CLASS, M+1, 0, 0, nit,
      tm, Mops, 
      "Random numbers generated",
      verified, NPBVERSION, COMPILETIME, 
      CS1, CS2, CS3, CS4, CS5, CS6, CS7,
      clu_GetDeviceTypeName(device_type), device_name);

  if (timers_enabled) {
    if (tm <= 0.0) tm = 1.0;
    tt = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);

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
        source_dir = (char*)malloc(strlen(optarg + 1));
        memcpy(source_dir, optarg, strlen(optarg) + 1);
        break;

      case 'o':
        memcpy(opt_level, optarg, strlen(optarg) + 1);
        l = atoi(opt_level);

        if (l == 0) {
          use_local_mem = false;
        }
        else if (l == 2) {
          use_local_mem = true;
        }
        else {
          exit(0);
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
  cl_int err_code;
  clu_ProfilerSetup();

  switch (CLASS) {
    case 'E':
      NUM_PARTITIONS = 16;
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
  cmd_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
  clu_CheckError(err_code, "clCreateCommandQueue()");

  // Build the program
  char *source_file = (use_local_mem) ? "ep_gpu_opt.cl" : "ep_gpu_base.cl";
  char build_option[256];
  sprintf(build_option, "-DM=%d -I.", M);
  program = clu_MakeProgram(context, device, source_dir, source_file,
                            build_option);

  // Create a kernel
  kernel = clCreateKernel(program, "embar", &err_code);
  clu_CheckError(err_code, "clCreateKernel()");
  
  // Create buffers
  buf_qq = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          NN / NUM_PARTITIONS / GROUP_SIZE * NQ * sizeof(double),
                          NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for qq");

  buf_psx = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for psx");

  buf_psy = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for psy");

  buf_lq = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          NN / NUM_PARTITIONS * NQ * sizeof(double),
                          NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for lq");

  buf_lsx = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           NN / NUM_PARTITIONS * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for lsx");

  buf_lsy = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           NN / NUM_PARTITIONS * sizeof(double),
                           NULL, &err_code);
  clu_CheckError(err_code, "clCreateBuffer() for lsy");

  qq = (double*)malloc(NN / NUM_PARTITIONS / GROUP_SIZE * NQ * sizeof(double));
  psx = (double*)malloc(NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double));
  psy = (double*)malloc(NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double));
}


void release_opencl()
{
  // Release the memory object 
  clReleaseMemObject(buf_qq);
  clReleaseMemObject(buf_psx);
  clReleaseMemObject(buf_psy);
  clReleaseMemObject(buf_lq);
  clReleaseMemObject(buf_lsx);
  clReleaseMemObject(buf_lsy);

  free(qq);
  free(psx);
  free(psy);

  // Release other object
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);

  clu_ProfilerRelease();
}
