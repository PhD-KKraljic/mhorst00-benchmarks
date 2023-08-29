//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB EP code. This CUDA® C  //
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

extern "C" {
#include "type.h"
#include "npbparams.h"
//#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#include "ep.h"
}

#include <cuda_runtime.h>
#include "cuda_util.h"
#include "ep_gpu.cu"

#define GROUP_SIZE 64

//--------------------------------------------------------------------
// CUDA part
//--------------------------------------------------------------------
static int              device;
static char             device_name[256];
static double           *qq, *psx, *psy;
static double           *buf_qq, *buf_psx, *buf_psy, *buf_lq, *buf_lsx, *buf_lsy;

cudaStream_t            cmd_queue;

int NUM_PARTITIONS;

static logical use_shared_mem = true;

void setup(int argc, char *argv[]);
void setup_cuda(int argc, char *argv[]);
void release_cuda();
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
    setup_cuda(argc, argv);

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
    printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-CUDA) - EP Benchmark\n");
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
    cuda_ProfilerStart();

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
        if (use_shared_mem) {
            int l_q_size, l_sx_size, l_sy_size;
            l_q_size = NQ * GROUP_SIZE;
            l_sx_size = GROUP_SIZE;
            l_sy_size = GROUP_SIZE;

            cuda_ProfilerStartEventRecord("k_embar_opt", cmd_queue);
            embar_opt<<<np / GROUP_SIZE, GROUP_SIZE,
                        (l_q_size + l_sx_size + l_sy_size) * sizeof(double), cmd_queue>>> (
                buf_qq, buf_psx, buf_psy, 
                l_q_size, l_sx_size, l_sy_size,
                k_offset, an, k_start, k_size);
            cuda_ProfilerEndEventRecord("k_embar_opt", cmd_queue);
        }
        else {
            cuda_ProfilerStartEventRecord("k_embar_base", cmd_queue);
            embar_base<<<np / GROUP_SIZE, GROUP_SIZE,
                          0, cmd_queue>>> (
                buf_qq, buf_psx, buf_psy,
                buf_lq, buf_lsx, buf_lsy,
                k_offset, an, k_start, k_size);
            cuda_ProfilerEndEventRecord("k_embar_base", cmd_queue);
        }
        CUCHK(cudaGetLastError());

        // Read buffers
        CUCHK(cudaMemcpyAsync(qq, buf_qq, k_size / GROUP_SIZE * NQ * sizeof(double), cudaMemcpyDeviceToHost, cmd_queue));
        CUCHK(cudaMemcpyAsync(psx, buf_psx, k_size / GROUP_SIZE * sizeof(double), cudaMemcpyDeviceToHost, cmd_queue));
        CUCHK(cudaMemcpyAsync(psy, buf_psy, k_size / GROUP_SIZE * sizeof(double), cudaMemcpyDeviceToHost, cmd_queue));
        CUCHK(cudaStreamSynchronize(0));

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

    cuda_ProfilerStop();
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    c_print_results("EP", CLASS, M+1, 0, 0, nit,
            tm, Mops, 
            "Random numbers generated",
            verified, NPBVERSION, COMPILETIME, 
            CS1, CS2, CS3, CS4, CS5, CS6, CS7,
            (const char *)prop.name);

    if (timers_enabled) {
        if (tm <= 0.0) tm = 1.0;
        tt = timer_read(0);
        printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);

        cuda_ProfilerPrintResult();
    }

    release_cuda();
    return 0;
}

void setup(int argc, char *argv[])
{
  int c, l;
  char opt_level[100];

  while ((c = getopt(argc, argv, "o:")) != -1) {
      switch (c) {
        case 'o':
          memcpy(opt_level, optarg, strlen(optarg) + 1);
          l = atoi(opt_level);

          if (l == 0) {
              use_shared_mem = false;
          }
          else if (l == 2) {
              use_shared_mem = true;
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
// Set up the CUDA environment.
//---------------------------------------------------------------------
void setup_cuda(int argc, char *argv[])
{
    cuda_ProfilerSetup();

    switch (CLASS) {
      case 'E':
        NUM_PARTITIONS = 16;
        break;
      default:
        NUM_PARTITIONS = 1;
        break;
    }

    // Find the default device type and get a device for the device type
    int devCount;

    CUCHK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        printf("Not Available Device\n");
        exit(EXIT_FAILURE);
    }
    CUCHK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUCHK(cudaGetDeviceProperties(&prop, device));
    strncpy(device_name, prop.name, 256);

    CUCHK(cudaStreamCreate(&cmd_queue));

    // Create buffers
    CUCHK(cudaMalloc(&buf_qq, NN / NUM_PARTITIONS / GROUP_SIZE * NQ * sizeof(double)));
    CUCHK(cudaMalloc(&buf_psx, NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double)));
    CUCHK(cudaMalloc(&buf_psy, NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lq, NN / NUM_PARTITIONS * NQ * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lsx, NN / NUM_PARTITIONS * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lsy, NN / NUM_PARTITIONS * sizeof(double)));
    qq = (double*)malloc(NN / NUM_PARTITIONS / GROUP_SIZE * NQ * sizeof(double));
    psx = (double*)malloc(NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double));
    psy = (double*)malloc(NN / NUM_PARTITIONS / GROUP_SIZE * sizeof(double));
    if (!qq || !psx || !psy) {
        printf("malloc() failed\n");
        exit(EXIT_FAILURE);
    }
}

void release_cuda()
{
    // Release the memory object 
    cudaFree(buf_qq);
    cudaFree(buf_psx);
    cudaFree(buf_psy);
    cudaFree(buf_lq);
    cudaFree(buf_lsx);
    cudaFree(buf_lsy);
    free(qq);
    free(psx);
    free(psy);

    cudaStreamDestroy(cmd_queue);

    cuda_ProfilerRelease();
}
