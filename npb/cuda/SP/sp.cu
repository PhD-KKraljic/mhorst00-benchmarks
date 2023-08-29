//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB SP code. This CUDA® C  //
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
// program SP
//---------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "header.h"
extern "C" {
#include "print_results.h"
}
#include "cuda_util.h"

// CUDA Variables
int device;
cudaStream_t cmd_queue[2];

double *buf_u[2];
double *buf_us;
double *buf_vs;
double *buf_ws;
double *buf_qs;
double *buf_rho_i;
double *buf_speed;
double *buf_square;
double *buf_rhs[2];
double *buf_forcing[2];

double *buf_lhs;
double *buf_lhsp;
double *buf_lhsm;

double *buf_u0, *buf_u1, *buf_u2, *buf_u3, *buf_u4;
double *buf_rhs0, *buf_rhs1, *buf_rhs2, *buf_rhs3, *buf_rhs4;
double *buf_forcing0, *buf_forcing1, *buf_forcing2, *buf_forcing3, *buf_forcing4;

double *buf_lhs0, *buf_lhs1, *buf_lhs2, *buf_lhs3, *buf_lhs4;
double *buf_lhsp0, *buf_lhsp1, *buf_lhsp2, *buf_lhsp3, *buf_lhsp4;
double *buf_lhsm0, *buf_lhsm1, *buf_lhsm2, *buf_lhsm3, *buf_lhsm4;
double *buf_temp;

cudaEvent_t write_event[MAX_PARTITONS];

int KMAXP_D, JMAXP_D, WORK_NUM_ITEM_K, WORK_NUM_ITEM_J;
int NUM_PARTITIONS;

int opt_level_t = 5;

void setup(int argc, char *argv[]);
void setup_cuda(int argc, char *argv[]);
void release_cuda();

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
    const char *t_names[t_last+1];

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

    printf("\n\n NAS Parallel Benchmarks (NPB3.3.1-CUDA) - SP Benchmark\n\n");

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
    setup_cuda(argc, argv);

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
    cuda_ProfilerStart();

    //---------------------------------------------------------------------
    // write buffer when NUM_PARTITIONS = 1
    //---------------------------------------------------------------------
    if (NUM_PARTITIONS == 1) {
        CUCHK(cudaMemcpyAsync(buf_forcing[0], forcing, sizeof(forcing),
                    cudaMemcpyHostToDevice, cmd_queue[0]));
        CUCHK(cudaMemcpyAsync(buf_u[0], u, sizeof(forcing),
                    cudaMemcpyHostToDevice, cmd_queue[0]));
        int nx = IMAXP + 1, ny = JMAXP + 1, nz = KMAX;
        int scatter_gws[] = {nx, ny, nz};
        int scatter_lws[] = {16, 16, 1};
        scatter_gws[0] = RoundWorkSize(scatter_gws[0], scatter_lws[0]);
        scatter_gws[1] = RoundWorkSize(scatter_gws[1], scatter_lws[1]);
        scatter_gws[2] = RoundWorkSize(scatter_gws[2], scatter_lws[2]);

        dim3 blockSize(scatter_gws[0]/scatter_lws[0],
                scatter_gws[1]/scatter_lws[1],
                scatter_gws[2]/scatter_lws[2]);
        dim3 threadSize(scatter_lws[0], scatter_lws[1], scatter_lws[2]);

        //---------------------------------------------------------------------
        // scatter buffer u -> u0, u1, u2, u3, u4
        //---------------------------------------------------------------------
        k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_u[0], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz
            );
        CUCHK(cudaGetLastError());

        //---------------------------------------------------------------------
        // scatter buffer forcing -> forcing0, forcing1, forcing2, forcing3, forcing4
        //---------------------------------------------------------------------
        k_scatter<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_forcing[0], buf_forcing0, buf_forcing1, buf_forcing2,
             buf_forcing3, buf_forcing4, nx, ny, nz
            );
        CUCHK(cudaGetLastError());
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
        int gather_gws[] = {nx, ny, nz};
        int gather_lws[] = {16, 16, 1};
        gather_gws[0] = RoundWorkSize(gather_gws[0], gather_lws[0]);
        gather_gws[1] = RoundWorkSize(gather_gws[1], gather_lws[1]);
        gather_gws[2] = RoundWorkSize(gather_gws[2], gather_lws[2]);

        dim3 blockSize(gather_gws[0]/gather_lws[0],
                gather_gws[1]/gather_lws[1],
                gather_gws[2]/gather_lws[2]);
        dim3 threadSize(gather_lws[0], gather_lws[1], gather_lws[2]);

        //---------------------------------------------------------------------
        // gather buffer u -> u0, u1, u2, u3, u4
        //---------------------------------------------------------------------

        k_gather<<< blockSize, threadSize, 0, cmd_queue[0] >>>
            (
             buf_u[0], buf_u0, buf_u1, buf_u2, buf_u3, buf_u4, nx, ny, nz
            );
        CUCHK(cudaGetLastError());

        CUCHK(cudaMemcpyAsync(u, buf_u[0], sizeof(u),
                    cudaMemcpyDeviceToHost, cmd_queue[0]));
        CUCHK(cudaStreamSynchronize(cmd_queue[0]));
    }
    //---------------------------------------------------------------------

    cuda_ProfilerStop();
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    c_print_results("SP", Class, grid_points[0], 
            grid_points[1], grid_points[2], niter, 
            tmax, mflops, "          floating point", 
            verified, NPBVERSION,COMPILETIME, CS1, CS2, CS3, CS4, CS5, 
            CS6, "(none)",
            (const char *)prop.name);

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
// Set up the CUDA environment.
//---------------------------------------------------------------------
void setup_cuda(int argc, char *argv[])
{
    int i;

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

    cuda_ProfilerSetup();

    // Create a command queue
    CUCHK(cudaStreamCreate(&cmd_queue[0]));
    CUCHK(cudaStreamCreate(&cmd_queue[1]));

    KMAXP_D = KMAX / NUM_PARTITIONS;
    JMAXP_D = JMAXP / NUM_PARTITIONS;
    WORK_NUM_ITEM_K = (NUM_PARTITIONS == 1) ? (KMAX) : (KMAXP_D+1+4);
    WORK_NUM_ITEM_J = (NUM_PARTITIONS == 1) ? (JMAXP+1) : (JMAXP_D+1+4);

    // Create buffers
    CUCHK(cudaMalloc(&buf_u[0], (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_u[1], (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_us, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_vs, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_ws, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_qs, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    CUCHK(cudaMalloc(&buf_rho_i, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_speed, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_square, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_rhs[0], (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_rhs[1], (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_forcing[0], (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_forcing[1], (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhs, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsp, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsm, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * 5 * sizeof(double)));

    // u0, u1, u2, u3, u4
    CUCHK(cudaMalloc(&buf_u0, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_u1, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_u2, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_u3, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_u4, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    // rhs0, rhs1, rhs2, rhs3, rhs4
    CUCHK(cudaMalloc(&buf_rhs0, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_rhs1, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_rhs2, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_rhs3, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_rhs4, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    // forcing0, forcing1, forcing2, forcing3, forcing4
    CUCHK(cudaMalloc(&buf_forcing0, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_forcing1, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_forcing2, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_forcing3, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_forcing4, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    // lhs0, lhs1, lhs2, lhs3, lhs4
    CUCHK(cudaMalloc(&buf_lhs0, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhs1, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhs2, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhs3, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhs4, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    // lhsp0, lhsp1, lhsp2, lhsp3, lhsp4
    CUCHK(cudaMalloc(&buf_lhsp0, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsp1, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsp2, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsp3, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsp4, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    // lhsm0, lhsm1, lhsm2, lhsm3, lhsm4
    CUCHK(cudaMalloc(&buf_lhsm0, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsm1, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsm2, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsm3, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));
    CUCHK(cudaMalloc(&buf_lhsm4, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    CUCHK(cudaMalloc(&buf_temp, (WORK_NUM_ITEM_K) * (JMAXP+1) * (IMAXP+1) * sizeof(double)));

    for (int i = 0; i < MAX_PARTITONS; ++i) {
        CUCHK(cudaEventCreate(&write_event[i]));
    }
}

void release_cuda()
{
    int i;

    cudaFree(buf_u[0]);
    cudaFree(buf_u[1]);
    cudaFree(buf_us);
    cudaFree(buf_vs);
    cudaFree(buf_ws);
    cudaFree(buf_qs);
    cudaFree(buf_rho_i);
    cudaFree(buf_speed);
    cudaFree(buf_square);
    cudaFree(buf_rhs[0]);
    cudaFree(buf_rhs[1]);
    cudaFree(buf_forcing[0]);
    cudaFree(buf_forcing[1]);
    cudaFree(buf_lhs);
    cudaFree(buf_lhsp);
    cudaFree(buf_lhsm);

    // Release other objects
    cudaStreamDestroy(cmd_queue[0]);
    cudaStreamDestroy(cmd_queue[1]);

    for (int i = 0; i < MAX_PARTITONS; ++i) {
        cudaEventDestroy(write_event[i]);
    }

    cuda_ProfilerRelease();
}
