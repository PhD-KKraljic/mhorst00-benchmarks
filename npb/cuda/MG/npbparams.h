/* CLASS = D */
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   
*/
#define NX_DEFAULT     1024
#define NY_DEFAULT     1024
#define NZ_DEFAULT     1024
#define NIT_DEFAULT    50
#define LM             10
#define LT_DEFAULT     10
#define DEBUG_DEFAULT  0
#define NDIM1          10
#define NDIM2          10
#define NDIM3          10
#define ONE            1

#define CONVERTDOUBLE  false
#define COMPILETIME "01 Aug 2023"
#define NPBVERSION "3.3.1"
#define CS1 "gcc"
#define CS2 "g++"
#define CS3 "-lm -L$(CUDA)/lib64 -L/var/tmp/likwid-nvidi..."
#define CS4 "-I../common -I/var/tmp/likwid-nvidia/include"
#define CS5 "-Wall -O2 -mcmodel=large -fopenmp"
#define CS6 "-O3 -mcmodel=large -fopenmp"
#define CS7 "randdp"

/* CUDA device info */
#define MAX_THREAD_BLOCK_SIZE  1024
#define MAX_THREAD_DIM_0       1024
#define MAX_THREAD_DIM_1       1024
#define MAX_THREAD_DIM_2       64
