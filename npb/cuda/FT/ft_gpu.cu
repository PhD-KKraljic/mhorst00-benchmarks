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

//---------------------------------------------------------------------------
// double complex
//---------------------------------------------------------------------------
//FIXME
#include "npbparams.h"
#include "global.h"

//typedef struct {
 // double real;
  //double imag;
//} dcomplex;

#ifndef decmplx
#define dcmplx(r,i)       (dcomplex){r, i}
#endif
#ifndef  dcmplx_add
#define dcmplx_add(a,b)   (dcomplex){(a).real+(b).real, (a).imag+(b).imag}
#endif
#ifndef  dcmplx_sub
#define dcmplx_sub(a,b)   (dcomplex){(a).real-(b).real, (a).imag-(b).imag}
#endif
#ifndef  dcmplx_mul
#define dcmplx_mul(a,b)   (dcomplex){((a).real*(b).real)-((a).imag*(b).imag),\
                                     ((a).real*(b).imag)+((a).imag*(b).real)}
#endif
#ifndef  dcmplx_mul2
#define dcmplx_mul2(a,b)  (dcomplex){(a).real*(b), (a).imag*(b)}
#endif
#ifndef  dcmplx_div2
#define dcmplx_div2(a,b)  (dcomplex){(a).real/(b), (a).imag/(b)}
#endif
#ifndef  dcmplx_abs
#define dcmplx_abs(x)     sqrt(((x).real*(x).real) + ((x).imag*(x).imag))
#endif

__device__ void cfftz_base_0(int is, int m, int n,
                             dcomplex *u,
                             dcomplex *x, 
                             dcomplex *y,
                             int lid, int lws);

__device__ void cfftz_base_1(int is, int m, int n,
                             dcomplex *u,
                             dcomplex *x, 
                             dcomplex *y,
                             int lid, int lws);

__device__ void cfftz_inplace(int is, int m, int n,
                              dcomplex *u,
                              dcomplex *x, 
                              int lid, int lws);

__device__ void fftz2_base_0(int is, int l, int m, int n,
                             dcomplex *u,
                             dcomplex *x, 
                             dcomplex *y, 
                             int lid, int lws);

__device__ void fftz2_base_1(int is, int l, int m, int n,
                             dcomplex *u,
                             dcomplex *x, 
                             dcomplex *y, 
                             int lid, int lws);

__device__ void fftz2_inplace(int is, int l, int m, int n,
                              dcomplex *u,
                              dcomplex *x,
                              int lid, int lws);

__device__ void transpose(int l, int m, int n,
                          dcomplex *x,
                          int lid, int lws);

__device__ inline void d_vranlc(int n, double *x, double a,  double *y);


//---------------------------------------------------------------------
// compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
// for time evolution exponent.
//---------------------------------------------------------------------
__global__ void k_compute_indexmap(double *twiddle,
                                   int d1, int d2,
                                   int ofs3, int len3,
                                   double ap)
{
  int i, j, k, kk, kk2, jj, kj2, ii;

  k = blockDim.z * blockIdx.z + threadIdx.z;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < d1 && j < d2 && k < len3) {
    kk = ((k + ofs3 + NZ/2) % NZ) - NZ/2;
    kk2 = kk*kk;
    jj = ((j + NY/2) % NY) - NY/2;
    kj2 = jj*jj + kk2;
    ii = ((i + NX/2) % NX) - NX/2;
    twiddle[k*d2*(d1+1) + j*(d1+1) + i] = exp(ap * (double) (ii*ii+kj2));
  }
}


//---------------------------------------------------------------------
// Go through by z planes filling in one square at a time.
//---------------------------------------------------------------------
__global__ void k_compute_initial_conditions(dcomplex *u1,
                                             double *starts,
                                             int d1,
                                             int d2,
                                             int d3)
{
  int k = blockDim.x * blockIdx.x + threadIdx.x;

  if (k < d3) {
    double x0 = starts[k];
    for (int j = 0; j < d2; j++) {
      d_vranlc(2*NX, &x0, A, ( double *) &u1[k*d2*(d1+1) + j*(d1+1)]);
    }
    starts[k] = x0;
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
__global__ void k_evolve(dcomplex *u0,
                         dcomplex *u1,
                         const double *twiddle,
                         int d1,
                         int d2,
                         int d3)
{
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (k < d3 && j < d2 && i < d1) {
    int idx = k*d2*(d1+1) + j*(d1+1) + i;
    u0[idx] = dcmplx_mul2(u0[idx], twiddle[idx]);
    u1[idx].real = u0[idx].real;
    u1[idx].imag = u0[idx].imag;
  }
}


__global__ void k_checksum(dcomplex *u1,
                           dcomplex *g_chk,
                           int d1,
                           int d2)
{
  extern __shared__ dcomplex l_chk[];
  int q, r, s;
  int j = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int lid = threadIdx.x;

  if (j <= 1024) {
    q = j % NX;
    r = 3*j % NY;
    s = 5*j % NZ;
    int u1_idx = s*d2*(d1+1) + r*(d1+1) + q;
    l_chk[lid].real = u1[u1_idx].real;
    l_chk[lid].imag = u1[u1_idx].imag;
  }
  else {
    l_chk[lid] = dcmplx(0.0, 0.0);
  }

  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (lid < i) {
      l_chk[lid] = dcmplx_add(l_chk[lid], l_chk[lid+i]);
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_chk[blockIdx.x].real = l_chk[0].real;
    g_chk[blockIdx.x].imag = l_chk[0].imag;
  }
}


__global__ void k_cffts1_base_0(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *tx,
                                dcomplex *ty,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd1)
{
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  dcomplex *tx_kj = &tx[k*d2*d1 + j*d1];
  dcomplex *ty_kj = &ty[k*d2*d1 + j*d1];

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx_kj[i] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_0(is, logd1, d1, u, tx_kj, ty_kj, lid, lws);

  if (logd1 % 2 == 0) {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = tx_kj[i];
    }
  }
  else {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = ty_kj[i];
    }
  }
}


__global__ void k_cffts1_base_1(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *tx,
                                dcomplex *ty,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd1)
{
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  dcomplex *tx_kj = &tx[k*d2*d1 + j*d1];
  dcomplex *ty_kj = &ty[k*d2*d1 + j*d1];

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx_kj[i] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_1(is, logd1, d1, u, tx_kj, ty_kj, lid, lws);

  if (logd1 % 2 == 0) {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = tx_kj[i];
    }
  }
  else {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = ty_kj[i];
    }
  }
}


__global__ void k_cffts1_base_2(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd1)
{
  extern __shared__ dcomplex tx[];
  dcomplex *ty = &tx[d1];

  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx[i] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_0(is, logd1, d1, u, tx, ty, lid, lws);

  if (logd1 % 2 == 0) {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = tx[i];
    }
  }
  else {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = ty[i];
    }
  }
}


__global__ void k_cffts1_base_3(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd1)
{
  extern __shared__ dcomplex tx[];
  dcomplex *ty = &tx[d1];

  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx[i] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_1(is, logd1, d1, u, tx, ty, lid, lws);

  if (logd1 % 2 == 0) {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = tx[i];
    }
  }
  else {
    for (int i = lid; i < d1; i += lws) {
      int xout_idx = kj_idx + i;
      xout[xout_idx] = ty[i];
    }
  }
}


__global__ void k_cffts2_base_0(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *tx,
                                dcomplex *ty,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd2)
{
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  dcomplex *tx_ki = &tx[k*d1*d2 + i*d2];
  dcomplex *ty_ki = &ty[k*d1*d2 + i*d2];

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx_ki[j] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_0(is, logd2, d2, u, tx_ki, ty_ki, lid, lws);

  if (logd2 % 2 == 0) {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = tx_ki[j];
    }
  }
  else {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = ty_ki[j];
    }
  }
}


__global__ void k_cffts2_base_1(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *tx,
                                dcomplex *ty,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd2)
{
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  dcomplex *tx_ki = &tx[k*d1*d2 + i*d2];
  dcomplex *ty_ki = &ty[k*d1*d2 + i*d2];

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx_ki[j] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_1(is, logd2, d2, u, tx_ki, ty_ki, lid, lws);

  if (logd2 % 2 == 0) {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = tx_ki[j];
    }
  }
  else {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = ty_ki[j];
    }
  }
}


__global__ void k_cffts2_base_2(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd2)
{
  extern __shared__ dcomplex tx[];
  dcomplex *ty = &tx[d2];

  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx[j] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_0(is, logd2, d2, u, tx, ty, lid, lws);

  if (logd2 % 2 == 0) {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = tx[j];
    }
  }
  else {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = tx[j];
    }
  }
}


__global__ void k_cffts2_base_3(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd2)
{
  extern __shared__ dcomplex tx[];
  dcomplex *ty = &tx[d2];

  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx[j] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_1(is, logd2, d2, u, tx, ty, lid, lws);

  if (logd2 % 2 == 0) {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = tx[j];
    }
  }
  else {
    for (int j = lid; j < d2; j += lws) {
      int xout_idx = ki_idx + j*(d1+1);
      xout[xout_idx] = ty[j];
    }
  }
}


__global__ void k_cffts3_base_0(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *tx,
                                dcomplex *ty,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd3)
{
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  dcomplex *tx_ji = &tx[j*d1*d3 + i*d3];
  dcomplex *ty_ji = &ty[j*d1*d3 + i*d3];

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx_ji[k] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_0(is, logd3, d3, u, tx_ji, ty_ji, lid, lws);

  if (logd3 % 2 == 0) {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = tx_ji[k];
    }
  }
  else {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = ty_ji[k];
    }
  }
}


__global__ void k_cffts3_base_1(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *tx,
                                dcomplex *ty,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd3)
{
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  dcomplex *tx_ji = &tx[j*d1*d3 + i*d3];
  dcomplex *ty_ji = &ty[j*d1*d3 + i*d3];

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx_ji[k] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_1(is, logd3, d3, u, tx_ji, ty_ji, lid, lws);

  if (logd3 % 2 == 0) {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = tx_ji[k];
    }
  }
  else {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = ty_ji[k];
    }
  }
}


__global__ void k_cffts3_base_2(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd3)
{
  extern __shared__ dcomplex tx[];
  dcomplex *ty = &tx[d3];

  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx[k] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_0(is, logd3, d3, u, tx, ty, lid, lws);

  if (logd3 % 2 == 0) {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = tx[k];
    }
  }
  else {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = ty[k];
    }
  }
}


__global__ void k_cffts3_base_3(dcomplex *xin,
                                dcomplex *xout,
                                dcomplex *u,
                                int is,
                                int d1,
                                int d2,
                                int d3,
                                int logd3)
{
  extern __shared__ dcomplex tx[];
  dcomplex *ty = &tx[d3];

  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx[k] = xin[xin_idx];
  }

  __syncthreads();

  cfftz_base_1(is, logd3, d3, u, tx, ty, lid, lws);

  if (logd3 % 2 == 0) {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = tx[k];
    }
  }
  else {
    for (int k = lid; k < d3; k += lws) {
      int xout_idx = ji_idx + k*d2*(d1+1);
      xout[xout_idx] = ty[k];
    }
  }
}

//---------------------------------------------------------------------
// Computes NY N-point complex-to-complex FFTs of X using an algorithm due
// to Swarztrauber.  X is both the input and the output array, while Y is a
// scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to
// perform FFTs, the array U must be initialized by calling CFFTZ with IS
// set to 0 and M set to MX, where MX is the maximum value of M for any
// subsequent call.
//---------------------------------------------------------------------
__device__ void cfftz_base_0(int is, int m, int n,
                             dcomplex *u,
                             dcomplex *x, 
                             dcomplex *y,
                             int lid, int lws)
{
  for (int l = 1; l <= m; l += 2) {
    fftz2_base_0(is, l, m, n, u, x, y, lid, lws);
    __syncthreads();

    if (l == m) break;

    fftz2_base_0(is, l+1, m, n, u, y, x, lid, lws);
    __syncthreads();
  }
}

__device__ void cfftz_base_1(int is, int m, int n,
                             dcomplex *u,
                             dcomplex *x, 
                             dcomplex *y,
                             int lid, int lws)
{
  for (int l = 1; l <= m; l += 2) {
    fftz2_base_1(is, l, m, n, u, x, y, lid, lws);
    __syncthreads();

    if (l == m) break;

    fftz2_base_1(is, l+1, m, n, u, y, x, lid, lws);
    __syncthreads();
  }
}


//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
__device__ void fftz2_base_0(int is, int l, int m, int n,
                             dcomplex *u,
                             dcomplex *x,
                             dcomplex *y,
                             int lid, int lws)
{
  int k, n1, li, lj, lk, ku, i, i11, i12, i21, i22;
  dcomplex u1, x11, x21, tmp;

  //---------------------------------------------------------------------
  // Set initial parameters.
  //---------------------------------------------------------------------
  n1 = n / 2;
  lk = 1 << (l - 1);
  li = 1 << (m - l);
  lj = 2 * lk;
  ku = li;

  for (i = 0; i <= li - 1; i++) {
    i11 = i * lk;
    i12 = i11 + n1;
    i21 = i * lj;
    i22 = i21 + lk;

    if (is >= 1) {
      u1 = u[ku+i];
    } else {
      u1 = dconjg(u[ku+i]);
    }

    for (k = lid; k <= lk - 1; k += lws) {
      x11.real = x[i11+k].real;
      x11.imag = x[i11+k].imag;
      x21.real = x[i12+k].real;
      x21.imag = x[i12+k].imag;
      y[i21+k] = dcmplx_add(x11, x21);
      tmp = dcmplx_sub(x11, x21);
      y[i22+k] = dcmplx_mul(u1, tmp);
    }
    __syncthreads();
  }
}

__device__ void fftz2_base_1(int is, int l, int m, int n,
                             dcomplex *u,
                             dcomplex *x,
                             dcomplex *y,
                             int lid, int lws)
{
  int k, n1, li, lj, lk, ku, i, i11, i12, i21, i22;
  dcomplex u1, x11, x21, tmp;

  n1 = n / 2;
  lk = 1 << (l - 1);
  li = 1 << (m - l);
  lj = lk * 2;

  int blk = (lk < lws) ? lws / lk : 1;
  int len = lk * blk;
  int ofs = lid / lk;

  for (i = 0; i < li; i += blk) {
    i11 = i * lk;
    i12 = i11 + n1;
    i21 = i * lj + ofs * lk;
    i22 = i21 + lk;
    ku = li + i + ofs;

    if (is >= 1) {
      u1.real = u[ku].real;
      u1.imag = u[ku].imag;
    }
    else {
      u1.real = u[ku].real;
      u1.imag = -u[ku].imag;
    }

    for (k = lid; k < len; k += lws) {
      x11.real = x[i11+k].real;
      x21.real = x[i12+k].real;
      x11.imag = x[i11+k].imag;
      x21.imag = x[i12+k].imag;

      y[i21+k] = dcmplx_add(x11, x21);
      tmp = dcmplx_sub(x11, x21);
      y[i22+k] = dcmplx_mul(u1, tmp);
    }
  }
}


__global__ void k_cffts1_inplace(dcomplex *xin,
                                 dcomplex *xout,
                                 dcomplex *u,
                                 int is,
                                 int d1,
                                 int d2,
                                 int d3,
                                 int logd1)
{
  extern __shared__ dcomplex tx[];

  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx[i].real = xin[xin_idx].real;
    tx[i].imag = xin[xin_idx].imag;
  }

  __syncthreads();

  cfftz_inplace(is, logd1, d1, u, tx, lid, lws);

  for (int i = lid; i < d1; i += lws) {
    int xout_idx = kj_idx + i;
    xout[xout_idx].real = tx[i].real;
    xout[xout_idx].imag = tx[i].imag;
  }
}


__global__ void k_cffts2_inplace(dcomplex *xin,
                                 dcomplex *xout,
                                 dcomplex *u,
                                 int is,
                                 int d1,
                                 int d2,
                                 int d3,
                                 int logd2)
{
  extern __shared__ dcomplex tx[];

  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx[j].real = xin[xin_idx].real;
    tx[j].imag = xin[xin_idx].imag;
  }

  __syncthreads();

  cfftz_inplace(is, logd2, d2, u, tx, lid, lws);

  for (int j = lid; j < d2; j += lws) {
    int xout_idx = ki_idx + j*(d1+1);
    xout[xout_idx].real = tx[j].real;
    xout[xout_idx].imag = tx[j].imag;
  }
}


__global__ void k_cffts3_inplace(dcomplex *xin,
                                 dcomplex *xout,
                                 dcomplex *u,
                                 int is,
                                 int d1,
                                 int d2,
                                 int d3,
                                 int logd3)
{
  extern __shared__ dcomplex tx[];

  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx[k].real = xin[xin_idx].real;
    tx[k].imag = xin[xin_idx].imag;
  }

  __syncthreads();

  cfftz_inplace(is, logd3, d3, u, tx, lid, lws);

  for (int k = lid; k < d3; k += lws) {
    int xout_idx = ji_idx + k*d2*(d1+1);
    xout[xout_idx].real = tx[k].real;
    xout[xout_idx].imag = tx[k].imag;
  }
}

//---------------------------------------------------------------------
// Computes NY N-point complex-to-complex FFTs of X using an algorithm due
// to Swarztrauber.  X is both the input and the output array, while Y is a
// scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to
// perform FFTs, the array U must be initialized by calling CFFTZ with IS
// set to 0 and M set to MX, where MX is the maximum value of M for any
// subsequent call.
//---------------------------------------------------------------------
__device__ void cfftz_inplace(int is, int m, int n,
                              dcomplex *u,
                              dcomplex *x,
                              int lid, int lws)
{
  for (int l = 1; l <= m; l++) {
    transpose(l, m, n, x, lid, lws);
    fftz2_inplace(is, l, m, n, u, x, lid, lws);
    __syncthreads();
  }
}

//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
__device__ void transpose(int l, int m, int n,
                          dcomplex *x,
                          int lid, int lws)
{
  int i0, i1, i2;
  dcomplex x1, x2;

  int lk = 1 << (l - 1);
  int li = 1 << (m - l);

  for (int i = 1; i < li; i <<= 1) {
    for (int k = lid; k < n / 4; k += lws) {
      i0 = 4 * i * (k / lk / i) + 2 * (k / lk % i);
      i1 = (i0 + 1) * lk + (k % lk);
      i2 = (i0 + 2 * i) * lk + (k % lk);

      x1 = x[i1];
      x2 = x[i2];

      x[i1] = x2;
      x[i2] = x1;
    }

    __syncthreads();
  }
}

//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
__device__ void fftz2_inplace(int is, int l, int m, int n,
                              dcomplex *u, dcomplex *x,
                              int lid, int lws)
{
  dcomplex uk, x1, x2, tmp;

  int lk = 1 << (l - 1);
  int ku = 1 << (m - 1);

  int work = (lk < lws) ? lws : lk;

  for (int i = 0; i < n; i += 2 * work) {
    for (int k = lid; k < work; k += lws) {
      if (is >= 1) {
        uk.real = u[(ku + k) / lk].real;
        uk.imag = u[(ku + k) / lk].imag;
      }
      else {
        uk.real = u[(ku + k) / lk].real;
        uk.imag = -u[(ku + k) / lk].imag;
      }

      int i1 = i + 2 * lk * (k / lk) + (k % lk);
      int i2 = i1 + lk;

      x1.real = x[i1].real;
      x1.imag = x[i1].imag;
      x2.real = x[i2].real;
      x2.imag = x[i2].imag;

      x[i1] = dcmplx_add(x1, x2);
      tmp = dcmplx_sub(x1, x2);
      x[i2] = dcmplx_mul(uk, tmp);
    }

    ku += work;
  }
}

__device__ inline void d_vranlc(int n, double *x, double a,  double *y)
{
  /*--------------------------------------------------------------------
    This routine generates N uniform pseudorandom double precision numbers in
    the range (0, 1) by using the linear congruential generator

    x_{k+1} = a x_k  (mod 2^46)

    where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
    before repeating.  The argument A is the same as 'a' in the above formula,
    and X is the same as x_0.  A and X must be odd double precision integers
    in the range (1, 2^46).  The N results are placed in Y and are normalized
    to be between 0 and 1.  X is updated to contain the new seed, so that
    subsequent calls to VRANLC using the same arguments will generate a
    continuous sequence.  If N is zero, only initialization is performed, and
    the variables X, A and Y are ignored.

    This routine is the standard version designed for scalar or RISC systems.
    However, it should produce the same results on any single processor
    computer with at least 48 mantissa bits in double precision floating point
    data.  On 64 bit systems, double precision should be disabled.
    --------------------------------------------------------------------*/

  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;

  double t1, t2, t3, t4, a1, a2, x1, x2, z;

  int i;

  //--------------------------------------------------------------------
  //  Break A into two parts such that A = 2^23 * A1 + A2.
  //--------------------------------------------------------------------
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;

  //--------------------------------------------------------------------
  //  Generate N results.   This loop is not vectorizable.
  //--------------------------------------------------------------------
  for (i = 0; i < n; i++) {
    //--------------------------------------------------------------------
    //  Break X into two parts such that X = 2^23 * X1 + X2, compute
    //  Z = A1 * X2 + A2 * X1  (mod 2^23), and then
    //  X = 2^23 * Z + A2 * X2  (mod 2^46).
    //--------------------------------------------------------------------
    t1 = r23 * (*x);
    x1 = (int) t1;
    x2 = *x - t23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int) (r23 * t1);
    z = t1 - t23 * t2;
    t3 = t23 * z + a2 * x2;
    t4 = (int) (r46 * t3);
    *x = t3 - t46 * t4;
    y[i] = r46 * (*x);
  }
}
