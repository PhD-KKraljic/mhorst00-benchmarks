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

//---------------------------------------------------------------------------
// double complex
//---------------------------------------------------------------------------
typedef struct {
  double real;
  double imag;
} dcomplex;

#define dcmplx(r,i)       (dcomplex){r, i}
#define dcmplx_add(a,b)   (dcomplex){(a).real+(b).real, (a).imag+(b).imag}
#define dcmplx_sub(a,b)   (dcomplex){(a).real-(b).real, (a).imag-(b).imag}
#define dcmplx_mul(a,b)   (dcomplex){((a).real*(b).real)-((a).imag*(b).imag),\
                                     ((a).real*(b).imag)+((a).imag*(b).real)}
#define dcmplx_mul2(a,b)  (dcomplex){(a).real*(b), (a).imag*(b)}
inline dcomplex dcmplx_div(dcomplex z1, dcomplex z2) {
  double a = z1.real;
  double b = z1.imag;
  double c = z2.real;
  double d = z2.imag;

  double divisor = c*c + d*d;
  double real = (a*c + b*d) / divisor;
  double imag = (b*c - a*d) / divisor;
  dcomplex result = (dcomplex){real, imag};
  return result;
}
#define dcmplx_div2(a,b)  (dcomplex){(a).real/(b), (a).imag/(b)}
#define dcmplx_abs(x)     sqrt(((x).real*(x).real) + ((x).imag*(x).imag))
#define dconjg(x)         (dcomplex){(x).real, -1.0*(x).imag}

void cfftz_base(int is, int m, int n,
                __global dcomplex *u,
                __local dcomplex *x,
                __local dcomplex *y,
                int lid, int lws);

void cfftz_inplace(int is, int m, int n,
                   __global dcomplex *u,
                   __local dcomplex *x,
                   int lid, int lws);

void fftz2_base(int is, int l, int m, int n,
                __global dcomplex *u,
                __local dcomplex *x,
                __local dcomplex *y,
                int lid, int lws);

void fftz2_inplace(int is, int l, int m, int n,
                   __global dcomplex *u,
                   __local dcomplex *x,
                   int lid, int lws);

void transpose(int l, int m, int n,
               __local dcomplex *x,
               int lid, int lws);

inline void vranlc(int n, double *x, double a, __global double *y);


//---------------------------------------------------------------------
// compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
// for time evolution exponent.
//---------------------------------------------------------------------
__kernel void compute_indexmap(__global double *twiddle,
                               int d1, int d2,
                               int ofs3, int len3,
                               double ap)
{
  int i, j, k, kk, kk2, jj, kj2, ii;

  k = get_global_id(2);
  j = get_global_id(1);
  i = get_global_id(0);

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
__kernel void compute_initial_conditions(__global dcomplex *u1,
                                         __global double *starts,
                                         int d1,
                                         int d2,
                                         int d3)
{
  int k = get_global_id(0);

  if (k < d3) {
    double x0 = starts[k];
    for (int j = 0; j < d2; j++) {
      vranlc(2*NX, &x0, A, (__global double *) &u1[k*d2*(d1+1) + j*(d1+1)]);
    }
    starts[k] = x0;
  }
}


//---------------------------------------------------------------------
// evolve u0 -> u1 (t time steps) in fourier space
//---------------------------------------------------------------------
__kernel void evolve(__global dcomplex *u0,
                     __global dcomplex *u1,
                     __global const double *twiddle,
                     int d1,
                     int d2,
                     int d3)
{
  int k = get_global_id(2);
  int j = get_global_id(1);
  int i = get_global_id(0);

  if (k < d3 && j < d2 && i < d1) {
    int idx = k*d2*(d1+1) + j*(d1+1) + i;
    u0[idx] = dcmplx_mul2(u0[idx], twiddle[idx]);
    u1[idx] = u0[idx];
  }
}


__kernel void checksum(__global dcomplex *u1,
                       __global dcomplex *g_chk,
                       __local dcomplex *l_chk,
                       int d1,
                       int d2)
{
  int q, r, s;
  int j = get_global_id(0) + 1;
  int lid = get_local_id(0);

  if (j <= 1024) {
    q = j % NX;
    r = 3*j % NY;
    s = 5*j % NZ;
    int u1_idx = s*d2*(d1+1) + r*(d1+1) + q;
    l_chk[lid] = u1[u1_idx];
  }
  else {
    l_chk[lid] = dcmplx(0.0, 0.0);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = get_local_size(0) / 2; i > 0; i >>= 1) {
    if (lid < i) {
      l_chk[lid] = dcmplx_add(l_chk[lid], l_chk[lid+i]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_chk[get_group_id(0)] = l_chk[0];
  }
}


__kernel void cffts1_base(__global dcomplex *xin,
                          __global dcomplex *xout,
                          __global dcomplex *u,
                          __local dcomplex *tx,
                          __local dcomplex *ty,
                          int is,
                          int d1,
                          int d2,
                          int d3,
                          int logd1)
{
  int k = get_global_id(1);
  int j = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx[i] = xin[xin_idx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  cfftz_base(is, logd1, d1, u, tx, ty, lid, lws);

  for (int i = lid; i < d1; i += lws) {
    int xout_idx = kj_idx + i;
    xout[xout_idx] = tx[i];
  }
}


__kernel void cffts2_base(__global dcomplex *xin,
                          __global dcomplex *xout,
                          __global dcomplex *u,
                          __local dcomplex *tx,
                          __local dcomplex *ty,
                          int is,
                          int d1,
                          int d2,
                          int d3,
                          int logd2)
{
  int k = get_global_id(1);
  int i = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx[j] = xin[xin_idx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  cfftz_base(is, logd2, d2, u, tx, ty, lid, lws);

  for (int j = lid; j < d2; j += lws) {
    int xout_idx = ki_idx + j*(d1+1);
    xout[xout_idx] = tx[j];
  }
}


__kernel void cffts3_base(__global dcomplex *xin,
                          __global dcomplex *xout,
                          __global dcomplex *u,
                          __local dcomplex *tx,
                          __local dcomplex *ty,
                          int is,
                          int d1,
                          int d2,
                          int d3,
                          int logd3)
{
  int j = get_global_id(1);
  int i = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx[k] = xin[xin_idx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  cfftz_base(is, logd3, d3, u, tx, ty, lid, lws);

  for (int k = lid; k < d3; k += lws) {
    int xout_idx = ji_idx + k*d2*(d1+1);
    xout[xout_idx] = tx[k];
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
void cfftz_base(int is, int m, int n,
                __global dcomplex *u,
                __local dcomplex *x,
                __local dcomplex *y,
                int lid, int lws)
{
  for (int l = 1; l <= m; l += 2) {
    fftz2_base(is, l, m, n, u, x, y, lid, lws);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l == m) {
      for (int j = lid; j < n; j += lws) {
        x[j] = y[j];
      }
    }
    else {
      fftz2_base(is, l + 1, m, n, u, y, x, lid, lws);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
void fftz2_base(int is, int l, int m, int n,
           __global dcomplex *u,
           __local dcomplex *x,
           __local dcomplex *y,
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}


__kernel void cffts1_inplace(__global dcomplex *xin,
                             __global dcomplex *xout,
                             __global dcomplex *u,
                             __local dcomplex *tx,
                             int is,
                             int d1,
                             int d2,
                             int d3,
                             int logd1)
{
  int k = get_global_id(1);
  int j = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int kj_idx = k*d2*(d1+1) + j*(d1+1);
  for (int i = lid; i < d1; i += lws) {
    int xin_idx = kj_idx + i;
    tx[i] = xin[xin_idx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  cfftz_inplace(is, logd1, d1, u, tx, lid, lws);

  for (int i = lid; i < d1; i += lws) {
    int xout_idx = kj_idx + i;
    xout[xout_idx] = tx[i];
  }
}


__kernel void cffts2_inplace(__global dcomplex *xin,
                             __global dcomplex *xout,
                             __global dcomplex *u,
                             __local dcomplex *tx,
                             int is,
                             int d1,
                             int d2,
                             int d3,
                             int logd2)
{
  int k = get_global_id(1);
  int i = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int ki_idx = k*d2*(d1+1) + i;
  for (int j = lid; j < d2; j += lws) {
    int xin_idx = ki_idx + j*(d1+1);
    tx[j] = xin[xin_idx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  cfftz_inplace(is, logd2, d2, u, tx, lid, lws);

  for (int j = lid; j < d2; j += lws) {
    int xout_idx = ki_idx + j*(d1+1);
    xout[xout_idx] = tx[j];
  }
}


__kernel void cffts3_inplace(__global dcomplex *xin,
                             __global dcomplex *xout,
                             __global dcomplex *u,
                             __local dcomplex *tx,
                             int is,
                             int d1,
                             int d2,
                             int d3,
                             int logd3)
{
  int j = get_global_id(1);
  int i = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int ji_idx = j*(d1+1) + i;
  for (int k = lid; k < d3; k += lws) {
    int xin_idx = ji_idx + k*d2*(d1+1);
    tx[k] = xin[xin_idx];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  cfftz_inplace(is, logd3, d3, u, tx, lid, lws);

  for (int k = lid; k < d3; k += lws) {
    int xout_idx = ji_idx + k*d2*(d1+1);
    xout[xout_idx] = tx[k];
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
void cfftz_inplace(int is, int m, int n,
                   __global dcomplex *u,
                   __local dcomplex *x,
                   int lid, int lws)
{
  for (int l = 1; l <= m; l++) {
    transpose(l, m, n, x, lid, lws);
    fftz2_inplace(is, l, m, n, u, x, lid, lws);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
void transpose(int l, int m, int n,
               __local dcomplex *x,
               int lid, int lws)
{
  int li = 1 << (m - l);
  int lk = 1 << (l - 1);

  n /= 4;

  for (int i = 1; i < li; i <<= 1) {
    for (int k = lid; k < n; k += lws) {
      int i0 = (k / lk / i * i * 4) + (k / lk % i * 2);
      int i1 = (i0 + 1) * lk + (k % lk);
      int i2 = (i0 + 2 * i) * lk + (k % lk);

      dcomplex x1 = x[i1];
      dcomplex x2 = x[i2];

      x[i1] = x2;
      x[i2] = x1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//---------------------------------------------------------------------
// Performs the L-th iteration of the second variant of the Stockham FFT.
//---------------------------------------------------------------------
void fftz2_inplace(int is, int l, int m, int n,
                   __global dcomplex *u,
                   __local dcomplex *x,
                   int lid, int lws)
{
  int li = 1 << (m - l);
  int lk = 1 << (l - 1);

  n /= 2;

  for (int k = lid; k < n; k += lws) {
    int ku = li + k / lk;
    dcomplex u1 = (is > 0) ? u[ku] : dconjg(u[ku]);

    int i1 = (k / lk * lk * 2) + (k % lk);
    int i2 = i1 + lk;

    dcomplex x1 = x[i1];
    dcomplex x2 = x[i2];

    x[i1] = dcmplx_add(x1, x2);
    dcomplex tmp = dcmplx_sub(x1, x2);
    x[i2] = dcmplx_mul(u1, tmp);
  }
}

inline void vranlc(int n, double *x, double a, __global double *y)
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
