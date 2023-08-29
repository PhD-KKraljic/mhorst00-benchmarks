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

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

#define MK        16
#define MM        (M - MK)
#define NN        (1 << MM)
#define NK        (1 << MK)
#define NQ        10
#define EPSILON   1.0e-8
#define A         1220703125.0
#define S         271828183.0

#define CHUNK_SIZE 128

inline double randlc( __private double *x, double a )
{
  /*
    This routine returns a uniform pseudorandom double precision number in the
    range (0, 1) by using the linear congruential generator

    x_{k+1} = a x_k  (mod 2^46)

    where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
    before repeating.  The argument A is the same as 'a' in the above formula,
    and X is the same as x_0.  A and X must be odd double precision integers
    in the range (1, 2^46).  The returned value RANDLC is normalized to be
    between 0 and 1, i.e. RANDLC = 2^(-46) * x_1.  X is updated to contain
    the new seed x_1, so that subsequent calls to RANDLC using the same
    arguments will generate a continuous sequence.

    This routine should produce the same results on any computer with at least
    48 mantissa bits in double precision floating point data.  On 64 bit
    systems, double precision should be disabled.

    David H. Bailey     October 26, 1990
  */

  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;

  double t1, t2, t3, t4, a1, a2, x1, x2, z;
  double r;

  //--------------------------------------------------------------------
  // Break A into two parts such that A = 2^23 * A1 + A2.
  //--------------------------------------------------------------------
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;

  //--------------------------------------------------------------------
  // Break X into two parts such that X = 2^23 * X1 + X2, compute
  // Z = A1 * X2 + A2 * X1  (mod 2^23), and then
  // X = 2^23 * Z + A2 * X2  (mod 2^46).
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
  r = r46 * (*x);

  return r;
}

inline void vranlc( int n, __private double *x, double a, __private double y[] )
{
  /*
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
  */

  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;

  double t1, t2, t3, t4, a1, a2, x1, x2, z;

  int i;

  //--------------------------------------------------------------------
  // Break A into two parts such that A = 2^23 * A1 + A2.
  //--------------------------------------------------------------------
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;

  //--------------------------------------------------------------------
  // Generate N results.   This loop is not vectorizable.
  //--------------------------------------------------------------------
  for ( i = 0; i < n; i++ ) {
    //--------------------------------------------------------------------
    // Break X into two parts such that X = 2^23 * X1 + X2, compute
    // Z = A1 * X2 + A2 * X1  (mod 2^23), and then
    // X = 2^23 * Z + A2 * X2  (mod 2^46).
    //--------------------------------------------------------------------
    t1 = r23 * (*x);
    x1 = (int) t1;
    x2 = *x - t23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int) (r23 * t1);
    z = t1 - t23 * t2;
    t3 = t23 * z + a2 * x2;
    t4 = (int) (r46 * t3) ;
    *x = t3 - t46 * t4;
    y[i] = r46 * (*x);
  }

  return;
}

__kernel void embar(
    __global double *qqs, __global double *psx, __global double *psy,
    __local double *l_q, __local double *l_sx, __local double *l_sy,
    int k_offset, double an, int k_start, int k_size)
{
  int g_id = get_global_id(0);

  int l_id = get_local_id(0);
  int l_size = get_local_size(0);
  
  double t1, t2, t3, t4, x1, x2, temp_t1;
  int i, ii, ik, kk, l, k;

  double sx = 0.0, sy = 0.0;
  __private double x[2*CHUNK_SIZE];

  for (i = 0; i < NQ; i++) {
    l_q[i * l_size + l_id] = 0.0;
  }

  k = k_start + g_id + 1;
  kk = k_offset + k; 
  t1 = S;
  t2 = an;

  // Find starting seed t1 for this kk.
  for (i = 1; i <= 100; i++) {
    ik = kk / 2;
    if ((2 * ik) != kk) t3 = randlc(&t1, t2);
    if (ik == 0) break;
    t3 = randlc(&t2, t2);
    kk = ik;
  }

  temp_t1 = t1;

  //--------------------------------------------------------------------
  // Compute Gaussian deviates by acceptance-rejection method and 
  // tally counts in concentri//square annuli.  This loop is not 
  // vectorizable. 
  //--------------------------------------------------------------------
  for (ii = 0; ii < NK; ii = ii + CHUNK_SIZE) {
    //--------------------------------------------------------------------
    // Compute uniform pseudorandom numbers.
    //--------------------------------------------------------------------
    vranlc(2 * CHUNK_SIZE, &temp_t1, A, x);
    
    for (i = 0; i < CHUNK_SIZE; i++) {
      x1 = 2.0 * x[2*i] - 1.0;
      x2 = 2.0 * x[2*i+1] - 1.0;
      t1 = x1 * x1 + x2 * x2;
      if (t1 <= 1.0) {
        t2    = sqrt(-2.0 * log(t1) / t1);
        t3    = (x1 * t2);
        t4    = (x2 * t2);
        l     = MAX(fabs(t3), fabs(t4));
        l_q[l * l_size + l_id] += 1.0;
        sx    = sx + t3;
        sy    = sy + t4;
      }
    }
  }
  l_sx[l_id] = sx;
  l_sy[l_id] = sy;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = 0; i < NQ; i++) {
    for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
      if (l_id < p) {
        l_q[i * l_size + l_id] += l_q[i * l_size + l_id + p];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
    if (l_id < p) {
      l_sx[l_id] += l_sx[l_id + p];
      l_sy[l_id] += l_sy[l_id + p];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l_id == 0) {
    for (i = 0; i < NQ; i++) {
      qqs[get_group_id(0) * NQ + i] = l_q[i * l_size + l_id];
    }
    psx[get_group_id(0)] = l_sx[0];
    psy[get_group_id(0)] = l_sy[0];
  }
}
