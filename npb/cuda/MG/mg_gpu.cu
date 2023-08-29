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


__global__ void kernel_zero3(double *u, int n1, int n2, int n3)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n1 * n2 * n3) {
    u[i] = 0.0;
  }
}


__global__ void kernel_psinv_opt(double *_r,
                                 double *_u,
                                 double *c,
                                 int n1,
                                 int n2,
                                 int len)
{
  extern __shared__ double r0[];
  double *r1 = &r0[len];

  int zid = blockDim.z * blockIdx.z + threadIdx.z + 1;
  int yid = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int ofs = (len - 2) * blockIdx.x;

  int lid = threadIdx.x;
  int lws = blockDim.x;

  double *r = _r + (zid * n2 + yid) * n1 + ofs;
  double *u = _u + (zid * n2 + yid) * n1 + ofs;

  for (int i = lid; i < len; i += lws) {
    r0[i] = r[i];

    r1[i] = r[i - n1] + r[i + n1]
          + r[i - n2 * n1] + r[i + n2 * n1];
  }

  __syncthreads();

  double c0 = c[0];
  double c1 = c[1];
  double c2 = c[2];

  for (int i = lid + 1; i < len - 1; i += lws) {
    double r2 = r[i - n2 * n1 - n1]
              + r[i - n2 * n1 + n1]
              + r[i + n2 * n1 - n1]
              + r[i + n2 * n1 + n1];

    u[i] = u[i]
         + c0 * r0[i]
         + c1 * (r1[i] + r0[i - 1] + r0[i + 1])
         + c2 * (r2 + r1[i - 1] + r1[i + 1]);
  }
}


__global__ void kernel_psinv_base(double *_r,
                                  double *_u,
                                  double *c,
                                  int n1,
                                  int n2,
                                  int len)
{
  int zid = blockDim.z * blockIdx.z + threadIdx.z + 1;
  int yid = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int ofs = (len - 2) * blockIdx.x;

  int lid = threadIdx.x;
  int lws = blockDim.x;

  double *r = _r + (zid * n2 + yid) * n1 + ofs;
  double *u = _u + (zid * n2 + yid) * n1 + ofs;

  double c0 = c[0];
  double c1 = c[1];
  double c2 = c[2];

  for (int i = lid + 1; i < len - 1; i += lws) {
    double r10 = r[i - n1] + r[i + n1]
               + r[i - n2 * n1] + r[i + n2 * n1];

    double r11 = r[i - n1 - 1] + r[i + n1 - 1]
               + r[i - n2 * n1 - 1] + r[i + n2 * n1 - 1];

    double r12 = r[i - n1 + 1] + r[i + n1 + 1]
               + r[i - n2 * n1 + 1] + r[i + n2 * n1 + 1];

    double r20 = r[i - n2 * n1 - n1] + r[i - n2 * n1 + n1]
               + r[i + n2 * n1 - n1] + r[i + n2 * n1 + n1];

    u[i] = u[i]
         + c0 * r[i]
         + c1 * (r10 + r[i - 1] + r[i + 1])
         + c2 * (r20 + r11 + r12);
  }
}


__global__ void kernel_resid_opt(double *r,
                                 double *u,
                                 double *v,
                                 double *a,
                                 int n1,
                                 int n2,
                                 int len)
{
  extern __shared__ double u1[];
  double *u2 = &u1[len];

  int zid = blockDim.z * blockIdx.z + threadIdx.z + 1;
  int yid = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int ofs = (len - 2) * blockIdx.x;

  int lid = threadIdx.x;
  int lws = blockDim.x;

  for (int i = lid; i < len; i += lws) {
    double u1_n = u[(zid * n2 + yid) * n1 + ofs + i - n1];
    double u1_s = u[(zid * n2 + yid) * n1 + ofs + i + n1];
    double u1_b = u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1];
    double u1_t = u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1];
    u1[i] = u1_n + u1_s + u1_b + u1_t;
  }

  for (int i = lid; i < len; i += lws) {
    double u2_bn = u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 - n1];
    double u2_bs = u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 + n1];
    double u2_tn = u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 - n1];
    double u2_ts = u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 + n1];
    u2[i] = u2_bn + u2_bs + u2_tn + u2_ts;
  }

  __syncthreads();

  double a0 = a[0];
  double a2 = a[2];
  double a3 = a[3];

  for (int i = lid + 1; i < len - 1; i += lws) {
    double u0 = u[(zid * n2 + yid) * n1 + ofs + i];

    r[(zid * n2 + yid) * n1 + ofs + i] = v[(zid * n2 + yid) * n1 + ofs + i]
                                       - a0 * u0
                                       - a2 * (u2[i] + u1[i - 1] + u1[i + 1])
                                       - a3 * (u2[i - 1] + u2[i + 1]);
  }
}


__global__ void kernel_resid_base(double *r,
                                  double *u,
                                  double *v,
                                  double *a,
                                  int n1,
                                  int n2,
                                  int len)
{
  int zid = blockDim.z * blockIdx.z + threadIdx.z + 1;
  int yid = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int ofs = (len - 2) * blockIdx.x;

  int lid = threadIdx.x;
  int lws = blockDim.x;

  for (int i = lid + 1; i < len - 1; i += lws) {
    double u11 = u[(zid * n2 + yid) * n1 + ofs + i - n1 - 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n1 - 1]
               + u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 - 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 - 1];

    double u12 = u[(zid * n2 + yid) * n1 + ofs + i - n1 + 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n1 + 1]
               + u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 + 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 + 1];

    double u20 = u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 - n1]
               + u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 + n1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 - n1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 + n1];

    double u21 = u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 - n1 - 1]
               + u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 + n1 - 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 - n1 - 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 + n1 - 1];

    double u22 = u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 - n1 + 1]
               + u[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 + n1 + 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 - n1 + 1]
               + u[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 + n1 + 1];

    r[(zid * n2 + yid) * n1 + ofs + i] = v[(zid * n2 + yid) * n1 + ofs + i]
                                       - a[0] * u[(zid * n2 + yid) * n1 + ofs + i]
                                       - a[2] * (u20 + u11 + u12)
                                       - a[3] * (u21 + u22);
  }
}


__global__ void kernel_rprj3(double *rk,
                             double *rj,
                             int m1k, int m2k,
                             int m1j, int m2j,
                             long long ofs_j)
{
  int j3 = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int j2 = (blockDim.x * blockIdx.x + threadIdx.x) / (m1j - 2) + 1;
  int j1 = (blockDim.x * blockIdx.x + threadIdx.x) % (m1j - 2) + 1;

  int i3 = 2 * j3 - 1;
  int i2 = 2 * j2 - 1;
  int i1 = 2 * j1 - 1;

  double *s = rj + (j3 * m2j + j2) * m1j + j1 + ofs_j;

  double *r0 = rk + (i3 * m2k + i2) * m1k + i1;
  double *r1 = r0 + m1k;
  double *r2 = r1 + m1k;
  double *r3 = r0 + m2k * m1k;
  double *r4 = r3 + m1k;
  double *r5 = r4 + m1k;
  double *r6 = r3 + m2k * m1k;
  double *r7 = r6 + m1k;
  double *r8 = r7 + m1k;

  double s1 = (r0[0] + r0[2])
            + (r2[0] + r2[2])
            + (r6[0] + r6[2])
            + (r8[0] + r8[2]);

  double s2 = (r0[1] + r2[1])
            + (r6[1] + r8[1])
            + (r1[0] + r1[2])
            + (r3[0] + r3[2])
            + (r5[0] + r5[2])
            + (r7[0] + r7[2]);

  double s3 = (r4[0] + r4[2])
            + (r1[1] + r3[1])
            + (r5[1] + r7[1]);

  double s4 = r4[1];

  *s = s1 * 0.0625 + s2 * 0.125 + s3 * 0.25 + s4 * 0.5;
}


__global__ void kernel_interp(double *uj,
                              double *uk,
                              int mm1, int mm2,
                              int n1, int n2,
                              long long ofs_j)
{
  int i3 = blockDim.z * blockIdx.z + threadIdx.z;
  int i2 = blockDim.y * blockIdx.y + threadIdx.y;
  int i1 = blockDim.x * blockIdx.x + threadIdx.x;

  int j3 = i3 * 2;
  int j2 = i2 * 2;
  int j1 = i1 * 2;

  double *z[4];
  double *u[4];

  if (i1 < mm1-1 && i2 < mm2-1) {
    double *z0 = uj + (i3 * mm2 + i2) * mm1 + i1 + ofs_j;
    double *z1 = z0 + mm1;
    double *z2 = z0 + mm2 * mm1;
    double *z3 = z2 + mm1;

    double *u0 = uk + (j3 * n2 + j2) * n1 + j1;
    double *u1 = u0 + n1;
    double *u2 = u0 + n2 * n1;
    double *u3 = u2 + n1;

    double u00 = z0[0];
    double u01 = z0[1];

    double u10 = z1[0] + u00;
    double u11 = z1[1] + u01;

    double u20 = z2[0] + u00;
    double u21 = z2[1] + u01;

    double u30 = z3[0] + u10 + u20 - u00;
    double u31 = z3[1] + u11 + u21 - u01;

    u0[0] += u00;
    u0[1] += 0.5 * (u00 + u01);

    u1[0] += 0.5 * u10;
    u1[1] += 0.25 * (u10 + u11);

    u2[0] += 0.5 * u20;
    u2[1] += 0.25 * (u20 + u21);

    u3[0] += 0.25 * u30;
    u3[1] += 0.125 * (u30 + u31);
  }
}


__global__ void kernel_norm2u3_opt(double *or_,
                                   int n1,
                                   int n2,
                                   double *g_sum,
                                   double *g_max)
{
  extern __shared__ double l_sum[];
  double *l_max = &l_sum[blockDim.x];

  int i3 = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int i2 = blockIdx.x + 1;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  double *r = or_ + (i3 * n2 + i2) * n1;

  double s = 0.0;
  double rnmu = 0.0;

  for (int i1 = lid+1; i1 < n1-1; i1 += lws) {
    double r_val = r[i1];
    s += r_val * r_val;
    double a = fabs(r_val);
    rnmu = (a > rnmu) ? a : rnmu;
  }

  l_sum[lid] = s;
  l_max[lid] = rnmu;

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      double l_max1 = l_max[lid];
      double l_max2 = l_max[lid + i];

      l_sum[lid] += l_sum[lid + i];
      l_max[lid] = (l_max1 > l_max2) ? l_max1 : l_max2;
    }

    __syncthreads();
  }

  if (lid == 0) {
    int g_idx = blockIdx.y * gridDim.x + blockIdx.x;

    g_sum[g_idx] = l_sum[0];
    g_max[g_idx] = l_max[0];
  }
}


__global__ void kernel_norm2u3_base(double *or_,
                                    int n1,
                                    int n2,
                                    double *g_sum,
                                    double *g_max,
                                    double *l_sum,
                                    double *l_max)
{
  int i3 = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int i2 = blockIdx.x + 1;
  int gid = blockIdx.y * gridDim.x * blockDim.y * blockDim.x
          + blockIdx.x * blockDim.y * blockDim.x
          + threadIdx.y * blockDim.x
          + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  double *r = or_ + (i3 * n2 + i2) * n1;

  double s = 0.0;
  double rnmu = 0.0;

  for (int i1 = lid+1; i1 < n1-1; i1 += lws) {
    double r_val = r[i1];
    s += r_val * r_val;
    double a = fabs(r_val);
    rnmu = (a > rnmu) ? a : rnmu;
  }

  l_sum[gid] = s;
  l_max[gid] = rnmu;

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      double l_max1 = l_max[gid];
      double l_max2 = l_max[gid + i];

      l_sum[gid] += l_sum[gid + i];
      l_max[gid] = (l_max1 > l_max2) ? l_max1 : l_max2;
    }

    __syncthreads();
  }

  if (lid == 0) {
    int bid = blockIdx.y * gridDim.x + blockIdx.x;

    g_sum[bid] = l_sum[gid];
    g_max[bid] = l_max[gid];
  }
}


__global__ void kernel_comm3_1(double *u, int n1, int n2, int n3)
{
  int i3 = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int i2 = blockDim.x * blockIdx.x + threadIdx.x + 1;

  int idx = (i3 * n2 + i2) * n1;

  if (i2 < n2-1) {
    u[idx] = u[idx + n1 - 2];
    u[idx + n1 - 1] = u[idx + 1];
  }
}


__global__ void kernel_comm3_2(double *u, int n1, int n2, int n3)
{
  int i3 = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int i1 = blockDim.x * blockIdx.x + threadIdx.x;

  int idx = i3 * n2 * n1 + i1;

  if (i1 < n1) {
    u[idx] = u[idx + (n2 - 2) * n1];
    u[idx + (n2 - 1) * n1] = u[idx + n1];
  }
}


__global__ void kernel_comm3_3(double *u, int n1, int n2, int n3)
{
  int i2 = blockDim.y * blockIdx.y + threadIdx.y;
  int i1 = blockDim.x * blockIdx.x + threadIdx.x;

  int idx = i2 * n1 + i1;

  if (i1 < n1) {
    u[idx] = u[idx + (n3 - 2) * n2 * n1];
    u[idx + (n3 - 1) * n2 * n1] = u[idx + n2 * n1];
  }
}
