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


__kernel void kernel_zero3(__global double *u, int n1, int n2, int n3)
{
  int i = get_global_id(0);

  if (i < n1 * n2 * n3) {
    u[i] = 0.0;
  }
}


__kernel void kernel_psinv(__global double *r,
                           __global double *u,
                           __global double *c,
                           int n1,
                           int n2,
                           int len,
                           __local double *r0,
                           __local double *r1)
{
  int zid = get_global_id(2) + 1;
  int yid = get_global_id(1) + 1;
  int ofs = (len - 2) * get_group_id(0);

  int lid = get_local_id(0);
  int lws = get_local_size(0);

  for (int i = lid; i < len; i += lws) {
    r0[i] = r[(zid * n2 + yid) * n1 + ofs + i];

    double r1_n = r[(zid * n2 + yid) * n1 + ofs + i - n1];
    double r1_s = r[(zid * n2 + yid) * n1 + ofs + i + n1];
    double r1_b = r[(zid * n2 + yid) * n1 + ofs + i - n2 * n1];
    double r1_t = r[(zid * n2 + yid) * n1 + ofs + i + n2 * n1];

    r1[i] = r1_n + r1_s + r1_b + r1_t;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  double c0 = c[0];
  double c1 = c[1];
  double c2 = c[2];

  for (int i = lid + 1; i < len - 1; i += lws) {
    double r2 = r[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 - n1]
              + r[(zid * n2 + yid) * n1 + ofs + i - n2 * n1 + n1]
              + r[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 - n1]
              + r[(zid * n2 + yid) * n1 + ofs + i + n2 * n1 + n1];

    u[(zid * n2 + yid) * n1 + ofs + i] = u[(zid * n2 + yid) * n1 + ofs + i]
                                       + c0 * r0[i]
                                       + c1 * (r1[i] + r0[i - 1] + r0[i + 1])
                                       + c2 * (r2 + r1[i - 1] + r1[i + 1]);
  }
}


__kernel void kernel_resid(__global double *r,
                           __global double *u,
                           __global double *v,
                           __global double *a,
                           int n1,
                           int n2,
                           int len,
                           __local double *u1,
                           __local double *u2)
{
  int zid = get_global_id(2) + 1;
  int yid = get_global_id(1) + 1;
  int ofs = (len - 2) * get_group_id(0);

  int lid = get_local_id(0);
  int lws = get_local_size(0);

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

  barrier(CLK_LOCAL_MEM_FENCE);

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


__kernel void kernel_rprj3(__global double *rk,
                           __global double *rj,
                           int m1k, int m2k,
                           int m1j, int m2j,
                           long ofs_j)
{
  int j3 = get_global_id(1) + 1;
  int j2 = get_global_id(0) / (m1j - 2) + 1;
  int j1 = get_global_id(0) % (m1j - 2) + 1;

  int i3 = 2 * j3 - 1;
  int i2 = 2 * j2 - 1;
  int i1 = 2 * j1 - 1;

  __global double *r0 = rk + ((i3 + 0) * m2k + i2 + 0) * m1k + i1;
  __global double *r1 = rk + ((i3 + 0) * m2k + i2 + 1) * m1k + i1;
  __global double *r2 = rk + ((i3 + 0) * m2k + i2 + 2) * m1k + i1;
  __global double *r3 = rk + ((i3 + 1) * m2k + i2 + 0) * m1k + i1;
  __global double *r4 = rk + ((i3 + 1) * m2k + i2 + 1) * m1k + i1;
  __global double *r5 = rk + ((i3 + 1) * m2k + i2 + 2) * m1k + i1;
  __global double *r6 = rk + ((i3 + 2) * m2k + i2 + 0) * m1k + i1;
  __global double *r7 = rk + ((i3 + 2) * m2k + i2 + 1) * m1k + i1;
  __global double *r8 = rk + ((i3 + 2) * m2k + i2 + 2) * m1k + i1;
  __global double *s0 = rj + (j3 * m2j + j2) * m1j + j1 + ofs_j;

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

  *s0 = s1 * 0.0625 + s2 * 0.125 + s3 * 0.25 + s4 * 0.5;
}


__kernel void kernel_interp(__global double *uj,
                            __global double *uk,
                            int mm1, int mm2,
                            int n1, int n2,
                            long ofs_j)
{
  int i3 = get_global_id(2);
  int i2 = get_global_id(1);
  int i1 = get_global_id(0);

  int j3 = i3 * 2;
  int j2 = i2 * 2;
  int j1 = i1 * 2;

  if (i1 < mm1-1 && i2 < mm2-1) {
    __global double *z0 = uj + ((i3 + 0) * mm2 + i2 + 0) * mm1 + i1 + ofs_j;
    __global double *z1 = uj + ((i3 + 0) * mm2 + i2 + 1) * mm1 + i1 + ofs_j;
    __global double *z2 = uj + ((i3 + 1) * mm2 + i2 + 0) * mm1 + i1 + ofs_j;
    __global double *z3 = uj + ((i3 + 1) * mm2 + i2 + 1) * mm1 + i1 + ofs_j;

    double z10 = z1[0];
    double z11 = z1[1];

    double z20 = z2[0];
    double z21 = z2[1];

    double u00 = z0[0];
    double u01 = z0[1];

    double u10 = z10 + u00;
    double u11 = z11 + u01;

    double u20 = z20 + u00;
    double u21 = z21 + u01;

    double u30 = z3[0] + z10 + z20 + u00;
    double u31 = z3[1] + z11 + z21 + u01;

    __global double *u0 = uk + ((j3 + 0) * n2 + j2 + 0) * n1 + j1;
    __global double *u1 = uk + ((j3 + 0) * n2 + j2 + 1) * n1 + j1;
    __global double *u2 = uk + ((j3 + 1) * n2 + j2 + 0) * n1 + j1;
    __global double *u3 = uk + ((j3 + 1) * n2 + j2 + 1) * n1 + j1;

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


__kernel void kernel_norm2u3(__global double *or,
                             int n1,
                             int n2,
                             __global double *g_sum,
                             __global double *g_max,
                             __local double *l_sum,
                             __local double *l_max)
{
  int i3 = get_global_id(1) + 1;
  int i2 = get_group_id(0) + 1;
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  __global double *r = or + (i3 * n2 + i2) * n1;

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

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      double l_max1 = l_max[lid];
      double l_max2 = l_max[lid + i];

      l_sum[lid] += l_sum[lid + i];
      l_max[lid] = (l_max1 > l_max2) ? l_max1 : l_max2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    int g_idx = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    g_sum[g_idx] = l_sum[0];
    g_max[g_idx] = l_max[0];
  }
}


__kernel void kernel_comm3_1(__global double *u, int n1, int n2, int n3)
{
  int i3 = get_global_id(1) + 1;
  int i2 = get_global_id(0) + 1;

  int idx = (i3 * n2 + i2) * n1;

  if (i2 < n2-1) {
    u[idx] = u[idx + n1 - 2];
    u[idx + n1 - 1] = u[idx + 1];
  }
}


__kernel void kernel_comm3_2(__global double *u, int n1, int n2, int n3)
{
  int i3 = get_global_id(1) + 1;
  int i1 = get_global_id(0);

  int idx = i3 * n2 * n1 + i1;

  if (i1 < n1) {
    u[idx] = u[idx + (n2 - 2) * n1];
    u[idx + (n2 - 1) * n1] = u[idx + n1];
  }
}


__kernel void kernel_comm3_3(__global double *u, int n1, int n2, int n3)
{
  int i2 = get_global_id(1);
  int i1 = get_global_id(0);

  int idx = i2 * n1 + i1;

  if (i1 < n1) {
    u[idx] = u[idx + (n3 - 2) * n2 * n1];
    u[idx + (n3 - 1) * n2 * n1] = u[idx + n2 * n1];
  }
}
