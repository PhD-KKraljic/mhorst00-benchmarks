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

//////////////////////////////////////////////////////////////////////////
// Kernels for main()
//////////////////////////////////////////////////////////////////////////
__kernel void main_0(__global double *x,
                     __global double *z,
                     __global double *g_norm_temp1,
                     __global double *g_norm_temp2,
                     __local double *l_norm_temp1,
                     __local double *l_norm_temp2,
                     int n)
{
  int j = get_global_id(0);
  int wid = get_group_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  if (j < n) {
    double x_val = x[j];
    double z_val = z[j];

    l_norm_temp1[lid] = x_val * z_val;
    l_norm_temp2[lid] = z_val * z_val;
  }
  else {
    l_norm_temp1[lid] = 0.0;
    l_norm_temp2[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_norm_temp1[lid] += l_norm_temp1[lid + i];
      l_norm_temp2[lid] += l_norm_temp2[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_norm_temp1[wid] = l_norm_temp1[0];
    g_norm_temp2[wid] = l_norm_temp2[0];
  }
}

__kernel void main_1(__global double *x,
                     __global double *z,
                     double norm_temp2,
                     int n)
{
  int j = get_global_id(0);

  if (j < n) {
    x[j] = norm_temp2 * z[j];
  }
}


//////////////////////////////////////////////////////////////////////////
// Kernels for conj_grad()
//////////////////////////////////////////////////////////////////////////
double mat_mul(__global double *a,
               __global double *b,
               __global int *colidx,
               int row_start,
               int row_end,
               int lid,
               int lws)
{
  double sum = 0.0;

  for (int k = row_start + lid; k < row_end; k += lws) {
    double a_val = a[k];
    double b_val = b[colidx[k]];

    sum += a_val * b_val;
  }

  return sum;
}


//---------------------------------------------------------------------
// Initialize the CG algorithm:
//---------------------------------------------------------------------
__kernel void conj_grad_0(__global double *q,
                          __global double *z,
                          __global double *r,
                          __global double *x,
                          __global double *p,
                          int n)
{
  int j = get_global_id(0);

  if (j < n) {
    q[j] = 0.0;
    z[j] = 0.0;

    double x_val = x[j];
    r[j] = x_val;
    p[j] = x_val;
  }
}


//---------------------------------------------------------------------
// rho = r.r
// Now, obtain the norm of r: First, sum squares of r elements locally...
//---------------------------------------------------------------------
__kernel void conj_grad_1(__global double *r,
                          __global double *g_rho,
                          __local double *l_rho,
                          int n)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  if (j < n) {
    double r_val = r[j];

    l_rho[lid] = r_val * r_val;
  }
  else {
    l_rho[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_rho[lid] += l_rho[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_rho[get_group_id(0)] = l_rho[0];
  }
}


//---------------------------------------------------------------------
// q = A.p
// The partition submatrix-vector multiply: use workspace w
//---------------------------------------------------------------------
__kernel void conj_grad_2(__global int *rowstr,
                          __global int *colidx,
                          __global double *a,
                          __global double *p,
                          __global double *q,
                          __local double *l_sum,
                          int vec_ofs,
                          int mat_ofs)
{
  int j = get_group_id(0) + vec_ofs;
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int row_start = rowstr[j] - mat_ofs;
  int row_end = rowstr[j+1] - mat_ofs;

  double sum = mat_mul(a, p, colidx, row_start, row_end, lid, lws);
  l_sum[lid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_sum[lid] += l_sum[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    q[j] = l_sum[0];
  }
}


//---------------------------------------------------------------------
// Obtain p.q
//---------------------------------------------------------------------
__kernel void conj_grad_3(__global double *p,
                          __global double *q,
                          __global double *g_d,
                          __local double *l_d,
                          int n)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  if (j < n) {
    double p_val = p[j];
    double q_val = q[j];

    l_d[lid] = p_val * q_val;
  }
  else {
    l_d[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_d[lid] += l_d[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_d[get_group_id(0)] = l_d[0];
  }
}


//---------------------------------------------------------------------
// Obtain z = z + alpha*p
// and    r = r - alpha*q
// then   rho = r.r
// Now, obtain the norm of r: First, sum squares of r elements locally..
//---------------------------------------------------------------------
__kernel void conj_grad_4(__global double *p,
                          __global double *q,
                          __global double *r,
                          __global double *z,
                          __global double *g_rho,
                          __local double *l_rho,
                          double alpha,
                          int n)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  if (j < n) {
    double z_val = z[j] + alpha * p[j];
    double r_val = r[j] - alpha * q[j];

    z[j] = z_val;
    r[j] = r_val;

    l_rho[lid] = r_val * r_val;
  }
  else {
    l_rho[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_rho[lid] += l_rho[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_rho[get_group_id(0)] = l_rho[0];
  }
}


//---------------------------------------------------------------------
// p = r + beta*p
//---------------------------------------------------------------------
__kernel void conj_grad_5(__global double *p,
                          __global double *r,
                          double beta,
                          int n)
{
  int j = get_global_id(0);

  if (j < n) {
    p[j] = r[j] + beta * p[j];
  }
}


//---------------------------------------------------------------------
// Compute residual norm explicitly:  ||r|| = ||x - A.z||
// First, form A.z
// The partition submatrix-vector multiply
//---------------------------------------------------------------------
__kernel void conj_grad_6(__global int *rowstr,
                          __global int *colidx,
                          __global double *a,
                          __global double *r,
                          __global double *z,
                          __local double *l_sum,
                          int vec_ofs,
                          int mat_ofs)
{
  int j = get_group_id(0) + vec_ofs;
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  int row_start = rowstr[j] - mat_ofs;
  int row_end = rowstr[j+1] - mat_ofs;

  double sum = mat_mul(a, z, colidx, row_start, row_end, lid, lws);
  l_sum[lid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_sum[lid] += l_sum[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    r[j] = l_sum[0];
  }
}


//---------------------------------------------------------------------
// At this point, r contains A.z
//---------------------------------------------------------------------
__kernel void conj_grad_7(__global double *r,
                          __global double *x,
                          __global double *g_d,
                          __local double *l_d,
                          int n)
{
  int j = get_global_id(0);
  int lid = get_local_id(0);
  int lws = get_local_size(0);

  if (j < n) {
    double d_val = x[j] - r[j];

    l_d[lid] = d_val * d_val;
  }
  else {
    l_d[lid] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_d[lid] += l_d[lid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    g_d[get_group_id(0)] = l_d[0];
  }
}
