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

//////////////////////////////////////////////////////////////////////////
// Kernels for main()
//////////////////////////////////////////////////////////////////////////
__global__ void main_0_base(double *x,
                            double *z,
                            double *g_norm_temp1,
                            double *g_norm_temp2,
                            double *l_norm_temp1,
                            double *l_norm_temp2,
                            int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int wid = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double x_val = x[j];
    double z_val = z[j];

    l_norm_temp1[j] = x_val * z_val;
    l_norm_temp2[j] = z_val * z_val;
  }
  else {
    l_norm_temp1[j] = 0.0;
    l_norm_temp2[j] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_norm_temp1[j] += l_norm_temp1[j + i];
      l_norm_temp2[j] += l_norm_temp2[j + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_norm_temp1[wid] = l_norm_temp1[j];
    g_norm_temp2[wid] = l_norm_temp2[j];
  }
}

__global__ void main_0_opt(double *x,
                           double *z,
                           double *g_norm_temp1,
                           double *g_norm_temp2,
                           int n)
{
  extern __shared__ double l_norm_temp1[];
  double *l_norm_temp2 = &l_norm_temp1[blockDim.x];

  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int wid = blockIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

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

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_norm_temp1[lid] += l_norm_temp1[lid + i];
      l_norm_temp2[lid] += l_norm_temp2[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_norm_temp1[wid] = l_norm_temp1[0];
    g_norm_temp2[wid] = l_norm_temp2[0];
  }
}

__global__ void main_1(double *x,
                       double *z,
                       double norm_temp2,
                       int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < n) {
    x[j] = norm_temp2 * z[j];
  }
}


//////////////////////////////////////////////////////////////////////////
// Kernels for conj_grad()
//////////////////////////////////////////////////////////////////////////
//---------------------------------------------------------------------
// Initialize the CG algorithm:
//---------------------------------------------------------------------
__global__ void conj_grad_0(double *q,
                            double *z,
                            double *r,
                            double *x,
                            double *p,
                            int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;

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
__global__ void conj_grad_1_base(double *r,
                                 double *g_rho,
                                 double *l_rho,
                                 int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double r_val = r[j];

    l_rho[j] = r_val * r_val;
  }
  else {
    l_rho[j] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_rho[j] += l_rho[j + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_rho[blockIdx.x] = l_rho[j];
  }
}

__global__ void conj_grad_1_opt(double *r,
                                double *g_rho,
                                int n)
{
  extern __shared__ double l_rho[];
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double r_val = r[j];

    l_rho[lid] = r_val * r_val;
  }
  else {
    l_rho[lid] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_rho[lid] += l_rho[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_rho[blockIdx.x] = l_rho[0];
  }
}


//---------------------------------------------------------------------
// q = A.p
// The partition submatrix-vector multiply: use workspace w
//---------------------------------------------------------------------
__global__ void conj_grad_2_base(int *rowstr,
                                 int *colidx,
                                 double *a,
                                 double *p,
                                 double *q,
                                 double *l_sum,
                                 int vec_ofs,
                                 int mat_ofs)
{
  int j = blockIdx.x + vec_ofs;
  int k = blockIdx.x * blockDim.x + threadIdx.x + vec_ofs;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int row_start = rowstr[j] - mat_ofs;
  int row_end = rowstr[j+1] - mat_ofs;

  double sum = 0.0;

  for (int k = row_start + lid; k < row_end; k += lws) {
    double a_val = a[k];
    double p_val = p[colidx[k]];

    sum += a_val * p_val;
  }

  l_sum[k] = sum;

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_sum[k] += l_sum[k + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    q[j] = l_sum[k];
  }
}

__global__ void conj_grad_2_opt(int *rowstr,
                                int *colidx,
                                double *a,
                                double *p,
                                double *q,
                                int vec_ofs,
                                int mat_ofs)
{
  extern __shared__ double l_sum[];
  int j = blockIdx.x + vec_ofs;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int row_start = rowstr[j] - mat_ofs;
  int row_end = rowstr[j+1] - mat_ofs;

  double sum = 0.0;

  for (int k = row_start + lid; k < row_end; k += lws) {
    double a_val = a[k];
    double p_val = p[colidx[k]];

    sum += a_val * p_val;
  }

  l_sum[lid] = sum;

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_sum[lid] += l_sum[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    q[j] = l_sum[0];
  }
}


//---------------------------------------------------------------------
// Obtain p.q
//---------------------------------------------------------------------
__global__ void conj_grad_3_base(double *p,
                                 double *q,
                                 double *g_d,
                                 double *l_d,
                                 int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double p_val = p[j];
    double q_val = q[j];

    l_d[j] = p_val * q_val;
  }
  else {
    l_d[j] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_d[j] += l_d[j + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_d[blockIdx.x] = l_d[j];
  }
}

__global__ void conj_grad_3_opt(double *p,
                                double *q,
                                double *g_d,
                                int n)
{
  extern __shared__ double l_d[];
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double p_val = p[j];
    double q_val = q[j];

    l_d[lid] = p_val * q_val;
  }
  else {
    l_d[lid] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_d[lid] += l_d[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_d[blockIdx.x] = l_d[0];
  }
}


//---------------------------------------------------------------------
// Obtain z = z + alpha*p
// and    r = r - alpha*q
// then   rho = r.r
// Now, obtain the norm of r: First, sum squares of r elements locally..
//---------------------------------------------------------------------
__global__ void conj_grad_4_base(double *p,
                                 double *q,
                                 double *r,
                                 double *z,
                                 double *g_rho,
                                 double *l_rho,
                                 double alpha,
                                 int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double z_val = z[j] + alpha * p[j];
    double r_val = r[j] - alpha * q[j];

    z[j] = z_val;
    r[j] = r_val;

    l_rho[j] = r_val * r_val;
  }
  else {
    l_rho[j] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_rho[j] += l_rho[j + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_rho[blockIdx.x] = l_rho[j];
  }
}

__global__ void conj_grad_4_opt(double *p,
                                double *q,
                                double *r,
                                double *z,
                                double *g_rho,
                                double alpha,
                                int n)
{
  extern __shared__ double l_rho[];
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

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

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_rho[lid] += l_rho[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_rho[blockIdx.x] = l_rho[0];
  }
}


//---------------------------------------------------------------------
// p = r + beta*p
//---------------------------------------------------------------------
__global__ void conj_grad_5(double *p,
                            double *r,
                            const double beta,
                            int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < n) {
    p[j] = r[j] + beta * p[j];
  }
}


//---------------------------------------------------------------------
// Compute residual norm explicitly:  ||r|| = ||x - A.z||
// First, form A.z
// The partition submatrix-vector multiply
//---------------------------------------------------------------------
__global__ void conj_grad_6_base(int *rowstr,
                                 int *colidx,
                                 double *a,
                                 double *r,
                                 double *z,
                                 double *l_sum,
                                 int vec_ofs,
                                 int mat_ofs)
{
  int j = blockIdx.x + vec_ofs;
  int k = blockIdx.x * blockDim.x + threadIdx.x + vec_ofs;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int row_start = rowstr[j] - mat_ofs;
  int row_end = rowstr[j+1] - mat_ofs;

  double sum = 0.0;

  for (long k = row_start + lid; k < row_end; k += lws) {
    double a_val = a[k];
    double z_val = z[colidx[k]];

    sum += a_val * z_val;
  }

  l_sum[k] = sum;

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_sum[k] += l_sum[k + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    r[j] = l_sum[k];
  }
}

__global__ void conj_grad_6_opt(int *rowstr,
                                int *colidx,
                                double *a,
                                double *r,
                                double *z,
                                int vec_ofs,
                                int mat_ofs)
{
  extern __shared__ double l_sum[];
  int j = blockIdx.x + vec_ofs;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  int row_start = rowstr[j] - mat_ofs;
  int row_end = rowstr[j+1] - mat_ofs;

  double sum = 0.0;

  for (long k = row_start + lid; k < row_end; k += lws) {
    double a_val = a[k];
    double z_val = z[colidx[k]];

    sum += a_val * z_val;
  }

  l_sum[lid] = sum;

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_sum[lid] += l_sum[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    r[j] = l_sum[0];
  }
}


//---------------------------------------------------------------------
// At this point, r contains A.z
//---------------------------------------------------------------------
__global__ void conj_grad_7_base(double *r,
                                 double *x,
                                 double *g_d,
                                 double *l_d,
                                 int n)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double d_val = x[j] - r[j];

    l_d[j] = d_val * d_val;
  }
  else {
    l_d[j] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_d[j] += l_d[j + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_d[blockIdx.x] = l_d[j];
  }
}

__global__ void conj_grad_7_opt(double *r,
                                double *x,
                                double *g_d,
                                int n)
{
  extern __shared__ double l_d[];
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int lid = threadIdx.x;
  int lws = blockDim.x;

  if (j < n) {
    double d_val = x[j] - r[j];

    l_d[lid] = d_val * d_val;
  }
  else {
    l_d[lid] = 0.0;
  }

  __syncthreads();

  for (int i = lws >> 1; i > 0; i >>= 1) {
    if (lid < i) {
      l_d[lid] += l_d[lid + i];
    }

    __syncthreads();
  }

  if (lid == 0) {
    g_d[blockIdx.x] = l_d[0];
  }
}
