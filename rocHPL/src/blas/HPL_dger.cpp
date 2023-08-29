/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#include "hpl.hpp"

void HPL_dger_omp(const enum HPL_ORDER ORDER,
                  const int            M,
                  const int            N,
                  const double         ALPHA,
                  const double*        X,
                  const int            INCX,
                  double*              Y,
                  const int            INCY,
                  double*              A,
                  const int            LDA,
                  const int            NB,
                  const int            II,
                  const int            thread_rank,
                  const int            thread_size) {

  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int mm = Mmin(NB - II, M);
    HPL_dger(ORDER, mm, N, ALPHA, X, INCX, Y, INCY, A, LDA);
  }
  ++tile;
  int i = NB - II;
  for(; i < M; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int mm = Mmin(NB, M - i);
      HPL_dger(ORDER, mm, N, ALPHA, X + i * INCX, INCX, Y, INCY, A + i, LDA);
    }
    ++tile;
  }
}
