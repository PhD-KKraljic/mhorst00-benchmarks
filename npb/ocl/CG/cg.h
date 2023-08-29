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

#ifndef __CG_H__
#define __CG_H__

//---------------------------------------------------------------------------
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#else
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
#endif

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf: enable
#endif
//---------------------------------------------------------------------------

#ifndef CLASS
#error "CLASS is not defined"
#endif

//----------
//  Class S:
//----------
#if CLASS == 'S'
#define NA        1400
#define NONZER    7
#define SHIFT     10
#define NITER     15
#define RCOND     1.0e-1
#endif

//----------
//  Class W:
//----------
#if CLASS == 'W'
#define NA        7000
#define NONZER    8
#define SHIFT     12
#define NITER     15
#define RCOND     1.0e-1
#endif

//----------
//  Class A:
//----------
#if CLASS == 'A'
#define NA        14000
#define NONZER    11
#define SHIFT     20
#define NITER     15
#define RCOND     1.0e-1
#endif

//----------
//  Class B:
//----------
#if CLASS == 'B'
#define NA        75000
#define NONZER    13
#define SHIFT     60
#define NITER     75
#define RCOND     1.0e-1
#endif

//----------
//  Class C:
//----------
#if CLASS == 'C'
#define NA        150000
#define NONZER    15
#define SHIFT     110
#define NITER     75
#define RCOND     1.0e-1
#endif

//----------
//  Class D:
//----------
#if CLASS == 'D'
#define NA        1500000
#define NONZER    21
#define SHIFT     500
#define NITER     100
#define RCOND     1.0e-1
#endif

//----------
//  Class E:
//----------
#if CLASS == 'E'
#define NA        9000000
#define NONZER    26
#define SHIFT     1500
#define NITER     100
#define RCOND     1.0e-1
#endif

#define NZ    ((size_t)NA*(NONZER+1)*(NONZER+1))
#define NAZ   ((size_t)NA*(NONZER+1))

#define TRUE    1
#define FALSE   0

typedef bool logical;

#endif //__CG_H__
