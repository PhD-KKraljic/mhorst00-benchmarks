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

#ifndef __MG_DIM_H__
#define __MG_DIM_H__

#define PSINV_DIM_CPU       2
#define RESID_DIM_CPU       2
#define RPRJ3_DIM_CPU       2
#define INTERP_1_DIM_CPU    2
#define NORM2U3_DIM_CPU     1
#define COMM3_1_DIM_CPU     1
#define COMM3_2_DIM_CPU     1
#define COMM3_3_DIM_CPU     1
#define ZERO3_DIM_CPU       2


#define PSINV_DIM_GPU       2
#define RESID_DIM_GPU       2
#define RPRJ3_DIM_GPU       2
#define INTERP_1_DIM_GPU    2
#define NORM2U3_DIM_GPU     2
#define COMM3_1_DIM_GPU     2
#define COMM3_2_DIM_GPU     2
#define COMM3_3_DIM_GPU     2
#define ZERO3_DIM_GPU       3


#endif //__MG_DIM_H__
