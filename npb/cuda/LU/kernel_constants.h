//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an CUDA® C version of the NPB LU code. This CUDA® C  //
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

#include "npbparams.h"

#define OMEGA_DEFAULT   1.2

//---------------------------------------------------------------------
// from setcoeff()
//---------------------------------------------------------------------
#define nx0     ISIZ1
#define ny0     ISIZ2
#define nz0     ISIZ3

#define dt      DT_DEFAULT
#define omega   OMEGA_DEFAULT


//---------------------------------------------------------------------
// set up coefficients
//---------------------------------------------------------------------
#define dxi 1.0/(nx0 - 1)
#define deta 1.0/(ny0 - 1)
#define dzeta 1.0/(nz0 - 1)

#define tx1 1.0/( dxi * dxi )
#define tx2 1.0/( 2.0 * dxi )
#define tx3 1.0/dxi
#define ty1 1.0/( deta * deta )
#define ty2 1.0/( 2.0 * deta )
#define ty3 1.0/deta
#define tz1 1.0/( dzeta * dzeta )
#define tz2 1.0/( 2.0 * dzeta )
#define tz3 1.0/dzeta


//---------------------------------------------------------------------
// diffusion coefficients
//---------------------------------------------------------------------
#define dx1 0.75
#define dx2 dx1
#define dx3 dx1
#define dx4 dx1
#define dx5 dx1

#define dy1 0.75
#define dy2 dy1
#define dy3 dy1
#define dy4 dy1
#define dy5 dy1

#define dz1 1.00
#define dz2 dz1
#define dz3 dz1
#define dz4 dz1
#define dz5 dz1


//---------------------------------------------------------------------
// fourth difference dissipation
//---------------------------------------------------------------------
#define dssp    (( max(max(dx1, dy1), dz1) ) / 4.0)
//---------------------------------------------------------------------

