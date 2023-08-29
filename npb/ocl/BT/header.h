//---------------------------------------------------------------------------//
//                                                                           //
//  This benchmark is an OpenCL C version of the NPB BT code. This OpenCL C  //
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

//---------------------------------------------------------------------
//---------------------------------------------------------------------
//
//  header.h
//
//---------------------------------------------------------------------
//---------------------------------------------------------------------
 
//---------------------------------------------------------------------
// The following include file is generated automatically by the
// "setparams" utility. It defines 
//      maxcells:      the square root of the maximum number of processors
//      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
//      dt_default:    default time step for this problem size if no
//                     config file
//      niter_default: default number of iterations for this problem size
//---------------------------------------------------------------------

#include "npbparams.h"
#include "type.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "cl_util.h"

#define AA            0
#define BB            1
#define CC            2
#define BLOCK_SIZE    5

/* common /global/ */
extern double elapsed_time;
extern int grid_points[3];
extern logical timeron;

/* common /constants/ */
extern double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
              dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
              dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
              ce[5][13], dxmax, dymax, dzmax, xxcon1, xxcon2, 
              xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
              dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
              yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
              zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
              dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
              dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
              c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1,
              dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
              c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
              c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

#define IMAX      PROBLEM_SIZE
#define JMAX      PROBLEM_SIZE
#define KMAX      PROBLEM_SIZE
#define IMAXP     IMAX/2*2
#define JMAXP     JMAX/2*2


// to improve cache performance, grid dimensions padded by 1 
// for even number sizes only.
/* common /fields/ */
extern double us     [KMAX][JMAXP+1][IMAXP+1];
extern double vs     [KMAX][JMAXP+1][IMAXP+1];
extern double ws     [KMAX][JMAXP+1][IMAXP+1];
extern double qs     [KMAX][JMAXP+1][IMAXP+1];
extern double rho_i  [KMAX][JMAXP+1][IMAXP+1];
extern double square [KMAX][JMAXP+1][IMAXP+1];
extern double forcing[KMAX][JMAXP+1][IMAXP+1][5];
extern double u      [KMAX][JMAXP+1][IMAXP+1][5];
extern double rhs    [KMAX][JMAXP+1][IMAXP+1][5];

/* common /work_1d/ */
extern double cuf[PROBLEM_SIZE+1];
extern double q  [PROBLEM_SIZE+1];
extern double ue [PROBLEM_SIZE+1][5];
extern double buf[PROBLEM_SIZE+1][5];
      

//-----------------------------------------------------------------------
// Timer constants
//-----------------------------------------------------------------------
#define t_total     1
#define t_rhsx      2
#define t_rhsy      3
#define t_rhsz      4
#define t_rhs       5
#define t_xsolve    6
#define t_ysolve    7
#define t_zsolve    8
#define t_rdis1     9
#define t_rdis2     10
#define t_add       11
#define t_last      11


void initialize();
void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
void exact_rhs();
void set_constants();
void adi();
void adi_init();
void adi_free();

void compute_rhs();
void compute_rhs_init(int iter);
void compute_rhs_release();
void compute_rhs_release_ev(int iter);
void compute_rhs_body(int work_step, 
                      int work_base, 
                      int work_num_item, 
                      int copy_buffer_base, 
                      int copy_num_item, 
                      int buf_idx,
                      int wait,
                      cl_event * ev_wb_end_ptr);

void x_solve_init(int iter);
void x_solve_release();
void x_solve_release_ev(int iter);
void x_solve(int work_step,
             int work_base,
             int work_num_item,
             int buf_idx);

void y_solve_init(int iter);
void y_solve_release();
void y_solve_release_ev(int iter);
cl_event* y_solve(int work_step,
                  int work_base,
                  int work_num_item,
                  int buf_idx);

void z_solve_init(int iter);
void z_solve_release();
void z_solve_release_ev(int iter);
void z_solve(int work_step,
             int work_base,
             int work_num_item,
             int buf_idx,
             cl_event *ev_wb_end_ptr);

void add(int work_step, 
         int work_base, 
         int work_num_item,
         int buf_idx);

void error_norm(double rms[5]);
void rhs_norm(double rms[5]);
void verify(int no_time_steps, char *class, logical *verified);

//-----------------------------------------------------------------------
// OpenCL - Macros
//-----------------------------------------------------------------------
#define t_memcpy_pre  20
#define t_memcpy_post 21
#define t_prof_cont   22
#define t_prof_rect   23

// Command Queue Types - default queue is kernel queue 
#define KERNEL_Q  0
#define DATA_Q    1
#define NUM_Q     2

#define VENDOR_ID_AMD 0x1002
#define VENDOR_ID_NVIDIA 0x10de

// Flags
//#define DETAIL_INFO

#ifdef DETAIL_INFO
#define DETAIL_LOG(fmt, ...) fprintf(stdout, " [OpenCL Detailed Info] " fmt "\n", ## __VA_ARGS__)
#else
#define DETAIL_LOG(fmt, ...) 
#endif

//-----------------------------------------------------------------------
// OpenCL - Variables
//-----------------------------------------------------------------------

/* OpenCL environment variables */
extern char               *device_name;
extern cl_device_type     device_type;
extern cl_device_id       device;
extern cl_context         context;
extern cl_command_queue   cmd_q[NUM_Q];
extern cl_program         p_rhs_baseline,
                          p_rhs_parallel, 
                          p_solve_baseline,
                          p_solve_parallel,
                          p_x_solve_memlayout,
                          p_y_solve_memlayout,
                          p_z_solve_memlayout,
                          p_solve_fullopt, 
                          p_add;

extern cl_kernel          k_add;

extern cl_mem             m_us, 
                          m_vs, 
                          m_ws, 
                          m_qs,
                          m_rho_i, 
                          m_square,
                          m_forcing[2], 
                          m_u[2], 
                          m_rhs[2],
                          m_lhsA, 
                          m_lhsB, 
                          m_lhsC;
extern cl_mem             m_fjac,
                          m_njac,
                          m_lhs;

/* OpenCL profiling variables */
extern cl_event           *loop1_ev_wb_start, 
                          *loop1_ev_wb_end,
                          *loop1_ev_rb_end,
                          *loop2_ev_wb_start, 
                          *loop2_ev_wb_end,
                          *loop2_ev_rb_start,
                          *loop2_ev_rb_end,
                          *loop2_ev_kernel_add;

/* OpenCL dynamic configuration flags */
extern int                split_flag, 
                          buffering_flag;

/* OpenCL device dependent variables */
extern size_t             max_work_item_sizes[3], 
                          max_work_group_size;
extern cl_uint            max_compute_unit;
extern cl_ulong           local_mem_size, 
                          max_mem_alloc_size,
                          gmem_size;
extern int                work_num_item_default, 
                          work_num_item_default_j;
extern int                loop1_work_max_iter, 
                          loop1_work_num_item_default,
                          loop2_work_max_iter, 
                          loop2_work_num_item_default;

/* OpenCL optimization levels enumerate */
enum OptLevel {
  OPT_BASELINE=0,
  OPT_PARALLEL,
  OPT_GLOBALMEM,
  OPT_MEMLAYOUT,
  OPT_SYNC,
  OPT_FULL,
};

enum DataTransferType {
  DT_CONT = 0,
  DT_RECT,
};


/* OpenCL optimization level*/
extern enum OptLevel      g_opt_level;

void setup_opencl(int argc, char *argv[]);
void release_opencl();
int get_loop1_copy_num_item(int work_base, int work_num_item);
int get_loop1_copy_buffer_base(int work_base);
int get_loop1_copy_host_base(int work_base);


