#ifndef CL_UTIL_H
#define CL_UTIL_H

#include "type.h"
#include <stdarg.h>
#include <CL/cl.h>

/****************************************************************************/
/* OpenCL Utility Functions                                                 */
/****************************************************************************/

/* Error Checking */
// Exit the host program with a message.
void clu_Exit(const char *format, ...);

// If err_code is not CU_SUCCESS, exit the host program with msg.
void clu_CheckErrorInternal(cl_int err_code, 
                            const char *msg,
                            const char *file,
                            int line);
#define clu_CheckError(e,m)  clu_CheckErrorInternal(e,m,__FILE__,__LINE__)
//#define clu_CheckError(e,m)

/* OpenCL Device */
// Find the device type from the environment variable OPENCL_DEVICE_TYPE. 
// - If the value of OPENCL_DEVICE_TYPE is "cpu", "gpu", or "accelerator",
//   return CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, or CL_DEVICE_ACCELERATOR,
//   respectively.
// - If it is not set or invalid, return CL_DEVICE_TYPE_DEFAULT.
cl_device_type clu_GetDefaultDeviceType();

// Return an available cl_device_id corresponding to device_type.
cl_device_id clu_GetAvailableDevice(cl_device_type device_type);

// Return the name of device. E.g., Geforce_GTX_480.
char *clu_GetDeviceName(cl_device_id device);

// Return the string of device_type. E.g., CL_DEVICE_TYPE_CPU.
const char *clu_GetDeviceTypeName(cl_device_type device_type);


/* Program Build */
// Load the source code from source_file and return the pointer of the source 
// code string. Length of source code is saved through source_len_ret pointer.
char *clu_LoadProgSource(const char *source_file, size_t *source_len_ret);

// Load the OpenCL program binary from binary_file and return the pointer of
// loaded binary. Size of binary is saved through binary_len_ret pointer.
unsigned char *clu_LoadProgBinary(const char *binary_file, 
                                  size_t *binary_len_ret);

// Create a program and build the program.
cl_program clu_MakeProgram(cl_context context,
                           cl_device_id device,
                           char *source_dir,
                           char *source_file, 
                           char *build_option);


/* Misc */
// Return the size that is rounded up to the multiple of group_size.
size_t clu_RoundWorkSize(size_t work_size, size_t group_size);


/****************************************************************************/
/* Constants                                                                */
/****************************************************************************/
#define DEV_VENDOR_NVIDIA       "NVIDIA"

/****************************************************************************/
/* OpenCL Profiling Functions                                               */
/****************************************************************************/

void clu_ProfilerSetup(void);
void clu_ProfilerRelease(void);
void clu_ProfilerStart(void);
void clu_ProfilerStop(void);
void clu_ProfilerClear(void);
void clu_ProfilerPrintResult(void);
void clu_ProfilerPrintElapsedTime(const char* name, double elapsed);

cl_int clu_Profiler_clEnqueueNDRangeKernel(cl_command_queue q, 
                                           cl_kernel k, 
                                           cl_uint wd, 
                                           const size_t *gws_off,
                                           const size_t *gws,
                                           const size_t *lws,
                                           cl_uint num_ev_wlist,
                                           const cl_event *ev_wlist,
                                           cl_event *ev, 
                                           const char *kernel_name);

cl_int clu_Profiler_clEnqueueWriteBuffer(cl_command_queue q,
                                         cl_mem buf,
                                         cl_bool block,
                                         size_t offset,
                                         size_t cb,
                                         const void *ptr,
                                         cl_uint num_ev_wlist,
                                         const cl_event *ev_wlist,
                                         cl_event *ev,
                                         const char* func);

cl_int clu_Profiler_clEnqueueReadBuffer(cl_command_queue q,
                                        cl_mem buf,
                                        cl_bool block,
                                        size_t offset,
                                        size_t cb,
                                        void *ptr,
                                        cl_uint num_ev_wlist,
                                        const cl_event *ev_wlist,
                                        cl_event *ev,
                                        const char* func);

cl_int clu_Profiler_clEnqueueCopyBuffer(cl_command_queue q,
                                        cl_mem src_buf,
                                        cl_mem dst_buf,
                                        size_t src_offset,
                                        size_t dst_offset,
                                        size_t cb,
                                        cl_uint num_ev_wlist,
                                        const cl_event *ev_wlist,
                                        cl_event *ev,
                                        const char* func);

cl_int clu_Profiler_clEnqueueWriteBufferRect(cl_command_queue q,
                                             cl_mem buf,
                                             cl_bool block,
                                             const size_t buffer_origin[3],
                                             const size_t host_origin[3],
                                             const size_t region[3],
                                             size_t buffer_row_pitch,
                                             size_t buffer_slice_pitch,
                                             size_t host_row_pitch,
                                             size_t host_slice_pitch,
                                             void *ptr,
                                             cl_uint num_ev_wlist,
                                             const cl_event *ev_wlist,
                                             cl_event *ev,
                                             const char* func);

cl_int clu_Profiler_clEnqueueReadBufferRect(cl_command_queue q,
                                            cl_mem buf,
                                            cl_bool block,
                                            const size_t buffer_origin[3],
                                            const size_t host_origin[3],
                                            const size_t region[3],
                                            size_t buffer_row_pitch,
                                            size_t buffer_slice_pitch,
                                            size_t host_row_pitch,
                                            size_t host_slice_pitch,
                                            void *ptr,
                                            cl_uint num_ev_wlist,
                                            const cl_event *ev_wlist,
                                            cl_event *ev,
                                            const char* func);


// Wrap original functions
#define clEnqueueNDRangeKernel(a, b, ...) clu_Profiler_clEnqueueNDRangeKernel(a, b, __VA_ARGS__, #b)
#define clEnqueueWriteBuffer(...) clu_Profiler_clEnqueueWriteBuffer(__VA_ARGS__, __func__)
#define clEnqueueReadBuffer(...) clu_Profiler_clEnqueueReadBuffer(__VA_ARGS__, __func__)
#define clEnqueueCopyBuffer(...) clu_Profiler_clEnqueueCopyBuffer(__VA_ARGS__, __func__)
#define clEnqueueWriteBufferRect(...) clu_Profiler_clEnqueueWriteBufferRect(__VA_ARGS__, __func__)
#define clEnqueueReadBufferRect(...) clu_Profiler_clEnqueueReadBufferRect(__VA_ARGS__, __func__)


#endif //CL_UTIL_H
