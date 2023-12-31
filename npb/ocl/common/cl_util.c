#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include <CL/cl.h>

#include "cl_util.h"

typedef int     BOOL;
#define TRUE    1
#define FALSE   0


/****************************************************************************/
/* OpenCL Utility Functions                                                 */
/****************************************************************************/

// Return the size that is rounded up to the multiple of group_size.
size_t clu_RoundWorkSize(size_t work_size, size_t group_size) {
  size_t rem = work_size % group_size;
  return (rem == 0) ? work_size : (work_size + group_size - rem);
}

// Return the string of OpenCL error code
static const char *clu_ErrorString(cl_int err_code) {
  static const char *errors[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
  };

  unsigned idx = (err_code < 0) ? -err_code : err_code;
  unsigned errorCount = sizeof(errors) / sizeof(errors[0]);
  return (idx < errorCount) ? errors[idx] : "Unknown Error";
}


// Exit with a message
void clu_Exit(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);

  exit(EXIT_FAILURE);
}


// Check err_code and if it is not CL_SUCCESS, exit the program
void clu_CheckErrorInternal(cl_int err_code, const char *msg,
                            const char *file, int line) {
  if (err_code != CL_SUCCESS) {
    clu_Exit("ERROR[%s:%d] %s (%d:%s)\n", 
      file, line, msg, err_code, clu_ErrorString(err_code));
  }
}


// Find the default OpenCL device from the environment variable
cl_device_type clu_GetDefaultDeviceType() {
  char *device_type = getenv("OPENCL_DEVICE_TYPE");
  if (device_type != NULL) {
    if (strcasecmp(device_type, "cpu") == 0) {
      return CL_DEVICE_TYPE_CPU;
    } else if (strcasecmp(device_type, "gpu") == 0) {
      return CL_DEVICE_TYPE_GPU;
    } else if (strcasecmp(device_type, "accelerator") == 0) {
      return CL_DEVICE_TYPE_ACCELERATOR;
    }
  }
  return CL_DEVICE_TYPE_CPU;
//  return CL_DEVICE_TYPE_DEFAULT;
}


// Find an available OpenCL device corresponding to device_type
cl_device_id clu_GetAvailableDevice(cl_device_type device_type) {
  cl_platform_id *platforms;
  cl_uint         num_platforms = 0;
  cl_device_id   *devices;
  cl_uint         num_devices;
  cl_int          err_code;
  cl_uint         i, k;

  // Get OpenCL platforms
  // 1) Get the number of available platforms
  err_code = clGetPlatformIDs(0, NULL, &num_platforms);
  if (num_platforms == 0) clu_Exit("No OpenCL platform!\n");

  // 2) Get platform IDs
  platforms = (cl_platform_id *)malloc(num_platforms*sizeof(cl_platform_id));
  err_code  = clGetPlatformIDs(num_platforms, platforms, NULL);
  clu_CheckError(err_code, "clGetPlatformIDs()");

  // Get the specified devices
  for (i = 0; i < num_platforms; i++) {
    // Get the number of available devices
    err_code = clGetDeviceIDs(platforms[i], device_type, 0, NULL,
                              &num_devices);
    if (err_code != CL_SUCCESS && err_code != CL_DEVICE_NOT_FOUND)
      clu_CheckError(err_code, "clGetDeviceIDs for num_devices");

    if (num_devices > 0) {
      // Get device IDs
      devices  = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
      err_code = clGetDeviceIDs(platforms[i], device_type, num_devices,
                                devices, NULL);
      clu_CheckError(err_code, "clGetDeviceIDs()");

      // Return the first available device
      for (k = 0; k < num_devices; k++) {
        cl_bool available;
        err_code = clGetDeviceInfo(devices[k],
                                   CL_DEVICE_AVAILABLE,
                                   sizeof(cl_bool), &available,
                                   NULL);
        clu_CheckError(err_code, "clGetDeviceInfo()");

        if (available == CL_TRUE) {
          cl_device_id ret_device = devices[k];
          free(devices);
          free(platforms);
          return ret_device;
        }
      }

      free(devices);
    }
  }

  free(platforms);

  // Error
  clu_Exit("ERROR: No device for %s\n", clu_GetDeviceTypeName(device_type));

  return 0;
}


// Return the name of device.
// Caller needs to free the pointer of the return string.
char *clu_GetDeviceName(cl_device_id device) {
  size_t name_size;
  cl_int err_code;

  // Get the length of device name
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &name_size);

  char *device_name = (char *)malloc(name_size + 1);
  err_code = clGetDeviceInfo(device, CL_DEVICE_NAME,
                             name_size, device_name, NULL);
  clu_CheckError(err_code, "clGetDeviceInfo()");

  // Remove unnecessary characters, such as space, @, (, ), etc.
  char ch;
  char *cur_pos = device_name;
  char *rst_pos = device_name;
  while ((ch = *cur_pos++)) {
    if (ch == '(' && *cur_pos == 'R' && *(cur_pos + 1) == ')') {
      cur_pos += 2;
      *rst_pos++ = '_';
      continue;
    } else if (ch == ' ' || ch == '@' || ch == '(' || ch == ')') {
      if (rst_pos != device_name && (*(rst_pos - 1) != '_'))
        *rst_pos++ = '_';
      continue;
    }
    *rst_pos++ = ch;
  }
  *rst_pos = '\0';

  return device_name;
}


// Get the string of the device type
const char *clu_GetDeviceTypeName(cl_device_type device_type) {
  switch (device_type) {
    case CL_DEVICE_TYPE_DEFAULT:     return "CL_DEVICE_TYPE_DEFAULT";
    case CL_DEVICE_TYPE_CPU:         return "CL_DEVICE_TYPE_CPU";
    case CL_DEVICE_TYPE_GPU:         return "CL_DEVICE_TYPE_GPU";
    case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
    case CL_DEVICE_TYPE_ALL:         return "CL_DEVICE_TYPE_DEFAULT";
    default: return "Unknown Device";
  }
}


// Load the source code from source_file and return the pointer of the source
//  code string.
//  - source_code   : file name including path
//  - source_len_ret: length of source code is saved through this pointer
char *clu_LoadProgSource(const char *source_file, size_t *source_len_ret) {
  // Open the OpenCL source code file
  FILE *fp_source = fopen(source_file, "rb");
  if (fp_source == NULL) {
    clu_Exit("ERROR: Failed to open %s\n", source_file);
    return NULL;
  }

  // Get the length of the source code
  fseek(fp_source, 0, SEEK_END);
  size_t length = (size_t)ftell(fp_source);
  *source_len_ret = length;
  rewind(fp_source);

  // Read the source code
  char *source_code = (char *)malloc(length + 1);
  if (fread(source_code, length, 1, fp_source) != 1) {
    fclose(fp_source);
    free(source_code);

    clu_Exit("ERROR: Failed to read %s\n", source_file);
    return NULL;
  }

  // Make the source code null-terminated
  source_code[length] = '\0';

  // Close the file
  fclose(fp_source);

  return source_code;
}


// Load the OpenCL program binary from binary_file and return the pointer of
//  loaded binary.
//  - binary_file   : file name including path
//  - binary_len_ret: size of binary is saved through this pointer
unsigned char *clu_LoadProgBinary(const char *binary_file,
                                  size_t *binary_len_ret) {
  // Open the OpenCL binary file
  FILE *fp_binary = fopen(binary_file, "rb");
  if (fp_binary == NULL) {
    clu_Exit("ERROR: Failed to open %s\n", binary_file);
    return NULL;
  }

  // Get the length of the binary
  fseek(fp_binary, 0, SEEK_END);
  size_t length = (size_t)ftell(fp_binary);
  *binary_len_ret = length;
  rewind(fp_binary);

  // Read the binary
  unsigned char *binary = (unsigned char *)malloc(length);
  if (fread(binary, length, 1, fp_binary) != 1) {
    fclose(fp_binary);
    free(binary);

    clu_Exit("ERROR: Failed to read %s\n", binary_file);
    return NULL;
  }

  return binary;
}


// Print the build log to stderr.
void clu_PrintBuildLog(cl_program program, cl_device_id device) {
  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                        0, NULL, &log_size);
  char *log = (char *)malloc(log_size + 1);
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                        log_size, log, NULL);
  fprintf(stderr, "\n======== BUILD LOG ========\n%s\n", log);
  fprintf(stderr, "=======================================================\n\n");
  free(log);
}


// Create a program and build the program.
cl_program clu_MakeProgram(cl_context context,
                           cl_device_id device,
                           char *source_dir,
                           char *source_file, 
                           char *build_option) {
  cl_program program;
  cl_int err_code;

  // Make a full path of source file
  char *source_path = source_file;
  char *build_opts = build_option;
  size_t sdir_len = strlen(source_dir);
  if (sdir_len > 0) {
    if (source_dir[sdir_len-1] == '/') {
      source_path = (char *)malloc(sdir_len + strlen(source_file) + 1);
      sprintf(source_path, "%s%s", source_dir, source_file);
    } else {
      source_path = (char *)malloc(sdir_len + strlen(source_file) + 2);
      sprintf(source_path, "%s/%s", source_dir, source_file);
    }
    build_opts = (char *)malloc(sdir_len + strlen(build_option) + 4);
    sprintf(build_opts, "%s -I%s", build_option, source_dir);
  }

  // Create a program with the source code
  size_t source_len;
  char *source_code = clu_LoadProgSource(source_path, &source_len);
  program = clCreateProgramWithSource(context,
                                      1, 
                                      (const char **)&source_code,
                                      &source_len,
                                      &err_code);
  free(source_code);
  clu_CheckError(err_code, "clCreateProgramWithSource()");

  // Build the program
  err_code = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
  if (err_code != CL_SUCCESS) {
    clu_PrintBuildLog(program, device);
    clu_CheckError(err_code, "clBuldProgram()");
  }

  //clu_PrintBuildLog(program, device);

  if (sdir_len > 0) {
    free(source_path);
    free(build_opts);
  }

  return program;
}

/****************************************************************************/
/* OpenCL Profiling Functions                                               */
/****************************************************************************/

#define MAX_NUM_LOG         65536
#define MAX_NUM_EVENT       100
#define MAX_EVENT_NAME      35
#define WRITE_EVENT_NAME    "Host to Device Data Transfer"
#define READ_EVENT_NAME     "Device to Host Data Transfer"
#define COPY_EVENT_NAME     "Device to Device Data Transfer"

typedef struct clu_ProfilerLog_t {
  char name[MAX_EVENT_NAME];
  cl_event ev;
} clu_ProfilerLog;

typedef struct clu_ProfilerEvent_t {
  char name[MAX_EVENT_NAME];
  double elapsed;
} clu_ProfilerEvent;

logical                     npb_timeron          = false;
logical                     clu_profiler_timeron = false;
clu_ProfilerLog             *clu_profiler_logs;
unsigned long long          clu_profiler_log_cnt = 0;
unsigned long long          clu_profiler_log_max = MAX_NUM_LOG;
clu_ProfilerEvent           *clu_profiler_events;
int                         clu_profiler_event_cnt = 0;

static double clu_GetEventElapsedTime(cl_event ev) {
  cl_int ecode; 
  cl_ulong start_ns, end_ns, elapsed_ns;

  ecode = clGetEventProfilingInfo(ev, 
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &start_ns,
      NULL);
  clu_CheckError(ecode, "clGetEventProfilingInfo");

  ecode = clGetEventProfilingInfo(ev, 
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &end_ns,
      NULL);
  clu_CheckError(ecode, "clGetEventProfilingInfo");

  elapsed_ns = end_ns - start_ns;

  return (double)elapsed_ns / 1.0E+9;
}

static void clu_ProfilerSaveEvent(const char* ev_name, cl_event ev) {

  // doubling
  if (clu_profiler_log_cnt == clu_profiler_log_max) {
    clu_ProfilerLog *tmp;
    tmp = (clu_ProfilerLog *)malloc(sizeof(clu_ProfilerLog)*clu_profiler_log_max*2);

    unsigned long long i;

    for (i = 0; i < clu_profiler_log_max; i++) {
      tmp[i].ev = clu_profiler_logs[i].ev;
      strcpy(tmp[i].name, clu_profiler_logs[i].name);
    }
    free(clu_profiler_logs);
    clu_profiler_logs = tmp;
    clu_profiler_log_max *= 2;
  }

  clu_profiler_logs[clu_profiler_log_cnt].ev = ev;
  strcpy(clu_profiler_logs[clu_profiler_log_cnt].name, ev_name);
  clu_profiler_log_cnt++;
}

static int clu_GetEventIndex(const char* name) {
  int i, ret;

  for (i = 0; i < clu_profiler_event_cnt; i++) {
    if (strcmp(name, clu_profiler_events[i].name) == 0)
      return i;
  }

  // does NOT find event

  if (i == MAX_NUM_EVENT) {
    fprintf(stderr, " Too many event types to profile\n");
    exit(EXIT_FAILURE);
  }

  strcpy(clu_profiler_events[clu_profiler_event_cnt].name, name);
  clu_profiler_events[clu_profiler_event_cnt].elapsed = 0.0;

  ret = clu_profiler_event_cnt++;

  return ret;
}

static void clu_ProfilerAccumLogs(void) {
  unsigned long long i;
  int ev_idx;
  double elapsed;

  for (i = 0; i < clu_profiler_log_cnt; i++) {
    elapsed = clu_GetEventElapsedTime(clu_profiler_logs[i].ev);
    ev_idx = clu_GetEventIndex(clu_profiler_logs[i].name);
    clu_profiler_events[ev_idx].elapsed += elapsed;
  }
}

static void clu_ProfilerPrintEventWithIndex(int ev_idx) {
  clu_ProfilerPrintElapsedTime(clu_profiler_events[ev_idx].name,
                               clu_profiler_events[ev_idx].elapsed);
}

// User APIs
void clu_ProfilerSetup(void) {
  FILE *fp;

  if ((fp = fopen("timer.flag", "r")) != NULL) {
    fclose(fp);
    npb_timeron = 1;
  }

  int i;
  clu_profiler_logs = (clu_ProfilerLog *)malloc(sizeof(clu_ProfilerLog)*clu_profiler_log_max);
  clu_profiler_events = (clu_ProfilerEvent *)malloc(sizeof(clu_ProfilerEvent)*MAX_NUM_EVENT);

  for (i = 0; i < MAX_NUM_EVENT; i++) {
    clu_profiler_events[i].elapsed = 0.0;
  }
}

void clu_ProfilerRelease(void) {
  unsigned long long i;
  for (i = 0; i < clu_profiler_log_cnt; i++) {
    clReleaseEvent(clu_profiler_logs[i].ev);
  }
  free(clu_profiler_logs);
  free(clu_profiler_events);
}

void clu_ProfilerStart(void) {
  clu_profiler_timeron = true;
}

void clu_ProfilerStop(void) {
  clu_profiler_timeron = false;
}

void clu_ProfilerClear(void) {
  unsigned long long i;
  for (i = 0; i < clu_profiler_log_cnt; i++) {
    clReleaseEvent(clu_profiler_logs[i].ev);
  }
  clu_profiler_log_cnt = 0;

  int j;
  for (j = 0; j < clu_profiler_event_cnt; j++) {
    clu_profiler_events[j].elapsed = 0.0;
  }
  clu_profiler_event_cnt = 0;
}

void clu_ProfilerPrintElapsedTime(const char* name, double elapsed) {
  if (elapsed != 0.0) {
    printf(" %35s : %12.6f sec\n", name, elapsed);
  }
}

void clu_ProfilerPrintResult(void) {
  int i;
  int wb_idx, rb_idx, copy_idx;

  clu_ProfilerAccumLogs();

  wb_idx = clu_GetEventIndex(WRITE_EVENT_NAME);
  rb_idx = clu_GetEventIndex(READ_EVENT_NAME);
  copy_idx = clu_GetEventIndex(COPY_EVENT_NAME);

  printf(" =========================================================================\n");
  printf("                       OpenCL Event Profiler Results\n");
  printf(" %35s   %12s \n",
      "Event Name    ", "      Time (sec)");
  for (i = 0; i < clu_profiler_event_cnt; i++) {
    if (i != wb_idx &&
        i != rb_idx &&
        i != copy_idx)
      clu_ProfilerPrintEventWithIndex(i);
  }
  clu_ProfilerPrintEventWithIndex(wb_idx);
  clu_ProfilerPrintEventWithIndex(rb_idx);
  clu_ProfilerPrintEventWithIndex(copy_idx);
}

// Wrapping Functions
cl_int clu_Profiler_clEnqueueNDRangeKernel(cl_command_queue q, 
                                           cl_kernel k, 
                                           cl_uint wd, 
                                           const size_t *gws_off,
                                           const size_t *gws,
                                           const size_t *lws,
                                           cl_uint num_ev_wlist,
                                           const cl_event *ev_wlist,
                                           cl_event *ev, 
                                           const char *kernel_name)
{
#undef clEnqueueNDRangeKernel

  if (!npb_timeron || !clu_profiler_timeron)
    return clEnqueueNDRangeKernel(q, k, wd, gws_off, gws, lws, num_ev_wlist, ev_wlist, ev);

  cl_int ecode;

  if (ev == NULL) {
    cl_event tmp_ev;
    ecode = clEnqueueNDRangeKernel(q, k, wd, gws_off, gws, lws, num_ev_wlist, ev_wlist, &tmp_ev);
    clu_ProfilerSaveEvent(kernel_name, tmp_ev);
  }
  else {
    ecode = clEnqueueNDRangeKernel(q, k, wd, gws_off, gws, lws, num_ev_wlist, ev_wlist, ev);
    clRetainEvent(*ev);
    clu_ProfilerSaveEvent(kernel_name, *ev);
  }

  return ecode;
}

cl_int clu_Profiler_clEnqueueWriteBuffer(cl_command_queue q,
                                         cl_mem buf,
                                         cl_bool block,
                                         size_t offset,
                                         size_t cb,
                                         const void *ptr,
                                         cl_uint num_ev_wlist,
                                         const cl_event *ev_wlist,
                                         cl_event *ev,
                                         const char* func)
{
#undef clEnqueueWriteBuffer

  if (!npb_timeron || !clu_profiler_timeron)
    return clEnqueueWriteBuffer(q, buf, block, offset, cb, ptr, num_ev_wlist, ev_wlist, ev);

  cl_int ecode;

  if (ev == NULL) {
    cl_event tmp_ev;
    ecode = clEnqueueWriteBuffer(q, buf, block, offset, cb, ptr, num_ev_wlist, ev_wlist, &tmp_ev);
    clu_ProfilerSaveEvent(WRITE_EVENT_NAME, tmp_ev);
  }
  else {
    ecode = clEnqueueWriteBuffer(q, buf, block, offset, cb, ptr, num_ev_wlist, ev_wlist, ev);
    clRetainEvent(*ev);
    clu_ProfilerSaveEvent(WRITE_EVENT_NAME, *ev);
  }

  return ecode;
}

cl_int clu_Profiler_clEnqueueReadBuffer(cl_command_queue q,
                                        cl_mem buf,
                                        cl_bool block,
                                        size_t offset,
                                        size_t cb,
                                        void *ptr,
                                        cl_uint num_ev_wlist,
                                        const cl_event *ev_wlist,
                                        cl_event *ev,
                                        const char* func)
{
#undef clEnqueueReadBuffer

  if (!npb_timeron || !clu_profiler_timeron)
    return clEnqueueReadBuffer(q, buf, block, offset, cb, ptr, num_ev_wlist, ev_wlist, ev);

  cl_int ecode;

  if (ev == NULL) {
    cl_event tmp_ev;
    ecode = clEnqueueReadBuffer(q, buf, block, offset, cb, ptr, num_ev_wlist, ev_wlist, &tmp_ev);
    clu_ProfilerSaveEvent(READ_EVENT_NAME, tmp_ev);
  }
  else {
    ecode = clEnqueueReadBuffer(q, buf, block, offset, cb, ptr, num_ev_wlist, ev_wlist, ev);
    clRetainEvent(*ev);
    clu_ProfilerSaveEvent(READ_EVENT_NAME, *ev);
  }
  
  return ecode;
}


cl_int clu_Profiler_clEnqueueCopyBuffer(cl_command_queue q,
                                        cl_mem src_buf,
                                        cl_mem dst_buf,
                                        size_t src_offset,
                                        size_t dst_offset,
                                        size_t cb,
                                        cl_uint num_ev_wlist,
                                        const cl_event *ev_wlist,
                                        cl_event *ev,
                                        const char* func)
{
#undef clEnqueueCopyBuffer
  if (!npb_timeron || !clu_profiler_timeron)
    return clEnqueueCopyBuffer(q, src_buf, dst_buf, src_offset, dst_offset, cb, num_ev_wlist, ev_wlist, ev);

  cl_int ecode;

  if (ev == NULL) {
    cl_event tmp_ev;
    ecode = clEnqueueCopyBuffer(q, src_buf, dst_buf, src_offset, dst_offset, cb, num_ev_wlist, ev_wlist, &tmp_ev);
    clu_ProfilerSaveEvent(COPY_EVENT_NAME, tmp_ev);
  }
  else {
    ecode = clEnqueueCopyBuffer(q, src_buf, dst_buf, src_offset, dst_offset, cb, num_ev_wlist, ev_wlist, ev);
    clRetainEvent(*ev);
    clu_ProfilerSaveEvent(COPY_EVENT_NAME, *ev);
  }

  return ecode;



}

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
                                             const char* func)
{
#undef clEnqueueWriteBufferRect
  if (!npb_timeron || !clu_profiler_timeron)
    return clEnqueueWriteBufferRect(q, buf, block, 
                                    buffer_origin, host_origin, region,
                                    buffer_row_pitch, buffer_slice_pitch,
                                    host_row_pitch, host_slice_pitch,
                                    ptr,
                                    num_ev_wlist,
                                    ev_wlist, ev);

  cl_int ecode;

  if (ev == NULL) {
    cl_event tmp_ev;
    ecode = clEnqueueWriteBufferRect(q, buf, block,
                                     buffer_origin, host_origin, region,
                                     buffer_row_pitch, buffer_slice_pitch,
                                     host_row_pitch, host_slice_pitch,
                                     ptr,
                                     num_ev_wlist,
                                     ev_wlist, &tmp_ev);
    clu_ProfilerSaveEvent(WRITE_EVENT_NAME, tmp_ev);
  }
  else {
    ecode = clEnqueueWriteBufferRect(q, buf, block,
                                     buffer_origin, host_origin, region,
                                     buffer_row_pitch, buffer_slice_pitch,
                                     host_row_pitch, host_slice_pitch,
                                     ptr,
                                     num_ev_wlist,
                                     ev_wlist, ev);
    clRetainEvent(*ev);
    clu_ProfilerSaveEvent(WRITE_EVENT_NAME, *ev);
  }

  return ecode;
}

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
                                            const char* func)
{
#undef clEnqueueReadBufferRect
  if (!npb_timeron || !clu_profiler_timeron)
    return clEnqueueReadBufferRect(q, buf, block, 
                                   buffer_origin, host_origin, region,
                                   buffer_row_pitch, buffer_slice_pitch,
                                   host_row_pitch, host_slice_pitch,
                                   ptr,
                                   num_ev_wlist,
                                   ev_wlist, ev);

  cl_int ecode;

  if (ev == NULL) {
    cl_event tmp_ev;
    ecode = clEnqueueReadBufferRect(q, buf, block,
                                    buffer_origin, host_origin, region,
                                    buffer_row_pitch, buffer_slice_pitch,
                                    host_row_pitch, host_slice_pitch,
                                    ptr,
                                    num_ev_wlist,
                                    ev_wlist, &tmp_ev);
    clu_ProfilerSaveEvent(READ_EVENT_NAME, tmp_ev);
  }
  else {
    ecode = clEnqueueReadBufferRect(q, buf, block,
                                    buffer_origin, host_origin, region,
                                    buffer_row_pitch, buffer_slice_pitch,
                                    host_row_pitch, host_slice_pitch,
                                    ptr,
                                    num_ev_wlist,
                                    ev_wlist, ev);
    clRetainEvent(*ev);
    clu_ProfilerSaveEvent(READ_EVENT_NAME, *ev);
  }

  return ecode;
}


#undef MAX_NUM_LOG
#undef MAX_NUM_EVENT
#undef MAX_EVENT_NAME
#undef WRITE_EVENT_NAME
#undef READ_EVENT_NAME
#undef COPY_EVENT_NAME
