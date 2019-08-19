#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if(err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

cl_platform_id platform;
cl_device_id device;
cl_uint numdevices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_event event;
cl_int err;

cl_mem cl_obj_create(cl_mem cl_obj, int m_size){
    // printf("> def cl_mem ");
    return clCreateBuffer(context, CL_MEM_READ_WRITE, m_size, NULL, &err);
    // CHECK_ERROR(err);
    // return cl_obj;
}