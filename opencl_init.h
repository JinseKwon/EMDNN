#include <CL/cl.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

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
char *kernel_source;
size_t kernel_source_size;
cl_event event;
cl_int err;

cl_kernel kernel1, kernel2, kernel3, kernel4, kernel5;

cl_mem cl_obj_create(cl_mem cl_obj, int m_size){
    // printf("> def cl_mem ");
    return clCreateBuffer(context, CL_MEM_READ_WRITE, m_size, NULL, &err);
    // CHECK_ERROR(err);
    // return cl_obj;
}
//ABSTRATION of CL MAPPING FUNCTION 
void cl_obj2mem(cl_mem cl_obj, float** host_mem, 
                cl_map_flags map_flag, int m_size){
    *host_mem = (float*)clEnqueueMapBuffer(queue, cl_obj, CL_TRUE, CL_MAP_WRITE, 0,
                                m_size, 0, NULL, NULL, &err);
    CHECK_ERROR(err);
    // printf("> cl_mem to mem ");
}
//ABSTRATION of CL UNMAPPING FUNCTION 
void mem2cl_obj(float* host_mem, cl_mem cl_obj){
    err = clEnqueueUnmapMemObject(queue, cl_obj, host_mem, 0, NULL, NULL);
    CHECK_ERROR(err);
    // printf("> mem to cl_mem ");
}
char *get_source_code(const char *file_name, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}