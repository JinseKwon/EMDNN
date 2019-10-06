#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include "opencv/highgui.h"
// #include "image_io.h"
#ifdef OPENCL
#include <CL/cl.h>
#endif
#ifdef NNPACK
#include <nnpack.h>
#endif

extern "C"

typedef enum{
    INPUT_LAYER,
    CONVOLUTIONAL,
    CONVOLUTIONAL_DW,
    CONNECTED,
    CONNECTED_T,
    MAXPOOL,
    SOFTMAX,
    AVGPOOL,
    SHORTCUT,
    DETECTION,
    CLASSIFICATION
}LAYER_TYPE;

typedef enum{
    CPU,
    GPU,
    NPU,
    PPU
}DEVICE_TYPE;

//layer print할때 enumerate 열거형 string 형태로 표기하기 위함
// char lay_type[8][20] = { 
//     "input   ",    "conv    ",    "conv_dw ",    "connect ",
//     "maxpool ",    "softmax ",    "avgpool ",    "shortcut"
// };

typedef enum{
    RELU, 
    LINEAR, 
    LEAKY
} ACTIVATION_TYPE;

typedef enum{
    HALF = 2, 
    SINGLE = 4, 
    DOUBLE = 8
} PRECISION;
// struct person{
//     int a;
// };

typedef struct LAYER{
    LAYER_TYPE TYPE;
    ACTIVATION_TYPE ACTIVATION;
    DEVICE_TYPE DEVICE;
    DEVICE_TYPE CUR_DEVICE;

    //layer index
    int NUM;
    
    //filter dimension
    int N;
    int C;
    int H;
    int W;
    int PAD;
    int STRIDE;
    int SCALE;
    

    //input dimension
    int IN_C;
    int IN_H;
    int IN_W;

    //output dimension
    int OUT_C;
    int OUT_H;
    int OUT_W;

    PRECISION XF;

    //im2col
    int IM2COL;

    //if tuning flag = on/off
    int TUNE;
    double TIME_CPU;
    double TIME_GPU;
    double TIME_PPU;
    double TIME_NPU;

    //DATA
    float* BIAS;
    float* WEIGHT;
    float* INPUT;
    float* OUTPUT;

#ifdef OPENCL
    cl_mem CL_WEIGHT;
    cl_mem CL_BIAS;
    cl_mem CL_INPUT;
    cl_mem CL_OUTPUT;
    
    //put in only layer 0 <== cl paramter 
    cl_command_queue *QUE;
    cl_event *EVT;
    
    cl_kernel *KER_IM2COL;
    cl_kernel *KER_MAXPOOL;
    cl_kernel *KER_BIAS;
    cl_kernel *KER_RELU;
    cl_kernel *KER_AVGPOOL;
#endif
#ifdef NNPACK
    pthreadpool_t PTHREAD;
#endif

//TODO
    //half_float
    //cl_mem
}LAYER;

//extern 안하면 undefined reference 에러 뜸...
extern void make_network(LAYER *l,
            float* net_weight, 
            int num,
            char *filename);

extern LAYER* layer_update(
// extern void layer_update(
            LAYER *l,
            LAYER_TYPE type, 
            ACTIVATION_TYPE act,
            DEVICE_TYPE device,
            int num, 
            int n, int c, int h, int w,
            int pad, int str, int scale);

extern void prepare_mem(LAYER *l, int num,
            DEVICE_TYPE pre_dev,
            DEVICE_TYPE cur_dev);

extern void tune_network(LAYER *l, 
            int num);

extern void print_network(LAYER *l, 
            int num);

extern void inference(LAYER *l, 
            int num);

void im2col(LAYER *l, int i,
            float *in, float *out, 
            int C, int H, int W, 
            int KER, int stride, int pad);
// void cpu_gemv(float *A, float *B, float *C, int M, int N, int K);  
// void cpu_gemm(float *A, float *B, float *C, int M, int N, int K);
void gemv(LAYER *l, int i,
          float* A,    float* B,    float* C,
          int ta,
          int M, int N, int K,
          float ALPHA, float BETA);
void gemm(LAYER *l, int i,
          float* A,    float* B,    float* C,
          int ta, int tb,
          int M, int N, int K,
          float ALPHA, float BETA);
void depth_gemm(LAYER *l, int i,
          float* A,    float* B,    float* C,
          int ta, int tb,
          int M, int N, int K,
          int offset,
          float ALPHA, float BETA);
void conv(LAYER *l, int i);
void depth_conv(LAYER *l, int i);
#ifdef NNPACK
void nnpack_depth_conv(LAYER *l, int i);
void nnpack_conv(LAYER *l, int i);
void nnpack_fc(LAYER *l, int i);
#endif
// void depthwise_conv(float *A, float *B, float *C, 
//                       int M, int N, int K,
//                       int offsetA);

float* input_bin_img(float* input, int C, int H, int W);
void batch_normalizaiton(float *bias, float *weight, 
                         int K, int C, int H, int W);
void bias_add(LAYER *l, int i,
              float* in, float* bias, 
              int C, int H, int W);
void activate_function(LAYER *l, int i,
                       float* in, ACTIVATION_TYPE act, 
                       int C, int H, int W);
void softmax(float *in, float *out, int C);
void transpose(float *INA, float *OUTA, 
               int ldx, int ldy);
void transpose_fc(float *INA,
               int ldx, int ldy);
void maxpool(LAYER *l,  int i,
             float *in, float *out, int C, int H, int W, int KER, int stride, int PAD);
void avgpool(LAYER *l,  int i,
             float *in, float *out, int C, int H, int W, int KER, int stride, int PAD);
void detection(float *grid_cell,float* box_out,
               int input_width,
               int det_h,
               int det_w,
               int det_BOX,
               int det_CLASS,
               float score_threshold,
               float iou_threshold,
               int tune);

void classification(float *output_score, 
                    int class_num,
                    int tune);//donghee

// void cam_read(float *image, int img_size);
IplImage* image_read(char *img_file, float *image, int img_size,
                     float mean_r,   float mean_g, float mean_b);

// void imagefile_read( float *image, int img_size, char *filename);
// void imagefile_read2( float *image, int img_size,int n);
void image_show(float* box_output, IplImage *readimg,
                double elapsed);
void image_free();

#ifdef OPENCL
void cl_obj2mem(cl_mem cl_obj, float** host_mem, 
                cl_map_flags map_flag, int m_size);
void mem2cl_obj(float* host_mem, cl_mem cl_obj);
#endif

//Now.. Timer function...
double get_time();