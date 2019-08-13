#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>

extern "C"

typedef enum{
    INPUT_LAYER,
    CONVOLUTIONAL,
    CONVOLUTIONAL_DW,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    AVGPOOL,
    SHORTCUT,
    DETECTION
}LAYER_TYPE;

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

    //DATA
    float* BIAS;
    float* WEIGHT;
    float* INPUT;
    float* OUTPUT;

//TODO
    //half_float
    //cl_mem
}LAYER;

//extern 안하면 undefined reference 에러 뜸...
extern void make_network(LAYER *l,
            float* net_weight, 
            int num,
            char *filename);

extern void layer_update(
            LAYER *l,
            LAYER_TYPE type, 
            ACTIVATION_TYPE act,
            int num, 
            int n, int c, int h, int w,
            int pad, int str, int scale);

extern void print_network(LAYER *l, 
            int num);

extern void inference(LAYER *l, 
            int num);

void im2col(float *in, float *out, 
            int C, int H, int W, 
            int KER, int stride, int pad);
void cpu_gemm(float *A, float *B, float *C, int M, int N, int K);
void cpu_stride_b_gemm(float *A, float *B, float *C, 
                      int M, int N, int K,
                      int stride_b, int batch_count);

float* input_img(float* input, int C, int H, int W);
void batch_normalizaiton(float *bias, float *weight, 
                         int K, int C, int H, int W);
void bias_add(float* in, float* bias, 
              int C, int H, int W);
void softmax(float *out, float *in, int K);
void activate_function(float* in, ACTIVATION_TYPE act, 
                       int C, int H, int W);

void transpose(float *INA, float *OUTA, 
               int ldx, int ldy);

void maxpool(float *in, float *out, int H, int W, int C, int KER, int stride);
void avgpool(float *in, float *out, int H, int W, int C, int KER, int stride);
void detection(float *grid_cell,float* box_out,
               int input_width,
               int det_h,
               int det_w,
               int det_BOX,
               int det_CLASS,
               float score_threshold,
               float iou_threshold);