#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <cblas.h>
#include "cpu_darktiny_init.h"
#include "timer.h"

#include <CL/cl.h>
#include <clblast_c.h>


/**************opencl part*********************/
cl_platform_id platform;
cl_device_id device;
cl_uint numdevices;
cl_context context;
cl_command_queue queue;
cl_int err;
cl_mem bufA, bufB, bufC;
cl_event event;

static int dark_DEBUG = 0;
// im2col(input, dark_cpu_o1_i2c,227,227,3,11,4); -> 55,55 - 3*11*11
// im2col(dark_cpu_o1_m, dark_cpu_o2_i2c,27,27,96,5,1,2);
inline void im2col(float *in, float *out, int H, int W, int C, int KER, int stride, int pad){
    double check_time = get_time();
    int start_xy;
    // int cccccc =0;
    if(pad == 0){
        start_xy = (KER/2);
    }else{
        start_xy = 0;
    }
    for(int hout = start_xy; hout <H-start_xy; hout+=stride){
        for(int wout = start_xy; wout <W-start_xy; wout+=stride){
            for(int c = 0; c < C; ++c){
                for(int r = 0; r < KER; ++r){
                    for(int s = 0; s < KER; ++s){
                        int h  = hout + r - (KER/2);
                        int w  = wout + s - (KER/2);
                        if(h < 0 || h >= H || w < 0 || w >= W){
                            out[ ( (hout-start_xy) / stride ) * (W-start_xy)/stride * C * KER * KER + ( (wout-start_xy) / stride ) * C * KER * KER + c * KER * KER + r * KER + s ] = 0;
                            // cccccc++;
                        }else{
                            out[ ( (hout-start_xy) / stride ) * (W-start_xy)/stride * C * KER * KER + ( (wout-start_xy) / stride ) * C * KER * KER + c * KER * KER + r * KER + s ] = 
                            in[  h * W * C + w * C + c ];
                            // cccccc++;
                        }
                        // printf("%d ", ((hout-start_xy) / stride ) * W * C * KER * KER + ( (wout-start_xy) / stride ) * C * KER * KER + c * KER * KER + r * KER + s );
                    }
                }
            }
        }
    }
    // printf("layer im2col output dimension : %d \n",cccccc);
    //TODO im2col 최적화....
    if(dark_DEBUG)  printf("i %.6f\n", get_time()-check_time);
}

static void gemm_blas(float *A, float *B, float *C, int M, int K, int N, int mode){
    double check_time = get_time();
    double op = M*N*(2*K-1.f) * 1e-9;
    if(mode == 0){
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    A, K,
                    B, N,
                    0.0f,
                    C, N);
    }else{
        clEnqueueWriteBuffer(queue  ,bufA  ,CL_FALSE, 0, M*K*sizeof(float), A,0,NULL,&event);
        clWaitForEvents(1, &event);
        clEnqueueWriteBuffer(queue  ,bufB  ,CL_FALSE, 0, K*N*sizeof(float), B,0,NULL,&event);
        clWaitForEvents(1, &event);

        CLBlastStatusCode status;
        status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                CLBlastTransposeNo, CLBlastTransposeNo,
                                M, N, K,
                                1.0f,
                                bufA, 0, K,
                                bufB, 0, N,
                                0.0f,
                                bufC, 0, N,
                                &queue, &event);
        if (status == CLBlastSuccess) {
            clWaitForEvents(1, &event);
        }
        err = clEnqueueReadBuffer(queue  ,bufC  ,CL_FALSE, 0, M * N*sizeof(float), C ,0,NULL,&event); 	
        clWaitForEvents(1, &event);
        clFinish(queue);
    }
    // // cpuversion check..
    // for(int m = 0; m < M; ++m){
    //     for(int n = 0; n < N; ++n){
    //         float sum = 0;
    //         for(int k = 0; k < K; ++k){
    //             sum += A[m*K+k] * B[k*N+n];
    //         }
    //         C[m*N+n] = sum;
    //     }
    // }
    check_time = get_time() - check_time;
    if(dark_DEBUG)  printf("g %.6f (%.3f Gflops) ops : %.3f\n", check_time, op/check_time, op);
}


inline void matrix_bias(float *C, float *bias, int M, int K, int N){
    double check_time = get_time();
    for(int m = 0; m < M; ++m){
        for(int n = 0; n < N; ++n){
            C[m*N+n] += bias[n];
        }
    }
    if(dark_DEBUG)  printf("b %.6f\n", get_time()-check_time);
}

/*
 * ReLU (in-place)
 * inout : (C, H, W)
 */
static void relu(float *inout, int CHW) {
    double check_time = get_time();
	float alpha = 0.1;
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = fmaxf(inout[chw], inout[chw]*alpha);
    }
    if(dark_DEBUG)  printf("r %.6f\n", get_time()-check_time);
}
/*
 * ReLU (leaky ReLU)
 * function : max(alpha *x, x)
 * inout : (C, H, W)
 */
// static void relu(float *inout, int CHW) {
//     double check_time = get_time();
// 	float alpha = 0.1;
//     for (int chw = 0; chw < CHW; ++chw) {
//         inout[chw] = fmaxf(inout[chw], inout[chw]*alpha);
//     }
//     if(dark_DEBUG)  printf("r%.6f ", get_time()-check_time);
// }
/*
 * MaxPooling
 * in  : (C, H, W)
 * out : (C, H/2, W/2)
 * 
 */
static void maxpool(float *in, float *out, int H, int W, int C, int KER, int stride){
    double check_time = get_time();
    int start_xy = KER/2;
    // printf("maxpool start_xy : %d\n",start_xy);
    for(int h = start_xy; h < H; h+=stride){
		for(int w = start_xy; w < W; w+=stride){
            for(int c = 0; c < C; c++){
                float maxp = -FLT_MAX;
                for(int r = -1; r < KER-1; ++r){
                    for(int s = -1; s < KER-1; ++s){
                        maxp = fmaxf(maxp, in[(h+r) * W * C + (w+s) * C + c ]);
                    }       
                }
				out[ (h/stride) * (W/stride) * C + (w/stride) * C + c] = maxp;
			}
		}
	}
    if(dark_DEBUG)  printf("m %.6f\n", get_time()-check_time);
}
/*
 * Fully-Connected Layer (matrix-vector multiplication)
 * in : (C)
 * out : (K)
 * weight : (K, C)
 * bias : (K)
 */
// void cblas_sgemv(   OPENBLAS_CONST enum CBLAS_ORDER order,  
//                     OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  
//                     OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
// 		            OPENBLAS_CONST float alpha, 
//                     OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  
//                     OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  
//                     OPENBLAS_CONST float beta,  
//                     float  *y, OPENBLAS_CONST blasint incy);

// static void fc_cblas(float *in, float *out, float *weight, int K, int C) {
//     double check_time = get_time();
//     // double op = M*N*(2*K-1.f) * 1e-9;
//     cblas_sgemm(CblasRowMajor,
//                 CblasNoTrans, CblasNoTrans,
//                 M, N, K,
//                 1.0f,
//                 A, K,
//                 B, N,
//                 0.0f,
//                 C, N);
//     check_time = get_time() - check_time;
//     if(dark_DEBUG)  printf("g%.6f (%.3f Gflops)", check_time, op/check_time);
// }
static void fc(float *in, float *out, float *weight, int K, int C) {
    double check_time = get_time();
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                K, C,
                1.0f,
                weight, C,
                in, 1,
                0.0f,
                out, 1);
    // for (int k = 0; k < K; ++k) {
    //     float s = 0.0f;
    //     for (int c = 0; c < C; ++c) {
    //         s += in[c] * weight[k * C + c];
    //     }
    //     out[k] = s;
    // }
    if(dark_DEBUG)  printf("f %.6f\n", get_time()-check_time);
}
inline void fc_bias(float *in, float *bias, int K){
    double check_time = get_time();
    for(int kk = 0; kk < K; ++kk){
        in[kk] += bias[kk];
    }
    if(dark_DEBUG)  printf("fb %.6f\n", get_time()-check_time);
}

//caffe model weight dimension = {KCHW => CHWK}
//K = output Feature Map
//C = input Channle
//R = Kernel Height
//S = Kernel Width
inline void weight_transpose_CHWK(float *weight, int K, int C, int R, int S){

  float *temp = (float*)malloc(sizeof(float) * K * C * R * S);

  for(int i = 0; i < K*C*R*S; ++i){
    temp[i] = weight[i];
  }
  for(int kk = 0; kk < K; ++kk){
    for(int csr = 0; csr < C*S*R; ++csr){
      weight[ K * csr + kk ] = temp[ C*S*R * kk + csr];
    }
  }

  free(temp);
}


inline void avgpool(float *out, float *in, int HW, int K){
    double check_time = get_time();
    float sum = 0;
    
    for (int kk = 0; kk < K; kk++) {
        sum = 0;
        for(int cc = 0; cc < HW; ++cc){
            sum += in[cc*K + kk];
        }
        out[kk] = sum/HW;
    }

    if(dark_DEBUG)  printf("avg %.6f\n", get_time()-check_time);
}

inline void softmax(float *out, float *in, int K){
    double check_time = get_time();
    float max = -FLT_MAX;
    
    for (int kk = 0; kk < K; kk++) {
        // printf("%.6f \n",in[kk]);
        if (in[kk] > max) {
            max = in[kk];
        }
    }

    float sum = 0.0f;
    for (int kk = 0; kk < K; kk++) {
        float e = expf(in[kk] - max);
        sum += e;
        out[kk] = e;
    }

    for (int kk = 0; kk < K; kk++) {
        
        out[kk] /= sum;
    }

    if(dark_DEBUG)  printf("soft %.6f\n", get_time()-check_time);
}

inline void weight_proc(float *bias, float *weight, int K, int C, int H, int W){
    //bias  K
    //gamma K
    //mean  K
    //var   K
    const float epsilon = 1e-3;
    
    for(int i=0; i<K; i++){
        
        float scale = bias[K*1 + i] / sqrt(bias[K*3 + i] + epsilon);
        bias[i] = bias[i] - bias[K*2 +i] * scale;
        for(int w=0; w<C*H*W; ++w){
            weight[i*C*H*W + w] = weight[i*C*H*W + w] * scale;
        }
    }
}

void mobilev1_openblas_init(){
    // network read
    FILE *fnet = fopen("models/tiny.weights", "rb");
    if (!fnet) {
        printf("Network file does not exist.\n");
        exit(0);
    }
    fseek(fnet, 0, SEEK_END);
    long fsz = ftell(fnet);
    if (fsz != 4185968) {
        printf("Network file is corrupted.\n");
        exit(0);
    }
    rewind(fnet);
    float *network = (float*)malloc(4185968);
    fread(network, 1, 4185968, fnet);
    fclose(fnet);

    network += 4;
    dark_cpu_conv1_b  = network; network +=   16 * 4;	
    dark_cpu_conv1_w  = network; network +=   16 *    3 * 3 * 3;
    dark_cpu_conv2_b  = network; network +=   32 * 4; 
    dark_cpu_conv2_w  = network; network +=   32 *   16 * 3 * 3;
    dark_cpu_conv3_b  = network; network +=   16 * 4;
    dark_cpu_conv3_w  = network; network +=   16 *   32 * 1 * 1;
    dark_cpu_conv4_b  = network; network +=  128 * 4;
    dark_cpu_conv4_w  = network; network +=  128 *   16 * 3 * 3;
    dark_cpu_conv5_b  = network; network +=   16 * 4;
    dark_cpu_conv5_w  = network; network +=   16 *  128 * 1 * 1;
    dark_cpu_conv6_b  = network; network +=  128 * 4;
    dark_cpu_conv6_w  = network; network +=  128 *   16 * 3 * 3;
    dark_cpu_conv7_b  = network; network +=   32 * 4; 
    dark_cpu_conv7_w  = network; network +=   32 *  128 * 1 * 1;
    dark_cpu_conv8_b  = network; network +=  256 * 4;
    dark_cpu_conv8_w  = network; network +=  256 *   32 * 3 * 3;
    dark_cpu_conv9_b  = network; network +=   32 * 4;
    dark_cpu_conv9_w  = network; network +=   32 *  256 * 1 * 1;
    dark_cpu_conv10_b = network; network +=  256 * 4;
    dark_cpu_conv10_w = network; network +=  256 *   32 * 3 * 3;
    dark_cpu_conv11_b = network; network +=   64 * 4;
    dark_cpu_conv11_w = network; network +=   64 *  256 * 1 * 1;
    dark_cpu_conv12_b = network; network +=  512 * 4; 
    dark_cpu_conv12_w = network; network +=  512 *   64 * 3 * 3;
    dark_cpu_conv13_b = network; network +=   64 * 4;
    dark_cpu_conv13_w = network; network +=   64 *  512 * 1 * 1;
    dark_cpu_conv14_b = network; network +=  512 * 4;
    dark_cpu_conv14_w = network; network +=  512 *   64 * 3 * 3;
    dark_cpu_conv15_b = network; network +=  128 * 4;
    dark_cpu_conv15_w = network; network +=  128 *  512 * 1 * 1;
    dark_cpu_conv16_b = network; network += 1000;
    dark_cpu_conv16_w = network; network += 1000 *  128 * 1 * 1;

    weight_proc(dark_cpu_conv1_b,  dark_cpu_conv1_w,    16,  3,3,3);
    
    weight_proc(dark_cpu_conv2_b,  dark_cpu_conv2_w,    32, 16,3,3);

    weight_proc(dark_cpu_conv3_b,  dark_cpu_conv3_w,    16, 32,1,1);
    weight_proc(dark_cpu_conv4_b,  dark_cpu_conv4_w,   128, 16,3,3);
    weight_proc(dark_cpu_conv5_b,  dark_cpu_conv5_w,    16,128,1,1);
    weight_proc(dark_cpu_conv6_b,  dark_cpu_conv6_w,   128, 16,3,3);

    weight_proc(dark_cpu_conv7_b,  dark_cpu_conv7_w,    32,128,1,1);
    weight_proc(dark_cpu_conv8_b,  dark_cpu_conv8_w,   256, 32,3,3);
    weight_proc(dark_cpu_conv9_b,  dark_cpu_conv9_w,    32,256,1,1);
    weight_proc(dark_cpu_conv10_b, dark_cpu_conv10_w,  256, 32,3,3);

    weight_proc(dark_cpu_conv11_b, dark_cpu_conv11_w,   64,256,1,1);
    weight_proc(dark_cpu_conv12_b, dark_cpu_conv12_w,  512, 64,3,3);
    weight_proc(dark_cpu_conv13_b, dark_cpu_conv13_w,   64,512,1,1);
    weight_proc(dark_cpu_conv14_b, dark_cpu_conv14_w,  512, 64,3,3);
    weight_proc(dark_cpu_conv15_b, dark_cpu_conv15_w,  128,512,1,1);
    // weight_proc(dark_cpu_conv16_b, dark_cpu_conv16_w, 1000,128,1,1);

    weight_transpose_CHWK(dark_cpu_conv1_w,    16,  3,3,3);
    weight_transpose_CHWK(dark_cpu_conv2_w,    32, 16,3,3);
    weight_transpose_CHWK(dark_cpu_conv3_w,    16, 32,1,1);
    weight_transpose_CHWK(dark_cpu_conv4_w,   128, 16,3,3);
    weight_transpose_CHWK(dark_cpu_conv5_w,    16,128,1,1);
    weight_transpose_CHWK(dark_cpu_conv6_w,   128, 16,3,3);
    weight_transpose_CHWK(dark_cpu_conv7_w,    32,128,1,1);
    weight_transpose_CHWK(dark_cpu_conv8_w,   256, 32,3,3);
    weight_transpose_CHWK(dark_cpu_conv9_w,    32,256,1,1);
    weight_transpose_CHWK(dark_cpu_conv10_w,  256, 32,3,3);
    weight_transpose_CHWK(dark_cpu_conv11_w,   64,256,1,1);
    weight_transpose_CHWK(dark_cpu_conv12_w,  512, 64,3,3);
    weight_transpose_CHWK(dark_cpu_conv13_w,   64,512,1,1);
    weight_transpose_CHWK(dark_cpu_conv14_w,  512, 64,3,3);
    weight_transpose_CHWK(dark_cpu_conv15_w,  128,512,1,1);
    weight_transpose_CHWK(dark_cpu_conv16_w, 1000,128,1,1);

    //opencl initialization
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);  
    bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, 416*416*1024*sizeof(float), NULL, &err);
    bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, 416*416*1024*sizeof(float), NULL, &err);
    bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, 416*416*1024*sizeof(float), NULL, &err);

}
void mobilev1_openblas_free(){
    free(dark_cpu_o1  );free(dark_cpu_o1_m);
    free(dark_cpu_o2  );free(dark_cpu_o2_m);
    free(dark_cpu_o3  );
    free(dark_cpu_o4  );
    free(dark_cpu_o5  );
    free(dark_cpu_o6  );free(dark_cpu_o6_m);
    free(dark_cpu_o7  );
    free(dark_cpu_o8  );
    free(dark_cpu_o9  );
    free(dark_cpu_o10 );free(dark_cpu_o10_m);
    free(dark_cpu_o11  );
    free(dark_cpu_o12  );
    free(dark_cpu_o13  );
    free(dark_cpu_o14  );
    free(dark_cpu_o15  );
    free(dark_cpu_o16  );

    free(dark_cpu_o1_i2c);
    free(dark_cpu_o2_i2c);
    free(dark_cpu_o4_i2c);
    free(dark_cpu_o6_i2c);
    free(dark_cpu_o8_i2c);
    free(dark_cpu_o10_i2c);
    free(dark_cpu_o12_i2c);
    free(dark_cpu_o14_i2c);
    free(dark_cpu_o16_avg);
}
void mobilev1_openblas(float *input, float *output, int mode){
    
    im2col(input, dark_cpu_o1_i2c, 224,224, 3,3,1,1);
    gemm_blas(dark_cpu_o1_i2c, dark_cpu_conv1_w, dark_cpu_o1, 224*224,  3*3*3, 16, mode);
    matrix_bias(dark_cpu_o1, dark_cpu_conv1_b, 224*224, 3*3*3, 16);
    relu(dark_cpu_o1,224*224*16);
   
    maxpool(dark_cpu_o1, dark_cpu_o1_m, 224, 224, 16, 2, 2); 

    im2col(dark_cpu_o1_m, dark_cpu_o2_i2c, 112,112,16,3,1,1);
    gemm_blas(dark_cpu_o2_i2c, dark_cpu_conv2_w, dark_cpu_o2, 112*112, 16*3*3, 32, mode);
    matrix_bias(dark_cpu_o2, dark_cpu_conv2_b, 112*112, 16*3*3, 32);
    relu(dark_cpu_o2,112*112*32);
    
    maxpool(dark_cpu_o2, dark_cpu_o2_m, 112, 112, 32, 2, 2);

    gemm_blas(dark_cpu_o2_m,     dark_cpu_conv3_w, dark_cpu_o3,  56* 56, 32*1*1, 16, mode);
    matrix_bias(dark_cpu_o3, dark_cpu_conv3_b, 56*56, 32*1*1, 16);
    relu(dark_cpu_o3,56*56*16);

    im2col(dark_cpu_o3, dark_cpu_o4_i2c,  56, 56,16,3,1,1);
    gemm_blas(dark_cpu_o4_i2c, dark_cpu_conv4_w, dark_cpu_o4,  56* 56, 16*3*3,128, mode);
    matrix_bias(dark_cpu_o4, dark_cpu_conv4_b, 56*56, 16*3*3, 128);
    relu(dark_cpu_o4,56*56*128);

    gemm_blas(dark_cpu_o4,     dark_cpu_conv5_w, dark_cpu_o5,  56* 56,128*1*1, 16, mode);
    matrix_bias(dark_cpu_o5, dark_cpu_conv5_b, 56*56, 128*1*1, 16);
    relu(dark_cpu_o5,56*56*16);

    im2col(dark_cpu_o5, dark_cpu_o6_i2c,  56, 56,16,3,1,1);    
    gemm_blas(dark_cpu_o6_i2c, dark_cpu_conv6_w, dark_cpu_o6,  56* 56, 16*3*3,128, mode);
    matrix_bias(dark_cpu_o6, dark_cpu_conv6_b, 56*56, 16*3*3, 128);
    relu(dark_cpu_o6,56*56*128);

    maxpool(dark_cpu_o6, dark_cpu_o6_m, 56, 56, 128, 2, 2);

    gemm_blas(dark_cpu_o6_m,     dark_cpu_conv7_w, dark_cpu_o7,  28* 28,128*1*1, 32, mode);
    matrix_bias(dark_cpu_o7, dark_cpu_conv7_b, 28*28, 128*1*1, 32);
    relu(dark_cpu_o7,28*28*32);

    im2col(dark_cpu_o7, dark_cpu_o8_i2c,  28, 28,32,3,1,1);
    gemm_blas(dark_cpu_o8_i2c, dark_cpu_conv8_w, dark_cpu_o8,  28* 28, 32*3*3,256, mode);
    matrix_bias(dark_cpu_o8, dark_cpu_conv8_b, 28*28, 32*3*3, 256);
    relu(dark_cpu_o8,28*28*256);

    gemm_blas(dark_cpu_o8,     dark_cpu_conv9_w, dark_cpu_o9,  28* 28,256*1*1, 32, mode);
    matrix_bias(dark_cpu_o9, dark_cpu_conv9_b, 28*28, 256*1*1, 32);
    relu(dark_cpu_o9,28*28*32);

    im2col(dark_cpu_o9, dark_cpu_o10_i2c, 28, 28,32,3,1,1);
    gemm_blas(dark_cpu_o10_i2c, dark_cpu_conv10_w, dark_cpu_o10,  28* 28, 32*3*3,256, mode);
    matrix_bias(dark_cpu_o10, dark_cpu_conv10_b, 28*28, 32*3*3, 256);
    relu(dark_cpu_o10,28*28*256);

    maxpool(dark_cpu_o10, dark_cpu_o10_m, 28, 28, 256, 2, 2);
    
    gemm_blas(dark_cpu_o10_m,     dark_cpu_conv11_w, dark_cpu_o11,  14* 14,256*1*1, 64, mode);
    matrix_bias(dark_cpu_o11, dark_cpu_conv11_b, 14*14, 256*3*3, 64);
    relu(dark_cpu_o11,14*14*64);

    im2col(dark_cpu_o11, dark_cpu_o12_i2c, 14, 14,64,3,1,1);
    gemm_blas(dark_cpu_o12_i2c, dark_cpu_conv12_w, dark_cpu_o12,  14* 14, 64*3*3,512, mode);
    matrix_bias(dark_cpu_o12, dark_cpu_conv12_b, 14*14, 64*3*3, 512);
    relu(dark_cpu_o12,14*14*512);

    gemm_blas(dark_cpu_o12,     dark_cpu_conv13_w, dark_cpu_o13,  14* 14,512*1*1, 64, mode);
    matrix_bias(dark_cpu_o13, dark_cpu_conv13_b, 14*14, 512*1*1, 64);
    relu(dark_cpu_o13,14*14*64);

    im2col(dark_cpu_o13, dark_cpu_o14_i2c, 14, 14,64,3,1,1);
    gemm_blas(dark_cpu_o14_i2c, dark_cpu_conv14_w, dark_cpu_o14,  14* 14, 64*3*3,512, mode);
    matrix_bias(dark_cpu_o14, dark_cpu_conv14_b, 14*14, 64*3*3, 512);
    relu(dark_cpu_o14,14*14*512);

    gemm_blas(dark_cpu_o14,     dark_cpu_conv15_w, dark_cpu_o15,  14* 14, 512*1*1,128, mode);
    matrix_bias(dark_cpu_o15, dark_cpu_conv15_b, 14*14, 512*1*1, 128);
    relu(dark_cpu_o15,14*14*128);

    gemm_blas(dark_cpu_o15,     dark_cpu_conv16_w, dark_cpu_o16,  14* 14,128*1*1,1000, mode);
    matrix_bias(dark_cpu_o16, dark_cpu_conv16_b, 14*14, 128*1*1, 1000);
    // relu(dark_cpu_o16,14*14*1000);

    avgpool(dark_cpu_o16_avg, dark_cpu_o16,14*14,1000);

    
    // for(int i = 0;i<1000;i++) printf("%.3f ",dark_cpu_o16_avg[i]);
    softmax(output,dark_cpu_o16_avg,1000);
    // for(int i = 0;i<1000;i++) printf("%.3f ",dark_cpu_o16[i]);
}