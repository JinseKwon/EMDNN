#include "emdnn.h"
#include <math.h>
#include <float.h>

#ifdef CLBLAST
#include <clblast_c.h>
#endif
#ifdef OPENBLAS
#include <cblas.h>
#endif

float* input_bin_img(float* input, int C, int H, int W){
    FILE *fin = fopen("dog.bin", "rb");
    if (!fin) {
        printf("Input file does not exist.\n");
        exit(0);
    }
    fseek(fin, 0, SEEK_END);
    long fsz = ftell(fin);
    rewind(fin);
    unsigned char *imgs = (unsigned char*)malloc(H * W * C);
    fread(imgs, 1, H * W * C, fin);
    fclose(fin);

    //CHW
    for(int j = 0; j< C; j++){
        for(int i = 0; i< H * W; i++){
		    input[j*H*W + i] = (float) imgs[i*C + j] /255.;
        }
	}

    return input;
}
void transpose(float *INA, float *OUTA, 
               int ldx, int ldy){
    
    for(int y = 0; y < ldy; ++y){
        for(int x = 0; x < ldx; ++x){
            OUTA[x * ldy + y] = INA[ y * ldx + x];
        }
    }
}
void transpose_fc(float *INA,
               int ldx, int ldy){
    float* temp = (float*)malloc(ldx * ldy * sizeof(float));
    for(int j = 0; j <ldx*ldy; ++j){
        temp[j] = INA[j];
    }
    for(int y = 0; y < ldy; ++y){
        for(int x = 0; x < ldx; ++x){
            INA[x * ldy + y] = temp[ y * ldx + x];
        }
    }
    free(temp);
}
//merged donghee's code
void cpu_gemv(float *A, float *B, float *C, 
               int M, int N, int K){  //N=1
    for (int m = 0; m < M; ++m) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[m * K + k] * B[k];
        }
        C[m] = sum;
    }
    //printf(" > gemv ");
}
void cpu_gemm(float *A, float *B, float *C, 
              int M, int N, int K){
    for(int m = 0; m < M; ++m){
        for(int n = 0; n < N; ++n){
            float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += A[m*K+k] * B[k*N+n];
            }
            C[m*N+n] = sum;
        }
    }
    // //C+= register block
    // for(int MN = 0; MN < M*N; ++MN){
    //     C[MN] = 0;
    // }
    // for(int m = 0; m < M; ++m){
    //     for(int k = 0; k < K; ++k){
    //         register float Apart = A[m*K+k];
    //         for(int n = 0; n < N; ++n){
    //             C[m*N+n] += Apart * B[k*N+n];
    //         }
    //     }
    // }
    //printf(" > cpu_gemm ");
}
void cpu_stride_b_gemm(float *A, float *B, float *C, 
                      int M, int N, int K,
                      int stride_b, int batch_count){
    for(int m = 0; m < M; ++m){
        for(int n = 0; n < N; ++n){
            float sum = 0;
            for(int bat = 0; bat < batch_count; ++bat){
                for(int k = 0; k < stride_b; ++k){
                    sum += A[m*K+k] * B[bat*stride_b*N + k*N + n];
                }
            }
            C[m*N+n] = sum;
        }
    }
    //printf(" > str_gemm ");
}
void gemv(LAYER *l, int i,
          float* A,    float* B,    float* C,
          int ta,
          int M, int N, int K,
          float ALPHA, float BETA){
    if(l[i].DEVICE == GPU){
#ifdef CLBLAST
        cl_mem cl_A = l[i  ].CL_WEIGHT; 
        cl_mem cl_B = l[i-1].CL_OUTPUT;
        cl_mem cl_C = l[i  ].CL_OUTPUT;
        CLBlastStatusCode status;
        status = CLBlastSgemv(CLBlastLayoutRowMajor,
                        CLBlastTransposeNo,
                        M, K,
                        1.0f,
                        cl_A, 0, K,
                        cl_B, 0, 1,
                        0.0f,
                        cl_C, 0, 1,
                        l[0].QUE, l[0].EVT);
        if (status == CLBlastSuccess) {
        clWaitForEvents(1, l[0].EVT);
        }
#endif
    }else if(l[i].DEVICE == CPU){
#ifdef OPENBLAS
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                M, K,
                ALPHA,
                A, K,
                B, 1,
                BETA,
                C, 1);
#else
    cpu_gemv(A,B,C,
             M,N,K);
#endif
    }
}
void gemm(LAYER *l, int i,
          float* A,    float* B,    float* C,
          int ta, int tb,
          int M, int N, int K,
          float ALPHA, float BETA){
    if(l[i].DEVICE == GPU){
#ifdef CLBLAST
        cl_mem cl_A = l[i].CL_WEIGHT; 
        cl_mem cl_B;
        cl_mem cl_C = l[i].CL_OUTPUT;
        if(l[i].IM2COL){
            cl_B = l[i].CL_INPUT;
        }else{
            cl_B = l[i-1].CL_OUTPUT;
        }
        // mem2cl_obj(A, cl_A);
        // mem2cl_obj(B, cl_B);
        // mem2cl_obj(C, cl_C);
        CLBlastStatusCode status;
        status = CLBlastSgemm(CLBlastLayoutRowMajor,
                        CLBlastTransposeNo, CLBlastTransposeNo,
                        M, N, K,
                        1.0f,
                        cl_A, 0, K,
                        cl_B , 0, N,
                        0.0f,
                        cl_C, 0, N,
                        l[0].QUE, l[0].EVT);
        if (status == CLBlastSuccess) {
        clWaitForEvents(1, l[0].EVT);
        }
        // cl_obj2mem(cl_A, &A,CL_MAP_WRITE, M*K*l[i].XF);
        // cl_obj2mem(cl_B, &B,CL_MAP_WRITE, K*N*l[i].XF);
        // cl_obj2mem(cl_C, &C,CL_MAP_WRITE, M*N*l[i].XF);
#endif
    }else if(l[i].DEVICE == CPU){
#ifdef OPENBLAS
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                A, K,
                B , N,
                0.0f,
                C, N);
#else
    cpu_gemm(A, B, C,
             M, N, K);
#endif
    }
}
//im2col은 >> CONV 시   weight * input으로 처리하도록 수정해야함.
// ex) input CHW로 들어오도록 처리해야함 (OpenCV는 HWC로 들어옴..)
// 따라서 Weight * Input으로 처리해야 darknet, CAFFE NCHW로 구성되어 있으므로 
//    weight를 수정하지 않을 수 있음.
// 물론 Input * Weight로 구성하여도 HWC * CHWN 로 곱해도 초기 weight 수정으로 가능하지만,
// weight 변조보다는 input transpose가 오버헤드가 적으므로 위와 같은 방식을 채용하도록 함.
void im2col(LAYER *l, int i,
            float *in, float *out, 
            int C, int H, int W,
            int KER, int stride, int pad){
    //CHW 순으로 정렬됨.
    if(l[i].DEVICE == CPU){
        int start_xy;
        if(pad == 0){
            start_xy = (KER/2);
        }else{
            start_xy = (KER/2)-pad;
        }
        for(int c = 0; c < C; ++c){
            for(int hout = start_xy; hout <H-start_xy; hout+=stride){
            for(int wout = start_xy; wout <W-start_xy; wout+=stride){
                for(int r = 0; r < KER; ++r){
                for(int s = 0; s < KER; ++s){
                    int h  = hout + r - (KER/2);
                    int w  = wout + s - (KER/2);
                    if(h < 0 || h >= H || w < 0 || w >= W){
                        out[((W - start_xy) / stride) * ((H   -start_xy) / stride) * KER * KER * c +  
                            ((W - start_xy) / stride) * ((H   -start_xy) / stride) * KER * r +   
                            ((W - start_xy) / stride) * ((H   -start_xy) / stride) * s +   
                            ((W - start_xy) / stride) * ((hout-start_xy) / stride) + 
                            ((wout-start_xy) /stride) 
                            ] = 0;
                    }else{
                        out[((W - start_xy) / stride) * ((H   -start_xy) / stride) * KER * KER * c +  
                            ((W - start_xy) / stride) * ((H   -start_xy) / stride) * KER * r +   
                            ((W - start_xy) / stride) * ((H   -start_xy) / stride) * s +   
                            ((W - start_xy) / stride) * ((hout-start_xy) / stride) + 
                            ((wout-start_xy) /stride) 
                            ] = in[ H * W * c + W * h + w ];
                    }
                }
                }
            }
            }
        }
    }
#ifdef OPENCL
    else if(l[i].DEVICE == GPU){
        //kernel build 한번에 하고 불러오는 형태로 수정해야함...
        // cl_kernel im2colKernel = clCreateKernel(*l[i].PROG, "im2colKernel",&err);  
        // if(err != CL_SUCCESS) { 
        //     printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); 
        //     exit(EXIT_FAILURE); 
        // }
        size_t global_size[2];
        size_t local_size[2];

        clSetKernelArg(*l[0].KER_IM2COL ,0,sizeof(cl_mem), &l[i-1].CL_OUTPUT);
        clSetKernelArg(*l[0].KER_IM2COL ,1,sizeof(cl_mem), &l[i].CL_INPUT);
        clSetKernelArg(*l[0].KER_IM2COL, 2,sizeof(cl_int), &C);
        clSetKernelArg(*l[0].KER_IM2COL, 3,sizeof(cl_int), &H); 
        clSetKernelArg(*l[0].KER_IM2COL, 4,sizeof(cl_int), &W);  
        clSetKernelArg(*l[0].KER_IM2COL ,5,sizeof(cl_int), &KER);
        clSetKernelArg(*l[0].KER_IM2COL ,6,sizeof(cl_int), &stride);
        clSetKernelArg(*l[0].KER_IM2COL ,7,sizeof(cl_int), &pad);

        // global_size[1] = H;
        // global_size[0] = W;
        
        int start_xy;
        if(pad == 0){
            start_xy = (KER/2);
        }else{
            start_xy = (KER/2)-pad;
        }

        global_size[1] = (H - start_xy) / stride;
        global_size[0] = (W - start_xy) / stride;
        
        clEnqueueNDRangeKernel(*l[0].QUE, *l[0].KER_IM2COL , 2, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(*l[0].QUE);
    }
#endif
    if(!l[i].TUNE)    printf(" > im2col ");
}

void maxpool(LAYER *l,  int i,
             float *in, float *out, 
             int C, int H, int W, 
             int KER, int stride, int PAD){
    int center =  KER/2;
    int offset = -KER/2;
    
    //pad 값 있는경우 추가 작성해야함..
    
    if(l[i].DEVICE == CPU){
        //NC HW RS
        for(int c = 0; c < C; c++){
            for(int h = center; h < H; h=h+stride){
                for(int w = center; w < W; w=w+stride){
                    float max = -FLT_MAX;
                    for(int r = offset; r<(KER+offset); ++r){
                        for(int s = offset; s<(KER+offset); ++s){
                            // if(w+s < W || h+r < H){
                            max = fmaxf(in[W * H * c + W*(h+r) + (w+s)], max);
                            // }
                        }
                    }
                    out[ (W/stride) * (H/stride) * c + 
                        (W/stride) * ((h-center)/stride) + 
                        ((w-center)/stride)] = max;
                }
            }
        }
    }
#ifdef OPENCL    
    else if(l[i].DEVICE == GPU){
        // cl_int err;
        // cl_kernel maxpoolKernel = clCreateKernel(*l[i].PROG, "maxpoolKernel",&err);  
        // if(err != CL_SUCCESS) { 
        //     printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); 
        //     exit(EXIT_FAILURE); 
        // }
        size_t global_size[2];
        size_t local_size[2];

        clSetKernelArg(*l[0].KER_MAXPOOL ,0,sizeof(cl_mem), &l[i-1].CL_OUTPUT);
        clSetKernelArg(*l[0].KER_MAXPOOL ,1,sizeof(cl_mem), &l[i].CL_OUTPUT);
        clSetKernelArg(*l[0].KER_MAXPOOL, 2,sizeof(cl_int), &C);
        clSetKernelArg(*l[0].KER_MAXPOOL, 3,sizeof(cl_int), &H); 
        clSetKernelArg(*l[0].KER_MAXPOOL, 4,sizeof(cl_int), &W);  
        clSetKernelArg(*l[0].KER_MAXPOOL ,5,sizeof(cl_int), &KER);
        clSetKernelArg(*l[0].KER_MAXPOOL ,6,sizeof(cl_int), &stride);
        clSetKernelArg(*l[0].KER_MAXPOOL ,7,sizeof(cl_int), &PAD);

        global_size[1] = H / stride;
        global_size[0] = W / stride;
        
        clEnqueueNDRangeKernel(*l[0].QUE, *l[0].KER_MAXPOOL , 2, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(*l[0].QUE);
    }
#endif
}

void avgpool(LAYER *l,  int i,
             float *in, float *out, 
             int C, int H, int W, 
             int KER, int stride, int PAD){

    float mean_n = KER*KER;
    if(l[i].DEVICE == CPU){

        for(int c = 0; c < C; c++){
            float avgp = 0.0f;
            for(int r = 0; r < KER; ++r){
                for(int s = 0; s < KER; ++s){
                    avgp += in[KER*KER* c + KER* r + s];
                }
            }
            out[ c ] = avgp/mean_n;
        }
    }
#ifdef OPENCL    
    else if(l[i].DEVICE == GPU){
        size_t global_size[1];
        size_t local_size[1];

        clSetKernelArg(*l[0].KER_AVGPOOL ,0,sizeof(cl_mem), &l[i-1].CL_OUTPUT);
        clSetKernelArg(*l[0].KER_AVGPOOL ,1,sizeof(cl_mem), &l[i].CL_OUTPUT);
        clSetKernelArg(*l[0].KER_AVGPOOL, 2,sizeof(cl_int), &C);
        clSetKernelArg(*l[0].KER_AVGPOOL, 3,sizeof(cl_int), &H); 
        clSetKernelArg(*l[0].KER_AVGPOOL, 4,sizeof(cl_int), &W);  
        clSetKernelArg(*l[0].KER_AVGPOOL ,5,sizeof(cl_int), &KER);
        clSetKernelArg(*l[0].KER_AVGPOOL ,6,sizeof(cl_int), &stride);
        clSetKernelArg(*l[0].KER_AVGPOOL ,7,sizeof(cl_int), &PAD);

        global_size[0] = C;
        
        clEnqueueNDRangeKernel(*l[0].QUE, *l[0].KER_AVGPOOL , 1, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(*l[0].QUE);
    }
#endif
}
void softmax(float *in, float *out, int C){
    // double check_time = get_time();
    float max = -FLT_MAX;
    
    for (int cc = 0; cc < C; cc++) {
        // printf("%.6f \n",in[cc]);
        if (in[cc] > max) {
            max = in[cc];
        }
    }

    float sum = 0.0f;
    for (int cc = 0; cc < C; cc++) {
        float e = expf(in[cc] - max);
        sum += e;
        out[cc] = e;
    }

    for (int cc = 0; cc < C; cc++) {
        
        out[cc] /= sum;
    }

    // if(dark_DEBUG)  printf("soft %.6f\n", get_time()-check_time);
}

void batch_normalizaiton(float *bias, float *weight, 
                         int K, int C, int H, int W){
    //bias  K
    //gamma K
    //mean  K
    //var   K
    const float epsilon = 1e-5;
    
    for(int i=0; i<K; i++){
        //scale = gamma[i] / sqrt(var[i] + epsilon)
        float scale = bias[K*1 + i] / sqrt(bias[K*3 + i] + epsilon);
        bias[i] = bias[i] - bias[K*2 +i] * scale;
        for(int w=0; w<C*H*W; ++w){
            weight[i*C*H*W + w] = weight[i*C*H*W + w] * scale;
        }
    }
}
void bias_add(LAYER *l, int i,
              float* in, float* bias, 
              int C, int H, int W){
    if(l[i].DEVICE == CPU){
        for(int c = 0; c<C; ++c){
            for(int hw = 0; hw < H*W; ++hw){
                in[c*H*W + hw] += bias[c];
            }
        }
    }
#ifdef OPENCL
    else if(l[i].DEVICE == GPU){  
        // cl_kernel biasKernel = clCreateKernel(*l[i].PROG, "biasKernel",NULL);       
        
        size_t global_size[2];
        int HW = H*W;
        clSetKernelArg(*l[0].KER_BIAS ,0,sizeof(cl_mem),&l[i].CL_OUTPUT);
        clSetKernelArg(*l[0].KER_BIAS ,1,sizeof(cl_mem),&l[i].CL_BIAS); 
        clSetKernelArg(*l[0].KER_BIAS ,2,sizeof(cl_int),&C);
        clSetKernelArg(*l[0].KER_BIAS ,3,sizeof(cl_int),&HW);

        global_size[1] = (int)C;
        global_size[0] = (int)HW;

        clEnqueueNDRangeKernel(*l[0].QUE, *l[0].KER_BIAS , 2, NULL, global_size, NULL, 0, NULL, NULL);
        clFinish(*l[0].QUE);
    }
#endif
    
    if(!l[i].TUNE)   printf(" > bias ");
}
void activate_function(LAYER *l, int i,
                       float* in, ACTIVATION_TYPE act, 
                       int C, int H, int W){
    switch(act){
    case RELU:
        if(l[i].DEVICE == CPU){
            for (int chw = 0; chw < C*H*W; ++chw) {
                in[chw] = (in[chw]>0) ? in[chw] : 0;
            }
        }
#ifdef OPENCL
        else if(l[i].DEVICE ==GPU){
            float alpha = 0.;
            // cl_kernel activationRelu = clCreateKernel(*l[i].PROG, "activationRelu",NULL);
            clSetKernelArg(*l[0].KER_RELU, 0, sizeof(cl_mem),   &l[i].CL_OUTPUT);
            clSetKernelArg(*l[0].KER_RELU, 1, sizeof(cl_float), &alpha);
        
            size_t global_size = C*H*W;
        
            clEnqueueNDRangeKernel(*l[0].QUE, *l[0].KER_RELU , 1, NULL, &global_size, NULL, 0, NULL, NULL);
            clFinish(*l[0].QUE);
        }
#endif
        break;
    case LEAKY:
        if(l[i].DEVICE == CPU){
            for (int chw = 0; chw < C*H*W; ++chw) {
                in[chw] = (in[chw]>0) ? in[chw] : .1*in[chw];
            }
        }
#ifdef OPENCL        
        else if(l[i].DEVICE ==GPU){
            float alpha = 0.1;
            // cl_kernel activationRelu = clCreateKernel(*l[i].PROG, "activationRelu",NULL);       
            clSetKernelArg(*l[0].KER_RELU, 0, sizeof(cl_mem),   &l[i].CL_OUTPUT);
            clSetKernelArg(*l[0].KER_RELU, 1, sizeof(cl_float), &alpha);
        
            size_t global_size = C*H*W;
        
            clEnqueueNDRangeKernel(*l[0].QUE, *l[0].KER_RELU , 1, NULL, &global_size, NULL, 0, NULL, NULL);
            clFinish(*l[0].QUE);
        }
#endif
        break;
    case LINEAR:
        break;
    default:
        return;
    }
    if(!l[i].TUNE)    printf(" > activation ");
}
struct timespec u_time;
double get_time(){
	clock_gettime(CLOCK_REALTIME, &u_time);
	return (u_time.tv_sec) + (u_time.tv_nsec) * 1e-9;
}	