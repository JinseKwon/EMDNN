#include "emdnn.h"
#include <math.h>
#include <limits.h>
#include <float.h>
#include <arm_neon.h>

#ifdef DEBUG_PRINT
#define DEBUG_PRINT 1
#else 
#define DEBUG_PRINT 0
#endif

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

float maximum(float *A,int N){
    float max=-FLT_MAX;
    for(long i = 0; i < N; ++i){
        if(fabs(A[i])>max){
            max = fabs(A[i]);
        }
    }
    return max;
}
void T_fp32_int8(float *A, int8_t *A_q8, 
               int N, int K, float range){
    
    float scale = 127/range;
    for(int i = 0; i < N; ++i){
        for(int k = 0; k < K; ++k){
            A_q8[i*K+k] = (int8_t)(A[k*N+i]*scale);
        }
    }
}
void fp32_int8(float *A, int8_t *A_q8, 
               long N,    float range){
    
    float scale = 127/range;
    for(long i = 0; i < N; ++i){
        A_q8[i] = (int8_t)(A[i]*scale);
    }
}
void int32_fp32(int *C_q, float *C,
               long N,    float range1, float range2){
    
    float scale1 = 127/range1;
    float scale2 = 127/range2;
    scale1 = scale1 * scale2;
    for(long i = 0; i < N; ++i){
        C[i] = (float)(C_q[i]/scale1);
    }
}
void AddDot8(int M, int N, int K, int i, int j, int8_t *A, int8_t *B, int *C){
    register int sum = 0;
    for(int k=0; k<K; k+=8){
        if(k+8 < K){
            int8x8_t a = vld1_s8(A + i*K + k);
            int8x8_t b = vld1_s8(B + j*K + k);
            int16x8_t c = {0,0,0,0, 0,0,0,0};
            c = vmlal_s8(c,a,b);
            int16_t t[8];
            vst1q_s16 (t, c);
            sum += (int)(t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]); 
        }else{
            // printf("k = %d K = %d \n", k, K);
            for(int kk = k; kk<K; kk++){
                sum += (int)(A[i * K + k] * B[j * K + k]);
            }
        }
    }
    C[i*N+j] = sum;
}
void AddDot8_gemm(int8_t *A_q8, int8_t *B_q8, int *C_q,int M, int N, int K){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            AddDot8(M,N,K,i,j,A_q8,B_q8,C_q);
        }
    }
}
void T_gemm_qu_de_q8(float *A, float *B, float *C,int M, int N, int K){
    int8_t *A_q8 = (int8_t*)malloc(M * K * sizeof(int8_t));
    int8_t *B_q8 = (int8_t*)malloc(K * N * sizeof(int8_t));
    int    *C_q  = (int*)   malloc(M * N * sizeof(int));

    /* 1. Scale Quantization */
        //1 = positive max interger range
    float rng1 = maximum(A,M*K);
    float rng2 = maximum(B,K*N);
    // printf("%.6f %.6f\n",rng1, rng2); 
    fp32_int8(A, A_q8, M*K, rng1); 
    T_fp32_int8(B, B_q8, N, K, rng2);
    /* 2. int8 multiplication */
    AddDot8_gemm(A_q8,B_q8,C_q,M,N,K);
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         int sum = 0;
    //         for (int k = 0; k < K; k++) {
    //             sum += A_q8[i * K + k] * B_q8[j * K + k];
    //         }
    //         /* 3. dequantization */
    //         C_q[i*N + j] = sum;
    //     }
    // }
    int32_fp32(C_q, C, M*N, rng1,rng2);
    free(A_q8);
    free(B_q8);
    free(C_q);
}
void cpu_gemm(float *A, float *B, float *C, 
              int M, int N, int K){
    // for(int m = 0; m < M; ++m){
    //     for(int n = 0; n < N; ++n){
    //         float sum = 0;
    //         for(int k = 0; k < K; ++k){
    //             sum += A[m*K+k] * B[k*N+n];
    //         }
    //         C[m*N+n] = sum;
    //     }
    // }

    T_gemm_qu_de_q8(A,B,C,M,N,K);
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
void T_gemv_qu_de_q8(float *A, float *B, float *C,int M, int N, int K){
    //N = 1;
    int8_t *A_q8 = (int8_t*)malloc(M * K * sizeof(int8_t));
    int8_t *B_q8 = (int8_t*)malloc(K * N * sizeof(int8_t));
    int    *C_q  = (int*)   malloc(M * N * sizeof(int));

    /* 1. Scale Quantization */
        //1 = positive max interger range
    float rng1 = maximum(A,M*K);
    float rng2 = maximum(B,K*N);
    // printf("%.6f %.6f\n",rng1, rng2); 
    fp32_int8(A, A_q8, M*K, rng1); 
    fp32_int8(B, B_q8, N*K, rng2);
    /* 2. int8 multiplication */
    for (int i = 0; i < M; i++) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A_q8[i * K + k] * B_q8[k];
        }
        /* 3. dequantization */
        C_q[i] = sum;
    }
    int32_fp32(C_q, C, M*N, rng1,rng2);
    free(A_q8);
    free(B_q8);
    free(C_q);
}
//merged donghee's code
void cpu_gemv(float *A, float *B, float *C, 
               int M, int N, int K){  //N=1
    
    T_gemv_qu_de_q8(A,B,C,M,N,K);
    // for (int m = 0; m < M; ++m) {
    //     float sum = 0;
    //     for (int k = 0; k < K; ++k) {
    //         sum += A[m * K + k] * B[k];
    //     }
    //     C[m] = sum;
    // }
    //printf(" > gemv ");
}
void cpu_depthwise_conv(float *A, float *B, float *C, 
                    int M, int N, int K,
                    int offset){
    for(int m = 0; m < M; ++m){
        for(int n = 0; n < N; ++n){
            float sum = 0;
            for(int k=0; k<K; ++k){
                sum += A[m*K+k] * B[m*offset*N + k*N + n];
            }
            C[m*N + n] = sum;
        }
    }
    // printf(" > str_gemm ");
}
void depth_conv(LAYER *l, int i){
    depth_gemm(l, i,
            l[i].WEIGHT, l[i].INPUT, l[i].OUTPUT,
            0, 0,
            l[i].IN_C,
            l[i].OUT_H * l[i].OUT_W,
            l[i].H*l[i].W,
            l[i].H*l[i].W,   //offset
            1.0f, 0.0f);
}
void depth_gemm(LAYER *l, int i,
          float* A,    float* B,    float* C,
          int ta, int tb,
          int M, int N, int K,
          int OFFSET,
          float ALPHA, float BETA){
    if(l[i].DEVICE == GPU){
#ifdef CLBLAST
        cl_mem cl_A = l[i].CL_WEIGHT; 
        cl_mem cl_B;
        cl_mem cl_C = l[i].CL_OUTPUT;
        if(l[i].IM2COL){
            cl_B = l[i].CL_INPUT;
            im2col(l, i,
                   l[i-1].OUTPUT, l[i].INPUT,
                   l[i].IN_C, l[i].IN_H, l[i].IN_W,
                   l[i].W,
                   l[i].STRIDE, l[i].PAD);
        }else{
            cl_B = l[i-1].CL_OUTPUT;
        }
        CLBlastStatusCode status;
        status = CLBlastSgemmStridedBatched(
                        CLBlastLayoutRowMajor,
                        CLBlastTransposeNo, CLBlastTransposeNo,
                        1, N, K,
                        ALPHA,
                        cl_A, 0, K, K,
                        cl_B, 0, N, K*N,
                        BETA,
                        cl_C, 0, N, N,
                        M,
                        l[0].QUE, l[0].EVT);
        if (status == CLBlastSuccess) {
        clWaitForEvents(1, l[0].EVT);
        }
#endif
    }else if(l[i].DEVICE == CPU){
        if(l[i].IM2COL){
            im2col(l, i,
                l[i-1].OUTPUT, l[i].INPUT,
                l[i].IN_C, l[i].IN_H, l[i].IN_W,
                l[i].W,
                l[i].STRIDE, l[i].PAD);
            B = l[i].INPUT;
        }else{
            B = l[i-1].OUTPUT;
        }
#ifdef OPENBLAS
        for(int itr=0;itr<M; ++itr){
            cblas_sgemv(CblasRowMajor,
                        CblasNoTrans,
                        N, K,
                        ALPHA,
                        B+(itr*K*N), K,
                        A+(itr*K), 1,
                        BETA,
                        C+(itr*N), 1);

            // cblas_sgemm(CblasRowMajor,
            //             CblasNoTrans, CblasNoTrans,
            //             1, N, K,
            //             1.0f,
            //             A+(itr*K), K,
            //             B+(itr*K*N), N,
            //             0.0f,
            //             C+(itr*N), N);
        }
#else
        cpu_depthwise_conv(A, B, C,
                        M, N, K, 
                        OFFSET);
#endif
    }
    else if(l[i].DEVICE == PPU){
#ifdef NNPACK
        nnpack_depth_conv(l, i);
#endif
    }
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > depth_c ");
}

#ifdef NNPACK
void nnpack_depth_conv(LAYER *l, int i){
    struct nnp_padding NP_in_pad   = { l[i].PAD   , l[i].PAD , 
                                       l[i].PAD   , l[i].PAD    };  
    struct nnp_size    NP_in_size  = { l[i].IN_H  , l[i].IN_W   };  
    struct nnp_size    NP_ker_size = { l[i].H     , l[i].W      };
    struct nnp_size    NP_str_size = { l[i].STRIDE, l[i].STRIDE };
    for(int ch = 0; ch< l[i].N; ++ch){
        nnp_convolution_inference(
            nnp_convolution_algorithm_implicit_gemm,
            nnp_convolution_transform_strategy_tuple_based,
            1,
            1,
            NP_in_size,
            NP_in_pad,
            NP_ker_size,
            NP_str_size,
            l[i-1].OUTPUT + (l[i].IN_H * l[i].IN_W * ch),
            l[i].WEIGHT   + (l[i].H    * l[i].W    * ch),
            NULL,
            l[i].OUTPUT   + (l[i].OUT_H* l[i].OUT_W* ch),
            l[0].PTHREAD,
            NULL
        );
    }
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > CONV ");
}
void nnpack_conv(LAYER *l, int i){
    struct nnp_padding NP_in_pad   = { l[i].PAD   , l[i].PAD , 
                                       l[i].PAD   , l[i].PAD    };  
    struct nnp_size    NP_in_size  = { l[i].IN_H  , l[i].IN_W   };  
    struct nnp_size    NP_ker_size = { l[i].H     , l[i].W      };
    struct nnp_size    NP_str_size = { l[i].STRIDE, l[i].STRIDE };
    nnp_convolution_inference(
        nnp_convolution_algorithm_implicit_gemm,
        nnp_convolution_transform_strategy_tuple_based,
        l[i].IN_C,
        l[i].N,
        NP_in_size,
        NP_in_pad,
        NP_ker_size,
        NP_str_size,
        l[i-1].OUTPUT,
        l[i].WEIGHT,
        NULL,
        l[i].OUTPUT,
        l[0].PTHREAD,
        NULL
    );
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > CONV ");
}
void nnpack_fc(LAYER *l, int i){
    nnp_fully_connected_inference(
        l[i].C,
        l[i].N,
        l[i-1].OUTPUT,
        l[i].WEIGHT,
        l[i].OUTPUT,
        l[0].PTHREAD
    );
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > FC ");
}
void nnpack_maxpool(LAYER *l, int i){
    struct nnp_padding NP_in_pad   = { 0          , l[i].PAD , 
                                       l[i].PAD   , 0           };  
    struct nnp_size    NP_in_size  = { l[i].IN_H  , l[i].IN_W   };  
    struct nnp_size    NP_pool_size ={ l[i].H     , l[i].W      };
    struct nnp_size    NP_str_size = { l[i].STRIDE, l[i].STRIDE };

    nnp_max_pooling_output(
        1,
        l[i].IN_C,
        NP_in_size,
        NP_in_pad,
        NP_pool_size,
        NP_str_size,
        l[i-1].OUTPUT,
        l[i].OUTPUT,
        l[0].PTHREAD);
}
#endif
void conv(LAYER *l, int i){
    gemm(l, i,
         l[i].WEIGHT,    l[i].INPUT,    l[i].OUTPUT,
         0, 0,
         l[i].N,
         l[i].OUT_H * l[i].OUT_W,
         l[i].C * l[i].H * l[i].W,
         1.0f, 0.0f);
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
            im2col(l, i,
                   l[i-1].OUTPUT, l[i].INPUT,
                   l[i].IN_C, l[i].IN_H, l[i].IN_W,
                   l[i].W,
                   l[i].STRIDE, l[i].PAD);
            cl_B = l[i].CL_INPUT;
        }else{
            cl_B = l[i-1].CL_OUTPUT;
        }
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
#endif
    }else if(l[i].DEVICE == CPU){
        if(l[i].IM2COL){
            im2col(l, i,
                l[i-1].OUTPUT, l[i].INPUT,
                l[i].IN_C, l[i].IN_H, l[i].IN_W,
                l[i].W,
                l[i].STRIDE, l[i].PAD);
            B = l[i].INPUT;
        }else{
            B = l[i-1].OUTPUT;
        }
#ifdef OPENBLAS
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    A, K,
                    B, N,
                    0.0f,
                    C, N);
#else
        cpu_gemm(A, B, C,
                M, N, K);
#endif
    }else if(l[i].DEVICE == PPU){
#ifdef NNPACK
        nnpack_conv(l, i);
#endif
    }
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > gemm ");
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
    }else if(l[i].DEVICE == PPU){
#ifdef NNPACK
        nnpack_fc(l, i);
#endif
    }
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > gemv ");
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
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > im2col ");
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
    else if(l[i].DEVICE == PPU){
#ifdef NNPACK
        nnpack_maxpool(l, i);
#endif
    }
}

void avgpool(LAYER *l,  int i,
             float *in, float *out, 
             int C, int H, int W, 
             int KER, int stride, int PAD){

    float mean_n = KER*KER;
    if(l[i].DEVICE == CPU || l[i].DEVICE == PPU){

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
    if(l[i].DEVICE == CPU || l[i].DEVICE == PPU){
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
    
    if(!l[i].TUNE && DEBUG_PRINT)   printf(" > bias ");
}
void activate_function(LAYER *l, int i,
                       float* in, ACTIVATION_TYPE act, 
                       int C, int H, int W){
    switch(act){
    case RELU:
        if(l[i].DEVICE == CPU || l[i].DEVICE == PPU){
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
        if(l[i].DEVICE == CPU || l[i].DEVICE == PPU){
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
    if(!l[i].TUNE && DEBUG_PRINT)    printf(" > activation ");
}
struct timespec u_time;
double get_time(){
	clock_gettime(CLOCK_REALTIME, &u_time);
	return (u_time.tv_sec) + (u_time.tv_nsec) * 1e-9;
}	