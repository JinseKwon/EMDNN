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
    printf(" > gemv ");
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
    printf(" > gemm ");
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
    printf(" > str_gemm ");
}

void gemm(LAYER *l, int i,
          int ta, int tb,
          int M, int N, int K,
          float ALPHA, float BETA){
#ifdef CLBLAST
    mem2cl_obj(l[i].CL_WEIGHT, l[i].WEIGHT);
    mem2cl_obj(l[i].CL_INPUT,  l[i].INPUT );
    mem2cl_obj(l[i].CL_OUTPUT, l[i].OUTPUT);
    CLBlastStatusCode status;
    status = CLBlastSgemm(CLBlastLayoutRowMajor,
                    CLBlastTransposeNo, CLBlastTransposeNo,
                    M, N, K,
                    1.0f,
                    l[i].CL_WEIGHT, 0, K,
                    l[i].CL_INPUT , 0, N,
                    0.0f,
                    l[i].CL_OUTPUT, 0, N,
                    l[i].QUE, l[i].EVT);
    if (status == CLBlastSuccess) {
    clWaitForEvents(1, l[i].EVT);
    }
    cl_obj2mem(l[i].CL_WEIGHT, &l[i].WEIGHT,CL_MAP_WRITE, M*K*l[i].XF);
    cl_obj2mem(l[i].CL_INPUT,  &l[i].INPUT, CL_MAP_WRITE, K*N*l[i].XF);
    cl_obj2mem(l[i].CL_OUTPUT, &l[i].OUTPUT,CL_MAP_WRITE, M*N*l[i].XF);
#else
#ifdef OPENBLAS
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                l[i].WEIGHT, K,
                l[i].INPUT , N,
                0.0f,
                l[i].OUTPUT, N);
#else
    cpu_gemm(l[i].WEIGHT, l[i].INPUT, l[i].OUTPUT,
             M, N, K);
#endif
#endif
}
//im2col은 >> CONV 시   weight * input으로 처리하도록 수정해야함.
// ex) input CHW로 들어오도록 처리해야함 (OpenCV는 HWC로 들어옴..)
// 따라서 Weight * Input으로 처리해야 darknet, CAFFE NCHW로 구성되어 있으므로 
//    weight를 수정하지 않을 수 있음.
// 물론 Input * Weight로 구성하여도 HWC * CHWN 로 곱해도 초기 weight 수정으로 가능하지만,
// weight 변조보다는 input transpose가 오버헤드가 적으므로 위와 같은 방식을 채용하도록 함.
void im2col(float *in, float *out, 
            int C, int H, int W,
            int KER, int stride, int pad){
    //CHW 순으로 정렬됨.
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
    printf(" > im2col ");
}

void maxpool(float *in, float *out, 
             int C, int H, int W, 
             int KER, int stride, int PAD){
    int center =  KER/2;
    int offset = -KER/2;
    
    //pad 값 있는경우 추가 작성해야함..
    
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

void avgpool(float *in, float *out, 
             int H, int W, int C, 
             int KER, int stride, int PAD){
    // double check_time = get_time();
    float mean_n = KER*KER;
    // printf("maxpool start_xy : %d\n",start_xy);
    for(int h = 0; h < H; h+=stride){
		for(int w = 0; w < W; w+=stride){
            for(int c = 0; c < C; c++){
                float avgp = 0;
                for(int r = 0; r < KER; ++r){
                    for(int s = 0; s < KER; ++s){
                        avgp += in[(h+r) * W * C + (w+s) * C + c ];
                    }
                }
				out[ (h/stride) * (W/stride) * C + (w/stride) * C + c] = avgp/mean_n;
			}
		}
	}
    // if(dark_DEBUG)  printf("m %.6f\n", get_time()-check_time);
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
    const float epsilon = 1e-3;
    
    for(int i=0; i<K; i++){
        //scale = gamma[i] / sqrt(var[i] + epsilon)
        float scale = bias[K*1 + i] / sqrt(bias[K*3 + i] + epsilon);
        bias[i] = bias[i] - bias[K*2 +i] * scale;
        for(int w=0; w<C*H*W; ++w){
            weight[i*C*H*W + w] = weight[i*C*H*W + w] * scale;
        }
    }
}
void bias_add(float* in, float* bias, 
              int C, int H, int W){
    for(int c = 0; c<C; ++c){
        for(int hw = 0; hw < H*W; ++hw){
            in[c*H*W + hw] += bias[c];
        }
    }
    printf(" > bias ");
}
void activate_function(float* in, ACTIVATION_TYPE act, 
                       int C, int H, int W){
    switch(act){
    case RELU:
        for (int chw = 0; chw < C*H*W; ++chw) {
            in[chw] = (in[chw]>0) ? in[chw] : 0;
        }
        break;
    case LEAKY:
        for (int chw = 0; chw < C*H*W; ++chw) {
            in[chw] = (in[chw]>0) ? in[chw] : .1*in[chw];
        }
        break;
    case LINEAR:
        break;
    default:
        return;
    }
    printf(" > activation ");
}