#include "emdnn.h"
#include <math.h>
#include <float.h>
float* input_img(float* input, int C, int H, int W){
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
        start_xy = 0;
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
             int KER, int stride){
    //NC HW RS
    for(int c = 0; c < C; c++){
        for(int h = 0; h < H; h=h+stride){
            for(int w = 0; w < W; w=w+stride){
                float max = -FLT_MAX;
                for(int r = 0; r<KER; ++r){
                    for(int s = 0; s<KER; ++s){
                        if(w+s < W || h+r < H){
                            max = fmaxf(in[W * H * c + W*(h+r) + (w+s)], max);
                        }
                    }
                }
                out[ H/stride * W/stride * c + W/stride * h/stride + w/stride] = max;
            }
        }
    }
}

void avgpool(float *in, float *out, int H, int W, int C, int KER, int stride){
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
void softmax(float *out, float *in, int K){
    // double check_time = get_time();
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
}