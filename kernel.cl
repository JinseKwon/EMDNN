__kernel void im2colKernel(__global float *in,
                           __global float *out,
                           int C, int H,  int W, 
                           int KER, int STR, int PAD){
    int start_xy;
    if(PAD == 0){
        start_xy = (KER/2);
    }else{
        start_xy = (KER/2)-PAD;
    }
    
    const int hout = get_global_id(1) * STR + start_xy;
    const int wout = get_global_id(0) * STR + start_xy;
    
    for(int c = 0; c < C; ++c){
        for(int r = 0; r < KER; ++r){
        for(int s = 0; s < KER; ++s){
            int h  = hout + r - (KER/2);
            int w  = wout + s - (KER/2);
            if(h < 0 || h >= H || w < 0 || w >= W){
                out[((W - start_xy) / STR) * ((H   -start_xy) / STR) * KER * KER * c +  
                    ((W - start_xy) / STR) * ((H   -start_xy) / STR) * KER * r +   
                    ((W - start_xy) / STR) * ((H   -start_xy) / STR) * s +   
                    ((W - start_xy) / STR) * ((hout-start_xy) / STR) + 
                    ((wout-start_xy) /STR) 
                    ] = 0;
            }else{
                out[((W - start_xy) / STR) * ((H   -start_xy) / STR) * KER * KER * c +  
                    ((W - start_xy) / STR) * ((H   -start_xy) / STR) * KER * r +   
                    ((W - start_xy) / STR) * ((H   -start_xy) / STR) * s +   
                    ((W - start_xy) / STR) * ((hout-start_xy) / STR) + 
                    ((wout-start_xy) /STR) 
                    ] = in[ H * W * c + W * h + w ];
            }
        }
        }
    }
}
__kernel void biasKernel(__global float *IN, 
                         __global  float *bias, 
                         int C,  int HW){
    
    const int c = get_global_id(1);
    const int hw = get_global_id(0);

    IN[c*HW + hw] = IN[c*HW + hw] +  bias[c];
    
}
__kernel void activationRelu(__global float *inout,
                              float alpha){

    int i = get_global_id(0);
    inout[i] = ((inout[i] >  inout[i]*alpha )? inout[i] : inout[i]*alpha);
  
}
__kernel void maxpoolKernel(__global float *in,__global float *out, 
                            int C, int H, int W,
                            int KER, int stride, int PAD){
    int center =  KER/2;
    int offset = -KER/2;
    
    int h = get_global_id(1) * stride + center;
    int w = get_global_id(0) * stride + center;

    //pad 값 있는경우 추가 작성해야함..
    
    //NC HW RS
    for(int c = 0; c < C; c++){
    // for(int h = center; h < H; h=h+stride){
    // for(int w = center; w < W; w=w+stride){
        float max = -FLT_MAX;
        for(int r = offset; r<(KER+offset); ++r){
            for(int s = offset; s<(KER+offset); ++s){
                // max = fmaxf(in[W * H * c + W*(h+r) + (w+s)], max);
                max = (in[W * H * c + W*(h+r) + (w+s)] > max ? 
                      in[W * H * c + W*(h+r) + (w+s)] : max);
            }
        }
        out[ (W/stride) * (H/stride) * c + 
             (W/stride) * ((h-center)/stride) + 
             ((w-center)/stride)] = max;
    // }
    // }
    }
}
__kernel void avgpoolKernel(__global float *in,__global float *out, 
                            int C, int H, int W,
                            int KER, int stride, int PAD){
    int c = get_global_id(0);

    float avgp = 0.0f;
    float mean_n = KER*KER;

    for(int r = 0; r < H; ++r){
        for(int s = 0; s < W; ++s){
            avgp += in[KER*KER* c + KER* r + s];
        }
    }
    out[c] = avgp/mean_n;
}