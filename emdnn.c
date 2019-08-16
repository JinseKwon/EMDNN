#include "emdnn.h"
#include <stdlib.h>
#ifdef OPENCL
#include "opencl_init.h"
#endif

// float* file_loader(float* network, int f_size, char *filename){
float* file_loader(float* network, char *filename){
    FILE *fnet = fopen(filename, "rb");
    if (!fnet) {
        printf("Network file does not exist.\n");
        exit(0);
    }
    fseek(fnet, 0, SEEK_END);
    long fsz = ftell(fnet);
    // if (fsz != f_size) {
    //     printf("Network file is corrupted.\n");
    //     exit(0);
    // }
    rewind(fnet);
    network = (float*)malloc(fsz);
    fread(network, 1, fsz, fnet);
    fclose(fnet);

    return network;
}
void make_network(LAYER *l,
            float* net_weight, 
            int num,
            char *filename){

#ifdef OPENCL
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
#endif

    //weight loader
    net_weight = file_loader(net_weight, filename);

    //TODO:Configuration
    net_weight += 4;    
    // int file_access = 4;

    //weight to layer set
    for(int i = 0; i<num; ++i){

        switch(l[i].TYPE){
        case INPUT_LAYER :
            l[i].OUT_C = l[i].C;
            l[i].OUT_H = l[i].H;
            l[i].OUT_W = l[i].W;

            l[i].OUTPUT = (float*)malloc(l[i].OUT_C * 
                                         l[i].OUT_H * l[i].OUT_W * 
                                         l[i].XF);
            // printf("input making...\n");
            break;

        case CONVOLUTIONAL :
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = (l[i].IN_H - l[i].H + 2*l[i].PAD) / l[i].STRIDE + 1;
            l[i].OUT_W = (l[i].IN_W - l[i].W + 2*l[i].PAD) / l[i].STRIDE + 1;

#ifdef OPENCL
            l[i].CL_BIAS = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                                l[i].N * l[i].SCALE * 
                                                sizeof(float), NULL, &err);
            l[i].BIAS = (float*)clEnqueueMapBuffer(queue, l[i].CL_BIAS, CL_TRUE, CL_MAP_WRITE, 0,
                                                l[i].N * l[i].SCALE * 
                                                sizeof(float), 0, NULL, NULL, &err);
            memcpy( l[i].BIAS, net_weight,
                    l[i].N * l[i].SCALE *sizeof(float));
            net_weight += l[i].N * l[i].SCALE;

            l[i].CL_WEIGHT = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                                l[i].N * l[i].C * l[i].H * l[i].W *
                                                sizeof(float), NULL, &err);
            l[i].WEIGHT = (float*)clEnqueueMapBuffer(queue, l[i].CL_WEIGHT, CL_TRUE, CL_MAP_WRITE, 0,
                                                l[i].N * l[i].C * l[i].H * l[i].W *
                                                sizeof(float), 0, NULL, NULL, &err);
            memcpy( l[i].WEIGHT, net_weight,
                    l[i].N * l[i].C * l[i].H * l[i].W *sizeof(float));
            net_weight += l[i].N * l[i].C * l[i].H * l[i].W;
            if(l[i].SCALE != 1){
                batch_normalizaiton(l[i].BIAS, l[i].WEIGHT, 
                l[i].N, l[i].C, l[i].H, l[i].W );
            }

            l[i].IM2COL = 0;
            if(l[i].W != 1){
                l[i].CL_INPUT = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                                l[i].IN_C * 
                                                l[i].OUT_H * l[i].OUT_W * 
                                                l[i].H * l[i].W * 
                                                l[i].XF, NULL, &err);
                l[i].INPUT = (float*)clEnqueueMapBuffer(queue, l[i].CL_INPUT, CL_TRUE, CL_MAP_WRITE, 0 ,
                                                l[i].IN_C * 
                                                l[i].OUT_H * l[i].OUT_W * 
                                                l[i].H * l[i].W * 
                                                l[i].XF, 0, NULL, NULL, &err);
                CHECK_ERROR(err);
                l[i].IM2COL = 1;
            }
            l[i].CL_OUTPUT = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                            l[i].OUT_C * 
                                            l[i].OUT_H * l[i].OUT_W * 
                                            l[i].XF, NULL, &err);
            l[i].OUTPUT = (float*)clEnqueueMapBuffer(queue, l[i].CL_OUTPUT, CL_TRUE, CL_MAP_WRITE, 0 ,
                                            l[i].OUT_C * 
                                            l[i].OUT_H * l[i].OUT_W * 
                                            l[i].XF, 0, NULL, NULL, &err);
#else
            l[i].BIAS   = net_weight; 
                          net_weight += l[i].N * l[i].SCALE;
            l[i].WEIGHT = net_weight; 
                          net_weight += l[i].N * l[i].C * l[i].H * l[i].W;
            if(l[i].SCALE != 1){
                batch_normalizaiton(l[i].BIAS, l[i].WEIGHT, 
                l[i].N, l[i].C, l[i].H, l[i].W );
            }
            //im2col mem space set
            l[i].IM2COL = 0;
            if(l[i].W != 1){
                l[i].INPUT  = (float *)malloc(l[i].IN_C  * 
                                              l[i].OUT_H * l[i].OUT_W * 
                                              l[i].H * l[i].W * 
                                              l[i].XF);
                l[i].IM2COL = 1;
            }

            l[i].OUTPUT = (float *)malloc(l[i].OUT_C * 
                                          l[i].OUT_H * l[i].OUT_W * 
                                          l[i].XF);
#endif
            // file_access += l[i].N * l[i].SCALE + l[i].N * l[i].C * l[i].H * l[i].W;
            // printf("CONV making...\n");
            break;

        case CONVOLUTIONAL_DW :
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = (l[i].IN_H - l[i].H + 2*l[i].PAD) / l[i].STRIDE + 1;
            l[i].OUT_W = (l[i].IN_W - l[i].W + 2*l[i].PAD) / l[i].STRIDE + 1;

            l[i].BIAS   = net_weight; 
                          net_weight += l[i].N * l[i].SCALE;
            l[i].WEIGHT = net_weight; 
                          net_weight += l[i].N * l[i].H * l[i].W;

            l[i].IM2COL = 0;
            if(l[i].W != 1){
                l[i].INPUT  = (float *)malloc(l[i].IN_C * 
                                              l[i].OUT_H * l[i].OUT_W * 
                                              l[i].H * l[i].W * 
                                              l[i].XF);
                l[i].IM2COL = 1;
            }
            l[i].OUTPUT = (float*)malloc(l[i].OUT_C * 
                                         l[i].OUT_H * l[i].OUT_W * 
                                         l[i].XF);

            // file_access += l[i].N * l[i].SCALE + l[i].N * l[i].H * l[i].W;
            break;

        case CONNECTED :
            l[i].IN_C = l[i-1].OUT_C * l[i-1].OUT_H * l[i-1].OUT_W;
            l[i].IN_H = 1;
            l[i].IN_W = 1;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = 1;
            l[i].OUT_W = 1;
            
            l[i].BIAS   = net_weight; 
                          net_weight += l[i].N * l[i].SCALE;
            //TODO : BIAS term processing...

            l[i].WEIGHT = net_weight; 
                          net_weight += l[i].N * l[i].C;
            l[i].OUTPUT = (float*)malloc(l[i].N * l[i].XF);

            // file_access += l[i].N * l[i].SCALE + l[i].N * l[i].C;
            break;

        case MAXPOOL:
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i].IN_C;
            l[i].OUT_H = l[i].IN_H / l[i].STRIDE;
            l[i].OUT_W = l[i].IN_W / l[i].STRIDE;
            l[i].OUTPUT = (float*)malloc(l[i].OUT_C * 
                                         l[i].OUT_H * l[i].OUT_W * 
                                         l[i].XF);
            break;

        case SOFTMAX :
            l[i].IN_C = l[i-1].OUT_C * l[i-1].OUT_H * l[i-1].OUT_W;
            l[i].IN_H = 1;
            l[i].IN_W = 1;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = 1;
            l[i].OUT_W = 1;
            l[i].OUTPUT = (float*)malloc(l[i].OUT_C * l[i].XF);
            break;

        case AVGPOOL :
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i].IN_C;
            l[i].OUT_H = l[i].IN_H / l[i].H;
            l[i].OUT_W = l[i].IN_W / l[i].W;
            l[i].OUTPUT = (float*)malloc(l[i].OUT_C * 
                                         l[i].OUT_H * l[i].OUT_W * 
                                         l[i].XF);
            break;

        case SHORTCUT :
            break;

        case DETECTION :
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i-1].C;
            l[i].OUT_H = l[i-1].OUT_H;
            l[i].OUT_W = l[i-1].OUT_W;

            l[i].INPUT = (float*)malloc(l[i].IN_C * 
                                         l[i].IN_H * l[i].IN_W * 
                                         l[i].XF);
            l[i].OUTPUT = (float*)malloc((l[i].N * l[i].C *
                                          l[i].OUT_H * l[i].OUT_W * 6 + 1) *
                                         sizeof(float));
            break;
        case CLASSIFICATION :
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = l[i].IN_H;
            l[i].OUT_W = l[i].IN_W;
        default :
            return;
        }
        // printf("file access : %d / %d \n", file_access*4, 17015472);
    }
}

LAYER* layer_update(
// void layer_update(
            LAYER *l,
            LAYER_TYPE type, 
            ACTIVATION_TYPE act,
            int num, 
            int n, int c, int h, int w,
            int pad, int str, int scale){
    if(num > 0){
        l = (LAYER *)realloc(l, (num+1)*sizeof(LAYER));
    }
    l[num].TYPE        = type;
    l[num].ACTIVATION  = act;
    l[num].NUM         = num;
    l[num].N           = n;
    l[num].C           = c;
    l[num].H           = h;
    l[num].W           = w;
    l[num].PAD         = pad;
    l[num].STRIDE      = str;
    l[num].SCALE       = scale;
    
    //TODO : precision hard coding
    l[num].XF          = SINGLE; 
    
    return l;   
}
void print_network(LAYER *l, 
            int num){
    char lay_type[10][20] = { 
        "input    ",    "conv     ",    "conv_dw  ",    "connect  ",
        "maxpool  ",    "softmax  ",    "avgpool  ",    "shortcut ",
        "detection",    "classify "
    };
    printf("layer :      type :     C * (      H *      W) \n");
    printf("============================================= \n");
    for(int j = 0; j<num; ++j){
        printf("%5d : %s : %5d * (  %5d *  %5d) \n", 
                j, lay_type[l[j].TYPE], l[j].OUT_C, l[j].OUT_H, l[j].OUT_W);
    }
}
void inference(LAYER *l, 
            int num){
    // for(int i = 0; i<3; ++i){            
    for(int i = 0; i<num; ++i){    
        printf("now layer >> %d ", i);
        switch(l[i].TYPE){
        case INPUT_LAYER :
            // l[i].OUTPUT = input_bin_img(l[i].OUTPUT, 
            //                         l[i].OUT_C,l[i].OUT_H,l[i].OUT_W);
// //debuging code
// for(int rr = 0; rr<3; ++rr){
//     printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 0]);
// }
// for(int rr = 0; rr<100; ++rr){
//     printf("%f ",l[i].OUTPUT[rr]);
// }
            break;

        case CONVOLUTIONAL :
            if(l[i].IM2COL){
                im2col( l[i-1].OUTPUT, l[i].INPUT,
                        l[i].IN_C, l[i].IN_H, l[i].IN_W,
                        l[i].W,
                        l[i].STRIDE, l[i].PAD);
            }else{
                l[i].INPUT = l[i-1].OUTPUT;
            }
//debuging code
// for(int rr = 0; rr<3*3*3; ++rr){
//     printf("%f ",l[i].INPUT[rr * l[i].IN_H * l[i].IN_W + 416*415]);
// }
// if(i == 3){
//     // for(int rr = 5*5*0; rr<5*5*2; ++rr){
//     for(int rr = 5*5*90; rr<5*5*96; ++rr){
//         printf("%f ",l[i].INPUT[rr * l[i].IN_H * l[i].IN_W + 0]);
//     }
// }
            cpu_gemm(l[i].WEIGHT, l[i].INPUT, l[i].OUTPUT,
                     l[i].N,
                     l[i].OUT_H*l[i].OUT_W,
                     l[i].C * l[i].H * l[i].W);
//debuging code
// for(int rr = 0; rr<16; ++rr){
//     printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 0]);
//     // printf("%f ",l[i].WEIGHT[rr ]);
// }
            bias_add(l[i].OUTPUT,l[i].BIAS,
                     l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);

            activate_function(l[i].OUTPUT, l[i].ACTIVATION,
                              l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);
//debuging code
// for(int rr = 0; rr<96; ++rr){
//     printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W +55]);
// }
// if(i == 3){
//     for(int rr = 0; rr<32; ++rr){
//         printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 0]);
//     }
// }
            break;

        case CONVOLUTIONAL_DW :
            if(l[i].IM2COL){
                im2col( l[i-1].OUTPUT, l[i].INPUT,
                        l[i].IN_C, l[i].IN_H, l[i].IN_W, 
                        l[i].W,
                        l[i].STRIDE, l[i].PAD);
            }else{
                l[i].INPUT = l[i-1].OUTPUT;
                // printf("memory leak? ");
            }
            cpu_stride_b_gemm(l[i].WEIGHT, l[i].INPUT, l[i].OUTPUT,
                l[i].OUT_H * l[i].OUT_W,
                l[i].OUT_C,
                l[i].N * l[i].H * l[i].W,
                l[i].H * l[i].W, 
                l[i].N);
            break;

        case CONNECTED :
            //$$merged donghee's code
            //M,K,N의 행렬 곱셈에서 N이 1인 GEMV
            cpu_gemv(l[i].WEIGHT, l[i-1].OUTPUT, l[i].OUTPUT,
                     l[i].N,
                     l[i].OUT_H*l[i].OUT_W, //N=1
                     l[i].C * l[i].H * l[i].W);
            bias_add(l[i].OUTPUT,l[i].BIAS,
                     l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);
            activate_function(l[i].OUTPUT, l[i].ACTIVATION,
                              l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);
//debuging code
// if(i == 11){
//     for(int rr = 0; rr<256; ++rr){
//         printf("%f ",l[i].OUTPUT[rr]);
//     }
// }
            break;

        case MAXPOOL:
            maxpool(l[i-1].OUTPUT, l[i].OUTPUT, 
                    l[i].IN_C, l[i].IN_H, l[i].IN_W,
                    l[i].W, l[i].STRIDE, l[i].PAD);
//debuging code
// for(int rr = 0; rr<96; ++rr){
//     printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 0]);
// }
// if(i == 8){
//     for(int rr = 0; rr<256; ++rr){
//         printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 0]);
//     }
// }
            break;

        case SOFTMAX :
            softmax(l[i-1].OUTPUT, l[i].OUTPUT, 
                    l[i].N);
// //debuging code
// if(i == 12){
//     for(int rr = 0; rr<256; ++rr){
//         printf("%f ",l[i].OUTPUT[rr]);
//     }
// }
            break;

        case AVGPOOL :
            avgpool(l[i-1].OUTPUT, l[i].OUTPUT,
                    l[i].IN_H, l[i].IN_W, l[i].IN_C, 
                    l[i].W, l[i].STRIDE, l[i].PAD);
            break;

        case SHORTCUT :
            break;

        case DETECTION :
            transpose(l[i-1].OUTPUT,l[i].INPUT,
                      l[i-1].OUT_H*l[i-1].OUT_W,
                      l[i-1].OUT_C);
            detection(l[i].INPUT,l[i].OUTPUT,
                      l[0].W,   //input w : 416
                      l[i].OUT_H, 
                      l[i].OUT_W,
                      l[i].C,
                      l[i].N,
                      0.3, 0.3);
                      //TODO : modify the hard code block...
            break;
        case CLASSIFICATION :
            //$$merged donghee's code
            classification(l[i-1].OUTPUT,1000);
            break;
        default :
            return;
        }
// debuging code
// if(i+1 == num){
// // for(int rr = 0; rr<13*13*5*20; ++rr){
//     // printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 13]);
//     char yolo_class[20][100] = {
//               "aeroplane", "bicycle",     "bird",      "boat",     "bottle",
//                     "bus",     "car",      "cat",     "chair",        "cow",
//             "diningtable",     "dog",    "horse", "motorbike",     "person",
//             "pottedplant",   "sheep",     "sofa",     "train",  "tvmonitor" };
//     printf("\n");
//     for(int b = 0; b < (int)l[i].OUTPUT[13*13*5*20*6]; ++b){
//         printf("%2d box : %14s class, %.3f\n",
//                     b+1,    yolo_class[(int)l[i].OUTPUT[b*6 + 1]], l[i].OUTPUT[b*6]);
//     }
//     // printf("\n detection boxes : %d ",(int)l[i].OUTPUT[13*13*5*20*6]);
// // }
// }
        printf("\n");
        // printf(" | complete \n");
        // printf("file access : %d / %d \n", file_access*4, 17015472);
    }
}
