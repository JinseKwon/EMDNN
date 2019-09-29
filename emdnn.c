#include "emdnn.h"
#include <stdlib.h>

//vscode 화면 뷰로 인함. 임시작성 배포시 삭제.
// #define OPENCL

#ifdef OPENCL
#include "opencl_init.h"

#include <clblast_c.h>
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

    kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, 
                                        (const char**)&kernel_source,
                                        &kernel_source_size, &err);

    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                    0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                    log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
    }
    CHECK_ERROR(err);
    
    l[0].QUE         = &queue;
    l[0].EVT         = &event;
    
    kernel1 = clCreateKernel(program, "im2colKernel",NULL);
    kernel2 = clCreateKernel(program, "maxpoolKernel",NULL);
    kernel3 = clCreateKernel(program, "biasKernel",NULL);
    kernel4 = clCreateKernel(program, "activationRelu",NULL);
    kernel5 = clCreateKernel(program, "avgpoolKernel",NULL);
    
    l[0].KER_IM2COL  = &kernel1;
    l[0].KER_MAXPOOL = &kernel2;
    l[0].KER_BIAS    = &kernel3;
    l[0].KER_RELU    = &kernel4;
    l[0].KER_AVGPOOL = &kernel5;

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
#ifdef OPENCL

            l[i].CL_OUTPUT =
            cl_obj_create(l[i].CL_OUTPUT,
                          l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * l[i].XF);
            cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                          l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * l[i].XF);
#else
            l[i].OUTPUT = (float*)malloc(l[i].OUT_C * 
                                         l[i].OUT_H * l[i].OUT_W * 
                                         l[i].XF);
#endif
            break;

        case CONVOLUTIONAL :
            l[i].IN_C = l[i-1].OUT_C;
            l[i].IN_H = l[i-1].OUT_H;
            l[i].IN_W = l[i-1].OUT_W;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = (l[i].IN_H - l[i].H + 2*l[i].PAD) / l[i].STRIDE + 1;
            l[i].OUT_W = (l[i].IN_W - l[i].W + 2*l[i].PAD) / l[i].STRIDE + 1;

#ifdef OPENCL
            //host memory mapping to CL OBJECT function
            //cl_obj2mem( CL_OBJ, mem_arry, size)
            l[i].CL_BIAS = 
            cl_obj_create(l[i].CL_BIAS,
                          l[i].N * l[i].SCALE * l[i].XF);
            cl_obj2mem(   l[i].CL_BIAS, &l[i].BIAS, CL_MAP_WRITE,
                          l[i].N * l[i].SCALE * l[i].XF);

            memcpy( l[i].BIAS, net_weight,
                    l[i].N * l[i].SCALE * l[i].XF);
            net_weight += l[i].N * l[i].SCALE;

            l[i].CL_WEIGHT = 
            cl_obj_create(l[i].CL_WEIGHT, 
                          l[i].N * l[i].C * l[i].H * l[i].W * l[i].XF);
            cl_obj2mem(   l[i].CL_WEIGHT, &l[i].WEIGHT, CL_MAP_WRITE,
                          l[i].N * l[i].C * l[i].H * l[i].W * l[i].XF);

            memcpy( l[i].WEIGHT, net_weight,
                    l[i].N * l[i].C * l[i].H * l[i].W * l[i].XF);
            net_weight += l[i].N * l[i].C * l[i].H * l[i].W;
            if(l[i].SCALE != 1){
                batch_normalizaiton(l[i].BIAS, l[i].WEIGHT, 
                l[i].N, l[i].C, l[i].H, l[i].W );
            }

            l[i].IM2COL = 0;
            if(l[i].W != 1){
                l[i].CL_INPUT =
                cl_obj_create(l[i].CL_INPUT,
                              l[i].IN_C * l[i].OUT_H * l[i].OUT_W * 
                              l[i].H * l[i].W * l[i].XF);
                cl_obj2mem(   l[i].CL_INPUT, &l[i].INPUT, CL_MAP_WRITE,
                              l[i].IN_C * l[i].OUT_H * l[i].OUT_W * 
                              l[i].H * l[i].W * l[i].XF);
                l[i].IM2COL = 1;
            }else{
                l[i].CL_INPUT = 
                cl_obj_create(l[i].CL_INPUT,
                              l[i].IN_C * l[i].IN_H * l[i].IN_W * l[i].XF);
                cl_obj2mem(   l[i].CL_INPUT, &l[i].INPUT, CL_MAP_WRITE,
                              l[i].IN_C * l[i].IN_H * l[i].IN_W * l[i].XF);
            }
            l[i].CL_OUTPUT = 
            cl_obj_create(l[i].CL_OUTPUT,
                          l[i].OUT_C *l[i].OUT_H * l[i].OUT_W * l[i].XF);
            cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                          l[i].OUT_C *l[i].OUT_H * l[i].OUT_W * l[i].XF);
            
            if(l[i].DEVICE == GPU){
                mem2cl_obj(l[i].BIAS, l[i].CL_BIAS);
                mem2cl_obj(l[i].WEIGHT, l[i].CL_WEIGHT);
                if(l[i].IM2COL){
                    mem2cl_obj(l[i].INPUT, l[i].CL_INPUT);
                }
                mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
            }else if(l[i].DEVICE == CPU){
                break;
            }else if(l[i].DEVICE == NPU){
                break;
            }
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

        case CONNECTED_T :
        case CONNECTED :
            l[i].IN_C = l[i-1].OUT_C * l[i-1].OUT_H * l[i-1].OUT_W;
            l[i].IN_H = 1;
            l[i].IN_W = 1;

            l[i].OUT_C = l[i].N;
            l[i].OUT_H = 1;
            l[i].OUT_W = 1;

#ifdef OPENCL
            l[i].CL_BIAS = 
            cl_obj_create(l[i].CL_BIAS,  
                          l[i].N * l[i].SCALE * l[i].XF);
            cl_obj2mem(   l[i].CL_BIAS, &l[i].BIAS, CL_MAP_WRITE,
                          l[i].N * l[i].SCALE * l[i].XF);
            memcpy( l[i].BIAS, net_weight,
                    l[i].N * l[i].SCALE * l[i].XF);
            net_weight += l[i].N * l[i].SCALE;

            l[i].CL_WEIGHT = 
            cl_obj_create(l[i].CL_WEIGHT, 
                          l[i].N * l[i].C * l[i].XF);
            cl_obj2mem(   l[i].CL_WEIGHT, &l[i].WEIGHT, CL_MAP_WRITE,
                          l[i].N * l[i].C * l[i].XF);
            memcpy( l[i].WEIGHT, net_weight,
                    l[i].N * l[i].C * l[i].XF);
            net_weight += l[i].N * l[i].C;

            if(l[i].SCALE != 1){
                batch_normalizaiton(l[i].BIAS, l[i].WEIGHT, 
                l[i].N, l[i].C, 1, 1 );
            }
            if(l[i].TYPE == CONNECTED_T){
                transpose_fc(l[i].WEIGHT, l[i].N, l[i].C);
            }
            l[i].CL_OUTPUT = 
            cl_obj_create(l[i].CL_OUTPUT,
                          l[i].N * l[i].XF);
            cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                          l[i].N * l[i].XF);
            
            if(l[i].DEVICE == GPU){
                mem2cl_obj(l[i].BIAS, l[i].CL_BIAS);
                mem2cl_obj(l[i].WEIGHT, l[i].CL_WEIGHT);
                mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
            }else if(l[i].DEVICE == CPU){
                break;
            }else if(l[i].DEVICE == NPU){
                break;
            }
#else
            l[i].BIAS   = net_weight; 
                          net_weight += l[i].N * l[i].SCALE;
            //TODO : BIAS term processing...

            l[i].WEIGHT = net_weight; 
                          net_weight += l[i].N * l[i].C;

            if(l[i].SCALE != 1){
                batch_normalizaiton(l[i].BIAS, l[i].WEIGHT, 
                l[i].N, l[i].C, 1, 1 );
            }
            if(l[i].TYPE == CONNECTED_T){
                transpose_fc(l[i].WEIGHT, l[i].N, l[i].C);
            }
            l[i].OUTPUT = (float*)malloc(l[i].N * l[i].XF);
#endif
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
#ifdef OPENCL
            l[i].CL_OUTPUT = 
            cl_obj_create(l[i].CL_OUTPUT,
                          l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * 
                          l[i].XF);
            cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                          l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * 
                          l[i].XF);
            if(l[i].DEVICE == GPU){
                mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
                break;
            }else if(l[i].DEVICE == CPU){
                break;
            }else if(l[i].DEVICE == NPU){
                break;
            }
#endif
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
#ifdef OPENCL
            l[i].CL_OUTPUT = 
            cl_obj_create(l[i].CL_OUTPUT,
                          l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * 
                          l[i].XF);
            cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                          l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * 
                          l[i].XF);
            if(l[i].DEVICE == GPU){
                mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
                break;
            }else if(l[i].DEVICE == CPU){
                break;
            }else if(l[i].DEVICE == NPU){
                break;
            }
#endif
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

void tune_parameter(LAYER *l,
                    int num){
    //weight to layer set
    for(int i = 0; i<num; ++i){
        switch(l[i].TYPE){
        case INPUT_LAYER :
            break;

        case CONVOLUTIONAL :
            if(l[i].CUR_DEVICE == CPU){
#ifdef OPENCL
                if(      l[i].DEVICE == GPU){
                    mem2cl_obj(l[i].BIAS, l[i].CL_BIAS);
                    mem2cl_obj(l[i].WEIGHT, l[i].CL_WEIGHT);
                    if(l[i].IM2COL){
                        mem2cl_obj(l[i].INPUT, l[i].CL_INPUT);
                    }
                    mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
                }else 
#endif
                if(l[i].DEVICE == NPU){

                }
            }
#ifdef OPENCL
            else if(l[i].CUR_DEVICE == GPU){
                if(      l[i].DEVICE == CPU){
                    cl_obj2mem(   l[i].CL_BIAS, &l[i].BIAS, CL_MAP_WRITE,
                          l[i].N * l[i].SCALE * l[i].XF);
                    cl_obj2mem(   l[i].CL_WEIGHT, &l[i].WEIGHT, CL_MAP_WRITE,
                                l[i].N * l[i].C * l[i].H * l[i].W * l[i].XF);
                    if(l[i].IM2COL == 1){
                        cl_obj2mem(   l[i].CL_INPUT, &l[i].INPUT, CL_MAP_WRITE,
                                    l[i].IN_C * l[i].OUT_H * l[i].OUT_W * 
                                    l[i].H * l[i].W * l[i].XF);
                    }
                    cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                                l[i].OUT_C *l[i].OUT_H * l[i].OUT_W * l[i].XF);
                }else if(l[i].DEVICE == NPU){

                }
            }
#endif
            else if(l[i].CUR_DEVICE == NPU){
                if(      l[i].DEVICE == CPU){
                
                }else if(l[i].DEVICE == GPU){

                }
            }
        case CONVOLUTIONAL_DW :
            break;

        case CONNECTED_T :
        case CONNECTED :
            if(l[i].CUR_DEVICE == CPU){
#ifdef OPENCL
                if(      l[i].DEVICE == GPU){
                    mem2cl_obj(l[i].BIAS, l[i].CL_BIAS);
                    mem2cl_obj(l[i].WEIGHT, l[i].CL_WEIGHT);
                    mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
                }else 
#endif
                if(l[i].DEVICE == NPU){

                }
            }
#ifdef OPENCL
            else if(l[i].CUR_DEVICE == GPU){
                if(      l[i].DEVICE == CPU){
                    cl_obj2mem(   l[i].CL_BIAS, &l[i].BIAS, CL_MAP_WRITE,
                          l[i].N * l[i].SCALE * l[i].XF);
                    cl_obj2mem(   l[i].CL_WEIGHT, &l[i].WEIGHT, CL_MAP_WRITE,
                                l[i].N * l[i].C * l[i].XF);
                    cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                                l[i].N * l[i].XF);
                }else if(l[i].DEVICE == NPU){

                }
            }
#endif
            else if(l[i].CUR_DEVICE == NPU){
                if(      l[i].DEVICE == CPU){
                
                }else if(l[i].DEVICE == GPU){

                }
            }

            break;

        case MAXPOOL:
            if(l[i].CUR_DEVICE == CPU){
#ifdef OPENCL
                if(      l[i].DEVICE == GPU){
                    mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
                }else 
#endif
                if(l[i].DEVICE == NPU){

                }
            }
#ifdef OPENCL
            else if(l[i].CUR_DEVICE == GPU){
                if(      l[i].DEVICE == CPU){
                    cl_obj2mem(   l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                                l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * 
                                l[i].XF);
                }else if(l[i].DEVICE == NPU){

                }
            }
#endif
            else if(l[i].CUR_DEVICE == NPU){
                if(      l[i].DEVICE == CPU){
                
                }else if(l[i].DEVICE == GPU){

                }
            }
            break;
        case SOFTMAX :
            break;
        case AVGPOOL :
            if(l[i].CUR_DEVICE == CPU){
#ifdef OPENCL
                if(l[i].DEVICE == GPU){
                    mem2cl_obj(l[i].OUTPUT, l[i].CL_OUTPUT);
                }else 
#endif
                if(l[i].DEVICE == NPU){

                }
            }
#ifdef OPENCL
            else if(l[i].CUR_DEVICE == GPU){
                if(      l[i].DEVICE == CPU){
                    cl_obj2mem( l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
                                l[i].OUT_C * l[i].OUT_H * l[i].OUT_W * 
                                l[i].XF);
                }else if(l[i].DEVICE == NPU){

                }
            }
#endif
            else if(l[i].CUR_DEVICE == NPU){
                if(      l[i].DEVICE == CPU){
                
                }else if(l[i].DEVICE == GPU){

                }
            }
            break;
        case SHORTCUT :
            break;
        case DETECTION :
            break;
        case CLASSIFICATION :
            break;
        default :
            return;
        }
        l[i].CUR_DEVICE = l[i].DEVICE;
    }
}
LAYER* layer_update(
            LAYER *l,
            LAYER_TYPE type, 
            ACTIVATION_TYPE act,
            DEVICE_TYPE device,
            int num, 
            int n, int c, int h, int w,
            int pad, int str, int scale){
    if(num > 0){
        l = (LAYER *)realloc(l, (num+1)*sizeof(LAYER));
    }
    l[num].TYPE        = type;
    l[num].ACTIVATION  = act;
    l[num].DEVICE      = device;
    l[num].CUR_DEVICE  = device;
    l[num].NUM         = num;
    l[num].N           = n;
    l[num].C           = c;
    l[num].H           = h;
    l[num].W           = w;
    l[num].PAD         = pad;
    l[num].STRIDE      = str;
    l[num].SCALE       = scale;
    l[num].TUNE        = 0;
    //TODO : precision hard coding
    l[num].XF          = SINGLE; 

    return l;   
}

void tune_network(LAYER *l, 
            int num){
    //TODO:coding the tuning stage !! gogo

    char lay_type[11][20] = { 
        "input    ",    "conv     ",    "conv_dw  ",    "fc       ",
        "connect_t",
        "maxpool  ",    "softmax  ",    "avgpool  ",    "shortcut ",
        "detection",    "classify "
    };
    char dev_type[3][20] = { 
        " CPU "," GPU ", " NPU "
    };
    
    double tic;
    double minimum = 1000.0;
    int    min_para = -1;

    int iter = 2049;
    int start_layer = 0;
    int end_layer = 0;
    for(int iii = 0; iii<l[0].C * l[0].H * l[0].W; ++iii){
        l[0].OUTPUT[iii] = 0.1;
    }
    // int search_space[iter][num];
    printf("  ");
    for(int i=0; i<num; ++i){
        l[i].TUNE = 1;
        l[i].DEVICE = CPU;
        printf("%c%c ",lay_type[l[i].TYPE][0],
                       lay_type[l[i].TYPE][1]);
        if(l[i].TYPE == INPUT_LAYER){
            start_layer = i+1;
        }
        if( l[i].TYPE == SOFTMAX || l[i].TYPE == CLASSIFICATION || 
            l[i].TYPE == DETECTION){
            if(end_layer == 0){
                end_layer = i-1;
            }
        }
    }
    printf(" latency || ");
    printf("[%d ~ %d(%d)]\n",start_layer,end_layer,num);
    printf("\n======================================================== \n");
    
    //::::::::: TODO list :::::::::
    //test image가 없음...오오...
    
    inference(l,num);
    printf("\n");
    //tuning iteration stage

    for(int k=0; k < iter+1; ++k){
        
        int bit_vec = k;
        //tuning result setting
        if(k == iter){
            bit_vec = min_para;
        }
        //input_layer
        printf("%d %c", k,dev_type[l[0].DEVICE][1]);
        
        //device tuning
        for(int rnd = start_layer-1; rnd <end_layer; ++rnd){
            //전반부
            if(k<1024){
                int bitmap = (bit_vec) << (start_layer-1);
                l[rnd+1].DEVICE = ((bitmap >> rnd) & 1) ? CPU : GPU;
            }

            //후반부
            if(k>=1024){
                int bitmap = (bit_vec - 1024 ) << (num-end_layer-1);
                l[rnd+1].DEVICE = ((bitmap >> end_layer-rnd+1) & 1) ? CPU : GPU;
                // l[rnd+1].DEVICE = ((bitmap >> end_layer-rnd+1) & 1) ?GPU : CPU;
            }
            // if(l[rnd].TYPE == SOFTMAX) l[rnd].DEVICE = CPU;
            // l[rnd].DEVICE = GPU;
            printf("%c",dev_type[l[rnd+1].DEVICE][1]);
        }
        for(int rnd = end_layer+1; rnd<num; rnd++){
            printf("%c", dev_type[l[num-1].DEVICE][1]);
        }
        tune_parameter(l,num);
        tic = get_time();

        inference(l,num);
        
        tic = get_time()-tic;
        printf(" %.6f\n", tic);
        if(minimum > tic){ 
            minimum = tic;
            min_para = k;
        }
    }
    

    printf("\n======================================================== \n");
    //tuning option off..
    for(int i =0; i<num; ++i){
        l[i].TUNE = 0;
    }

}
void print_network(LAYER *l, 
            int num){
    char lay_type[11][20] = { 
        "input    ",    "conv     ",    "conv_dw  ",    "connect  ",
        "connect_t",
        "maxpool  ",    "softmax  ",    "avgpool  ",    "shortcut ",
        "detection",    "classify "
    };
    char dev_type[3][20] = { 
        " CPU "," GPU ", " NPU "
    };
    printf("layer :      type :     C * (      H *      W) || DEVICE  \n");
    printf("======================================================== \n");
    for(int j = 0; j<num; ++j){
        printf("%5d : %s : %5d * (  %5d *  %5d)  || %s\n", 
                j, lay_type[l[j].TYPE], l[j].OUT_C, l[j].OUT_H, l[j].OUT_W,
                dev_type[l[j].DEVICE]);
    }
}
void inference(LAYER *l, 
            int num){
    // for(int i = 0; i<3; ++i){            
    double tic;
    for(int i = 0; i<num; ++i){    
        if(!l[i].TUNE)   printf("now layer >> %d ", i);
        
        tic = get_time();

        if(l[i-1].DEVICE == CPU){
#ifdef OPENCL
            if(l[i].DEVICE == GPU){
                mem2cl_obj(l[i-1].OUTPUT, l[i-1].CL_OUTPUT);
            }else
#endif
            if(l[i].DEVICE == NPU){
                
            }
        }
#ifdef OPENCL        
        else if(l[i-1].DEVICE == GPU){
            if(l[i].DEVICE == CPU){
                cl_obj2mem(l[i-1].CL_OUTPUT, &l[i-1].OUTPUT, CL_MAP_WRITE,
                            l[i-1].OUT_C *l[i-1].OUT_H * l[i-1].OUT_W * l[i-1].XF);
            }else if(l[i].DEVICE == NPU){

            }
        }
#endif
        else if(l[i-1].DEVICE == NPU){
            if(l[i].DEVICE == CPU){
                
            }else if(l[i].DEVICE == GPU){

            }
        }
        if(l[i].TUNE) printf("#%.6f",get_time()-tic);
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
            tic = get_time();
            if(l[i].IM2COL){
                im2col(l, i,
                       l[i-1].OUTPUT, l[i].INPUT,
                       l[i].IN_C, l[i].IN_H, l[i].IN_W,
                       l[i].W,
                       l[i].STRIDE, l[i].PAD);
                gemm(l, i,
                     l[i].WEIGHT,    l[i].INPUT,    l[i].OUTPUT,
                     0, 0,
                     l[i].N,
                     l[i].OUT_H * l[i].OUT_W,
                     l[i].C * l[i].H * l[i].W,
                     1.0f, 0.0f);
            }else{
                gemm(l, i,
                     l[i].WEIGHT,    l[i-1].OUTPUT,    l[i].OUTPUT,
                     0, 0,
                     l[i].N,
                     l[i].OUT_H * l[i].OUT_W,
                     l[i].C * l[i].H * l[i].W,
                     1.0f, 0.0f);
            }
            bias_add(l, i,
                     l[i].OUTPUT,l[i].BIAS,
                     l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);

            activate_function(l, i,
                              l[i].OUTPUT, l[i].ACTIVATION,
                              l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);
            if(l[i].TUNE) printf(" %.3f",get_time()-tic);

//debuging code
// for(int rr = 0; rr<96; ++rr){
//     printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W +55]);
// }
// cl_obj2mem(l[i].CL_OUTPUT, &l[i].OUTPUT, CL_MAP_WRITE,
//                             l[i].OUT_C *l[i].OUT_H * l[i].OUT_W * l[i].XF);
// if(i == 1){
//     for(int rr = 0; rr<32; ++rr){
//         printf("%f ",l[i].OUTPUT[rr * l[i].OUT_H * l[i].OUT_W + 0]);
//     }
// }
            break;

        case CONVOLUTIONAL_DW :
            if(l[i].IM2COL){
                im2col( l, i,
                        l[i-1].OUTPUT, l[i].INPUT,
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

        case CONNECTED_T :
        case CONNECTED   :
            tic = get_time();
            //$$merged donghee's code
            //M,K,N의 행렬 곱셈에서 N이 1인 GEMV
            gemv(l, i,
                 l[i].WEIGHT, l[i-1].OUTPUT, l[i].OUTPUT,
                 0,
                 l[i].N,
                 l[i].OUT_H*l[i].OUT_W, //N=1
                 l[i].C * l[i].H * l[i].W,
                 1.0f, 0.0f);
            bias_add(l, i,
                     l[i].OUTPUT,l[i].BIAS,
                     l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);
            activate_function(l, i,
                              l[i].OUTPUT, l[i].ACTIVATION,
                              l[i].OUT_C, l[i].OUT_H,l[i].OUT_W);
            if(l[i].TUNE) printf(" %.3f",get_time()-tic);
//debuging code
// if(i == 11){
//     for(int rr = 0; rr<256; ++rr){
//         printf("%f ",l[i].OUTPUT[rr]);
//     }
// }
            break;

        case MAXPOOL:
            tic = get_time();
            maxpool(l, i,
                    l[i-1].OUTPUT, l[i].OUTPUT, 
                    l[i].IN_C, l[i].IN_H, l[i].IN_W,
                    l[i].W, l[i].STRIDE, l[i].PAD);
            if(l[i].TUNE) printf(" %.3f",get_time()-tic);
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
            tic = get_time();
            softmax(l[i-1].OUTPUT, l[i].OUTPUT, 
                    l[i].N);
            if(l[i].TUNE) printf(" %.3f",get_time()-tic);
// //debuging code
// if(i == 12){
//     for(int rr = 0; rr<256; ++rr){
//         printf("%f ",l[i].OUTPUT[rr]);
//     }
// }
            break;

        case AVGPOOL :
            tic = get_time();
            avgpool(l, i,
                    l[i-1].OUTPUT, l[i].OUTPUT,
                    l[i].IN_C, l[i].IN_H, l[i].IN_W,
                    l[i].W, l[i].STRIDE, l[i].PAD);
            if(l[i].TUNE) printf(" %.3f",get_time()-tic);
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
                      0.3, 0.3,
                      l[i].TUNE);
                      //TODO : modify the hard code block...
            break;
        case CLASSIFICATION :
            //$$merged donghee's code
            classification(l[i-1].OUTPUT,1000,l[i].TUNE);
            break;
        default :
            return;
        }

        tic = get_time();
        if(l[i-1].DEVICE == CPU){
#ifdef OPENCL
            if(l[i].DEVICE == GPU){
                cl_obj2mem(l[i-1].CL_OUTPUT, &l[i-1].OUTPUT, CL_MAP_WRITE,
                            l[i-1].OUT_C *l[i-1].OUT_H * l[i-1].OUT_W * l[i-1].XF);
                
            }else 
#endif
            if(l[i].DEVICE == NPU){
                
            }
        }
#ifdef OPENCL
        else if(l[i-1].DEVICE == GPU){
            if(l[i].DEVICE == CPU){
                mem2cl_obj(l[i-1].OUTPUT, l[i-1].CL_OUTPUT);
            }else if(l[i].DEVICE == NPU){

            }
        }
#endif
        else if(l[i-1].DEVICE == NPU){
            if(l[i].DEVICE == CPU){
                
            }else if(l[i].DEVICE == GPU){

            }
        }
        if(l[i].TUNE) printf(" %.6f",get_time()-tic);
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
        if(!l[i].TUNE)   printf("\n");
        // printf(" | complete \n");
        // printf("file access : %d / %d \n", file_access*4, 17015472);
    }
}
