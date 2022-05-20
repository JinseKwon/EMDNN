#include "emdnn.h"
#include <stdlib.h>

// void fp32_int8(float *A, int8_t *A_q8, 
//                int N,    float range){
    
//     float scale = 127/range;
//     for(int i = 0; i < N*N; ++i){
//         A_q8[i] = (int8_t)(A[i]*scale);
//     }
// }
// void int32_fp32(int *C_q, float *C,
//                int N,    float range){
    
//     float scale = 127/range;
//     scale = scale * scale;
//     for(int i = 0; i < N*N; ++i){
//         C[i] = (float)(C_q[i]/scale);
//     }
// }

void quant_network(LAYER *l, int num){
    //TODO:Configuration
    int file_access = 0;

    //weight to layer set
    for(int i = 0; i<num; ++i){
        switch(l[i].TYPE){
        case INPUT_LAYER :
            break;

        case CONVOLUTIONAL :
            l[i].Q8_BIAS   = (int8_t*)malloc(l[i].N);
            l[i].Q8_WEIGHT = (int8_t*)malloc(l[i].N * l[i].C * l[i].H * l[i].W);
            // l[i].BIAS   = net_weight; 
            //               net_weight += l[i].N;
            // l[i].WEIGHT = net_weight; 
            //               net_weight += l[i].N * l[i].C * l[i].H * l[i].W;
            file_access += l[i].N + l[i].N * l[i].C * l[i].H * l[i].W;
            // printf("CONV making...\n");
            break;

        case CONVOLUTIONAL_DW :
            l[i].Q8_BIAS   = (int8_t*)malloc(l[i].N);
            l[i].Q8_WEIGHT = (int8_t*)malloc(l[i].N * l[i].H * l[i].W);
            // l[i].BIAS   = net_weight; 
            //               net_weight += l[i].N;
            // l[i].WEIGHT = net_weight; 
            //               net_weight += l[i].N * l[i].H * l[i].W;
            file_access += l[i].N * l[i].SCALE + l[i].N * l[i].H * l[i].W;
            break;

        case CONNECTED_T :
        case CONNECTED :
            l[i].Q8_BIAS   = (int8_t*)malloc(l[i].N);
            l[i].Q8_WEIGHT = (int8_t*)malloc(l[i].N * l[i].C);
            // l[i].BIAS   = net_weight; 
            //               net_weight += l[i].N;
            // l[i].WEIGHT = net_weight; 
            //               net_weight += l[i].N * l[i].C;
            file_access += l[i].N * l[i].SCALE + l[i].N * l[i].C;
            break;

        case MAXPOOL:
            break;

        case SOFTMAX :
            break;

        case AVGPOOL :
            break;

        case SHORTCUT :
            break;

        case DETECTION :
            break;
        case CLASSIFICATION :
        default :
            return;
        }

        printf("file access : (int8_t)%d /(fp32) %d \n",file_access, file_access*4);
    }
}
