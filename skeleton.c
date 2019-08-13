#include "emdnn.h"

int main(){
    char f_name[100] = "yolov2-tiny-voc.weights";
    float *net_weight;

    int num_layer=17;   //input layer = 0;
    LAYER l[num_layer];

    int i = 0;
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    layer_update(l, INPUT_LAYER     , LINEAR,i++,   0,   3,416,416, 0  ,  0,   0);
    //CONV
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,  16,   3,  3,  3, 1  ,  1,   4);
    layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,  32,  16,  3,  3, 1  ,  1,   4);
    layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,  64,  32,  3,  3, 1  ,  1,   4);
    layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++, 128,  64,  3,  3, 1  ,  1,   4);
    layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++, 256, 128,  3,  3, 1  ,  1,   4);
    layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++, 512, 256,  3,  3, 1  ,  1,   4);
    layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  1,   0);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,1024, 512,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,1024,1024,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , LINEAR,i++, 125,1024,  1,  1, 0  ,  1,   1);
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    layer_update(l, DETECTION       , LINEAR,i++,  20,   5,  0,  0, 0  ,  0,   0);

    //conv(input,output)
    make_network(l,net_weight,num_layer,f_name);
    print_network(l,num_layer);
    inference(l,num_layer);
    
    // while(1){

    // }
    return 0;
}   