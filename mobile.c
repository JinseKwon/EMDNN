#include "emdnn.h"

int main(){
    char f_name[100] = "mobilenet_v1.weights";
    float *net_weight;

    int num_layer=31;   //input layer = 0;
    LAYER l[num_layer];

    int i = 0;
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,   N,   C,  H,  W, PAD,STR,SCALE
    //input
    layer_update(l, INPUT_LAYER     , LINEAR,i++,   0,   3,224,224, 0  ,  0,   4);
    //CONV1
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++,  32,   3,  3,  3, 1  ,  2,   4);
    //CONV2
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++,  32,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++,  64,  32,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++,  64,   0,  3,  3, 1  ,  2,   4);
    //CONV3
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 128,  64,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 128,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 128, 128,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 128,   0,  3,  3, 1  ,  2,   4);
    //CONV4
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 256, 128,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 256,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 256, 256,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 256,   0,  3,  3, 1  ,  2,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 256,  1,  1, 0  ,  1,   4);
    //CONV5
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 512,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 512,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 512,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 512,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 512,   0,  3,  3, 1  ,  1,   4);
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++, 512,   0,  3,  3, 1  ,  2,   4);
    //CONV6
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++,1024, 512,  1,  1, 0  ,  1,   4);
    layer_update(l, CONVOLUTIONAL_DW, RELU  ,i++,1024,   0,  3,  3, 1  ,  1,   4);
    //CONV7
    layer_update(l, CONVOLUTIONAL   , RELU  ,i++,1024,1024,  1,  1, 0  ,  1,   4);
    layer_update(l, AVGPOOL         , LINEAR,i++,   0,   0,  7,  7, 0  ,  7,   0);

    layer_update(l, CONNECTED       , LINEAR,i++,1000,1024,  0,  0, 0  ,  1,   1);
    layer_update(l, SOFTMAX         , LINEAR,i++,1000,   0,  0,  0, 0  ,  1,   0);

    //conv(input,output)
    make_network(l,net_weight,num_layer,f_name);
    print_network(l,num_layer);
    inference(l,num_layer);

    // while(1){

    // }
    return 0;
}   