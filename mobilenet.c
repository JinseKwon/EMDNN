#include "emdnn.h"

int main(){
    char f_name[100] = "mobilenet_v1_72.weights";
    char img_name[100] = "dog.jpg"; // "cat.jpg"; //

    float *net_weight;
    LAYER *l = (LAYER *)malloc(sizeof(LAYER));

    int i = 0;
    //       LAYER, LAYER_TYPE,    ACTIVATION,  dev, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR, CPU, i++,   0,   3,256,256, 0  ,  0,   5);
    //CONV1
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  32,   3,  3,  3, 1  ,  2,   4);
    //CONV2
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++,  32,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  64,  32,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++,  64,   0,  3,  3, 1  ,  2,   4);
    //CONV3
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 128,  64,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 128,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 128, 128,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 128,   0,  3,  3, 1  ,  2,   4);
    //CONV4
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 256, 128,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 256,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 256, 256,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 256,   0,  3,  3, 1  ,  2,   4);
    //CONV5
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 256,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++, 512,   0,  3,  3, 1  ,  2,   4);
    //CONV6
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,1024, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , GPU, i++,1024,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,1024,1024,  1,  1, 0  ,  1,   4);
    l=layer_update(l, AVGPOOL         , LINEAR, CPU, i++,   0,   0,  8,  8, 0  ,  0,   0);
    //CONV7
    l=layer_update(l, CONNECTED       , LINEAR, GPU, i++,1000,1024,  1,  1, 0  ,  0,   1);
    l=layer_update(l, SOFTMAX         , LINEAR, CPU, i++,1000,   0,  0,  0, 0  ,  1,   0);
    
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, CLASSIFICATION  , LINEAR, CPU, i++,1000,   0,  0,  0, 0  ,  0,   0);

    int num_layer = i;
    make_network(l,net_weight,num_layer,f_name);
    //tune_network(l,num_layer);

    print_network(l,num_layer);
    for(int rr =0 ; rr <5; ++rr){
        IplImage *cvimg = image_read(img_name, l[0].OUTPUT, l[0].W,
                                    //  103.94f,  116.78f,  123.68f);
                                     0.0f,  0.0f,  0.0f);
        
        double tic = get_time();
        
        inference(l,num_layer);
        
        printf("%.6f times \n\n",get_time()-tic);
        // image_show(l[num_layer-1].OUTPUT, cvimg);
    }

    return 0;
}   
