#include "emdnn.h"

int main(){
    char f_name[100] = "yolov2-tiny-voc.weights";
    char img_name[100] = "dog.jpg";

    float *net_weight;
    LAYER *l = (LAYER *)malloc(sizeof(LAYER));
    int i = 0;

    //       LAYER, LAYER_TYPE,  ACTIVATION, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR,i++,   0,   3,416,416, 0  ,  0,   0);
    //CONV
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,  16,   3,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,  32,  16,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,  64,  32,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++, 128,  64,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++, 256, 128,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++, 512, 256,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  1,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,1024, 512,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY ,i++,1024,1024,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LINEAR,i++, 125,1024,  1,  1, 0  ,  1,   1);
    // //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, DETECTION       , LINEAR,i++,  20,   5,  0,  0, 0  ,  0,   0);
    
    int num_layer = i;
    make_network(l,net_weight,num_layer,f_name);
    print_network(l,num_layer);
    
    // l[0].CL_WEIGHT = clCreateBuffer(context, CL_MEM_READ_WRITE, l[0].H*l[0].W*sizeof(float), NULL, NULL);
    printf("%d \n",sizeof(LAYER));
    for(int rr =0 ; rr <5; ++rr){
        IplImage *cvimg = image_read(img_name, l[0].OUTPUT, l[0].W);
        inference(l,num_layer);
        image_show(l[num_layer-1].OUTPUT, cvimg);
    }
    
    image_free();
    // while(1){

    // }
    return 0;
}   