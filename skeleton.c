#include "emdnn.h"

int main(){
    char f_name[100] = "yolov2-tiny-voc.weights";
    char img_name[100] = "dog.jpg";

    float *net_weight;
    LAYER *l = (LAYER *)malloc(sizeof(LAYER));
    
    int i = 0;
    //       LAYER, LAYER_TYPE,    ACTIVATION,  dev, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR, CPU, i++,   0,   3,416,416, 0  ,  0,   4);
    //CONV
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  16,   3,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  32,  16,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  64,  32,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 128,  64,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, PPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 256, 128,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, PPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512, 256,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, PPU, i++,   0,   0,  2,  2, 1  ,  1,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,1024, 512,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,1024,1024,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LINEAR, GPU, i++, 125,1024,  1,  1, 0  ,  1,   1);
    // //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, DETECTION       , LINEAR, CPU, i++,  20,   5,  0,  0, 0  ,  0,   0);
    int num_layer = i;

    make_network(l,net_weight,num_layer,f_name);
    tune_network(l,num_layer);

    print_network(l,num_layer);
    
    printf("%d \n",sizeof(LAYER));
    for(int rr =0 ; rr <10; ++rr){
        IplImage *cvimg = image_read(img_name, l[0].OUTPUT, l[0].W, 0.0f, 0.0f, 0.0f);
        double tic = get_time();
        inference(l,num_layer);
        tic = get_time() - tic;
        printf("%.6f times \n\n",tic);
        image_show(l[num_layer-1].OUTPUT, cvimg, tic);
    }
   
    // image_free();

    return 0;
}   
