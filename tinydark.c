#include "emdnn.h"

int main(){
    char f_name[100] = "tinydark.weights";
    char img_name[100] = "dog.jpg";

    float *net_weight;
    LAYER *l = (LAYER *)malloc(sizeof(LAYER));

    int i = 0;
    //       LAYER, LAYER_TYPE,    ACTIVATION,  dev, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR, CPU, i++,   0,   3,224,224, 0  ,  0,   4);
    //CONV
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  16,   3,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  32,  16,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  16,  32,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 128,  16,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  16, 128,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 128,  16,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  32, 128,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 256,  32,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  32, 256,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 256,  32,  3,  3, 1  ,  1,   4);
    l=layer_update(l, MAXPOOL         , LINEAR, GPU, i++,   0,   0,  2,  2, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  64, 256,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512,  64,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,  64, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 512,  64,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++, 128, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , GPU, i++,1000, 128,  1,  1, 0  ,  1,   1);
    l=layer_update(l, AVGPOOL         , LINEAR, GPU, i++,   0,   0, 14, 14, 0  ,  0,   0);
    l=layer_update(l, SOFTMAX         , LINEAR, CPU, i++,1000,   0,  0,  0, 0  ,  1,   0);
    
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, CLASSIFICATION  , LINEAR, CPU, i++, 1000,   0,  0,  0, 0  ,  0,   0);

    int num_layer = i;
    make_network(l,net_weight,num_layer,f_name);
    // tune_network(l,num_layer);

    print_network(l,num_layer);
    for(int rr =0 ; rr <2; ++rr){
        IplImage *cvimg = image_read(img_name, l[0].OUTPUT, l[0].W, 0.0f, 0.0f, 0.0f);
        
        double tic = get_time();
        
        inference(l,num_layer);
        
        printf("%.6f times \n\n",get_time()-tic);
        // image_show(l[num_layer-1].OUTPUT, cvimg);
    }
    // while(1){

    // }
    return 0;
}
