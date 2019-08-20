#include "emdnn.h"

int main(){
    char f_name[100] = "models/vgg-16.weights";
    char img_name[100] = "dog.jpg";
    float *net_weight;

   
     LAYER *l = (LAYER *)malloc(sizeof(LAYER));
    int i = 0;
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,   N,   C,  H,  W,PAD ,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR,i++,   0,   3,224,224, 0  ,  0,   0);
    //conv1
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++,  64,   3,  3,  3, 1  ,  1,   1);
    //conv2
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++,  64,  64,  3,  3, 1  ,  1,   1);
    //max1
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    //conv3
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 128,  64,  3,  3, 1  ,  1,   1);
    //conv4
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 128, 128,  3,  3, 1  ,  1,   1);
    //max2
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    //conv5
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 256, 128,  3,  3, 1  ,  1,   1);
    //conv6
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 256, 256,  3,  3, 1  ,  1,   1);
    //conv7
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 256, 256,  3,  3, 1  ,  1,   1);
    //max3
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    //conv8
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 256,  3,  3, 1  ,  1,   1);
    //conv9
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  3,  3, 1  ,  1,   1);
    //conv10
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  3,  3, 1  ,  1,   1);
    //max4
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    //conv11
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  3,  3, 1  ,  1,   1);
    //conv12
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  3,  3, 1  ,  1,   1);
    //conv13
    l=layer_update(l, CONVOLUTIONAL   , RELU  ,i++, 512, 512,  3,  3, 1  ,  1,   1);
    //max5
    l=layer_update(l, MAXPOOL         , LINEAR,i++,   0,   0,  2,  2, 0  ,  2,   0);
    
    
    //fc1
    l=layer_update(l, CONNECTED       , RELU  ,i++,4096,7*7*512,  1,  1, 0  ,  0,   1);
    //fc2
    l=layer_update(l, CONNECTED       , RELU  ,i++,4096,4096,  1,  1, 0  ,  0,   1);
    //fc3
    l=layer_update(l, CONNECTED       , LINEAR,i++,1000,4096,  1,  1, 0  ,  0,   1);
    //softmax
    l=layer_update(l, SOFTMAX         , LINEAR,i++,1000,   0,  0,  0, 0  ,  1,   0);
    
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, CLASSIFICATION  , LINEAR,i++, 1000,   0,  0,  0, 0  ,  0,   0);


    //conv(input,output)
    int num_layer=i;
    make_network(l,net_weight,num_layer,f_name);
    print_network(l,num_layer);
       
       
        IplImage *cvimg = image_read(img_name, l[0].OUTPUT, l[0].W);
        inference(l,num_layer);


    // while(1){

    // }
    return 0;
}


