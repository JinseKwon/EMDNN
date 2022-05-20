#include "emdnn.h"

int main(int argc, char *argv[]){
    char f_name[100] = "mobilenet_v1_72.weights";
    char img_name[100] = "dog.jpg"; // "cat.jpg"; //
    char vid_name[100] = "video.mp4";

    char c = '0';
    if(argc > 1)
        if(argv[1][0] == 'b') c = 'b';

    DEVICE_TYPE sel_dev;
    DEVICE_TYPE sel_dev2;
    int opt_mode = 0;
    switch (c) {
        case 'b':             /* print help message */
            if(!strcmp(argv[2],"openblas")){
                sel_dev = CPU;
                sel_dev2 = CPU;
            }else if(!strcmp(argv[2],"clblast")){
                sel_dev = GPU;
                sel_dev2 = GPU;
            }else if(!strcmp(argv[2],"opt_blas")){
                sel_dev  = GPU;
                sel_dev2 = CPU;
                opt_mode = 1;
            }
            break;
        default:
            sel_dev  = CPU;
            sel_dev2 = CPU;
            break;
    }

    float *net_weight;
    LAYER *l = (LAYER *)malloc(sizeof(LAYER));

    int i = 0;
    //       LAYER, LAYER_TYPE,    ACTIVATION,  dev, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR, CPU, i++,   0,   3,224,224, 0  ,  0,   5);
    //CONV1
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++,  32,   3,  3,  3, 1  ,  2,   4);
    //CONV2
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++,  32,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++,  64,  32,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++,  64,   0,  3,  3, 1  ,  2,   4);
    //CONV3
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 128,  64,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 128,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 128, 128,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 128,   0,  3,  3, 1  ,  2,   4);
    //CONV4
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 256, 128,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 256,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 256, 256,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 256,   0,  3,  3, 1  ,  2,   4);
    //CONV5
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 512, 256,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 512,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++, 512, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++, 512,   0,  3,  3, 1  ,  2,   4);
    //CONV6
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++,1024, 512,  1,  1, 0  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL_DW, LEAKY , PPU, i++,1024,   0,  3,  3, 1  ,  1,   4);
    l=layer_update(l, CONVOLUTIONAL   , LEAKY , PPU, i++,1024,1024,  1,  1, 0  ,  1,   4);
    l=layer_update(l, AVGPOOL         , LINEAR, GPU, i++,   0,   0,  7,  7, 0  ,  0,   0);
    //CONV7
    l=layer_update(l, CONNECTED       , LINEAR, GPU, i++,1000,1024,  1,  1, 0  ,  0,   1);
    l=layer_update(l, SOFTMAX         , LINEAR, CPU, i++,1000,   0,  0,  0, 0  ,  1,   0);
    
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, CLASSIFICATION  , LINEAR, CPU, i++,1000,   0,  0,  0, 0  ,  0,   0);

    int num_layer = i;
    make_network(l,net_weight,num_layer,f_name);
    if(opt_mode) tune_network(l,num_layer);

    print_network(l,num_layer,opt_mode);

    char* class_result;
    IplImage *cvimg;
    CvCapture* pCapture = NULL; 
    if( !(pCapture = cvCaptureFromFile(img_name)) )
        printf("Video Capture Wrong!\n");
    for(int rr =0 ; rr <1000; ++rr){
        // cvimg = Ipl_read(cvQueryFrame( pCapture ), l[0].OUTPUT, l[0].W, 0.0f, 0.0f, 0.0f);
        cvimg = image_read(img_name, l[0].OUTPUT, l[0].W, 0.0f, 0.0f, 0.0f);
        
        double tic = get_time();
        inference(l,num_layer);
        tic = get_time() - tic;
        printf("%.6f times\n",tic);   
        class_result = class_print();
        // image_show_class(class_result, cvimg, tic);
    }

    return 0;
}   