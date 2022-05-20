#include "emdnn.h"

int main(int argc, char *argv[]){
    char f_name[100] = "alexnet.weights";
    char img_name[100] = "dog.jpg";
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
	    }else if(!strcmp(argv[2],"nnpack")){
                sel_dev = PPU;
                sel_dev2 = PPU;
            }else if(!strcmp(argv[2],"opt_blas")){
                sel_dev  = GPU;
                sel_dev2 = CPU;
                opt_mode = 1;
            }
            break;
        default:
            sel_dev  = GPU;
            sel_dev2 = PPU;
            break;
    }

    float *net_weight;
    LAYER *l = (LAYER *)malloc(sizeof(LAYER));

    int i = 0;
    //       LAYER, LAYER_TYPE,    ACTIVATION,  dev, num,   N,   C,H-r,W-s, PAD,STR,SCALE
    //input
    l=layer_update(l, INPUT_LAYER     , LINEAR, CPU, i++,   0,   3,227,227, 0  ,  0,   4);
    //CONV
    l=layer_update(l, CONVOLUTIONAL   , RELU  , sel_dev, i++,  96,   3, 11, 11, 0  ,  4,   1);
    l=layer_update(l, MAXPOOL         , LINEAR, sel_dev, i++,   0,   0,  3,  3, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , RELU  , sel_dev2, i++, 256,  96,  5,  5, 2  ,  1,   1);
    l=layer_update(l, MAXPOOL         , LINEAR, sel_dev, i++,   0,   0,  3,  3, 0  ,  2,   0);
    l=layer_update(l, CONVOLUTIONAL   , RELU  , sel_dev2, i++, 384, 256,  3,  3, 1  ,  1,   1);
    l=layer_update(l, CONVOLUTIONAL   , RELU  , sel_dev, i++, 384, 384,  3,  3, 1  ,  1,   1);
    l=layer_update(l, CONVOLUTIONAL   , RELU  , sel_dev, i++, 256, 384,  3,  3, 1  ,  1,   1);
    l=layer_update(l, MAXPOOL         , LINEAR, sel_dev, i++,   0,   0,  3,  3, 0  ,  2,   0);
    //FC
    l=layer_update(l, CONNECTED       , RELU  , sel_dev2, i++,4096,9216,  1,  1, 0  ,  0,   1);
    l=layer_update(l, CONNECTED       , RELU  , sel_dev2, i++,4096,4096,  1,  1, 0  ,  0,   1);
    l=layer_update(l, CONNECTED       , LINEAR, sel_dev2, i++,1000,4096,  1,  1, 0  ,  0,   1);
    l=layer_update(l, SOFTMAX         , LINEAR, CPU, i++,1000,   0,  0,  0, 0  ,  1,   0);
    
    //       LAYER, LAYER_TYPE,  ACTIVATION, num,Nclass,Box 
    l=layer_update(l, CLASSIFICATION  , LINEAR, CPU, i++, 1000,   0,  0,  0, 0  ,  0,   0);
    
    int num_layer = i;

    make_network(l,net_weight,num_layer,f_name);
    if(opt_mode) tune_network(l,num_layer);
    char* class_result;
    print_network(l,num_layer,opt_mode);
    
    IplImage *cvimg;
    CvCapture* pCapture = NULL; 
    if( !(pCapture = cvCaptureFromFile(img_name)) ){
        printf("Video Capture Wrong!\n");
        fflush(stdout);
    }
        

    for(int rr =0 ; rr <1000; ++rr){
        // cvimg = Ipl_read(cvQueryFrame( pCapture ), l[0].OUTPUT, l[0].W, 0.0f, 0.0f, 0.0f);
        cvimg = image_read(img_name, l[0].OUTPUT, l[0].W, 0.0f, 0.0f, 0.0f);
        double tic = get_time();
        inference(l,num_layer);
        tic = get_time() - tic;
        printf("%.6f times\n",tic);
        fflush(stdout);
        class_result = class_print();
        // image_show_class(class_result, cvimg, 33);
    }



    return 0;
}


