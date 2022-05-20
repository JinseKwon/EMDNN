#include "opencv/highgui.h"
#include "string.h"
#include <stdlib.h>

#include "timer.h"
#include "imagenet_class.h"

// #define ALEX
// #define VGG

// const char color_map[20][3] ={  {204,23,134},{153,63,118},{171,4,255},{255,251,141},{204,179,23},
//                             {155,204,39},{250,255,0},{255,65,0},{137,255,196},{20,204,60},
//                             {204,172,23},{0,255,134},{187,204,180},{66,214,255},{255,228,23},
//                             {204,150,192},{153,63,134},{148,4,255},{255,242,66},{204,173,23}};

void table_swap(int *a, int *b);
void swap(float* a, float* b);
void quick_sort(float* array, int start, int end, int* table);

CvCapture* capture = cvCaptureFromCAM(0);
IplImage *input_img;
#if defined ALEX || defined ALEXCL ||defined ALEXMVNC
    IplImage *readimg = cvCreateImage(cvSize(227,227),IPL_DEPTH_8U,3);
    char *imgs = (char*)malloc(227 * 227 * 3);
#endif
#if defined VGG || defined VGGCL || defined VGGMVNC
    IplImage *readimg = cvCreateImage(cvSize(224,224),IPL_DEPTH_8U,3);
    char *imgs = (char*)malloc(224 * 224 * 3);
#endif
#if defined TINY || defined TINYCL || TINYMVNC
    IplImage *readimg = cvCreateImage(cvSize(224,224),IPL_DEPTH_8U,3);
    char *imgs = (char*)malloc(224 * 224 * 3);
#endif
IplImage *viewimg = cvCreateImage(cvSize(500,500),IPL_DEPTH_8U,3);



CvFont font, font_guide, font_txt;

void cam_read(float *image, int img_size){
    input_img = cvQueryFrame(capture);
    int x = input_img -> width;
    int y = input_img -> height;
    printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
        printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
        printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    // cvResize(input_img, viewimg, 0);
    imgs = readimg->imageData;
    if(img_size%2 == 1){
        for( int h=0; h<img_size; ++h){
            for( int w=0; w<img_size; ++w){
                image[h*img_size*3 + w*3 + 0] = (float)(imgs[h*(img_size+1)*3 + w*3 + 2]/255.f);
                image[h*img_size*3 + w*3 + 1] = (float)(imgs[h*(img_size+1)*3 + w*3 + 1]/255.f);
                image[h*img_size*3 + w*3 + 2] = (float)(imgs[h*(img_size+1)*3 + w*3 + 0]/255.f);
            }
        }
    }else{
        for(int i = 0; i<img_size*img_size*3; ++i){
            image[i+0] = (float)(imgs[i+2]/255.f);
            image[i+1] = (float)(imgs[i+1]/255.f);
            image[i+2] = (float)(imgs[i+0]/255.f);
        }
    }
    // cvReleaseImage(&input_img);
}
void image_read( float *image, int img_size,char *filename){
    // input_img = cvLoadImage("images/dog.jpg");
    input_img = cvLoadImage(filename);
    // input_img = cvLoadImage("images/512_ElectricGuitar.jpg");
    // input_img = cvLoadImage("images/512_ElectricGuitar.jpg");
    // input_img = cvLoadImage("images/dog_1x1.jpg");
    int x = input_img -> width;
    int y = input_img -> height;
    //printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
      //  printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    imgs = readimg->imageData;
   // printf("height,width : %d %d (%d?)\n",readimg -> height, readimg -> width, img_size);

    if(img_size%2 == 1){
        for( int h=0; h<img_size; ++h){
            for( int w=0; w<img_size; ++w){
                image[h*img_size*3 + w*3 + 0] = (float)(imgs[h*(img_size+1)*3 + w*3 + 2]/255.f);
                image[h*img_size*3 + w*3 + 1] = (float)(imgs[h*(img_size+1)*3 + w*3 + 1]/255.f);
                image[h*img_size*3 + w*3 + 2] = (float)(imgs[h*(img_size+1)*3 + w*3 + 0]/255.f);
            }
        }
    }else{
        for(int i = 0; i<img_size*img_size*3; ++i){
            image[i+0] = (float)(imgs[i+2]/255.f);
            image[i+1] = (float)(imgs[i+1]/255.f);
            image[i+2] = (float)(imgs[i+0]/255.f);
        }
    }
    // cvReleaseImage(&input_img);
}
void image_check(float *img,int dim){
    IplImage *chkimg = cvCreateImage(cvSize(dim,dim),IPL_DEPTH_8U,3);
    for(int c=0; c<3; ++c){
        for( int h=0; h<dim; ++h){
            for( int w=0; w<dim; ++w){
                chkimg->imageData[ h*dim*3 + w*3 + c] = (char)(img[c*dim*dim + h*dim + w]*255.f);
            }
        }
    }
    cvShowImage("Check Image",chkimg);
    cvWaitKey(50000);
    cvReleaseImage(&chkimg);
}
int image_show(float *output_score,int class_num){
   
    int index_table[1000];
    for(int i = 0; i<1000; ++i){
        index_table[i] = i;
    }
    quick_sort(output_score, 0, class_num-1, index_table);   
    
    char text[50];
    cvResize(input_img, viewimg, 0);
    // cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);
    // cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    // cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);

    double elapsed = timer_stop(0);
    for(int i = 0; i<5; ++i){
        sprintf(text, "top%d :(idx : %d) %2.3f ( %s ) ",i,index_table[i], output_score[i]*100, class_name[index_table[i]] );
        // cvPutText(viewimg, text, cvPoint(10, 100+30*i), &font, CV_RGB(255,255,41));
        printf("%s \n", text);
    }
    // sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    // cvPutText(viewimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    // cvPutText(viewimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt

    printf("Elapsed time  %.6f   \n", elapsed);

    // printf( "top%d :(idx : %d) %2.3f ( %s ) \n",0,index_table[0], output_score[0]*100, class_name[index_table[0]] );
    // sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    // cvPutText(viewimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    // cvPutText(viewimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val

    cvShowImage("Image Classification",viewimg);
    cvWaitKey(10000);
    // return 1;//
    // cvWaitKey(33);
    // //if(cvWaitKey(33) == 1048691){   "s" key input
}
void image_free(){
    cvReleaseCapture(&capture);
    cvReleaseImage(&readimg);
    cvReleaseImage(&viewimg);
    cvDestroyWindow("Image Classification");
    free(imgs);
}
void table_swap(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
void swap(float* a, float* b){
    float tmp=*a;
    *a=*b;
    *b=tmp;
}
void quick_sort(float* array, int start, int end, int* table){
 
    if(start>=end) return;
 
    int mid=(start+end)/2;
    float pivot=array[mid];
 
    swap(&array[start],&array[mid]);
    table_swap(&table[start],&table[mid]);
    
    int left=start+1;
    int right=end;

    while(1){
        while(array[left]>=pivot){ left++; }
        while(array[right]<pivot){ right--; }
        if(left>right) break;
        swap(&array[left],&array[right]);
        table_swap(&table[left],&table[right]);
    }
 
    swap(&array[start],&array[right]);
    table_swap(&table[start],&table[right]);
 
    quick_sort(array,start,right-1,table);
    quick_sort(array,right+1,end,table);
}
