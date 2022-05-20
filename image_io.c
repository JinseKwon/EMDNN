#include "opencv/highgui.h"

#include "emdnn.h"
#include "string.h"
#include <stdlib.h>

const char class_name[20][20] = {"aeroplane", "bicycle", "bird", "boat", "bottle", 
                                "bus", "car", "cat", "chair", "cow", 
                                "diningtable", "dog", "horse", "motorbike", "person", 
                                "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
const char color_map[20][3] ={
  {(char)204,(char)23,(char)134},{(char)153,(char)63,(char)118},{(char)171,(char)4,(char)255},{(char)255,(char)251,(char)141},{(char)204,(char)179,(char)23},
  {(char)155,(char)204,(char)39},{(char)250,(char)255,(char)0},{(char)255,(char)65,(char)0},{(char)137,(char)255,(char)196},{(char)20,(char)204,(char)60},
  {(char)204,(char)172,(char)23},{(char)0,(char)255,(char)134},{(char)187,(char)204,(char)180},{(char)66,(char)214,(char)255},{(char)255,(char)228,(char)23},
  {(char)204,(char)150,(char)192},{(char)153,(char)63,(char)134},{(char)148,(char)4,(char)255},{(char)255,(char)242,(char)66},{(char)204,(char)173,(char)23}
};

float scale;
// CvCapture* capture = cvCaptureFromCAM(0);
// void cam_read(float *image, int img_size){
//     input_img = cvQueryFrame(capture);
//     int x = input_img -> width;
//     int y = input_img -> height;
//     //printf("h,w : %d %d\n",y, x);
//     if(x > y){
//         int new_x = (x-y)/2;
//         cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
//         //printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
//         // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
//     }else if(y > x){
//         int new_y = (y-x)/2;
//         cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
//         //printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
//     }
//     cvResize(input_img, readimg, 0);
//     imgs = readimg->imageData;
//     for(int i = 0; i<img_size*img_size*3; i++){
//         image[i] = imgs[i] /255.;
//     }
//     //===========================//
//     //gray scale conversion//
//     // imgs = readimg->imageData;
//     // float gray;        
//     // for(int i = 0; i<img_size*img_size*3; i+=3){
//     //     gray = (readimg->imageData[i]*0.05 + readimg->imageData[i+1]*0.02 + readimg->imageData[i+2]*0.03)/255;
//     //     image[i  ] = gray;//imgs[i  ] /255.;
//     //     image[i+1] = gray;//imgs[i+1] /255.;
//     //     image[i+2] = gray;//imgs[i+2] /255.;
//     //     readimg->imageData[i  ] = (char)(gray*255);
//     //     readimg->imageData[i+1] = (char)(gray*255);
//     //     readimg->imageData[i+2] = (char)(gray*255);
//     // }
//     //===========================//
//     // cvReleaseImage(&input_img);
// }
IplImage* image_read(char *img_file, float *image, int img_size,
                     float mean_r,   float mean_g, float mean_b){   
    //  input_img = cvLoadImage("images/dog.jpg");    
    IplImage *input_img = cvLoadImage(img_file); 
    IplImage *readimg = cvCreateImage(cvSize(img_size,img_size),IPL_DEPTH_8U,3);
    //  input_img = cvLoadImage("images/black_dog.jpg");  
    // input_img = cvLoadImage("images/horses.jpg");    
    int x = input_img -> width;
    int y = input_img -> height;
    // //printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        scale = (float)y / (float)img_size;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        scale = (float)x / (float)img_size;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    // char *imgs = (char*)malloc(416 * 416 * 3);
    unsigned char *imgs = (unsigned char*)readimg->imageData;
    
    //opencv 4 times padding for speed up
    int img_padding = readimg->widthStep;
    
    for(int h =0; h<img_size; ++h){
        for(int w =0; w<img_size; ++w){
            // for(int c = 0; c<3; c++){
            //     image[c*img_size*img_size + h*img_size + w] = imgs[ h*img_padding + w*3 + c] / 255.;
            image[0*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 0]-mean_r)/ 255.;
            image[1*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 1]-mean_g)/ 255.;
            image[2*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 2]-mean_b)/ 255.;
            // }
        }
    }
    // free(imgs);
    // for(int kk = 0; kk < img_size * img_size * 3; ++kk){
    //     readimg->imageData[kk] = (char)(image[kk] * 255 );
    // }
    // cvShowImage("YOLO tiny with OpenCL",readimg);
    // cvWaitKey(33);
    // cvReleaseImage(&input_img);
    return input_img;
}
IplImage* Ipl_read(IplImage* input_img, float *image, int img_size,
                     float mean_r,   float mean_g, float mean_b){   
    
    IplImage *readimg = cvCreateImage(cvSize(img_size,img_size),IPL_DEPTH_8U,3);
    int x = input_img -> width;
    int y = input_img -> height;
    // //printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        scale = (float)y / (float)img_size;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        scale = (float)x / (float)img_size;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    // char *imgs = (char*)malloc(416 * 416 * 3);
    unsigned char *imgs = (unsigned char*)readimg->imageData;
    
    //opencv 4 times padding for speed up
    int img_padding = readimg->widthStep;
    
    for(int h =0; h<img_size; ++h){
        for(int w =0; w<img_size; ++w){
            // for(int c = 0; c<3; c++){
            //     image[c*img_size*img_size + h*img_size + w] = imgs[ h*img_padding + w*3 + c] / 255.;
            image[0*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 0]-mean_r)/ 255.;
            image[1*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 1]-mean_g)/ 255.;
            image[2*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 2]-mean_b)/ 255.;
            // }
        }
    }
    // free(imgs);
    // for(int kk = 0; kk < img_size * img_size * 3; ++kk){
    //     readimg->imageData[kk] = (char)(image[kk] * 255 );
    // }
    // cvShowImage("YOLO tiny with OpenCL",readimg);
    // cvWaitKey(33);
    // cvReleaseImage(&input_img);
    return input_img;
}
// void imagefile_read2( float *image, int img_size,  int i ){   
//     // input_img = cvLoadImage("images/dog.jpg");    
  
//     switch(i){ 
//         case 0: input_img = cvLoadImage("images/eagle.jpg");    break;
//         case 1: input_img = cvLoadImage("images/dog.jpg");  break;
//         case 2: input_img = cvLoadImage("images/cat.jpg"); break;
//         case 3: input_img = cvLoadImage("images/horses.jpg"); break;
//     }
//     //input_img = cvLoadImage("images/dog.jpg");    
//     int x = input_img -> width;
//     int y = input_img -> height;
//     //printf("h,w : %d %d\n",y, x);
//       if(x > y){
//         int new_x = (x-y)/2;
//         cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
//        // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
//         // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
//     }else if(y > x){
//         int new_y = (y-x)/2;
//         cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
//       //  printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
//     }
//     cvResize(input_img, readimg, 0);
//     imgs = readimg->imageData;
//    // printf("height,width : %d %d (%d?)\n",readimg -> height, readimg -> width, img_size);

//     if(img_size%2 == 1){
//         for( int h=0; h<img_size; ++h){
//             for( int w=0; w<img_size; ++w){
//                 image[h*img_size*3 + w*3 + 0] = (float)(imgs[h*(img_size+1)*3 + w*3 + 2]/255.f);
//                 image[h*img_size*3 + w*3 + 1] = (float)(imgs[h*(img_size+1)*3 + w*3 + 1]/255.f);
//                 image[h*img_size*3 + w*3 + 2] = (float)(imgs[h*(img_size+1)*3 + w*3 + 0]/255.f);
//             }
//         }
//     }else{
//         for(int i = 0; i<img_size*img_size*3; ++i){
//             image[i+0] = (float)(imgs[i+2]/255.f);
//             image[i+1] = (float)(imgs[i+1]/255.f);
//             image[i+2] = (float)(imgs[i+0]/255.f);
//         }
//     }
//     // cvReleaseImage(&input_img);
// }

// void imagefile_read( float *image, int img_size, char *filename){   
//     input_img = cvLoadImage(filename);    
//     int x = input_img -> width;
//     int y = input_img -> height;
//     //printf("h,w : %d %d\n",y, x);
//     if(x > y){
//         int new_x = (x-y)/2;
//         cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
//      //   printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
//         // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
//     }else if(y > x){
//         int new_y = (y-x)/2;
//         cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
//        // printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
//     }
//     cvResize(input_img, readimg, 0);
//     imgs = readimg->imageData;
//     for(int i = 0; i<img_size*img_size*3; i++){
//         image[i] = imgs[i] /255.;
//     }
//     // cvReleaseImage(&input_img);
//     //printf("hihih");
// }
void image_show_class(char* class_text, IplImage *readimg,
                      double elapsed, int show_time){
    char text[20];
    CvFont font, font_guide, font_txt;
        
    cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 0.7, 0.7, 0 ,2);//, 0, 1, CV_AA);
    cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);

    // double elapsed = timer_stop(0);
    // double elapsed = 1;
    IplImage *newimg = cvCreateImage(cvSize(500,500),IPL_DEPTH_8U,3);
    cvResize(readimg,newimg);
    // classificaiton result print
    // printf("%s\n",class_text);
    char *ptr;
    ptr = strtok(class_text, "\n");
    for(int i=0; i<5; ++i){
        cvPutText(newimg, ptr, cvPoint(5,375+i*25), 
                  &font, CV_RGB(237,193,26));
        ptr = strtok(NULL,"\n");
    }
    sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    cvPutText(newimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    cvPutText(newimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt

    // printf("Elapsed time  %.6f \n", elapsed);
    // printf("%s\n", text);
    sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    cvPutText(newimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    cvPutText(newimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val

    cvShowImage("Image Classification",newimg);
    // cvWaitKey(10);
    cvWaitKey(show_time);

    //if(cvWaitKey(33) == 1048691){   "s" key input
}
void image_show_yolo(float* box_output, IplImage *readimg,
                double elapsed, int show_time){
    char text[20];
    CvFont font, font_guide, font_txt;

    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0 ,2);//, 0, 1, CV_AA);
    cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);

    // double elapsed = timer_stop(0);
    // double elapsed = 1;
    // printf("printing box... %d <<< \n",(int)box_output[13*13*5*20*6]);
    if((int)box_output[13*13*5*20*6] > 0){
       for(int i = 0; i < (int)box_output[13*13*5*20*6]; i++){
            sprintf(text, "%s(%.2f)", class_name[(int)box_output[i*6 + 1]], box_output[i*6] );//1/(get_time()-end_time));
            printf("%s\n", text);
            int class_i = (int)box_output[i*6 + 1];
            cvPutText(readimg, text, 
                      cvPoint((int)(box_output[i*6 + 2]*scale), 
                              (int)(box_output[i*6 + 3]*scale)), 
                      &font, 
                      CV_RGB(color_map[class_i][0],
                             color_map[class_i][1],
                             color_map[class_i][2])
                      );
            cvRectangle(readimg, 
                        cvPoint((int)(box_output[i*6 + 2]*scale), 
                                (int)(box_output[i*6 + 3]*scale)), 
                        cvPoint((int)(box_output[i*6 + 4]*scale), 
                                (int)(box_output[i*6 + 5]*scale)),
                        CV_RGB(color_map[class_i][0],
                               color_map[class_i][1],
                               color_map[class_i][2]),
                        2);
        }
    }
    sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    cvPutText(readimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    cvPutText(readimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt

    // printf("Elapsed time  %.6f \n", elapsed);
    // printf("%s\n", text);
    sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    cvPutText(readimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    cvPutText(readimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val

    cvShowImage("Object Detection",readimg);
    cvWaitKey(show_time);

    //if(cvWaitKey(33) == 1048691){   "s" key input
}
void image_free(){
    // cvReleaseCapture(&capture);
    // cvReleaseImage(&readimg);
    cvDestroyWindow("YOLO tiny with OpenCL");
}
    
