#include "string.h"
#include <stdlib.h>
#include "emdnn.h"
#include "classification_class.h"

// #define ALEX
// #define VGG

// const char color_map[20][3] ={  {204,23,134},{153,63,118},{171,4,255},{255,251,141},{204,179,23},
//                             {155,204,39},{250,255,0},{255,65,0},{137,255,196},{20,204,60},
//                             {204,172,23},{0,255,134},{187,204,180},{66,214,255},{255,228,23},
//                             {204,150,192},{153,63,134},{148,4,255},{255,242,66},{204,173,23}};

// void table_swap2(int *a, int *b);
// void swap2(float* a, float* b);
// void quick_sort2(float* array, int start, int end, int* table);



void table_swap2(int *a, int *b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
void swap2(float* a, float* b){
    float tmp=*a;
    *a=*b;
    *b=tmp;
}
void quick_sort2(float* array, int start, int end, int* table){
 
    if(start>=end) return;
 
    int mid=(start+end)/2;
    float pivot=array[mid];
 
    swap2(&array[start],&array[mid]);
    table_swap2(&table[start],&table[mid]);
    
    int left=start+1;
    int right=end;

    while(1){
        while(array[left]>=pivot){ left++; }
        while(array[right]<pivot){ right--; }
        if(left>right) break;
        swap2(&array[left],&array[right]);
        table_swap2(&table[left],&table[right]);
    }
 
    swap2(&array[start],&array[right]);
    table_swap2(&table[start],&table[right]);
 
    quick_sort2(array,start,right-1,table);
    quick_sort2(array,right+1,end,table);
}



void classification(float *output_score,int class_num){
    int index_table[1000];
    for(int i = 0; i<1000; ++i){
        index_table[i] = i;
    }
    quick_sort2(output_score, 0, class_num-1, index_table);   
    
    char text[50];
    //cvResize(input_img, viewimg, 0);
    // cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);
    // cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    // cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);

    printf("\n");
    for(int i = 0; i<5; ++i){
        sprintf(text, "top%d :(idx : %d) %2.3f ( %s ) ",i,index_table[i], output_score[i]*100, class_name[index_table[i]] );
        // cvPutText(viewimg, text, cvPoint(10, 100+30*i), &font, CV_RGB(255,255,41));
        printf("%s \n", text);
    }
    // sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    // cvPutText(viewimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    // cvPutText(viewimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt


    // printf( "top%d :(idx : %d) %2.3f ( %s ) \n",0,index_table[0], output_score[0]*100, class_name[index_table[0]] );
    // sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    // cvPutText(viewimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    // cvPutText(viewimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val

//    cvShowImage("Image Classification",viewimg);
    // cvWaitKey(10000);
    // return 1;//
 //   cvWaitKey(33);
    // //if(cvWaitKey(33) == 1048691){   "s" key input
}