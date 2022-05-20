#include "string.h"
#include <stdlib.h>
#include "emdnn.h"
#include "classification_class.h"
char RESULT_TXT[2000];
float top1_s;
int   top1_i;
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



void classification(float *output_score,
                    int class_num, 
                    int tune,
                    char *result){
    int index_table[class_num];

    // printf("class num : %d\n",class_num);
    for(int i = 0; i<class_num; ++i){
        index_table[i] = i;
    }
    // result = (char*)malloc(1000);
    quick_sort2(output_score, 0, class_num-1, index_table);   
    char text[1000]={""};
    ////// if(!tune) printf("\n");
    for(int i = 0; i<5; ++i){
        if(class_num == 10){
            sprintf(text, "%stop%d :(idx : %d) %2.3f\% \n",text,i,index_table[i], output_score[i]*100);    
        }else{
            sprintf(text, "%stop%d :(idx : %d) %2.3f\% ( %s ) \n",text,i,index_table[i], output_score[i]*100, class_name[index_table[i]] );
        }
        // cvPutText(viewimg, text, cvPoint(10, 100+30*i), &font, CV_RGB(255,255,41));
    }
    // printf("%s",text);
    // if(!tune) printf("%s", text);
    // RESULT_TXT = (char*)malloc(1024);
    top1_s = output_score[0];
    top1_i = index_table[0];
    sprintf(RESULT_TXT, "%s", text);
    
}
char* class_print(){
    return RESULT_TXT;
}
float top1_score(){
    return top1_s;
}
int top1_idx(){
    return top1_i;
}