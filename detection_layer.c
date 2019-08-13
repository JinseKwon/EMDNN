#include "emdnn.h"

#include <stdio.h>
#include <math.h>
#include <float.h>

#define I_MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define I_MIN(X,Y) ((X) < (Y) ? (X) : (Y))

struct Det_boxes{
    float *score;
    int   *coord;
    int   counter;
};

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
float det_sigmoid(float x){
    return (float)1./(1. + exp(-x));
}
int det_softmax(float *in, int count){  //range modify in 0~1 // return best score;

    //find max
    float max = -FLT_MAX;      // input val max
    //float out
    float temp = 1.0;             //hinton T = 1; https://jamiekang.github.io/page2/
    for(int i = 0; i < count; ++i){
        if(in[i] > max){
            max = in[i];
        }
    }

    //Summation
    float sum =0 ;
    for(int i = 0; i < count; ++i){
        float e = expf(in[i]/temp - max/temp);
        in[i] = e;
        sum += e;
    }
    //Each val/summation
    //test val
    float best_score = -FLT_MAX;
    int max_idx = -1;
    for(int i =0; i < count; ++i){
        in[i] = in[i] / sum;
        if(in[i] > best_score){
            best_score = in[i];
            max_idx = i;
            //printf(" best_score : %f, idx : %d",best_score,max_idx);
        }
    }
    
    return max_idx;//return class index;
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
float iou(int A_lt_x, int A_lt_y, int A_rb_x, int A_rb_y, 
          int B_lt_x, int B_lt_y, int B_rb_x, int B_rb_y){
    
    float result;

    // determine the (x, y)-coordinates of the intersection rectangle
    int xA = I_MAX(A_lt_x, B_lt_x);
    int yA = I_MAX(A_lt_y, B_lt_y);
    int xB = I_MIN(A_rb_x, B_rb_x);
    int yB = I_MIN(A_rb_y, B_rb_y);

    // compute the area of intersection rectangle
    int interArea = I_MAX(0, xB - xA + 1) * I_MAX(0, yB - yA + 1);

    // compute the area of both the prediction and ground-truth rectangles
    int boxA_Area = (A_rb_x - A_lt_x + 1) * (A_rb_y - A_lt_y + 1);
    int boxB_Area = (B_rb_x - B_lt_x + 1) * (B_rb_y - B_lt_y + 1);

    result = (float)interArea / (float)( boxA_Area + boxB_Area - interArea);

    return result;

}
void non_maximal_suppression(struct Det_boxes *nms_box, float *out,
                            int det_h,
                            int det_w,
                            int det_BOX,
                            int det_CLASS,
                            float iou_threshold){

    int next_box = 0;
    int table[det_h*det_w*det_BOX];
    for(int class_i = 0; class_i < det_CLASS; ++class_i){
        
        for(int table_i = 0; table_i< nms_box[class_i].counter+1; ++table_i){
            table[table_i] = table_i;
        }
        if(nms_box[class_i].counter >= 0){
            quick_sort(nms_box[class_i].score,0,nms_box[class_i].counter+1,table);
        }
                
        for(int box_i = 0; box_i < nms_box[class_i].counter+1; ++box_i){
        
            if(table[box_i] < 0 || box_i == nms_box[class_i].counter){
                continue;
            } 
            for(int box_rmv = box_i+1; box_rmv < nms_box[class_i].counter+1; ++box_rmv){
                float iou_val = iou(    nms_box[class_i].coord[table[box_i]*4  ],nms_box[class_i].coord[table[box_i]*4+1],
                                        nms_box[class_i].coord[table[box_i]*4+2],nms_box[class_i].coord[table[box_i]*4+3],
                                        nms_box[class_i].coord[table[box_rmv]*4  ],nms_box[class_i].coord[table[box_rmv]*4+1],
                                        nms_box[class_i].coord[table[box_rmv]*4+2],nms_box[class_i].coord[table[box_rmv]*4+3] );
                if(iou_val > iou_threshold) {
                    table[box_rmv] = -1;
                }
            }
        }
        for(int box_i = 0; box_i < nms_box[class_i].counter+1; ++box_i){
            if(table[box_i] >= 0){
                out[next_box * 6    ] = nms_box[class_i].score[box_i];
                out[next_box * 6 + 1] = class_i;
                out[next_box * 6 + 2] = nms_box[class_i].coord[ table[box_i]*4];
                out[next_box * 6 + 3] = nms_box[class_i].coord[ table[box_i]*4 + 1];
                out[next_box * 6 + 4] = nms_box[class_i].coord[ table[box_i]*4 + 2];
                out[next_box * 6 + 5] = nms_box[class_i].coord[ table[box_i]*4 + 3];
                next_box++;
            }
        }

        out[det_h*det_w*det_BOX*det_CLASS*6] = next_box * 1.0;
    }
}
void detection(float *grid_cell,float* box_out,
               int input_width,
               int det_h,
               int det_w,
               int det_BOX,
               int det_CLASS,
               float score_threshold,
               float iou_threshold){

    // float score_threshold = 0.3;
    // float iou_threshold = 0.3;
    // int det_h = 13;
    // int det_w = 13;
    // int det_BOX = 5;
    // int det_CLASS = 20

    //box_coord = 4; //left top(x, y) || right down(x, y)
    float grid_size = (float)(input_width / det_w);
    
    struct Det_boxes dbox[det_CLASS];
    
    // struct Det_boxes{
    // for(int i = 0; i < det_CLASS; ++i){
    //         float *score[det_h * det_w * det_BOX];
    //         int   *coord[det_h * det_w * det_BOX * 4];
    //         int   counter;
    // }
    // }dbox[det_CLASS];

    char class_name[20][100] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
    //float anchor[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
//ANCHOR : yolov2-tiny-voc.cfg
    float anchor[10] =  {1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52};
//ANCHOR : yolov2-voc.cfg
    // float anchor[10] =  {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};
    
    int total_box = 0;
 
    
    for(int i =0; i<det_CLASS; ++i){
        dbox[i].score = (float*)malloc(det_h * det_w * det_BOX * sizeof(float));
        dbox[i].coord = (int*)malloc(det_h * det_w * det_BOX * 4 * sizeof(int));
        dbox[i].counter = -1;
    }
    for(int row = 0; row < det_h; ++row){
        for(int col = 0; col < det_w; ++col){
            for(int box = 0; box < det_BOX; ++box){
                //output.bin : ROW, COL, BOX, CLASS => 13,13,5,"25" <- this section
                // box center x,y / box size width,height / box confidence
                float *grid_map = grid_cell;
                //  printf("detection\n");
                float tx = grid_map[ 0 ];
                float ty = grid_map[ 1 ];
                float tw = grid_map[ 2 ];
                float th = grid_map[ 3 ];
                float tc = grid_map[ 4 ];
                
                
                float cen_x = ((float)col + det_sigmoid(tx))* grid_size;
                float cen_y = ((float)row + det_sigmoid(ty))* grid_size;
               
                float roi_w = expf(tw) * anchor[2*box + 0] * grid_size;
                float roi_h = expf(th) * anchor[2*box + 1] * grid_size;

                float final_confidence = det_sigmoid(tc);
                //printf("%d : %.6f\n",row*W*BOX + col*BOX + box,final_confidence);
                int left   = (int)(cen_x - roi_w/2.);
                int right  = (int)(cen_x + roi_w/2.);
                int top    = (int)(cen_y - roi_h/2.);
                int bottom = (int)(cen_y + roi_h/2.);
                
                grid_cell += det_BOX;
                
                float *classsoft = grid_cell;

                //softmax
                //best class 구분
                int best_class = det_softmax(classsoft, det_CLASS);
                
                //best class score 계산
                float best_class_score = grid_cell[ best_class ];
                
                //if final_confidence
                if((final_confidence *= best_class_score) > score_threshold){
                    
                    total_box++;
                    dbox[best_class].counter += 1;
                    dbox[best_class].score[dbox[best_class].counter ] = final_confidence;
                    
                    dbox[best_class].coord[dbox[best_class].counter * 4     ] = left;
                    dbox[best_class].coord[dbox[best_class].counter * 4 + 1 ] = top;
                    dbox[best_class].coord[dbox[best_class].counter * 4 + 2 ] = right;
                    dbox[best_class].coord[dbox[best_class].counter * 4 + 3 ] = bottom;
                }
                grid_cell += det_CLASS;
            }
        }
    }

    non_maximal_suppression(dbox, box_out,
                            det_h,
                            det_w,
                            det_BOX,
                            det_CLASS,
                            iou_threshold);

    //printf("\nfinal bounding box count =  %d\n",(int)box_out[13*13*5*6]);
}