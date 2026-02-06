#include<stdio.h>
#include "mat_add.cu.h"

__global__ void matAdd(float* mat_a, float* mat_b, float* mat_c, int row_num, int col_num) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ele_cnt = row_num * col_num;
    int stride = gridDim.x * blockDim.x;
    //printf("matAddGpu: threadid=%d, total_ele_cnt=%d, stride=%d\n", thread_id, total_ele_cnt, stride);
    while (thread_id < total_ele_cnt) {
       float ax = *(mat_a + thread_id);
       float bx = *(mat_b + thread_id);
       *(mat_c + thread_id) = ax + bx;
        thread_id += stride;
        if (thread_id > total_ele_cnt) {
            //printf("matAddGpu: thread_id: %d, total_ele_cnt: %d\n", thread_id, total_ele_cnt);
        }
    }
}

void matAddCpu(float* mat_a, float* mat_b, float* mat_c, int row_num, int col_num) {
    for (int row = 0; row < row_num; row++) {
        for (int col = 0; col < col_num; col++) {
            *(mat_c + row*col_num + col) = *(mat_a + row*col_num + col) + *(mat_b + row*col_num + col);
        }
    }
}

void lanuchMatAddGpu(float* mat_a, float* mat_b, float* mat_c, int row_num, int col_num) {
    int block_size = 512;
    int total_num = row_num * col_num;
    if (total_num > 10000) {
        total_num = 10000;
    }
    int grid_size  = (total_num + block_size - 1) / block_size; 
    matAdd<<<grid_size, block_size>>>(mat_a, mat_b, mat_c, row_num, col_num);
    cudaDeviceSynchronize();
}

void lanuchMatAddCpu(float* mat_a, float* mat_b, float* mat_c, int row_num, int col_num) {
    matAddCpu(mat_a, mat_b, mat_c, row_num, col_num);     
}
