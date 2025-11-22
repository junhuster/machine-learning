#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include "../gpu-kernal/mat_add.cu.h"
#include "../tool/util.h"

const int ROW_NUM = 2000;
const int COL_NUM = 8000;
int alloc_bytes = sizeof(float) * ROW_NUM * COL_NUM + sizeof(uint64_t);
int main() {
    
    float* mat_a = (float*)malloc(alloc_bytes);
    float* mat_b = (float*)malloc(alloc_bytes);
    float* mat_c = (float*)malloc(alloc_bytes);
    float* mat_c_0 = (float*)malloc(alloc_bytes);
    
    MyTime timer;
    timer.start();
    memset(mat_c, 0, alloc_bytes);
    memset(mat_c_0, 0, alloc_bytes);
    std::cout << "begin construct random matrix" << std::endl;
    float last_num = 0.123;
    for (int i = 0; i < ROW_NUM; i++) {
        for (int j = 0; j < COL_NUM; j++) {
            if (i % 50 == 0 || j % 50 == 0) {
                last_num = getRandomFloat();
                *(mat_a + i * COL_NUM + j) = last_num;
                *(mat_b + i * COL_NUM + j) = last_num + 0.003;
            } else {
                *(mat_a + i * COL_NUM + j) = last_num + i * j * 0.0003;
                *(mat_b + i * COL_NUM + j) = last_num + i * j * 0.000001;
            }
        }
    }
    std::cout << "construct mat_a mat_b cost " << timer.interval() << " us, RowNum: " << ROW_NUM << ", ColNum: " << COL_NUM << std::endl;

    lanuchMatAddCpu(mat_a, mat_b, mat_c, ROW_NUM, COL_NUM);
    std::cout << "mat_add cpu cost: " << timer.interval() << " us" << std::endl;
   
    void* dmat_a;
    void* dmat_b;
    void* dmat_c;
    {
        MyTime time0;
        time0.start();
        auto ret = cudaMalloc((void**)&dmat_a, alloc_bytes);
        //std::cout << "singel cudaMalloc cost: " << time0.interval() << std::endl;
        if (ret != cudaSuccess) {
            std::cout << "cuda alloc dmat_a failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
        ret = cudaMemcpy(dmat_a, mat_a, alloc_bytes, cudaMemcpyHostToDevice); 
        //std::cout << "singel cudaMemcpy cost: " << time0.interval() << std::endl;
        if (ret != cudaSuccess) {
            std::cout << "cuda copy dmat_a failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
    }
    {
        MyTime timer0;
        timer0.start();
        auto ret = cudaMalloc((void**)&dmat_b, alloc_bytes);
        //std::cout << "single2 cudaMalloc cost: " << timer0.interval() << " us" << std::endl;
        if (ret != cudaSuccess) {
            std::cout << "cuda alloc dmat_b failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
        ret = cudaMemcpy(dmat_b, mat_b, alloc_bytes, cudaMemcpyHostToDevice); 
        //std::cout << "single2 cudaMemcpy cost: " << timer0.interval() << " us" << std::endl;
        if (ret != cudaSuccess) {
            std::cout << "cuda copy dmat_b failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
    }
    {
        auto ret = cudaMalloc((void**)&dmat_c, alloc_bytes);
        if (ret != cudaSuccess) {
            std::cout << "cuda alloc dmat_b failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
    }
    std::cout << "cuda malloc/memcpy cost: " << timer.interval() << " us" << std::endl;
    lanuchMatAddGpu((float*)dmat_a, (float*)dmat_b, (float*)dmat_c, ROW_NUM, COL_NUM);
    std::cout << "mat_add gpu cost: " << timer.interval() << " us" << std::endl;

    auto ret = cudaMemcpy(mat_c_0, dmat_c, alloc_bytes, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
        std::cout << "cuda copy dmat_c failed, msg: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }
    std::cout << "cuda copy deviceToHost cost us: " << timer.interval() << " us" << std::endl;
    bool ret0 = compare_mat_diff(mat_c, mat_c_0, ROW_NUM, COL_NUM);
    if (ret0) {
        std::cout << "matadd_cpu equal matadd_gpu" << std::endl;
    }
    std::cout << "mat_a_00: " << *mat_a << ", mat_b_00: " << *mat_b << ", mat_c_00: " << *mat_c << ", g_matc_00: " << *mat_c_0 << std::endl;
    return 0;
}
