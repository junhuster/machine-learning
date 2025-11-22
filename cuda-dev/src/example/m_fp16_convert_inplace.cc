#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Core>
#include "../gpu-kernal/fp16.cu.h"
#include "../tool/util.h"
#include "../tool/fp16_cpu.h"

const int VAL_NUM = 300000;
int alloc_bytes = sizeof(float) * VAL_NUM + sizeof(uint64_t);
int main() {
    
    float* fp16_a = (float*)malloc(alloc_bytes);
    float* fp32_b = (float*)malloc(alloc_bytes);
    float* fp32_b_0 = (float*)malloc(alloc_bytes);
    
    MyTime timer;
    timer.start();
    std::cout << "begin construct random float" << std::endl;
    float last_num = 0.123;
    uint16_t* fp16_buffer = (uint16_t*)fp16_a;
    for (int i = 0; i < VAL_NUM; i++) {
       if (i % 50 == 0) {
            last_num = getRandomFloat();
       } else {
            last_num = last_num + (i / 1000) * 0.0003;
            if (last_num > 5.0) {
                last_num = 0.5678;
            }
       }
       if (i == 0) {
            std::cout << "origin fp32_0: " << last_num << std::endl;
       }
       Eigen::half fp16 = eigenFp32ToFp16(last_num);
       *(fp16_buffer + i) = fp16.x;
    }
    std::cout << "construct fp16 cost " << timer.interval() << " us, total val num: " << VAL_NUM  << std::endl;
    
    lanuchFp16ToFp32Cpu((uint16_t*)fp16_a, fp32_b, VAL_NUM);
    std::cout << "fp16_2_fp32 cpu cost: " << timer.interval() << " us" << std::endl;
   
    void* dfp16_a;
    {
        MyTime time0;
        time0.start();
        auto ret = cudaMalloc((void**)&dfp16_a, alloc_bytes);
        //std::cout << "singel cudaMalloc cost: " << time0.interval() << std::endl;
        if (ret != cudaSuccess) {
            std::cout << "cuda alloc dfp16_a failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
        ret = cudaMemcpy(dfp16_a, fp16_a, alloc_bytes, cudaMemcpyHostToDevice); 
        std::cout << "singel cudaMemcpy cost: " << time0.interval() << std::endl;
        if (ret != cudaSuccess) {
            std::cout << "cuda copy dfp16_a failed, msg: " << cudaGetErrorString(ret) << std::endl;
            return -1;
        }
    }
    std::cout << "cuda malloc/memcpy cost: " << timer.interval() << " us" << std::endl;
    lanuchFp16ToFp32GpuInplace((uint16_t*)dfp16_a, VAL_NUM);
    std::cout << "fp16_2_fp32 gpu cost: " << timer.interval() << " us" << std::endl;

    auto ret = cudaMemcpy(fp32_b_0, dfp16_a, alloc_bytes, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
        std::cout << "cuda copy dfp32_b failed, msg: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }
    std::cout << "cuda copy deviceToHost cost us: " << timer.interval() << " us" << std::endl;
    bool ret0 = compare_float_diff(fp32_b, fp32_b_0, VAL_NUM);
    if (ret0) {
        std::cout << "fp16_2_fp32 cpu equal gpu" << std::endl;
    }
    std::cout << "cpu_0: " << *(fp32_b) << ", gpu_0: " << *(fp32_b_0) << std::endl;
    return 0;
}
