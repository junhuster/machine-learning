#include<stdio.h>
#include "fp16_cpu.h"

float eigenFp16ToFp32(uint16_t fp16) {
    Eigen::half half = *(Eigen::half*)(&fp16);
    return Eigen::half_impl::half_to_float(half);
}

Eigen::half eigenFp32ToFp16(float fp32) {
    return Eigen::half_impl::float_to_half_rtne(fp32);
}

void fp16ToFp32Cpu(uint16_t* fp16_buffer, float* fp32_buffer, int val_num) {
    #pragma unroll
    for (int i = 0; i < val_num; i++) {
        *(fp32_buffer + i) = eigenFp16ToFp32(*(fp16_buffer + i));
    }
}

void lanuchFp16ToFp32Cpu(uint16_t* fp16_buffer, float* fp32_buffer, int val_num) {
    fp16ToFp32Cpu(fp16_buffer, fp32_buffer, val_num);
}
