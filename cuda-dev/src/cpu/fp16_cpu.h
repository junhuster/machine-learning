#pragma once
#include <iostream>
#include <eigen3/Eigen/Core>

void lanuchFp16ToFp32Cpu(uint16_t* fp16_buffer, float* fp32_buffer, int val_num);
float eigenFp16ToFp32(uint16_t fp16);
Eigen::half eigenFp32ToFp16(float fp32);
