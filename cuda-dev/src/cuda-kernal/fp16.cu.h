#pragma once
#include <iostream>

void lanuchFp16ToFp32Gpu(uint16_t* fp16_buffer, float* fp32_buffer, int val_num);
int lanuchFp16ToFp32GpuInplace(uint16_t* fp16_buffer, int val_num);
