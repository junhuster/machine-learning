#pragma once
#include <iostream>

void lanuchWarmup(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output);
void lanuchReduceSumBase(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output);
void lanuchReduceSumBase2(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);
void lanuchReduceSumShamem(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);
void lanuchReduceSumWarpOpt(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);
void lanuchReduceSumBankOpt(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);
void lanuchReduceSumBankOpt8byte(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);
void lanuchReduceSumIdleOpt(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);
void lanuchReduceSumRollup(uint64_t* input_buffer, int val_num, int block_size, uint64_t* block_output, const ::cudaStream_t stream = 0);




