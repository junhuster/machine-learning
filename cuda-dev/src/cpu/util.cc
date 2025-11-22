#include <iostream>
#include <random>
#include <algorithm>
#include "util.h"

const int precison = 6;
bool compare_float_diff(float* left, float* right, int val_num) {
    int diff_num = 0;
    float delta = 1.0 / std::pow(10, precison);
    for (int i = 0; i < val_num; i++) {
        if (std::abs(*(left + i) - *(right + i)) > delta) {
            diff_num += 1;
        }
    }
    if (diff_num == 0) {
        return true;
    }
    std::cout << "compare float diff, total_ele_cnt: " << val_num << ", diff_num: " << diff_num << ", rate: " << diff_num / (val_num * 1.0) << std::endl;
    return false;
}

bool compare_mat_diff(float* mat_a, float* mat_b, int row_num, int col_num) {
    int total_ele_cnt = row_num * col_num;
    int diff_num = 0;
    float delta = 1.0 / std::pow(10, precison);
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            float a = *(mat_a + i * col_num + j);
            float b = *(mat_b + i * col_num + j);
            if (std::abs(a - b) > delta) {
                diff_num += 1;
            }
        }
    }
    std::cout << "compare mat diff, total_ele_cnt: " << total_ele_cnt << ", diff_num: " << diff_num << ", rate: " << diff_num / (total_ele_cnt * 1.0) << std::endl;
    if (diff_num == 0) {
        return true;
    }
    return false;
}

float getRandomFloat() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    float random_float = dis(gen);
    return random_float;
}

uint64_t getRandomInt() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, 255);
    return dis(gen);
}

void constructInts(uint64_t* output_buffer, int val_num) {
    uint64_t last_num = 0;
    for (int i = 0; i < val_num; i++) {
        if (i % 500 == 0) {
            last_num = getRandomInt();
        } else {
            last_num = last_num + i;
        }
        if (last_num > 255) last_num = last_num % 255;
        *(output_buffer + i) = last_num;
    }
}

void getDeviceProp() {
    int device_cnt = -1;
    cudaGetDeviceCount(&device_cnt);
    if (device_cnt == 0) {
        std::cout << "cuda device cnt: " << device_cnt << std::endl;
    }
    for (int device_id = 0; device_id < device_cnt; device_id++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        std::cout << " device idx: " << device_id << "\n"
                    << " device name: " << prop.name << "\n"
                    << " compute capability: " << prop.major << "." << prop.minor << "\n"
                    << " Total Global Memeory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n"
                    << " Shared Memory Per Block: " << prop.sharedMemPerBlock << " Bytes\n"
                    << " regs per block: " << prop.regsPerBlock << "\n"
                    << " warp size: " << prop.warpSize << "\n"
                    << " totalConstMem: " << prop.totalConstMem << "\n"
                    << " Max Threads per block: " << prop.maxThreadsPerBlock << "\n"
                    << " Max Block Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n"
                    << " Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " <<  prop.maxGridSize[1] << ", " <<  prop.maxGridSize[2] << ")\n"
                    << " Multiprocessor Count: " << prop.multiProcessorCount << "\n" 
                    << " concurrentKernels: " << prop.concurrentKernels << "\n"
                    << " maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << "\n"
                    << " maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl << "\n";
    }
}
