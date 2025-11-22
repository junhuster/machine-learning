#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Core>
#include "../gpu-kernal/reduce_sum.cu.h"
#include "../tool/util.h"

const int VAL_NUM = 1 << 24;
const int BLOCK_SIZE = 512;

int alloc_bytes = sizeof(uint64_t) * VAL_NUM + sizeof(uint64_t);
void reduceSumCpu(uint64_t* input_buffer, int val_num, uint64_t& output);
int main(int argc, char** argv) {
    std::cout << "-------------------before prework-----------" << std::endl;
    if (argc != 2) {
        std::cout << "argc cnt: " << argc << " is invalid" << std::endl;
        return -1;
    }

    std::string kindstr = argv[1];
    int kind = std::atoi(kindstr.c_str());
    std::cout << "cmd argv kind:" << kind << std::endl;
    
    uint64_t* h_input = (uint64_t*)malloc(alloc_bytes);
    MyTime timer;
    timer.start();
    constructInts(h_input, VAL_NUM);
    std::cout << "construct ints cost " << timer.interval() << " us, total val num: " << VAL_NUM  << std::endl;
    
    uint64_t output_cpu = 0;
    uint64_t* d_input;
    uint64_t* block_output;
    
    int grid_size = (VAL_NUM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t* block_output_cpu = (uint64_t*)malloc(grid_size * sizeof(uint64_t));
    {
        auto ret = cudaMalloc((uint64_t**)&d_input, alloc_bytes);
        if (ret != cudaSuccess) {
            std::cout << "cuda malloc d_input failed, msg: " << cudaGetErrorString(ret) << std::endl;
        }
        std::cout << "cuda malloc d_input " << alloc_bytes << " bytes cost: " << timer.interval() << "us" << std::endl;
        ret = cudaMalloc(&block_output, grid_size * sizeof(uint64_t));
        if (ret != cudaSuccess) {
            std::cout << "cuda malloc d_input failed, msg: " << cudaGetErrorString(ret) << std::endl;
        }
        std::cout << "cuda malloc block_output " << grid_size * sizeof(uint64_t) << " bytes cost: " << timer.interval() << "us" << std::endl;
    }

    {
        auto ret = cudaMemcpy(d_input, h_input, alloc_bytes, cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) {
            std::cout << "cuda memcpy d_input failed, msg: " << cudaGetErrorString(ret) << std::endl;
        }
        std::cout << "cuda memcpy " << alloc_bytes << " bytes cost: " << timer.interval() << "us\n" << std::endl;
    }
    std::cout << "-------------------after prework-----------" << std::endl;
    std::cout << "\n*****************begin compute***********\n" << std::endl;
    uint64_t cpu_cost = 0;
    {
        reduceSumCpu(h_input, VAL_NUM, output_cpu);
        cpu_cost = timer.interval();
    }
    {
        //lanuchWarmup(d_input, VAL_NUM, BLOCK_SIZE, block_output);
        //std::cout << "warmup cost:" << timer.interval() << " us" << std::endl;
    }
    timer.stop();
    timer.start();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (kind == 0) {
        lanuchReduceSumBase(d_input, VAL_NUM, BLOCK_SIZE , block_output);
        uint64_t tm1 = timer.interval();
        cudaMemcpy(block_output_cpu, block_output, sizeof(uint64_t) * grid_size, cudaMemcpyDeviceToHost);
        uint64_t output_gpu = 0;
        for (int i = 0; i < grid_size; i++) {
            output_gpu += *(block_output_cpu + i);
        }
        int tm2 = timer.interval();
        std::cout << "launchReduceSumBase, cuda kernal res:" << output_gpu << ", cpu res:" << output_cpu;
        if (output_cpu == output_gpu) {
            std::cout << ", gpu res eq cpu" << std::endl;;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        std::cout << "launchReduceSumBase2, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;
    } else if (kind == 1) {
        lanuchReduceSumBase2(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "launchReduceSumBase2, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "launchReduceSumBase2, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;
    } else if (kind == 2) {
        lanuchReduceSumShamem(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "lanuchReduceSumShamem, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "lanuchReduceSumShamem, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;

    } else if (kind == 3) {
        lanuchReduceSumWarpOpt(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "lanuchReduceSumWarpOpt, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "lanuchReduceSumWarpOpt, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;

    } else if (kind == 4) {
        lanuchReduceSumBankOpt(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "lanuchReduceSumBankOpt, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "lanuchReduceSumBankOpt, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;
    } else if (kind == 5) {
        lanuchReduceSumBankOpt8byte(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "lanuchReduceSumBankOpt8byte, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "lanuchReduceSumBankOpt8byte, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;

    } else if (kind == 6) {
        lanuchReduceSumIdleOpt(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "lanuchReduceSumIdleOpt, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "lanuchReduceSumIdleOpt, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;

    } else if (kind == 7) {
        lanuchReduceSumRollup(d_input, VAL_NUM, BLOCK_SIZE, block_output, stream);
        cudaStreamSynchronize(stream);
        uint64_t tm1 = timer.interval();
        uint64_t gpu_res;
        cudaMemcpy(&gpu_res, block_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "lanuchReduceSumRollup, cuda kernal res:" << gpu_res << ", cpu res:" << output_cpu;
        if (output_cpu == gpu_res) {
            std::cout << ", gpu res eq cpu" << std::endl;
        } else {
            std::cout << ", gpu res not eq cpu" << std::endl;
        }
        uint64_t tm2 = timer.interval();
        std::cout << "lanuchReduceSumRollup, gpu cost:(" << tm1 + tm2 << "," << tm1 << "," << tm2 << "), cpu_cost:" << cpu_cost << std::endl;

    }
    std::cout << "\n*****************after compute***********\n" << std::endl;
    return 0;
}

void reduceSumCpu(uint64_t* input_buffer, int val_num, uint64_t& output) {
    output = 0;
    for (int i = 0; i < val_num; i++) {
        output += *(input_buffer + i);
    }
}
