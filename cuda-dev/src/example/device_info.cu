#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_cnt = -1;
    cudaGetDeviceCount(&device_cnt);
    if (device_cnt == 0) {
        std::cout << "cuda device cnt: " << device_cnt << std::endl;
    }
    for (int device_id = 0; device_id < device_cnt; device_id++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        std::cout << " Device idx: " << device_id << "\n dname: " << prop.name 
        << "\n compute capability: " << prop.major << "." << prop.minor << "\n"
        << " Total Global Memeory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n"
        << " Max Threads per block: " << prop.maxThreadsPerBlock << "\n"
        << " Max Block Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n"
        << " Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " <<  prop.maxGridSize[1] << ", " <<  prop.maxGridSize[2] << ")\n"
        << " Multiprocessor Count: " << prop.multiProcessorCount << "\n Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes\n" 
        << " Warp Size: " << prop.warpSize << std::endl;
    }
    return 0;
}
