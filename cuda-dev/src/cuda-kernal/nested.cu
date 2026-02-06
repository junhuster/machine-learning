#include "nested.cu.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nested(int threads_num, int depth) {
    if (depth == 0 || threads_num == 0) return;
    printf("Depth:%d, Hello World from Tid:%d,BlockId:%d\n", depth, threadIdx.x, blockIdx.x);
    int tid_num = blockDim.x / 2;
    if (tid_num > 0 && threadIdx.x == 0) {
         nested<<<1, tid_num>>>(tid_num, --depth);
    }
}

void lanuchNested() {
    nested<<<1, 8>>>(8, 4);
    cudaDeviceSynchronize();
}