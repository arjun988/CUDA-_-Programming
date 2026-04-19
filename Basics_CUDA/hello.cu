#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello from GPU!\n");
}

int main() {
    helloCUDA<<<1,1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    return 0;
}