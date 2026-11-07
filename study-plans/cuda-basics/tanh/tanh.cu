#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    // Calculate the global index of this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't access outside the array
    if (idx < N) {
        output[idx] = tanhf(input[idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}