#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx < N) {

        // Load input once into a register
        float x = input[idx];

        // Calculate sigmoid
        float sigmoid = 1.0f / (1.0f + expf(-x));

        // Calculate Swish
        output[idx] = x * sigmoid;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
