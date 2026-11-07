#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Only one thread performs the complete calculation.
    if (blockIdx.x == 0 && threadIdx.x == 0) {

        // Step 1: Find maximum value
        float max_val = input[0];

        for (int i = 1; i < N; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        // Step 2: Calculate sum of exponentials
        float sum = 0.0f;

        for (int i = 0; i < N; i++) {
            sum += expf(input[i] - max_val);
        }

        // Step 3: Normalize
        for (int i = 0; i < N; i++) {
            output[i] = expf(input[i] - max_val) / sum;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}