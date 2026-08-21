#include <stdio.h>
#include <cuda_runtime.h>

__global__ void quant_naive(
    float*Q, // quantization output
    const float* X, // input (N, d)
    const float* C, // codebook (K, d)
    int* indicies, // (N,)
    int N, int K, int d){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float min_dist = 1e30f;
        int best_k = -1;

        for (int k = 0; k < K; ++k) {
            float dist = 0.0f;

            for (int dim = 0; dim < d; ++dim) {
                float diff = X[i * d + dim] - C[k * d + dim];
                dist += diff * diff
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_k = k;
            }
        }
    indicies[i] = best_k
}

int main() {
    const int N = 1000000, d_bytes = N * sizeof(int);
    const int C = 256, c_bytes = C * sizeof(int);


    int *h_X = (int*)malloc(d_bytes);
    int *h_Q = (int*)malloc(d_bytes);
    int *h_C = (int*)malloc(c_bytes);
    
    for (int i = 0; i < N; i++) { 
        h_X[i] = i / 4;
    }
    for (int i = 0; i < K; i++) { 
        h_C[i] = i * 10000;
    }

    int *dev_X, *dev_C, *dev_Q;

    cudaMalloc(&dev_X, d_bytes)
    cudaMalloc(&dev_C, c_bytes)
    cudaMalloc(&dev_Q, d_bytes)

    cudaMemcpy(dev_X, h_X, d_bytes, cudaMemcpyHostToDevices);
    cudaMemcpy(dev_C, h_C, c_bytes, cudaMemcpyHostToDevices);

    int threadsPerBlock = 256, blocksPerGrid = (size + 255) / 256;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);

    cudaMemcpy(h_Q, dev_Q, d_bytes, cudaMemcpyDeviceToHost);

    printf("Quantization of %d elements\n", N)
    printf("Verification: Q[0]=%d, c[999999]=%d\n", h_Q[0], h_Q[999999])
}