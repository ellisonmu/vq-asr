#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float), bytes_B = K * N * sizeof(float), bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A), *h_B = (float*)malloc(bytes_B), *h_C = (float*)malloc(bytes_C);
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A); cudaMalloc(&d_B, bytes_B); cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16), numBlocks((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K);
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    printf("Naive Matrix Mult: %dx%dx%d\n", M, K, N);
    printf("Verification: C[0] = %.1f\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}