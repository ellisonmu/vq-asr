#include <stdio.h>
#include <stlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE
#define D 80
#define MAX_ITERATIONS 30

__global__ void vector_quantizer_naive(
    const float* X, const float*, 
    int* I, float* D,
    int N, int K, int d
){
    int i = blockIdx.x * blockDim.x + threadIdx.x
    if (i >= N) {
        return;
    }
    float best_dist = INFINITY;
    int best_idx = -1;
    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        for (int j = 0; j < d; j++){
            float diff = C[k * d + j] - X[i * d + j]
            dist += diff * diff
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = k
        }
    }
    I[i] = best_idx;
    D[i] = best_dist;
}
    
__global__ void vector_quantizer_tiled(
    const float* X, const float* C, 
    int* I, float* D,
    int N, int K, int d
){
    __shared__ float C_tile[TILE_SIZE][80]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i >= N)
        return;
    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
        for (int idx = tid; idx < TILE_SIZE * d; idx += blockDim.x){
            int k = idx / d;
            int j = idx % d;
            if (k0 + k < K) {
                C_tile[k][j] = C[(k0 + k)*d + j];
            }
        }
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            if (k0 + k >= K)
                break;
            float dist = 0.0f;
            for (int j = 0; j < d; j++){
                float diff = C_tile[k][j] - X[i * d + j];
                dist += diff * diff;
            }
            if (dist < best_dist) { 
                best_dist = dist;
                best_idx = k0 + k;
            }
        }
        __syncthreads();
    }
    I[i] = best_idx;
    D[i] = best_dist;
}

__global__ void accumulate_clusters(
    const float* X, const int* I,
    int* counts, float* sums,
    int N, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    
    int k = I[i];
    atomicAdd(&counts[k], 1);
    for (int j = 0; j < d; j++) {
        atomicAdd(&sums[k * d + j], X[i * d + j]);
    }
}

__global__ coid update_centroids(
    float *C, const float* sums, 
    const int* counts, 
    int K, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= K * d)
        return;
    
        int k = idx / d;
        int j = idx % d;

        if (counts[k] > 0) {
            C[k * d + j] = sums[k * d + j] / (float)counts[k];
        }

}

int main() {
    const int N = 1000, d = 80, K = 256 // 8 bits

    const int bytes_X = N * d * sizeof(float);
    const int bytes_C = K * d * sizeof(float);
    const int bytes_I = N * sizeof(int);
    const int bytes_D = N * sizeof(float);
    const int bytes_CT = K * sizeof(int);
    const int bytes_SM = K * d * sizeof(float);
    
    float *h_x = (float*)malloc(bytes_X);
    float *h_c = (float*)malloc(bytes_C);
    int *h_i = (int*)malloc(bytes_I);
    float *h_d = (float*)malloc(bytes_D);

    // somehow get input and codebook matricies
    float *dev_x; 
    float *dev_c;
    float *dev_d;
    int *dev_i; 
    int* dev_counts;
    float* dev_sums;

    cudaMalloc(&dev_x, bytes_X);
    cudaMalloc(&dev_c, bytes_C);
    cudaMalloc(&dev_d, bytes_D);
    cudaMalloc(&dev_i, bytes_I);
    cudaMalloc(&dev_counts, bytes_CT);
    cudaMalloc(&dev_sums, bytes_SM);

    cudaMemcpy(dev_x, h_x, bytes_X, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, h_c, bytes_C, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock-1) / threadsPerBlock; 
    int cbElements = K * d;
    int cbBlocks = (cbElements + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        vector_quantizer_tiled<<<blocksPerGrid, threadsPerBlock>>>(
            dev_x, dev_c, dev_i, dev_d,
            N, K, d
        );

        cudaDeviceSychronize();

        cudaMemcpy(h_i, dev_i, bytes_I, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d, dev_d, bytes_D, cudaMemcpyDeviceToHost);

        cudaMemset(dev_counts, 0, K * sizeof(int));
        cudaMemset(dev_sums, 0, K * d * sizeof(float));

        accumulate_clusters<<blocksPerGrid, threadsPerBlock>>>(
            dev_x, dev_i, dev_counts, dev_sums,
            N, d
        );

        update_centroids<<<cbBlocks, threadsPerBlock>>>(
            dev_c, dev_sums, dev_counts,
            K, d
        );

        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_c, dev_c, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_c);
    cudaFree(dev_d);
    cudaFree(dev_i);
    cudaFree(dev_counts);
    cudaFree(dev_sums);

    free(h_x);
    free(h_c);
    free(h_i);
    free(h_d);

    return 0;
}