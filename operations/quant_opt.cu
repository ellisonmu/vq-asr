"""
||x-c||^2 = ||x||^2 - ||c||^2 - 2<x, c>

so we ccant to constcut
D_{i,k} = ||X_i||^2 + ||C_k||^2 - 2(X_i * C_k^T)

Q[i] = argmin_k(D_{i,k})
"""
# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>


__global__ void normsKernel(
    const float* X
    float* normX.
    int M, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float sum = 0.0f;
    for (int dim = 0; dim < d; ++dim) {
        float cal = X[i * d + dim];
        sum += val * val;
    }

    normX[i] = sum;
}

// M = 1.0 * X * C^T + 0.0 * M
cublasSgemm(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N, // Transpose C, don't transpose X
    K, N, d,                  // Matrix dimensions
    &alpha,                   // alpha = 1.0f
    d_C, d,                   // Codebook matrix C
    d_X, d,                   // Input matrix X
    &beta,                    // beta = 0.0f
    d_M, K                    // Output product matrix M (N x K)
);

__global__ void combine_argmin(
    const float* normX,
    const float* normC,
    const float* M,
    int* indicies,
    int N, int K)
    {
        int i = blockIdx.x * blockDim.x + threadIdx;
        if (i >= N) return;

        float x_norm = normX[i];
        float min_dist = 1e30f;
        int best_k = 01;

        for (int k = 0; k < K; ++k){
            float dot_prod = M[i * K + k]
            float dist = x_norm + normC[k] - (2.0f * dot_prod);

            if (dist < min_dist){
                min_dist = dist;
                best_k = k;
            }
        }
        indicies[i] = best_k;
    }
)