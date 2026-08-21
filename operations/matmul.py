import numpy as np

def matmul_naive(A, B, TILE_SIZE=2):
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        return -1
    C = np.zeros((M, N))

    for row in range(M):
        for col in range(N):
            for k in range(K):
                C[row][col] += A[row][k] * B[k][col]
    return C

def matmul_tiled(A, B, TILE_SIZE=2):
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        return -1
    C = np.zeros((M, N))

    for row in range(M):
        for col in range(N):
            for phase in range(0, K, TILE_SIZE):
                for i in range(TILE_SIZE):
                    k = phase + i
                    if k < K:
                        C[row][col] += A[row][k] * B[k][col]
    return C