import numpy as np
from operations.quantization import vector_quantizer

def kmeans(X, K, d=80, eps=1e-5, MAX_ITERATIONS=100):
    # lets assume that speech
    C = np.random.uniform(low=X.min(), high=X.max(), size=(K, d))
    vq = vector_quantizer(C)
    I, D = vq.quantizer(X)
    error_prev = np.inf
    error_curr = np.mean(D)
    delta = abs(error_curr-error_prev)
    m = 0
    while m < MAX_ITERATIONS:
        for k in range(K):
            mask = I == k
            if np.any(mask):
                C[k] = X[mask].mean(axis=0)
            # else: no points assigned this iteration, keep previous centroid
        vq = vector_quantizer(C)
        I, D = vq.quantizer(X)
        error_prev = error_curr
        error_curr = np.mean(D)
        delta = abs(error_curr - error_prev)
        if delta < eps:
            break
        m += 1
    return C


# def gumbel_softmax():

# def gmm():

# def deeplearn():