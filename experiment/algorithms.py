import numpy as np
import torch
from tqdm import tqdm
from operations.quantization import vector_quantizer, get_device

def kmeans(X, K, d=80, eps=1e-5, MAX_ITERATIONS=100, device=None):
    device = device or get_device()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)

    # lets assume that speech
    C = torch.empty(K, d, device=device).uniform_(float(X_t.min()), float(X_t.max()))
    vq = vector_quantizer(C, device=device)
    I, D = vq.quantizer(X_t, return_tensor=True)
    error_prev = float("inf")
    error_curr = D.mean().item()
    m = 0
    pbar = tqdm(total=MAX_ITERATIONS, desc=f"kmeans K={K} ({device})")
    while m < MAX_ITERATIONS:
        counts = torch.bincount(I, minlength=K).float()  # (K,)
        sums = torch.zeros(K, d, device=device, dtype=torch.float32).index_add_(0, I, X_t)
        nonempty = counts > 0
        C[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        # empty clusters keep their previous centroid

        vq = vector_quantizer(C, device=device)
        I, D = vq.quantizer(X_t, return_tensor=True)
        error_prev = error_curr
        error_curr = D.mean().item()
        delta = abs(error_curr - error_prev)
        pbar.update(1)
        if delta < eps:
            break
        m += 1
    pbar.close()
    return C.cpu().numpy()


# def gumbel_softmax():

# def gmm():

# def deeplearn():
