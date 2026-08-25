import numpy as np
import torch
from tqdm import tqdm
from operations.quantization import vector_quantizer, get_device

def kmeans(X, K, d=80, eps=1e-5, MAX_ITERATIONS=100, device=None):
    device = device or get_device()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)

    #bad random intialization for speech. 
    # x_min, x_max = X_t.min(dim=0).values, X_t.max(dim=0).values 
    # C = torch.rand(K, d, device=device) * (x_max - x_min) + x_min

    #better initialization for speech properties
    idx = torch.randperm(X_t.shape[0], device=device)[:K]
    C = X_t[idx].clone()
    
    vq = vector_quantizer(C, device=device)
    I, D = vq.quantizer(X_t, return_tensor=True)
    error_prev = float("inf")
    error_curr = D.mean().item()
    m = 0
    pbar = tqdm(total=MAX_ITERATIONS, desc=f"kmeans K={K} ({device})")
    while m < MAX_ITERATIONS:
        counts = torch.bincount(I, minlength=K).float()  # (K,)
        sums = torch.zeros(K, d, device=device, dtype=torch.float32).index_add_(0, I, X_t)
        nonempty = counts > 0 # empty clusters keep their previous centroid
        C[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1) 
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

def gmmem(X, K, d=80, eps=1e-5, MAX_ITERATIONS=100, device=None):
    device = device or get_device()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    N = X_t.shape[0]

    idx = torch.randperm(N, device=device)[:K]
    means = X_t[idx].clone()
    variance = torch.ones(K, d, device=device)
    weights = torch.ones(K, device=device) / K
    ll_prev = float("-inf")
    m = 0
    diff = X_t[:, None, :] - means[None, :, :]  # x_n - mu_k, (N, K, d)
    pbar = tqdm(total=MAX_ITERATIONS, desc=f"gmmem K={K} ({device})")
    while m < MAX_ITERATIONS:
        log_gaussian = -0.5 * torch.sum(diff**2 / variance[None] + torch.log(variance[None]), dim=2)
        log_gaussian -= 0.5 * d * np.log(2 * np.pi)
        assert log_gaussian.shape == (N, K), "incorrect gaussian matrix dimensions"

        log_numerator = torch.log(weights)[None, :] + log_gaussian
        log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)
        R = torch.exp(log_numerator - log_denominator)
        ll_curr = log_denominator.sum().item()

        # Maximization
        Nk = R.sum(dim=0)
        alive = Nk > eps

        means_new = means.clone()
        means_new[alive] = (R[:, alive].T @ X_t) / Nk[alive].unsqueeze(1)

        diff = X_t[:, None, :] - means_new[None, :, :]
        weighted_s_diff = R[:, :, None] * diff**2
        variance_new = variance.clone()
        variance_new[alive] = weighted_s_diff[:, alive, :].sum(dim=0) / Nk[alive].unsqueeze(1)
        variance_new = torch.clamp(variance_new, min=1e-6)

        weights_new = weights.clone()
        weights_new[alive] = Nk[alive] / N

        means, variance, weights = means_new, variance_new, weights_new

        delta = abs(ll_curr - ll_prev)
        pbar.update(1)
        if delta < eps:
            break
        ll_prev = ll_curr
        m += 1
    pbar.close()
    return means.cpu().numpy() # this is C
