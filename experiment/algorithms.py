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

def gmmem(X, K, d=80, eps=1e-5, MAX_ITERATIONS=100, device=None, batch_size=16384):
    device = device or get_device()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    N = X_t.shape[0]

    idx = torch.randperm(N, device=device)[:K]
    means = X_t[idx].clone()
    variance = torch.ones(K, d, device=device)
    weights = torch.ones(K, device=device) / K
    ll_prev = float("-inf")
    m = 0
    log_2pi_d = d * np.log(2 * np.pi)
    pbar = tqdm(total=MAX_ITERATIONS, desc=f"gmmem K={K} ({device})")
    while m < MAX_ITERATIONS:
        # diagonal-covariance Mahalanobis distance, expanded to avoid an (N, K, d) tensor:
        # sum_d (x_d - mu_kd)^2 / var_kd = x^2 @ (1/var)^T - 2 x @ (mu/var)^T + sum_d mu_kd^2/var_kd
        inv_var = 1.0 / variance                                    # (K, d)
        mu_invvar = means * inv_var                                 # (K, d)
        mu2_invvar_sum = (means ** 2 * inv_var).sum(dim=1)          # (K,)
        log_var_sum = torch.log(variance).sum(dim=1)                # (K,)
        log_weights = torch.log(weights)

        ll_curr = 0.0
        Nk = torch.zeros(K, device=device)
        sum_x = torch.zeros(K, d, device=device)
        sum_x2 = torch.zeros(K, d, device=device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X_t[start:end]      # (b, d)
            Xb2 = Xb ** 2
            mahal = Xb2 @ inv_var.T - 2 * (Xb @ mu_invvar.T) + mu2_invvar_sum[None, :]  # (b, K)
            log_gaussian = -0.5 * (mahal + log_var_sum[None, :] + log_2pi_d)

            log_numerator = log_weights[None, :] + log_gaussian
            log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)
            R = torch.exp(log_numerator - log_denominator)  # (b, K)
            ll_curr += log_denominator.sum().item()

            Nk += R.sum(dim=0)
            sum_x += R.T @ Xb
            sum_x2 += R.T @ Xb2

        # Maximization
        alive = Nk > eps

        means_new = means.clone()
        means_new[alive] = sum_x[alive] / Nk[alive].unsqueeze(1)

        # Var[x] = E[x^2] - E[x]^2, computed from the batched sufficient statistics above
        variance_new = variance.clone()
        variance_new[alive] = sum_x2[alive] / Nk[alive].unsqueeze(1) - means_new[alive] ** 2
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
