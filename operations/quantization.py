import numpy as np
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class scalar:
    def quantize(x: np.ndarray):
        """int8 = [127 / (max - min)] * (float32 - min )
        float32 = [(max - min) / 127] * int8 + min
        these are lossy transformations """

        x_min = np.min(x).item()
        x_max = np.max(x).item()

        if x_max == x_min:
            return np.zeros_like(x, dtype=np.int8), 1.0, x_min

        scale = 127 / (x_max - x_min)

        q = np.round((x - x_min) * scale)
        q = np.clip(q, 0, 127).astype(np.int8)

        return q, scale, x_min

    def dequantize(x: np.ndarray, scale: float, x_min: float) -> np.ndarray:
        return (x.astype(np.float32) / scale) + x_min

class vector_quantizer:
    def __init__(self, codebook, device=None):
        self.device = device or get_device()
        if isinstance(codebook, torch.Tensor):
            self.codebook = codebook.to(self.device, dtype=torch.float32)
        else:
            self.codebook = torch.as_tensor(np.asarray(codebook), dtype=torch.float32, device=self.device)
        self.K, self.d = self.codebook.shape

    def quantizer(self, X, batch_size: int = 16384, return_tensor: bool = False):
        X_t = X.to(self.device, dtype=torch.float32) if isinstance(X, torch.Tensor) \
            else torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        assert X_t.shape[1] == self.d, f"Expected vector dimension {self.d}, got {X_t.shape[1]}"

        N = X_t.shape[0]
        indices = torch.empty(N, dtype=torch.int64, device=self.device)
        min_distances = torch.empty(N, dtype=torch.float32, device=self.device)
        C_sq = (self.codebook ** 2).sum(dim=1)  # (K,)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X_t[start:end]
            X_sq = (Xb ** 2).sum(dim=1, keepdim=True)  # (b, 1)
            cross = Xb @ self.codebook.T  # (b, K)
            distances = X_sq + C_sq.unsqueeze(0) - 2 * cross
            idx = torch.argmin(distances, dim=1)
            indices[start:end] = idx
            min_distances[start:end] = distances.gather(1, idx.unsqueeze(1)).squeeze(1)

        if return_tensor:
            return indices, min_distances
        return indices.cpu().numpy().astype(np.int32), min_distances.cpu().numpy()

    def dequantizer(self, indices):
        idx_t = indices.to(self.device, dtype=torch.long) if isinstance(indices, torch.Tensor) \
            else torch.as_tensor(np.asarray(indices), dtype=torch.long, device=self.device)
        out = self.codebook[idx_t]
        return out if isinstance(indices, torch.Tensor) else out.cpu().numpy()

class mid_riser:
    def quantize(x_15: np.ndarray, target_bits: int = 8, m: int = 0):
        shift = 16 - target_bits
        rounding_offset = (1 << (shift - 1)) if m == 1 else 0
        return (x_15.astype(np.int32) + rounding_offset) >> shift 

    def dequantize(q: np.ndarray, shift: int, m: int = 0):
        x_hat = q.astype(np.int32) << shift
        half_delta = (1 << (shift - 1)) if m == 0 else 0
        return (x_hat + half_delta).astype(np.int16)