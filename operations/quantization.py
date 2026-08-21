import numpy as np

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
    def __init__(self, codebook: np.ndarray):
        self.codebook = codebook.astype(np.float32)
        self.K, self.d = codebook.shape

    def quantizer(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        assert X.shape[1] == self.d, f"Expected vector dimension {self.d}, got {X.shape[1]}"

        N = X.shape[0]
        indices = np.empty(N, dtype=np.int32)
        min_distances = np.empty(N, dtype=np.float32)
        C_sq = np.sum(self.codebook ** 2, axis=1)  # (K,)

        # batched, avoids materializing the (N, K, d) diff tensor
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X[start:end]
            X_sq = np.sum(Xb ** 2, axis=1, keepdims=True)  # (b, 1)
            cross = Xb @ self.codebook.T  # (b, K)
            distances = X_sq + C_sq[np.newaxis, :] - 2 * cross
            idx = np.argmin(distances, axis=1)
            indices[start:end] = idx
            min_distances[start:end] = distances[np.arange(end - start), idx]

        return indices, min_distances

    def dequantizer(self, indices: np.ndarray) -> np.ndarray:
        return self.codebook[indices]

class mid_riser:
    def quantize(x_15: np.ndarray, target_bits: int = 8, m: int = 0):
        shift = 16 - target_bits
        rounding_offset = (1 << (shift - 1)) if m == 1 else 0
        return (x_15.astype(np.int32) + rounding_offset) >> shift 

    def dequantize(q: np.ndarray, shift: int, m: int = 0):
        x_hat = q.astype(np.int32) << shift
        half_delta = (1 << (shift - 1)) if m == 0 else 0
        return (x_hat + half_delta).astype(np.int16)