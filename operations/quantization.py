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
        return (q.astype(np.float32) / scale) + x_min

class vector_quantizer:
    def __init__(self, codebook: np.ndarray):
        self.codebook = codebook.astype(np.float32)
        self.K, self.d = codebook.shape

    def quantizer(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == self.d, f"Expected vector dimension {self.d}, got {X.shape[1]}"

        # C: (1, K, d)
        diff = self.codebook[np.newaxis, :, :] - X[:, np.newaxis, :] 
        distances = np.sum(diff ** 2, axis=2)

        indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        return indices.astype(np.int32), min_distances

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