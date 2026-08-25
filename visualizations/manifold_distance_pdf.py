import numpy as np
import torch
import matplotlib.pyplot as plt

from operations.quantization import get_device
from visualizations.spec_prob import load_frame_vectors

N_ANCHORS = 1000     # M: number of anchor points per set
N_BINS = 100

def sample_data_anchors(X, m, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=m, replace=False)
    return X[idx], idx

def sample_box_anchors(X, m, seed=0):
    # per-dimension uniform, within the real per-dim min/max (the "fixed" init from algorithms.py)
    rng = np.random.default_rng(seed)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    return rng.uniform(x_min, x_max, size=(m, X.shape[1])).astype(np.float32)

def pairwise_distances(anchors, X, device, exclude_idx=None, batch_size=256):
    """
    For each anchor, distance to every point in X. Returns (M, N) numpy array.
    exclude_idx: if the anchors ARE rows of X (self-comparison), the index of each
    anchor within X, so we can drop the zero self-distance.
    """
    A = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    out = torch.empty(A.shape[0], X_t.shape[0], device=device)
    for start in range(0, A.shape[0], batch_size):
        end = min(start + batch_size, A.shape[0])
        out[start:end] = torch.cdist(A[start:end], X_t)
    dists = out.cpu().numpy()
    if exclude_idx is not None:
        dists[np.arange(len(exclude_idx)), exclude_idx] = np.nan
    return dists

def average_pdf(dists, bins, value_range):
    """Histogram each anchor's distance row separately, then average the histograms."""
    hists = []
    for row in dists:
        row = row[~np.isnan(row)]
        h, edges = np.histogram(row, bins=bins, range=value_range, density=True)
        hists.append(h)
    return np.mean(hists, axis=0), edges

def correlation_sum(dists, radii):
    """C(r) = fraction of anchor-to-X distances that fall within radius r, for each r."""
    valid = ~np.isnan(dists)
    n_valid = valid.sum()
    flat = dists[valid]
    return np.array([(flat < r).sum() / n_valid for r in radii])

def main():
    device = get_device()
    X = load_frame_vectors()

    data_anchors, data_idx = sample_data_anchors(X, N_ANCHORS)
    box_anchors = sample_box_anchors(X, N_ANCHORS)

    data_dists = pairwise_distances(data_anchors, X, device, exclude_idx=data_idx)
    box_dists = pairwise_distances(box_anchors, X, device)

    all_finite = np.concatenate([data_dists[~np.isnan(data_dists)], box_dists.ravel()])
    value_range = (0.0, np.percentile(all_finite, 99.5))

    data_pdf, edges = average_pdf(data_dists, N_BINS, value_range)
    box_pdf, _ = average_pdf(box_dists, N_BINS, value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radii = np.logspace(np.log10(max(value_range[1] * 1e-3, 1e-6)), np.log10(value_range[1]), 50)
    data_C = correlation_sum(data_dists, radii)
    box_C = correlation_sum(box_dists, radii)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(centers, data_pdf, label="real data anchors", color="tab:blue")
    ax1.plot(centers, box_pdf, label="per-dim uniform box anchors", color="tab:orange")
    ax1.set_xlabel("distance to dataset point")
    ax1.set_ylabel("average density")
    ax1.set_title("Averaged pairwise-distance PDF")
    ax1.legend()

    ax2.plot(radii, data_C, label="real data anchors", color="tab:blue")
    ax2.plot(radii, box_C, label="per-dim uniform box anchors", color="tab:orange")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("radius r (log)")
    ax2.set_ylabel("C(r): fraction of points within r (log)")
    ax2.set_title("Correlation sum C(r)")
    ax2.legend()

    fig.suptitle(f"Real data vs. per-dim uniform box samples ({N_ANCHORS} anchors each)")
    fig.tight_layout()
    fig.savefig("visualizations/manifold_distance_pdf.png", dpi=150)
    print("saved visualizations/manifold_distance_pdf.png")

if __name__ == "__main__":
    main()
