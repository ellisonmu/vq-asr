import io
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datasets import load_dataset, Audio
import whisper

from operations.spectrogram import log_mel_spectrogram

N_UTTERANCES = 1000
FRAMES_PER_UTTERANCE = 50   # subsample frames per utterance so the point cloud stays tractable


def load_frame_vectors(n_utterances=N_UTTERANCES, frames_per_utterance=FRAMES_PER_UTTERANCE, seed=0):
    """
    Returns X: (N, 80) array of mel frame vectors, one row per time frame,
    pooled across a subset of LibriSpeech clean utterances.
    """
    dataset = load_dataset("openslr/librispeech_asr", "all", split="test.clean")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.shuffle(seed=seed).select(range(n_utterances))

    rng = np.random.default_rng(seed)
    vectors = []
    for utterance in dataset:
        audio, sr = sf.read(io.BytesIO(utterance["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == whisper.audio.SAMPLE_RATE, f"expected {whisper.audio.SAMPLE_RATE} Hz, got {sr}"
        mel = whisper.log_mel_spectrogram(audio)      # (80, frames)
        mel = mel.numpy()
        frames = mel.T                        # (frames, 80)

        if len(frames) > frames_per_utterance:
            idx = rng.choice(len(frames), size=frames_per_utterance, replace=False)
            frames = frames[idx]

        vectors.append(frames)

    return np.concatenate(vectors, axis=0)


def pca(X, n_components=2):
    reducer = PCA(n_components=n_components)
    return reducer.fit_transform(X), reducer


def tsne(X, n_components=2, perplexity=30):
    reducer = TSNE(n_components=n_components, perplexity=perplexity, init="pca", random_state=0)
    return reducer.fit_transform(X), reducer


def plot_projection(Z, title, ax):
    ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")


def main():
    X = load_frame_vectors()

    Z_pca, pca_model = pca(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_projection(Z_pca, f"PCA (explained var: {pca_model.explained_variance_ratio_.sum():.2f})", ax)

    fig.suptitle(f"Mel frame vectors ({X.shape[0]} frames, 80 channels) — LibriSpeech clean")
    fig.tight_layout()
    fig.savefig("visualizations/spec_prob_pca.png", dpi=150)
    print("saved visualizations/spec_prob_pca.png")


if __name__ == "__main__":
    main()
