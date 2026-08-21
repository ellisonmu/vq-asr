import numpy as np
import jiwer
from operations.quantization import vector_quantizer
from datasets import load_dataset
from itertools import islice
import whisper
from whisper.normalizers import EnglishTextNormalizer
import torch
from experiment.algorithms import kmeans

normalizer = EnglishTextNormalizer()

def normalize(input):
    peak = np.max(abs(input))
    return input / peak if peak > 0 else 0

def front_end(utterance, quantize=False, codebook=None):
    x = normalize(utterance["audio"])
    x = whisper.pad_or_trim(x)
    x_mel = whisper.log_mel_spectrogram(x)
    mel = x_mel.numpy()
    ref = utterance["text"]
    ref_norm = normalizer(ref)
    if quantize == True:
        X = mel.T.astype(np.float32)
        vq = vector_quantizer(codebook)
        indices = vq.quantizer(X)
        X_hat = vq.dequantizer(indices)
        mel_q = X_hat.T
    else:
        mel_q = mel
    return mel_q, ref_norm #numpy arr

def train(bitdepth=8):
    K = 2**bitdepth
    ds = load_dataset("librispeech_asr", "all", split="train.clean.100", streaming=True)
    train_subset = list(islice(ds, 5000))
    X = np.concatenate([front_end(u, quantize=False)[0].T for u in train_subset], axis=0)  # (total_frames, 80)
    return kmeans(X, K)

def evaluate(codebook):
    wer_hist, cer_hist = [], []
    model = whisper.load_model("base").to("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = load_dataset("librispeech_asr", "all", split="test.clean").shuffle().select(range(1000))

    for utterance in test_dataset:
        mel_q, ref_norm = front_end(utterance, quantize=True, codebook=codebook)
        mel_torch = torch.from_numpy(mel_q).to(model.device)
        options = whisper.DecodingOptions()
        pred = whisper.decode(model, mel_torch, options)
        pred_norm = normalizer(pred.text)
        wer_hist.append(jiwer.wer(ref_norm, pred_norm))
        cer_hist.append(jiwer.cer(ref_norm, pred_norm))
    return np.mean(wer_hist), np.mean(cer_hist), np.std(wer_hist), np.std(cer_hist)

def plot_sweep(bitdepth, mean_wer_hist, mean_cer_hist, std_wer_hist, std_cer_hist):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(bitdepth, mean_wer_hist, yerr=std_wer_hist, label="WER", marker="o", capsize=3)
    ax.errorbar(bitdepth, mean_cer_hist, yerr=std_cer_hist, label="CER", marker="o", capsize=3)
    ax.set_xlabel("bit depth")
    ax.set_ylabel("error rate")
    ax.set_title("WER / CER vs codebook bit depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig("experiment/wer_cer_sweep.png", dpi=150)
    print("saved experiment/wer_cer_sweep.png")

def main():
    mean_wer_hist, mean_cer_hist, std_wer_hist, std_cer_hist = [], [], [], []
    bitdepth = list(range(1, 13))
    for b in bitdepth:
        codebook = train(b)
        wer_mean, cer_mean, wer_std, cer_std = evaluate(codebook)
        mean_wer_hist.append(wer_mean)
        mean_cer_hist.append(cer_mean)
        std_wer_hist.append(wer_std)
        std_cer_hist.append(cer_std)
    plot_sweep(bitdepth, mean_wer_hist, mean_cer_hist, std_wer_hist, std_cer_hist)

if __name__ == "__main__":
    main()
    





