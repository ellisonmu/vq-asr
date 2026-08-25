import io
import numpy as np
import jiwer
import soundfile as sf
from tqdm import tqdm
from operations.quantization import vector_quantizer, get_device
from datasets import load_dataset, Audio
from itertools import islice
import whisper
from whisper.normalizers import EnglishTextNormalizer
import torch
from experiment.algorithms import kmeans, gmmem

ALGORITHMS = {"kmeans": kmeans, "gmmem": gmmem}

normalizer = EnglishTextNormalizer()

def load_split(split, streaming=False):
    ds = load_dataset("openslr/librispeech_asr", "all", split=split, streaming=streaming)
    return ds.cast_column("audio", Audio(decode=False))

def normalize(input):
    peak = np.max(abs(input))
    return input / peak if peak > 0 else 0

def read_audio(utterance):
    # audio column is left undecoded (see load_split), so decode the raw FLAC here
    audio = utterance["audio"]
    x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == whisper.audio.SAMPLE_RATE, f"expected {whisper.audio.SAMPLE_RATE} Hz, got {sr}"
    return x

def front_end(utterance, quantize=False, codebook=None):
    x = normalize(read_audio(utterance))
    x = whisper.pad_or_trim(x)
    x_mel = whisper.log_mel_spectrogram(x)
    mel = x_mel.numpy()
    ref = utterance["text"]
    ref_norm = normalizer(ref)
    if quantize == True:
        X = mel.T.astype(np.float32)
        vq = vector_quantizer(codebook)
        indices, _ = vq.quantizer(X)
        X_hat = vq.dequantizer(indices)
        mel_q = X_hat.T
    else:
        mel_q = mel
    return mel_q, ref_norm #numpy arr

def extract_train_features():
    ds = load_split("train.clean.100", streaming=True)
    train_subset = tqdm(islice(ds, 5000), total=5000, desc="extracting train features")
    return np.concatenate([front_end(u, quantize=False)[0].T for u in train_subset], axis=0)  # (total_frames, 80)

def train(X, bitdepth=8, algorithm="kmeans"):
    K = 2**bitdepth
    return ALGORITHMS[algorithm](X, K)

def decode_with_fallback(model, mel_torch, fp16, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                          compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6):
    # mirrors whisper.transcribe()'s decode_with_fallback: retries at increasing temperature
    # when the greedy decode looks like a repetition loop or garbage, instead of trusting
    # a single low-level whisper.decode() pass on out-of-distribution (quantized) audio.
    result = None
    for t in temperatures:
        options = whisper.DecodingOptions(fp16=fp16, language="en", task="transcribe", temperature=t)
        result = whisper.decode(model, mel_torch, options)

        needs_fallback = False
        if compression_ratio_threshold is not None and result.compression_ratio > compression_ratio_threshold:
            needs_fallback = True
        if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
            needs_fallback = True
        if (no_speech_threshold is not None and result.no_speech_prob > no_speech_threshold
                and logprob_threshold is not None and result.avg_logprob < logprob_threshold):
            needs_fallback = False
        if not needs_fallback:
            break
    return result

def evaluate(model, test_dataset, codebook=None, quantize=True):
    wer_hist, cer_hist = [], []
    fp16 = model.device.type == "cuda"  # fp16 decoding is unsupported on cpu/mps
    for utterance in tqdm(test_dataset, desc="evaluating"):
        mel_q, ref_norm = front_end(utterance, quantize=quantize, codebook=codebook)
        mel_torch = torch.from_numpy(mel_q).to(model.device)
        pred = decode_with_fallback(model, mel_torch, fp16)
        pred_norm = normalizer(pred.text)
        wer_hist.append(jiwer.wer(ref_norm, pred_norm))
        cer_hist.append(jiwer.cer(ref_norm, pred_norm))
    return np.mean(wer_hist), np.mean(cer_hist), np.std(wer_hist), np.std(cer_hist)

def plot_sweep(bitdepth, mean_wer_hist, mean_cer_hist, std_wer_hist, std_cer_hist, algorithm="kmeans", baseline_wer=None, baseline_cer=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(bitdepth, mean_wer_hist, yerr=std_wer_hist, label="WER", marker="o", capsize=3, color="tab:blue")
    ax.errorbar(bitdepth, mean_cer_hist, yerr=std_cer_hist, label="CER", marker="o", capsize=3, color="tab:orange")
    if baseline_wer is not None:
        ax.axhline(baseline_wer, color="tab:blue", linestyle="--", label="WER (unquantized)")
    if baseline_cer is not None:
        ax.axhline(baseline_cer, color="tab:orange", linestyle="--", label="CER (unquantized)")
    ax.set_xlabel("bit depth")
    ax.set_ylabel("error rate")
    ax.set_title(f"WER / CER vs codebook bit depth ({algorithm})")
    ax.legend()
    fig.tight_layout()
    out_path = f"experiment/wer_cer_sweep_{algorithm}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

def main(algorithm="kmeans"):
    mean_wer_hist, mean_cer_hist, std_wer_hist, std_cer_hist = [], [], [], []
    bitdepth = list(range(1, 10))
    X = extract_train_features()
    model = whisper.load_model("base").to(get_device())
    test_dataset = load_split("test.clean").shuffle().select(range(1000))
    baseline_wer, baseline_cer, _, _ = evaluate(model, test_dataset, quantize=False)
    print(f"unquantized baseline: WER={baseline_wer:.3f} CER={baseline_cer:.3f}")
    for b in bitdepth:
        codebook = train(X, b, algorithm=algorithm)
        wer_mean, cer_mean, wer_std, cer_std = evaluate(model, test_dataset, codebook)
        mean_wer_hist.append(wer_mean)
        mean_cer_hist.append(cer_mean)
        std_wer_hist.append(wer_std)
        std_cer_hist.append(cer_std)
    plot_sweep(bitdepth, mean_wer_hist, mean_cer_hist, std_wer_hist, std_cer_hist, algorithm=algorithm,
               baseline_wer=baseline_wer, baseline_cer=baseline_cer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="kmeans")
    args = parser.parse_args()
    main(algorithm=args.algorithm)
    





