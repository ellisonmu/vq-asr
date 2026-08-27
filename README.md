# vq-asr

## Goal

Explore a learned vector quantizer for speech. Such a quantizer would let us represent a speech
representation with a small set of codebook vectors. We expect the quantizer to degrade at
ultra-low resolution (without retraining the ASR model), since low bit depths force more
collisions between distinct speech frames onto the same codebook entry.

We measure quality as ASR performance (WER / CER).

## Pipeline

```
speech waveform -> mel spectrogram -> quantized mel spectrogram -> ASR (Whisper)
```

We quantize the feature vectors of the 80-channel mel spectrogram with a codebook of size
`K = 2^b`, where `b` is the bit depth. The codebook matrix `C` has shape `(K, d=80)`, and the
input mel spectrogram `X` has shape `(N, d=80)` (`N` frames).

## Plan

1. Build the mel spectrogram (matching Whisper's STFT settings) and quantize it with the
   vector quantizer in `operations/quantization.py`.
2. Implement and test at least three codebook learning methods:
   - k-means / Lloyd's algorithm
   - Gumbel-Softmax VQ
   - GMM / EM
   - neural network
3. Implement and test ASR performance on Whisper (base) across a sweep of codebook bit depths.
4. Write up findings and potential directions for continued study.
5. Optimize the codebook training for CUDA.
