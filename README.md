# vq-asr

## Goal

Explore a learned vector quantizer for speech. Such a quantizer would let us represent a speech
representation with a small set of codebook vectors. We expect the quantizer to degrade at
ultra-low resolution (without retraining the ASR model), since resolution forces more
collisions between distinct speech frames onto the same codebook entry.

We measure quality as ASR performance (WER / CER).

This is an ongoing project.

## Pipeline

```
speech waveform -> mel spectrogram -> quantized mel spectrogram -> ASR (Whisper)
```

We quantize the feature vectors of the 80-channel mel spectrogram with a codebook of size
`K`. The codebook matrix `C` has shape `(K, d=80)`, and the
input mel spectrogram `X` has shape `(N, d=80)` (`N` frames).

## Plan

1. Build the mel spectrogram (matching Whisper's STFT settings) and quantize it with the
   vector quantizer in `operations/quantization.py`.
2. Implement and test at least three codebook learning methods:
   - k-means / Lloyd's algorithm
   - GMM / EM
3. Implement and test ASR performance on Whisper (base) across a sweep of codebook resolutions.
4. Write up findings and potential directions for continued study.
5. Optimize the codebook training for CUDA.

K-means/lloyds: Soruce:https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/
- 
GMM / EM

Soruce:https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/

Gaussian Mixtre aModation (GMM)
- handels overelapping cluster effectively
- cluster shape ids dfiend by the mean and covairnce of a pdf
- assume data is a mied of K ghaussian distirbutions (clusters) with mean $\mu_k$ and covariance $\Sigma_k$ and mixing weight $\pi_k$.

Probability that a given vector $x_n$ belong to codebook vector $k$

Probability of obeve $x_n$ under all the entie moxture models

Expectation Maximation for estimating GMM Parameters

Expection:
$$$
\gamma_{n,k} = \P(z_n=k | x_n) = \frac{\pi_k \cdot \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j \cdot \mathcal{N}(x_n|\mu_j,\Sigma_j)}
$$$
where $z_n$ is a latenet variable incdicating cluster assignment.
and 
$$$
\mathcal{N}(x_n|\mu_k,\Sigma_k) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp{\left[-\frac{1}{2}(x_n-\mu_n)^T\sigma_K^{-1}(x_n-\mu_k)\right]}
$$$

Maximation: update meanas, coevariance, mixing coeffincet

Mean: $\mu_k = \frac{\sum_{n=1}^N \gamma_{n,k}x_n}{\sum_{n=1}^N \gamma_{n,k}}$
Covariances: $\Sigma_k = \frac{\sum_{n=1}^N \gamma_{n,k}(x_n-\mu_k)(x_n-\mu_k)^T}{\sum_{n=1}^N \gamma_{n,k}}$
Mixing Weights: $\pi_k = \frac{1}{N}\sum_{n=1}^N\gamma_{n,k}$
Notably, $\sum_{k=1}^K \pi_k = 1$. 

Objective that EM optimzies is 
$$$
\mathcal{L} = \Pi_{n-1}^{N} \sum_{k=1}^{K}\pi_{k}\cdot\mathca{N}\right(x_n|\mu_k,\Sigma_k\left)
$$$

### Numerical stability: computing the E-step in log-space

At `d=80`, evaluating $\mathcal{N}(x_n|\mu_k,\Sigma_k)$ directly (as written above) underflows to 0 almost
everywhere: the density of a high-dimensional Gaussian is astronomically small unless $x_n$ sits
extremely close to $\mu_k$. Once the numerator underflows, $\gamma_{n,k}$ becomes `0/0` (`NaN`) within
the first few iterations. `gmmem` in `experiment/algorithms.py` avoids this by never
exponentiating until the very last step.

We use a diagonal covariance ($\Sigma_k = \mathrm{diag}(\sigma_{k,1}^2, \dots, \sigma_{k,d}^2)$), so the
log-density has a closed form with no matrix inverse or determinant:

$$$
\log \mathcal{N}(x_n|\mu_k,\Sigma_k) = -\frac{1}{2}\sum_{i=1}^{d}\left[\frac{(x_{n,i}-\mu_{k,i})^2}{\sigma_{k,i}^2} + \log \sigma_{k,i}^2\right] - \frac{d}{2}\log(2\pi)
$$$

Adding $\log \pi_k$ gives the log-numerator of the responsibility, $\log(\pi_k \mathcal{N}(x_n|\mu_k,\Sigma_k))$,
for every $(n, k)$ pair. The E-step responsibility

$$$
\gamma_{n,k} = \frac{\pi_k \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j \mathcal{N}(x_n|\mu_j,\Sigma_j)}
$$$

then only needs the log of that denominator, $\log\sum_j \exp(\log\text{-numerator}_{n,j})$, which is exactly
the `logsumexp` operation: it computes $\log\sum_j e^{a_j}$ by subtracting off $\max_j a_j$ before
exponentiating, so nothing overflows or underflows regardless of how large or small the log-numerators
are. We recover $\gamma_{n,k}$ at the end via $\exp(\log\text{-numerator}_{n,k} - \log\text{-denominator}_n)$ —
the only exponentiation in the whole E-step, applied to an already-normalized (safely negative) value.

This also gives the per-iteration log-likelihood for free: $\log\text{-denominator}_n$ is precisely
$\log\sum_k \pi_k \mathcal{N}(x_n|\mu_k,\Sigma_k)$, so summing it over all $n$ gives $\log \mathcal{L}$
directly, without ever computing $\mathcal{L}$ itself (which would also underflow/overflow).

The implementation batches this computation over frames and expands the Mahalanobis term
$\sum_i (x_i-\mu_{k,i})^2/\sigma_{k,i}^2$ algebraically into
$x^2 \cdot (1/\sigma^2)^T - 2\, x \cdot (\mu/\sigma^2)^T + \sum_i \mu_{k,i}^2/\sigma_{k,i}^2$
so it never materializes an `(N, K, d)` tensor — only `(N, K)` and `(K, d)` — which is what keeps the
M-step's sufficient statistics (`sum_x`, `sum_x2`) memory-tractable at large `K`.

## Discussion
For both methods, we observe that increasing the codebook resolution improves WER and CER. We observe little difference between the GMM-derived and K-means-derived codebooks across the evaluated resolutions. Future work will investigate alternative methods for deriving the codebook, such as Gumbel-Softmax, as well as other optimization objectives.
