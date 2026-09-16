"""Pasangan skip-gram + skip-gram dengan negative sampling, dari nol.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_embeddings.py.

Matematika (satu pasangan per langkah):

    loss  = −log σ(v_t · u_c)  +  Σ_neg −log σ(−v_t · u_neg)
    grad  = (σ(z) − y) · konteks       dengan y = 1 (asli) / 0 (noise)

Urutan yang benar per pasangan:
    1. hitung semua gradien pada v, u ASLI (sebelum update!)
    2. update W_out untuk noise & konteks
    3. update W_in dan W_out[c]

Trik word2vec yang dipakai: noise distribution = unigram^0.75, lr decay linear.
"""

import numpy as np

__all__ = ["sigmoid", "pasangan_skipgram", "frekuensi_unigram", "train_skipgram"]


def sigmoid(z):
    """Sigmoid stabil: z di-clip ke [-30, 30] sebelum exp."""
    # TODO: 1 / (1 + exp(-clip(z, -30, 30)))
    raise NotImplementedError


def pasangan_skipgram(kalimat_list, w2i, window=2):
    """Semua pasangan (target, konteks) dalam jendela `window` DUA ARAH.

    Kalimat [a, b, c] window 1 -> (a,b), (b,a), (b,c), (c,b). Tanpa diri-sendiri.
    kalimat_list: list list-token (sudah tokenize).
    """
    # TODO
    raise NotImplementedError


def frekuensi_unigram(kalimat_list, w2i):
    """Distribusi unigram (array (V,), jumlah = 1.0) untuk sampling noise.

    Paksa sel terakhir = 1 - sum(sisanya) supaya jumlah TEPAT 1.0 (presisi float).
    """
    # TODO
    raise NotImplementedError


def train_skipgram(pairs, vocab_size, freq_unigram,
                   dim=16, epochs=5, lr=0.05, k_neg=3, seed=0):
    """Skip-gram + negative sampling.

    pairs        : list (target_id, context_id)
    freq_unigram : array (vocab_size,) probabilitas unigram (untuk noise)
    Return (W_in, loss_history): W_in (vocab_size, dim) = embedding matrix,
    loss_history panjang `epochs` (rata-rata loss per pasangan tiap epoch).

    Detail:
    - noise = freq_unigram ** 0.75, dinormalisasi
    - lr_ep = lr * (1.0 - 0.5 * ep / epochs)   (decay linear)
    - per pasangan: 1 sampel asli + k_neg noise (abaikan noise == konteks)
    - W_out[neg] langsung di-update; gradien untuk W_in & W_out[c]
      diakumulasi pada nilai ASLI lalu di-update di akhir
    """
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0, 0.1, (vocab_size, dim))
    W_out = rng.normal(0, 0.1, (vocab_size, dim))
    noise = np.asarray(freq_unigram, dtype=float) ** 0.75
    noise = noise / noise.sum()
    loss_hist = []
    for ep in range(epochs):
        lr_ep = lr * (1.0 - 0.5 * ep / epochs)
        tot, cnt = 0.0, 0
        for (t, c) in pairs:
            # TODO 1: pasangan asli — s = sigmoid(W_in[t] @ W_out[c]),
            #         grad_v = (s - 1) * W_out[c], grad_u = (s - 1) * W_in[t]
            # TODO 2: k_neg noise: sn = sigmoid(W_in[t] @ W_out[neg]);
            #         grad_v += sn * W_out[neg]; W_out[neg] -= lr_ep * (sn * W_in[t])
            #         loss += -log(1 - sn)  (lewati noise == c)
            # TODO 3: W_in[t] -= lr_ep * grad_v ; W_out[c] -= lr_ep * grad_u
            #         loss += -log(s) ; tot += loss ; cnt += 1
            raise NotImplementedError
        loss_hist.append(tot / cnt)
    return W_in, loss_hist
