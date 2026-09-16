"""Bigram language model: smoothing, cross-entropy, sampling temperature.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_languagemodel.py.

Konsep kunci:
- Model bahasa = distribusi token berikutnya: P(next | prev).
- Smoothing (1 + alpha) di SEMUA sel -> tidak ada probabilitas nol.
- Temperature: p -> p^(1/T) sebelum dinormalisasi.
    T -> 0  : deterministik (argmax)
    T = 1   : sesuai distribusi asli
    T > 1   : lebih acak / kreatif
"""

import numpy as np

__all__ = ["bigram_matrix", "ee_bigram", "prob_baris_dengan_temp", "sample_bigram"]


def bigram_matrix(seq, vocab_size, alpha=0.5):
    """Matriks hitungan bigram + smoothing. Return P (baris = distribusi next).

    counts diisi (1 + alpha) untuk SEMUA sel, +1 untuk tiap transisi a->b
    yang berturutan di seq, lalu dinormalisasi PER BARIS (jumlah = 1).
    """
    # TODO
    raise NotImplementedError


def ee_bigram(seq, P):
    """Cross-entropy rata-rata (nats/char): -mean log P[next | prev]."""
    # TODO: gunakan zip(seq[:-1], seq[1:]); tambahkan 1e-12 di dalam log
    raise NotImplementedError


def prob_baris_dengan_temp(baris, temp):
    """Terapkan temperature pada SATU baris distribusi: p -> p^(1/temp).

    Baris nol (semua nol) boleh dikembalikan apa adanya. Selain itu:
    naikkan ke 1/temp, lalu normalisasi (jumlah = 1).
    """
    # TODO
    raise NotImplementedError


def sample_bigram(P, seed_id, n_token, temp, rng):
    """Generate n_token id dengan temperature sampling.

    seed_id : id karakter awal (posisi baris pertama)
    Return list id sepanjang n_token (TIDAK termasuk seed).
    Per langkah: baris = prob_baris_dengan_temp(P[cur], temp); next = rng.choice(V, p=baris).
    """
    # TODO
    raise NotImplementedError
