"""Bagian 3 — Embedding: makna sebagai geometri.  TODO: kamu yang mengisi."""
import numpy as np


def cos_sim(a, b):
    """Cosine similarity dua vektor: dot / (norm * norm). Return float.

    Pembagi nol -> 0.0 (konvensi produksi: jangan crash di serving).
    """
    # TODO
    raise NotImplementedError


def cari_mirip(W, vektor, exclude=(), top_n=3):
    """Index token paling mirip (cosine tertinggi) terhadap `vektor`.

    W: (V, d). Skip indeks di `exclude`. Return list of int sepanjang top_n,
    terurut menurun menurut cosine.
    """
    # TODO
    raise NotImplementedError


def analogi(a, b, c, W):
    """a : b :: c : ? — cari_mirip ke target W[b] - W[a] + W[c], exclude (a,b,c).

    Return satu int (top-1).
    """
    # TODO
    raise NotImplementedError


def latih_embedding(idx_vokal=(0, 4, 6), vocab=8, d=8, epochs=200, seed=9):
    """Fake task yang sama dengan lab: tiap epoch, token 'vokal' digeser
    +SIGNAL (1.5, 1.5, 0, ...) lalu semua embedding didecay 0.99.

    Return matrix (vocab, d). Struktur kelompok harus terbaca dari cosine.
    """
    # GIVEN — dipakai tests untuk membuktikan struktur muncul dari data
    signal = np.zeros(d)
    signal[:2] = 1.5
    r = np.random.default_rng(seed)
    W = r.normal(0, 1, (vocab, d))
    for _ in range(epochs):
        W[list(idx_vokal)] += signal
        W *= 0.99
    return W
