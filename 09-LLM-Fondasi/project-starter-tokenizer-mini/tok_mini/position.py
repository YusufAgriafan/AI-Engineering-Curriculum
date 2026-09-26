"""Bagian 4 — Positional encoding sinusoidal.  TODO: kamu yang mengisi."""
import math

import numpy as np


def positional_encoding(max_len, d):
    """Return (max_len, d): kolom genap sin, kolom ganjil cos.

    freq kolom (2i, 2i+1) = 10000^(-2i/d).
    Petunjuk vektorisasi:
        pos = np.arange(max_len)[:, None]
        i   = np.arange(0, d, 2)
        div = np.exp(-math.log(10000.0) * i / d)
        PE[:, 0::2] = np.sin(pos * div); PE[:, 1::2] = np.cos(pos * div)
    """
    # TODO
    raise NotImplementedError


def rpe(pos, d):
    """PE untuk SATU posisi — baris ke-`pos` dari tabel di atas. Return (d,) float."""
    # TODO
    raise NotImplementedError
