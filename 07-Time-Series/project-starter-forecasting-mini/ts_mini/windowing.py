"""Windowing & split temporal — fondasi dataset supervised dari satu seri.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_windowing.py.

Aturan emas: SPLIT TEMPORAL. Urutan waktu adalah informasi — mengacaknya
berarti memasukkan masa depan ke training (data leakage temporal).
"""

import numpy as np

__all__ = ["buat_window", "split_waktu", "buat_window_multi"]


def buat_window(series, W):
    """Windowing 1-step: X[i] = series[i:i+W], y[i] = series[i+W].

    Return (X, y): X berbentuk (n, W) dengan n = len(series) - W,
    y berbentuk (n,). W harus >= 1 dan < len(series).
    """
    # TODO
    raise NotImplementedError


def split_waktu(X, y, rasio_train=0.8):
    """Split BERURUTAN waktu: X[:cut] train, X[cut:] val, cut = int(n·rasio).

    Return (Xtr, ytr, Xva, yva). TIDAK boleh mengacak.
    """
    # TODO
    raise NotImplementedError


def buat_window_multi(series, W, h):
    """Windowing multi-step: X[i] = series[i:i+W], y[i] = series[i+W:i+W+h].

    Return (X, y): X (n, W), y (n, h) dengan n = len(series) - W - h + 1.
    """
    # TODO
    raise NotImplementedError
