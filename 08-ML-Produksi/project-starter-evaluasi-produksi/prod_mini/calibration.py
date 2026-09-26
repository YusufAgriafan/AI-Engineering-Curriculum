"""TODO: Kalibrasi probability — sigmoid stabil, binning, ECE, Platt scaling.

Spesifikasi lengkap: tests/test_calibration.py
"""

import numpy as np


def sigmoid(z):
    """Numerik-stabil: 1/(1+exp(-z)) dengan z di-clip ke [-60, 60]. Return array/float."""
    # TODO
    raise NotImplementedError


def binned_calibration(y_true01, p, n_bins=10):
    """Kelompokkan prediksi per bin probabilitas.

    idx = min(int(p * n_bins), n_bins - 1)  -- p=1.0 masuk bin terakhir.
    Per bin: {'lo', 'hi', 'n', 'conf', 'acc'}; conf=mean(p), acc=mean(label);
    bin kosong: conf=0.0, acc=0.0, n=0.
    Return list sepanjang n_bins.
    """
    # TODO
    raise NotImplementedError


def ece(y_true01, p, n_bins=10):
    """Expected Calibration Error = Σ (n_bin/n)·|acc − conf|, bin kosong dilewati.

    Return float.
    """
    # TODO
    raise NotImplementedError


def fit_platt(score, y01, lr=0.1, epochs=3000, seed=8):
    """Platt scaling: p = sigmoid(a·s + b) via gradient descent pada cross-entropy.

    Gradien: dL/da = mean((p − y)·s), dL/db = mean(p − y).
    Inisialisasi a=1.0, b=0.0. Return (a, b).
    """
    # TODO
    raise NotImplementedError


def apply_platt(a, b, score):
    """Terapkan hasil Platt: p = sigmoid(a·score + b). Return array/float."""
    # TODO
    raise NotImplementedError
