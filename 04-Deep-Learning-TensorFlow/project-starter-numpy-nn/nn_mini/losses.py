"""Losses + turunannya — numpy murni.

Tugasmu: isi fungsi bertanda TODO. Kontrak ada di docstring dan di
tests/test_losses.py.
"""

import numpy as np


def bce(y_true, y_pred, eps=1e-12):
    """Binary cross-entropy rata-rata (input sudah probabilitas).

    - clip probabilitas ke [eps, 1-eps] sebelum log
    - return float (python), bukan np.float64
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    if y.shape != p.shape:
        raise ValueError("bentuk y_true dan y_pred harus sama")
    # TODO: rumus BCE rata-rata
    raise NotImplementedError


def bce_with_logits_grad(y_pred, y_true, eps=1e-12):
    """Gradien BCE terhadap PROBABILITAS (untuk backprop manual).

    d/dp [ -(y log p + (1-y) log(1-p)) ] / m  =  (p - y) / (p (1 - p)) / m

    Return array float sebesar p. Clip p dengan eps yang sama agar pembilang
    dan penyebut konsisten.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    m = len(y)
    # TODO
    raise NotImplementedError


def mse(y_true, y_pred):
    """Mean squared error rata-rata. Return float."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError("bentuk y_true dan y_pred harus sama")
    # TODO
    raise NotImplementedError


def mse_grad(y_pred, y_true):
    """Gradien MSE terhadap prediksi: 2 (p - y) / m. Return array."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    # TODO
    raise NotImplementedError
