"""Logistic regression dari nol — numpy murni (full-batch gradient descent).

Tugasmu: isi semua bagian bertanda TODO. Kontrak fungsi ada di docstring
dan di tests/test_model.py — baca test-nya dulu (TDD).
"""

import numpy as np


def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z), stabil untuk z sangat negatif.

    - z >= 0  : 1 / (1 + exp(-z))   (aman)
    - z < 0   : exp(z) / (1 + exp(z))  (hindari overflow exp(700+))
    Return array numpy dengan bentuk sama seperti z.
    """
    z = np.asarray(z, dtype=float)
    # TODO: implementasikan versi stabil per cabang
    raise NotImplementedError


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """BCE rata-rata: -mean( y·log(p) + (1-y)·log(1-p) ), dengan clipping eps.

    Return float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    if y_true.shape != y_pred.shape:
        raise ValueError("bentuk y_true dan y_pred harus sama")
    # TODO
    raise NotImplementedError


class LogisticRegressionGD:
    """Logistic regression dilatih dengan full-batch gradient descent.

    Gradien (turunan BCE terhadap parameter):
        ∂L/∂w = Xᵀ(p − y) / m      (p = σ(Xw + b))
        ∂L/∂b = sum(p − y) / m
    """

    def __init__(self, lr=0.1, n_iter=2000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.n_iter):
            # TODO: forward pass (p = σ(Xw+b)), hitung loss, simpan ke
            # loss_history, hitung gradien dw/db, update w dan b.
            raise NotImplementedError
        return self

    def predict_proba(self, X):
        """Return σ(Xw + b) — probabilitas kelas 1 (array 1D)."""
        if self.w is None:
            raise RuntimeError("model belum di-fit — panggil fit() dulu")
        X = np.asarray(X, dtype=float)
        # TODO
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        """Return array 0/1: probabilitas >= threshold → 1."""
        # TODO: panggil predict_proba lalu threshold
        raise NotImplementedError
