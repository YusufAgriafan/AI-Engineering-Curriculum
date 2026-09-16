"""SGD + iter_batch + train_model — SUDAH LENGKAP, tidak perlu diubah.

Kembaran mini dari nn_mini/train.py Bab 4 (kontrak sama, lihat tests Bab 4).
Disediakan supaya fokus project Bab 5 tetap pada konvolusi, pooling, dan
augmentasi. Kontrak:

    SGD.step(params, grads)      : update in-place `p -= lr * g` (params == grads urutannya)
    iter_batch(X, y, batch_size, seed) : yield (Xb, yb) dengan shuffle per epoch,
                                         batch terakhir boleh lebih kecil
    train_model(model, ...)      : BCE + SGD mini-batch + early stopping pada val_loss
"""

import numpy as np

from .nnet import bce, bce_with_logits_grad

__all__ = ["SGD", "iter_batch", "train_model"]


class SGD:
    def __init__(self, lr):
        self.lr = float(lr)

    def step(self, params, grads):
        if len(params) != len(grads):
            raise ValueError("params dan grads harus sama panjang")
        for p, g in zip(params, grads):
            p -= self.lr * g


def iter_batch(X, y, batch_size, seed):
    n = len(X)
    if batch_size < 1:
        raise ValueError("batch_size minimal 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for awal in range(0, n, batch_size):
        pilih = idx[awal:awal + batch_size]
        yield X[pilih], y[pilih]


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=300, batch_size=32, lr=0.1, seed=0,
                patience=None, verbose=False):
    """Latih model BCE + SGD mini-batch, dengan optional early stopping.

    Return dict riwayat: {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}
    Panjang tiap list = jumlah epoch yang benar-benar dijalankan.

    Early stopping (hanya jika patience bukan None): pantau val_loss;
    'membaik' = val_loss < best - 1e-6; berhenti setelah `patience` epoch
    berturut-turut tanpa perbaikan.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    opt = SGD(lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = np.inf
    wait = 0

    for epoch in range(epochs):
        for Xb, yb in iter_batch(X_train, y_train, batch_size, seed=seed + epoch):
            p = model.forward(Xb).ravel()
            g = bce_with_logits_grad(p, yb).reshape(-1, 1)
            model.backward(g)
            opt.step(model.parameters(), model.grads())

        ptr = model.predict_proba(X_train)
        pva = model.predict_proba(X_val)
        history["train_loss"].append(bce(y_train, ptr))
        history["val_loss"].append(bce(y_val, pva))
        history["val_acc"].append(float(((pva >= 0.5) == y_val).mean()))

        if verbose:
            print(f"epoch {epoch + 1:>3}: train={history['train_loss'][-1]:.4f} "
                  f"val={history['val_loss'][-1]:.4f} acc={history['val_acc'][-1]:.3f}")

        if patience is not None:
            if history["val_loss"][-1] < best_val - 1e-6:
                best_val = history["val_loss"][-1]
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    return history
