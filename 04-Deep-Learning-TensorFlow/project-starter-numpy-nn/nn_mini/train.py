"""SGD optimizer, batching, dan training loop — numpy murni.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak ada di docstring dan di
tests/test_train.py.

Inilah versi mini dari `model.fit()` — dan fondasi custom training loop
dengan GradientTape di Bab 4 serta fine-tuning di Bab 13.
"""

import numpy as np

from .losses import bce, bce_with_logits_grad


class SGD:
    """Vanilla SGD: param -= lr * grad."""

    def __init__(self, lr=0.1):
        self.lr = float(lr)

    def step(self, params, grads):
        """Update semua parameter in-place. params & grads: list sejajar."""
        if len(params) != len(grads):
            raise ValueError("params dan grads harus sama panjang")
        # TODO
        raise NotImplementedError


def iter_batch(X, y, batch_size, seed=None):
    """Yield (X_batch, y_batch) dengan urutan acak tiap epoch.

    - shuffle indeks dengan np.random.default_rng(seed).permutation
    - batch terakhir boleh lebih kecil
    """
    if len(X) != len(y):
        raise ValueError("X dan y harus sama panjang")
    if batch_size < 1:
        raise ValueError("batch_size harus >= 1")
    rng = np.random.default_rng(seed)
    # TODO: shuffle, lalu yield irisan per batch_size
    raise NotImplementedError


def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32,
                lr=0.1, patience=None, seed=0, verbose=False):
    """Latih model BCE + SGD mini-batch, dengan optional early stopping.

    Return dict riwayat:
        {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}
    (panjang tiap list = jumlah epoch yang benar-benar dijalankan)

    Early stopping (hanya jika patience bukan None):
    - pantau val_loss; jika TIDAK membaik lebih dari 1e-6 selama `patience`
      epoch berturut-turut -> berhenti
    - 'membaik' = val_loss < best_so_far - 1e-6 ; best diinisialisasi np.inf
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
        # TODO 1: loop batch dari iter_batch(seed=seed + epoch):
        #   - p = model.forward(Xb).ravel()
        #   - g = bce_with_logits_grad(p, yb).reshape(-1, 1)
        #   - model.backward(g)
        #   - opt.step(model.parameters(), grads) di mana grads URUTANNYA SAMA
        #     dengan model.parameters(): [L.dW, L.db untuk tiap layer L]
        raise NotImplementedError

        # ---- evaluasi akhir epoch (sudah diberikan) ----
        ptr = model.predict_proba(X_train)
        pva = model.predict_proba(X_val)
        history["train_loss"].append(bce(y_train, ptr))
        history["val_loss"].append(bce(y_val, pva))
        history["val_acc"].append(float(((pva >= 0.5) == y_val).mean()))

        if verbose:
            print(f"epoch {epoch + 1:>3}: train={history['train_loss'][-1]:.4f} "
                  f"val={history['val_loss'][-1]:.4f} acc={history['val_acc'][-1]:.3f}")

        # TODO 2: early stopping sesuai kontrak di docstring
        raise NotImplementedError

    return history
