"""CNNMini — merakit blok-bab 5 menjadi CNN utuh.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak ada di docstring dan
tests/test_model.py.

Arsitektur (satu kanal, gambar 8x8):
    input (n, 8, 8)
    → Conv2D(3x3, pad=1)        → (n, 8, 8)   detektor lokal
    → ReLU                      → (n, 8, 8)
    → MaxPool2x2                → (n, 4, 4)   invarian posisi kasar
    → Flatten                   → (n, 16)
    → Dense(1, sigmoid)         → (n,)        probabilitas "vertikal"

CNNMini adalah keras.Sequential mini: daftar blok yang masing-masing punya
forward/backward (dan parameters/grads kalau punya bobot). Tidak ada magic —
yang berbeda dari MLP Bab 4 hanya JENIS bloknya.
"""

import numpy as np

from .conv import Conv2D, Flatten, conv2d_forward, maxpool2x2_backward, maxpool2x2_forward
from .nnet import Dense, Sequential, relu, relu_grad

__all__ = ["ReLU", "MaxPool2x2", "CNNMini"]


class ReLU:
    """Blok aktivasi tanpa parameter: forward simpan z, backward pakai relu_grad."""

    def __init__(self):
        self._z = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        self._z = X
        # TODO: return relu(X)
        raise NotImplementedError

    def backward(self, grad):
        # TODO: return grad * relu_grad(self._z)
        raise NotImplementedError


class MaxPool2x2:
    """Blok pooling tanpa parameter: backward butuh input forward (posisi argmax)."""

    def __init__(self):
        self._X = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("MaxPool2x2 butuh (n, h, w)")
        self._X = X
        # TODO: return maxpool2x2_forward(X)
        raise NotImplementedError

    def backward(self, grad):
        # TODO: return maxpool2x2_backward(self._X, grad)
        raise NotImplementedError


class CNNMini:
    """Conv2D → ReLU → MaxPool2x2 → Flatten → Dense(1, sigmoid).

    Hanya Conv2D dan Dense yang punya bobot — parameters()/grads() dikumpulkan
    dari keduanya, urutannya konsisten antara kedua method (kontrak SGD).
    """

    def __init__(self, kh=3, kw=3, pad=1, seed=0):
        self.conv = Conv2D(kh, kw, pad=pad, stride=1, seed=seed)
        self.relu = ReLU()
        self.pool = MaxPool2x2()
        self.flat = Flatten()
        self.dense = Dense(1, activation="sigmoid", seed=seed + 1)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("input CNNMini harus (n, h, w)")
        # TODO: rangkai forward lewat self.conv, self.relu, self.pool,
        #       self.flat, self.dense — simpan hanya yang dibutuhkan backward
        raise NotImplementedError

    def backward(self, grad):
        # TODO: teruskan grad mundur lewat dense → flat → pool → relu → conv,
        #       return grad terakhir (dL/dX input)
        raise NotImplementedError

    def parameters(self):
        if self.conv.K is None or self.dense.W is None:
            raise RuntimeError("forward belum pernah dipanggil")
        return [self.conv.K, self.dense.W, self.dense.b]

    def grads(self):
        return [self.conv.dK, self.dense.dW, self.dense.db]

    def predict_proba(self, X):
        return np.asarray(self.forward(X)).ravel()


# ----------------------------------------------------------------- demo util
def fitur_conv(X, K, stride=1):
    """Helper untuk eksperimen di notebook: keluaran konvolusi mentah (n, oh, ow)."""
    return conv2d_forward(np.asarray(X, dtype=float), np.asarray(K, dtype=float), stride)


def plot_filter_response(K, judul=""):
    """Heatmap respons filter K terhadap dirinya (autokorelasi) — visual kecil."""
    import matplotlib.pyplot as plt
    resp = conv2d_forward(K[None, :, :], K, stride=1)[0]
    plt.imshow(resp, cmap="viridis")
    plt.title(judul or "respons filter")
    plt.colorbar()
    plt.axis("off")
