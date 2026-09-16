"""Implementasi referensi CNN mini — untuk cek mandiri SETELAH selesai.

Self-contained: hanya butuh numpy + nn_mini.nnet (Dense/bce yang sudah diberikan).
Dipakai oleh _verify_project.py untuk mengisi semua TODO lalu menjalankan
notebook. Jangan dibuka sebelum mencoba — sama seperti kunci jawaban kuis.
"""

import numpy as np

from nn_mini.nnet import Dense, bce, bce_with_logits_grad, relu, relu_grad

__all__ = ["pad_zero", "conv2d_forward", "conv2d_backward", "maxpool2x2_forward",
           "maxpool2x2_backward", "Flatten", "Conv2D", "flip_horizontal",
           "flip_vertical", "random_noise", "random_shift", "augment_batch",
           "ReLU", "MaxPool2x2", "CNNMini", "bce", "bce_with_logits_grad",
           "Dense", "relu", "relu_grad"]


# ------------------------------------------------------------------ conv ---
def pad_zero(X, pad):
    if pad < 0:
        raise ValueError("pad tidak boleh negatif")
    X = np.asarray(X, dtype=float)
    if pad == 0:
        return X.copy()
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0.0)


def conv2d_forward(X, K, stride=1):
    X = np.asarray(X, dtype=float)
    K = np.asarray(K, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    if K.ndim != 2:
        raise ValueError("K harus (kh, kw)")
    if stride < 1:
        raise ValueError("stride minimal 1")
    if K.shape[0] > X.shape[1] or K.shape[1] > X.shape[2]:
        raise ValueError("kernel lebih besar dari input")
    n, h, w = X.shape
    kh, kw = K.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((n, oh, ow))
    for u in range(kh):
        for v in range(kw):
            out += K[u, v] * X[:, u:u + (oh - 1) * stride + 1:stride,
                               v:v + (ow - 1) * stride + 1:stride]
    return out


def conv2d_backward(Xp, K, grad, stride=1):
    Xp = np.asarray(Xp, dtype=float)
    grad = np.asarray(grad, dtype=float)
    oh, ow = grad.shape[1], grad.shape[2]
    dXp = np.zeros_like(Xp)
    dK = np.zeros_like(K)
    for r in range(oh):
        for c in range(ow):
            g = grad[:, r, c]
            potong = Xp[:, r * stride:r * stride + K.shape[0],
                        c * stride:c * stride + K.shape[1]]
            dK += (g[:, None, None] * potong).sum(axis=0)
            dXp[:, r * stride:r * stride + K.shape[0],
                c * stride:c * stride + K.shape[1]] += g[:, None, None] * K[None, :, :]
    return dXp, dK


def maxpool2x2_forward(X):
    X = np.asarray(X, dtype=float)
    n, h, w = X.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("h dan w harus genap untuk max-pool 2x2")
    out = np.zeros((n, h // 2, w // 2))
    for r in range(h // 2):
        for c in range(w // 2):
            out[:, r, c] = X[:, 2 * r:2 * r + 2, 2 * c:2 * c + 2].max(axis=(1, 2))
    return out


def maxpool2x2_backward(X, grad):
    X = np.asarray(X, dtype=float)
    grad = np.asarray(grad, dtype=float)
    n, h, w = X.shape
    dX = np.zeros_like(X)
    for r in range(h // 2):
        for c in range(w // 2):
            jendela = X[:, 2 * r:2 * r + 2, 2 * c:2 * c + 2].reshape(n, 4)
            am = jendela.argmax(axis=1)
            ar, ac = am // 2, am % 2
            dX[np.arange(n), 2 * r + ar, 2 * c + ac] = grad[:, r, c]
    return dX


class Flatten:
    def __init__(self):
        self._bentuk = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("Flatten butuh (n, h, w)")
        self._bentuk = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return np.asarray(grad, dtype=float).reshape(self._bentuk)


class Conv2D:
    def __init__(self, kh, kw, pad=1, stride=1, seed=None):
        if kh < 1 or kw < 1:
            raise ValueError("ukuran kernel minimal 1x1")
        self.kh, self.kw = int(kh), int(kw)
        self.pad = int(pad)
        self.stride = int(stride)
        self._rng = np.random.default_rng(seed)
        self.K = None
        self.dK = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("input Conv2D harus (n, h, w)")
        if self.K is None:
            self.K = self._rng.normal(0.0, np.sqrt(2.0 / (self.kh * self.kw)),
                                      size=(self.kh, self.kw))
        self._Xp = pad_zero(X, self.pad)
        return conv2d_forward(self._Xp, self.K, self.stride)

    def backward(self, grad):
        if self.K is None:
            raise RuntimeError("forward belum pernah dipanggil")
        dXp, self.dK = conv2d_backward(self._Xp, self.K, grad, self.stride)
        if self.pad > 0:
            p = self.pad
            return dXp[:, p:-p, p:-p]
        return dXp


# --------------------------------------------------------------- augment ---
def flip_horizontal(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    return X[:, :, ::-1].copy()


def flip_vertical(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    return X[:, ::-1, :].copy()


def random_noise(X, rng, p=0.1, strength=0.5):
    X = np.asarray(X, dtype=float).copy()
    n, h, w = X.shape
    n_kacau = int(p * n)
    if n_kacau == 0:
        return X
    idx = rng.choice(n, size=n_kacau, replace=False)
    for i in idx:
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        X[i, r, c] += rng.uniform(0.4, 0.4 + strength)
    return X


def random_shift(X, rng, max_shift=2):
    X = np.asarray(X, dtype=float)
    n, h, w = X.shape
    out = np.zeros_like(X)
    for i in range(n):
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        # konten bergeser sebesar (dy, dx): out[r+dy, c+dx] = X[r, c]
        s_r0, s_r1 = max(0, -dy), min(h, h - dy)
        s_c0, s_c1 = max(0, -dx), min(w, w - dx)
        if s_r0 >= s_r1 or s_c0 >= s_c1:
            continue
        out[i, s_r0 + dy:s_r1 + dy, s_c0 + dx:s_c1 + dx] = X[i, s_r0:s_r1, s_c0:s_c1]
    return out


def augment_batch(X, y, rng, p_noise=0.1):
    X = np.asarray(X, dtype=float).copy()
    n = len(X)
    idx = rng.choice(n, size=n // 2, replace=False)
    X[idx] = flip_horizontal(X[idx])
    X = random_noise(X, rng, p=p_noise)
    return X, y


# ----------------------------------------------------------------- model ---
class ReLU:
    def __init__(self):
        self._z = None

    def forward(self, X):
        self._z = np.asarray(X, dtype=float)
        return relu(self._z)

    def backward(self, grad):
        return grad * relu_grad(self._z)


class MaxPool2x2:
    def __init__(self):
        self._X = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("MaxPool2x2 butuh (n, h, w)")
        self._X = X
        return maxpool2x2_forward(X)

    def backward(self, grad):
        return maxpool2x2_backward(self._X, grad)


class CNNMini:
    """Conv2D → ReLU → MaxPool2x2 → Flatten → Dense(1, sigmoid)."""

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
        a = self.conv.forward(X)
        a = self.relu.forward(a)
        a = self.pool.forward(a)
        a = self.flat.forward(a)
        return self.dense.forward(a)

    def backward(self, grad):
        g = self.dense.backward(grad)
        g = self.flat.backward(g)
        g = self.pool.backward(g)
        g = self.relu.backward(g)
        return self.conv.backward(g)

    def parameters(self):
        if self.conv.K is None or self.dense.W is None:
            raise RuntimeError("forward belum pernah dipanggil")
        return [self.conv.K, self.dense.W, self.dense.b]

    def grads(self):
        return [self.conv.dK, self.dense.dW, self.dense.db]

    def predict_proba(self, X):
        return np.asarray(self.forward(X)).ravel()
