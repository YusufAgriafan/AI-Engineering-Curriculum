"""Blok dense & loss — SUDAH LENGKAP, tidak perlu diubah.

Ini kembaran mini dari nn_mini Bab 4 (Dense + BCE). Modul ini disediakan
supaya fokus project Bab 5 tetap pada hal BARU: konvolusi, pooling,
augmentasi, dan cara merakitnya jadi CNN.

Kontrak Dense sama persis dengan Bab 4:
    forward : z = X @ W + b ; a = aktivasi(z)      (lazy init W saat forward pertama)
    backward: dz = grad * aktivasi'(z) ; dW = Xᵀ @ dz ; db = dz.sum(0) ; return dz @ Wᵀ
"""

import numpy as np

__all__ = ["relu", "relu_grad", "sigmoid", "sigmoid_from_logits_stable",
           "bce", "bce_with_logits_grad", "Dense", "Sequential"]


# ---------------------------------------------------------------- activations
def relu(z):
    return np.maximum(z, 0.0)


def relu_grad(z):
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoid stabil dua-cabang (tanpa overflow)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_from_logits_stable(z):
    """Return (p, dp/dz) untuk logits z, stabil-numerik.

    Trik: p = sigmoid(z) lalu dp/dz = p(1-p). Karena z sudah dari forward,
    p dihitung sekali dan turunannya ikut stabil.
    """
    p = sigmoid(z)
    return p, p * (1.0 - p)


# --------------------------------------------------------------------- losses
def bce(y_true, p_pred, eps=1e-12):
    """Binary cross-entropy rata-rata, dengan clipping."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p_pred, dtype=float).ravel(), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bce_with_logits_grad(p_pred, y_true, eps=1e-12):
    """dL/dp untuk BCE rata-rata — gradien yang diteruskan ke layer output."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p_pred, dtype=float).ravel(), eps, 1 - eps)
    m = len(y)
    return (p - y) / (p * (1 - p) * m)


# ---------------------------------------------------------------------- dense
class Dense:
    """Fully-connected layer dengan aktivasi pilihan (relu/sigmoid/linear)."""

    def __init__(self, units, activation="relu", seed=None):
        if activation not in ("relu", "sigmoid", "linear"):
            raise ValueError(f"aktivasi tidak dikenal: {activation!r}")
        self.units = int(units)
        self.activation = activation
        self._rng = np.random.default_rng(seed)
        self.W = None
        self.b = np.zeros(self.units)
        self.dW = None
        self.db = None

    def _init_weights(self, fan_in):
        std = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
        self.W = self._rng.normal(0.0, std, size=(fan_in, self.units))

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("input Dense harus matriks 2D (n, fan_in)")
        if self.W is None:
            self._init_weights(X.shape[1])
        self._X = X
        z = X @ self.W + self.b
        self._z = z
        if self.activation == "relu":
            return relu(z)
        if self.activation == "sigmoid":
            return sigmoid(z)
        return z

    def backward(self, grad):
        if self.W is None:
            raise RuntimeError("forward belum pernah dipanggil")
        if self.activation == "relu":
            dz = grad * relu_grad(self._z)
        elif self.activation == "sigmoid":
            _, dpdz = sigmoid_from_logits_stable(self._z)
            dz = grad * dpdz
        else:
            dz = grad
        self.dW = self._X.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T


class Sequential:
    """Tumpukan layer — versi mini dari keras.Sequential."""

    def __init__(self, layers):
        if not layers:
            raise ValueError("Sequential butuh minimal 1 layer")
        self.layers = list(layers)

    def forward(self, X):
        out = X
        for L in self.layers:
            out = L.forward(out)
        return out

    def backward(self, grad):
        for L in reversed(self.layers):
            grad = L.backward(grad)
        return grad

    def parameters(self):
        params = []
        for L in self.layers:
            if getattr(L, "W", None) is None:
                raise RuntimeError("ada layer yang belum pernah forward")
            params.extend([L.W, L.b])
        return params

    def grads(self):
        grads = []
        for L in self.layers:
            grads.extend([L.dW, L.db])
        return grads

    def predict_proba(self, X):
        return np.asarray(self.forward(X)).ravel()
