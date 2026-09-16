"""Layer Dense dari nol — forward, backward, parameter.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak ada di docstring dan di
tests/test_layers.py.

Matematika per layer:
    forward:  z = X @ W + b   ;   a = aktivasi(z)
    backward: dz = grad * aktivasi'(z)
              dW = Xᵀ @ dz    ;   db = dz.sum(axis=0)
              grad untuk layer sebelumnya = dz @ Wᵀ
"""

import numpy as np

from .activations import relu, relu_grad, sigmoid, sigmoid_from_logits_stable

_AKTIVASI = ("relu", "sigmoid", "linear")


class Dense:
    """Fully-connected layer dengan aktivasi pilihan.

    - relu    : hidden layer (init He: std = sqrt(2/fan_in))
    - sigmoid : output biner (init Xavier: std = sqrt(1/fan_in))
    - linear  : output regresi (init Xavier)
    """

    def __init__(self, units, activation="relu", seed=None):
        if activation not in _AKTIVASI:
            raise ValueError(f"aktivasi harus salah satu dari {_AKTIVASI}, dapat {activation!r}")
        self.units = int(units)
        self.activation = activation
        self._rng = np.random.default_rng(seed)
        self.W = None          # (fan_in, units) — diisi saat forward pertama
        self.b = np.zeros(self.units)

    # ------------------------------------------------------------------
    def _init_weights(self, fan_in):
        """Inisialisasi W sesuai jenis aktivasi (He untuk relu, Xavier lainnya)."""
        std = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
        # TODO: self.W = self._rng.normal(0, std, size=(fan_in, self.units))
        raise NotImplementedError

    # ------------------------------------------------------------------
    def forward(self, X):
        """Simpan X dan z untuk backward, return a = aktivasi(X @ W + b)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("input Dense harus matriks 2D (n, fan_in)")
        if self.W is None:
            self._init_weights(X.shape[1])
        self._X = X
        # TODO: hitung z, hitung a sesuai self.activation, simpan keduanya
        #       (a. relu -> np.maximum ; b. sigmoid -> fungsi dari activations ;
        #        c. linear -> a = z), lalu return a
        raise NotImplementedError

    # ------------------------------------------------------------------
    def backward(self, grad):
        """Terima dL/da, return dL/dX. Simpan dW dan db untuk optimizer.

        Untuk sigmoid, pakai sigmoid_from_logits_stable agar dL/dz didapat
        secara stabil (p * (1 - p)) tanpa menghitung ulang sigmoid.
        """
        if self.W is None:
            raise RuntimeError("forward belum pernah dipanggil")
        if self.activation == "relu":
            dz = grad * relu_grad(self._z)
        elif self.activation == "sigmoid":
            p, dpdz = sigmoid_from_logits_stable(self._z)
            dz = grad * dpdz
        else:
            dz = grad
        # TODO: dW = Xᵀ @ dz ; db = dz.sum(axis=0) ; return dz @ Wᵀ
        raise NotImplementedError


class Sequential:
    """Tumpukan Dense — versi mini dari keras.Sequential."""

    def __init__(self, layers):
        if not layers:
            raise ValueError("Sequential butuh minimal 1 layer")
        self.layers = list(layers)

    def forward(self, X):
        """Propagasi maju lewat semua layer. Return output layer terakhir."""
        # TODO
        raise NotImplementedError

    def backward(self, grad):
        """Propagasi mundur (urutan TERBALIK). Return dL/dX input."""
        # TODO: loop reversed(self.layers)
        raise NotImplementedError

    def parameters(self):
        """Return list [W1, b1, W2, b2, ...] — urutan penting untuk optimizer."""
        params = []
        for L in self.layers:
            if L.W is None:
                raise RuntimeError("ada layer yang belum pernah forward — panggil forward() dulu")
            params.extend([L.W, L.b])
        return params

    def predict_proba(self, X):
        """Forward + ravel — output sigmoid 1D probabilitas (asumsi output layer sigmoid)."""
        return np.asarray(self.forward(X)).ravel()
