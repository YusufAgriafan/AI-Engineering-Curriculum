"""Implementasi referensi — untuk memverifikasi tests/ bisa hijau.

Buka HANYA setelah mencoba sendiri atau benar-benar mentok.
"""

import numpy as np


# ---------------- activations ----------------

def relu(z):
    return np.maximum(np.asarray(z, dtype=float), 0.0)


def relu_grad(z):
    return (np.asarray(z, dtype=float) > 0).astype(float)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_from_logits_stable(logits):
    logits = np.asarray(logits, dtype=float)
    p = sigmoid(logits)
    return p, p * (1.0 - p)


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------- losses ----------------

def bce(y_true, y_pred, eps=1e-12):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    if y.shape != p.shape:
        raise ValueError("bentuk y_true dan y_pred harus sama")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bce_with_logits_grad(y_pred, y_true, eps=1e-12):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    m = len(y)
    return (p - y) / (p * (1 - p)) / m


def mse(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError("bentuk y_true dan y_pred harus sama")
    return float(np.mean((p - y) ** 2))


def mse_grad(y_pred, y_true):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return 2.0 * (p - y) / len(y)


# ---------------- layers ----------------

class Dense:
    def __init__(self, units, activation="relu", seed=None):
        if activation not in ("relu", "sigmoid", "linear"):
            raise ValueError(f"aktivasi harus salah satu dari ('relu', 'sigmoid', 'linear'), dapat {activation!r}")
        self.units = int(units)
        self.activation = activation
        self._rng = np.random.default_rng(seed)
        self.W = None
        self.b = np.zeros(self.units)

    def _init_weights(self, fan_in):
        std = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
        self.W = self._rng.normal(0, std, size=(fan_in, self.units))

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("input Dense harus matriks 2D (n, fan_in)")
        if self.W is None:
            self._init_weights(X.shape[1])
        self._X = X
        self._z = X @ self.W + self.b
        if self.activation == "relu":
            self._a = np.maximum(self._z, 0.0)
        elif self.activation == "sigmoid":
            self._a = sigmoid(self._z)
        else:
            self._a = self._z
        return self._a

    def backward(self, grad):
        if self.W is None:
            raise RuntimeError("forward belum pernah dipanggil")
        if self.activation == "relu":
            dz = grad * (self._z > 0).astype(float)
        elif self.activation == "sigmoid":
            p, dpdz = sigmoid_from_logits_stable(self._z)
            dz = grad * dpdz
        else:
            dz = grad
        self.dW = self._X.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T


class Sequential:
    def __init__(self, layers):
        if not layers:
            raise ValueError("Sequential butuh minimal 1 layer")
        self.layers = list(layers)

    def forward(self, X):
        a = np.asarray(X, dtype=float)
        for L in self.layers:
            a = L.forward(a)
        return a

    def backward(self, grad):
        g = np.asarray(grad, dtype=float)
        for L in reversed(self.layers):
            g = L.backward(g)
        return g

    def parameters(self):
        params = []
        for L in self.layers:
            if L.W is None:
                raise RuntimeError("ada layer yang belum pernah forward — panggil forward() dulu")
            params.extend([L.W, L.b])
        return params

    def predict_proba(self, X):
        return np.asarray(self.forward(X)).ravel()


# ---------------- train ----------------

class SGD:
    def __init__(self, lr=0.1):
        self.lr = float(lr)

    def step(self, params, grads):
        if len(params) != len(grads):
            raise ValueError("params dan grads harus sama panjang")
        for P, G in zip(params, grads):
            P -= self.lr * G


def iter_batch(X, y, batch_size, seed=None):
    if len(X) != len(y):
        raise ValueError("X dan y harus sama panjang")
    if batch_size < 1:
        raise ValueError("batch_size harus >= 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    for s in range(0, len(y), batch_size):
        idx = order[s:s + batch_size]
        yield X[idx], y[idx]


def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32,
                lr=0.1, patience=None, seed=0, verbose=False):
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
            grads = []
            for L in model.layers:
                grads.extend([L.dW, L.db])
            opt.step(model.parameters(), grads)

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
