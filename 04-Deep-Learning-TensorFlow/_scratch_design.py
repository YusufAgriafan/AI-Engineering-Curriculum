"""Scratch: freeze two-moons dataset + verify baselines for Bab 4 project tests."""
import json

import numpy as np


def buat_moons(n=600, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n_per = n // 2
    theta = rng.uniform(0, np.pi, n_per)
    moon0 = np.column_stack([np.cos(theta), np.sin(theta)])
    moon1 = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
    X = np.vstack([moon0, moon1]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n_per + [1] * n_per)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def split(X, y, fracs, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n1 = int(len(y) * fracs[0])
    n2 = int(len(y) * fracs[1])
    return (X[idx[:n1]], y[idx[:n1]],
            X[idx[n1:n1 + n2]], y[idx[n1:n1 + n2]],
            X[idx[n1 + n2:]], y[idx[n1 + n2:]])


X, y = buat_moons()
X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, (0.70, 0.15, 0.15))
print("splits:", len(y_tr), len(y_va), len(y_te), "| pos per split:",
      int(y_tr.sum()), int(y_va.sum()), int(y_te.sum()))
print("X range:", X.min(0).round(3), X.max(0).round(3))

np.savez("AI-Engineering-Curriculum/04-Deep-Learning-TensorFlow/project-starter-numpy-nn/nn_mini/data.npz",
         X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
print("saved npz")


# ---- logistic regression baseline (should struggle on moons) ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


m, n = X_tr.shape
w = np.zeros(n); b = 0.0
for _ in range(3000):
    p = sigmoid(X_tr @ w + b)
    w -= 0.5 * (X_tr.T @ (p - y_tr)) / m
    b -= 0.5 * np.sum(p - y_tr) / m
acc_lr = float(((sigmoid(X_va @ w + b) >= 0.5) == y_va).mean())
print("LR baseline val acc:", round(acc_lr, 4))


# ---- reference MLP (the solusi implementation, ahead of writing files) ----
class Dense:
    def __init__(self, units, activation="relu", input_dim=None, seed=None):
        self.units = units
        self.activation = activation
        r = np.random.default_rng(seed)
        self.W = None
        self.b = np.zeros(units)
        self._seed = seed
        self._r = r
        self.input_dim = input_dim

    def _init(self, fan_in):
        if self.activation == "relu":
            std = np.sqrt(2.0 / fan_in)          # He
        else:
            std = np.sqrt(1.0 / fan_in)          # Xavier-ish
        self.W = self._r.normal(0, std, size=(fan_in, self.units))

    def forward(self, X):
        if self.W is None:
            self._init(X.shape[1])
        self._X = X
        self._z = X @ self.W + self.b
        if self.activation == "relu":
            self._a = np.maximum(self._z, 0.0)
        elif self.activation == "sigmoid":
            self._a = 1.0 / (1.0 + np.exp(-self._z))
        else:
            self._a = self._z
        return self._a

    def backward(self, grad):
        if self.activation == "relu":
            dz = grad * (self._z > 0)
        elif self.activation == "sigmoid":
            dz = grad * self._a * (1 - self._a)
        else:
            dz = grad
        self.dW = self._X.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bce_grad(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return (p - y) / (p * (1 - p)) / len(y)


def build(sizes, seed=0):
    layers = []
    for i, (u, act) in enumerate(sizes):
        layers.append(Dense(u, act, seed=seed + i))
    return layers


def forward_all(layers, X):
    a = X
    for L in layers:
        a = L.forward(a)
    return a


layers = build([(16, "relu"), (8, "relu"), (1, "sigmoid")], seed=7)
epochs, bs, lr, patience = 400, 32, 0.1, 40
best_val, best_ep, wait = np.inf, 0, 0
hist = []
rng = np.random.default_rng(0)
for ep in range(epochs):
    idx = rng.permutation(len(y_tr))
    for s in range(0, len(y_tr), bs):
        bidx = idx[s:s + bs]
        Xb, yb = X_tr[bidx], y_tr[bidx]
        p = forward_all(layers, Xb).ravel()
        g = bce_grad(p, yb).reshape(-1, 1)
        for L in reversed(layers):
            g = L.backward(g)
        for L in layers:
            L.W -= lr * L.dW
            L.b -= lr * L.db
    pv = forward_all(layers, X_va).ravel()
    vloss = bce(y_va, pv)
    hist.append(vloss)
    if vloss < best_val - 1e-6:
        best_val, best_ep, wait = vloss, ep, 0
    else:
        wait += 1
        if wait >= patience:
            print(f"early stop at epoch {ep} (best {best_ep})")
            break

pt = forward_all(layers, X_te).ravel()
acc_te = float(((pt >= 0.5) == y_te).mean())
pv2 = forward_all(layers, X_va).ravel()
acc_va = float(((pv2 >= 0.5) == y_va).mean())
print("MLP val acc:", round(acc_va, 4), "| test acc:", round(acc_te, 4),
      "| best val bce:", round(best_val, 4), "| epochs run:", len(hist))
print("loss awal (epoch 0):", round(hist[0], 4), "-> best:", round(best_val, 4))

out = {"lr_val_acc": acc_lr, "mlp_val_acc": acc_va, "mlp_test_acc": acc_te,
       "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
       "pos_train": int(y_tr.sum()), "pos_val": int(y_va.sum()), "pos_test": int(y_te.sum())}
with open("AI-Engineering-Curriculum/04-Deep-Learning-TensorFlow/_scratch_reference.json", "w") as f:
    json.dump(out, f, indent=1)
print("done")
