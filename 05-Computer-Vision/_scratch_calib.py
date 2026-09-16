"""Scratch: kalibrasi hyperparameter latih_cnn untuk lab Bab 5."""
import sys
from pathlib import Path

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


# --- ref impls (identik dengan _verify_lab.py) ---
def pad_zero(img, pad):
    if pad == 0:
        return np.asarray(img, dtype=float).copy()
    return np.pad(np.asarray(img, dtype=float), pad, mode='constant')


def conv2d_forward(img, K, pad=0, stride=1):
    xp = pad_zero(img, pad)
    win = sliding_window_view(xp, K.shape)[::stride, ::stride]
    return np.tensordot(win, K, axes=([2, 3], [0, 1]))


def conv2d_backward(dOut, img, K, pad=0, stride=1):
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    Ho, Wo = dOut.shape
    xp = pad_zero(img, pad)
    winK = sliding_window_view(xp, K.shape)[::stride, ::stride]
    dK = np.tensordot(dOut, winK, axes=([0, 1], [0, 1]))
    dil = np.zeros(((Ho - 1) * stride + 1, (Wo - 1) * stride + 1))
    dil[::stride, ::stride] = dOut
    Kf = K[::-1, ::-1]
    d = np.pad(dil, [(K.shape[0] - 1, K.shape[0] - 1),
                     (K.shape[1] - 1, K.shape[1] - 1)])
    dX_full = np.tensordot(sliding_window_view(d, K.shape), Kf,
                           axes=([2, 3], [0, 1]))
    dX = dX_full[pad:pad + H, pad:pad + W]
    return dK, dX


def maxpool2x2_backward(dOut, x):
    H, W = x.shape
    dX = np.zeros_like(x, dtype=float)
    for i in range(H // 2):
        for j in range(W // 2):
            win = x[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            idx = np.unravel_index(np.argmax(win), win.shape)
            dX[2 * i + idx[0], 2 * j + idx[1]] += dOut[i, j]
    return dX


# --- layer (ref) ---
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        return dout * self.mask

    def step(self):
        pass


class Pool2x2:
    def forward(self, X):
        self.X = X
        n, H, W = X.shape
        return X.reshape(n, H // 2, 2, W // 2, 2).max(axis=(2, 4))

    def backward(self, dOut):
        dX = np.zeros_like(self.X, dtype=float)
        for s in range(self.X.shape[0]):
            dX[s] = maxpool2x2_backward(dOut[s], self.X[s])
        return dX

    def step(self):
        pass


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def step(self):
        pass


class ConvLayer:
    def __init__(self, ksize, rng, lr=0.05, pad=0):
        self.K = rng.normal(0, np.sqrt(2.0 / (ksize * ksize)), (ksize, ksize))
        self.lr = lr
        self.pad = pad

    def forward(self, X):
        self.X = X
        return np.stack([conv2d_forward(x, self.K, pad=self.pad) for x in X])

    def backward(self, dOut):
        self.dK = np.zeros_like(self.K)
        dX = np.empty_like(self.X)
        for s in range(self.X.shape[0]):
            dKs, dXs = conv2d_backward(dOut[s], self.X[s], self.K, pad=self.pad)
            self.dK += dKs
            dX[s] = dXs
        return dX

    def step(self):
        self.K -= self.lr * self.dK


class Dense:
    def __init__(self, n_in, n_out, rng, lr=0.1):
        self.W = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.lr = lr

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(0)
        return dout @ self.W.T

    def step(self):
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- dataset Bars (identik dgn notebook) ---
def make_bars(n=600, img=8, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.05, (n, img, img))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        p = rng.integers(0, img)
        if rng.random() < 0.5:
            X[i, p, :] += 1.0
        else:
            y[i] = 1
            X[i, :, p] += 1.0
        for _ in range(4):
            X[i, rng.integers(img), rng.integers(img)] += rng.choice([-1., 1.]) * 0.8
    return X, y


X_all, y_all = make_bars()
X_tr, y_tr = X_all[:420], y_all[:420].astype(float)
X_va, y_va = X_all[420:510], y_all[420:510]
X_te, y_te = X_all[510:600], y_all[510:600]


def latih_cnn(epochs=30, lr=0.5, batch_size=32, seed=0):
    r = np.random.default_rng(seed)
    layers = [ConvLayer(3, r, lr=lr), ReLU(), Pool2x2(), Flatten(),
              Dense(9, 1, r, lr=lr)]
    hist = []
    for ep in range(epochs):
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(X_tr))
        for s in range(0, len(X_tr), batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = X_tr[idx], y_tr[idx]
            a = Xb
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - yb) / len(yb)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
        av = X_va
        for L in layers:
            av = L.forward(av)
        pv = (sigmoid(av.ravel()) >= 0.5).astype(int)
        # train loss (batch besar, tanpa update)
        tr_pred = []
        for s in range(0, len(X_tr), 128):
            a = X_tr[s:s + 128]
            for L in layers:
                a = L.forward(a)
            tr_pred.append(sigmoid(a.ravel()))
        ptr = np.concatenate(tr_pred)
        hist.append((bce(y_tr, ptr), float((pv == y_va).mean())))
    return layers, hist


def make_bars_pos(n, img=8, positions=None, seed=0):
    """Bars dgn posisi garis dibatasi ke daftar `positions` (baris/kolom)."""
    rng = np.random.default_rng(seed)
    if positions is None:
        positions = list(range(img))
    X = rng.normal(0, 0.05, (n, img, img))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        p = rng.choice(positions)
        if rng.random() < 0.5:
            X[i, p, :] += 1.0
        else:
            y[i] = 1
            X[i, :, p] += 1.0
        for _ in range(4):
            X[i, rng.integers(img), rng.integers(img)] += rng.choice([-1., 1.]) * 0.8
    return X, y


def latih_mlp(Xtr, ytr, epochs=40, lr=0.5, batch_size=32, seed=0, hidden=16):
    r = np.random.default_rng(seed)
    layers = [Flatten(), Dense(64, hidden, r, lr=lr), ReLU(), Dense(hidden, 1, r, lr=lr)]
    for ep in range(epochs):
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(Xtr))
        for s in range(0, len(Xtr), batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = Xtr[idx], ytr[idx]
            a = Xb
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - yb) / len(yb)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    return layers


def latih_cnn2(Xtr, ytr, Xva, yva, epochs=60, lr=0.5, batch_size=32, seed=0, dense_units=1):
    r = np.random.default_rng(seed)
    layers = [ConvLayer(3, r, lr=lr), ReLU(), Pool2x2(), Flatten(),
              Dense(9, dense_units, r, lr=lr)]
    if dense_units != 1:
        layers += [ReLU(), Dense(dense_units, 1, r, lr=lr)]
    hist = []
    for ep in range(epochs):
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(Xtr))
        for s in range(0, len(Xtr), batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = Xtr[idx], ytr[idx]
            a = Xb
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - yb) / len(yb)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
        a = Xtr
        for L in layers:
            a = L.forward(a)
        tr_loss = bce(ytr, sigmoid(a.ravel()))
        tr_acc = float(((sigmoid(a.ravel()) >= 0.5) == ytr).mean())
        a = Xva
        for L in layers:
            a = L.forward(a)
        pv = sigmoid(a.ravel())
        hist.append((tr_loss, tr_acc, bce(yva, pv), float(((pv >= 0.5) == yva).mean())))
    return layers, hist


def akurasi(model, X, y):
    a = X
    for L in model:
        a = L.forward(a)
    p = (sigmoid(a.ravel()) >= 0.5).astype(int)
    return float((p == y).mean())


X_ek, y_ek = make_bars_pos(90, positions=[0, 7], seed=99)
X_pos, y_pos = make_bars_pos(510, positions=[1, 2, 3, 4, 5, 6], seed=42)
Xtr_p, ytr_p = X_pos[:420], y_pos[:420].astype(float)
Xva_p, yva_p = X_pos[420:], y_pos[420:]

print("\n=== CNN pad=0 TANPA dense bias (posisi 1-6) ===")


class DenseNB(Dense):
    """Dense tanpa bias — leak shortcut kecerahan tertutup."""

    def __init__(self, n_in, n_out, rng, lr=0.5):
        super().__init__(n_in, n_out, rng, lr)
        self.b = np.zeros(n_out)

    def backward(self, dout):
        self.dW = self.x.T @ dout
        return dout @ self.W.T

    def step(self):
        self.W -= self.lr * self.dW


for sd in (0, 1, 2, 3, 4, 5):
    r = np.random.default_rng(sd)
    layers = [ConvLayer(3, r, lr=0.5, pad=0), ReLU(), Pool2x2(), Flatten(),
              DenseNB(9, 8, r, lr=0.5), ReLU(), DenseNB(8, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_p))
        for s in range(0, len(Xtr_p), 32):
            idx = order[s:s + 32]
            a = Xtr_p[idx]
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_p, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_p, yva_p):.3f} "
          f"| terlarang={akurasi(layers, X_ek, y_ek):.3f}")

print("\n=== CNN pad=0 TANPA bias + STANDARDISASI ===")


def standardize(X):
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd_ = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - mu) / sd_


Xtr_s = standardize(Xtr_p)
Xva_s = standardize(Xva_p)
Xek_s = standardize(X_ek)

for sd in (0, 1, 2, 3, 4, 5):
    r = np.random.default_rng(sd)
    layers = [ConvLayer(3, r, lr=0.5, pad=0), ReLU(), Pool2x2(), Flatten(),
              DenseNB(9, 8, r, lr=0.5), ReLU(), DenseNB(8, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_s))
        for s in range(0, len(Xtr_s), 32):
            idx = order[s:s + 32]
            a = Xtr_s[idx]
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_s, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_s, yva_p):.3f} "
          f"| terlarang={akurasi(layers, Xek_s, y_ek):.3f}")

print("\n=== DENGAN AUGMENTASI SHIFT (max_shift=2) ===")


def random_shift_arr(img, rng, max_shift=2):
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    dy = int(rng.integers(-max_shift, max_shift + 1))
    dx = int(rng.integers(-max_shift, max_shift + 1))
    out = np.zeros_like(img)
    ys, xs = slice(max(0, -dy), min(H, H - dy)), slice(max(0, -dx), min(W, W - dx))
    yd, xd = slice(max(0, dy), min(H, H + dy)), slice(max(0, dx), min(W, W + dx))
    out[yd, xd] = img[ys, xs]
    return out


print("--- CNN pad=1 + shift aug ---")
for sd in (0, 1, 2, 3, 4):
    r = np.random.default_rng(sd)
    rng_aug = np.random.default_rng(sd + 100)
    layers = [ConvLayer(3, r, lr=0.5, pad=1), ReLU(), Pool2x2(), Flatten(),
              Dense(16, 8, r, lr=0.5), ReLU(), Dense(8, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_p))
        for s in range(0, len(Xtr_p), 32):
            idx = order[s:s + 32]
            Xb = np.stack([random_shift_arr(x, rng_aug, 2) for x in Xtr_p[idx]])
            a = Xb
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_p, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_p, yva_p):.3f} "
          f"| terlarang={akurasi(layers, X_ek, y_ek):.3f}")

print("--- MLP + shift aug ---")
for sd in (0, 1, 2, 3, 4):
    r = np.random.default_rng(sd)
    rng_aug = np.random.default_rng(sd + 100)
    layers = [Flatten(), Dense(64, 16, r, lr=0.5), ReLU(), Dense(16, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_p))
        for s in range(0, len(Xtr_p), 32):
            idx = order[s:s + 32]
            Xb = np.stack([random_shift_arr(x, rng_aug, 2) for x in Xtr_p[idx]])
            a = Xb
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_p, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_p, yva_p):.3f} "
          f"| terlarang={akurasi(layers, X_ek, y_ek):.3f}")

print("\n=== DENGAN STANDARDISASI PER-GAMBAR (buang shortcut kecerahan) ===")

def standardize(X):
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - mu) / sd

Xtr_s = standardize(Xtr_p)
Xva_s = standardize(Xva_p)
Xek_s = standardize(X_ek)

print("--- CNN pad=1, standardisasi ---")
for sd in (0, 1, 2, 3, 4):
    r = np.random.default_rng(sd)
    layers = [ConvLayer(3, r, lr=0.5, pad=1), ReLU(), Pool2x2(), Flatten(),
              Dense(16, 8, r, lr=0.5), ReLU(), Dense(8, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_s))
        for s in range(0, len(Xtr_s), 32):
            idx = order[s:s + 32]
            a = Xtr_s[idx]
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_s, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_s, yva_p):.3f} "
          f"| terlarang={akurasi(layers, Xek_s, y_ek):.3f}")

print("--- MLP standardisasi ---")
for sd in (0, 1, 2, 3, 4):
    m = latih_mlp(Xtr_s, ytr_p, epochs=40, seed=sd)
    print(f"seed {sd}: in={akurasi(m, Xtr_s, ytr_p):.3f} "
          f"| terlarang={akurasi(m, Xek_s, y_ek):.3f}")

print("--- CNN pad=0, standardisasi ---")
for sd in (0, 1, 2, 3, 4):
    r = np.random.default_rng(sd)
    layers = [ConvLayer(3, r, lr=0.5, pad=0), ReLU(), Pool2x2(), Flatten(),
              Dense(9, 8, r, lr=0.5), ReLU(), Dense(8, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_s))
        for s in range(0, len(Xtr_s), 32):
            idx = order[s:s + 32]
            a = Xtr_s[idx]
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_s, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_s, yva_p):.3f} "
          f"| terlarang={akurasi(layers, Xek_s, y_ek):.3f}")

print("\n--- CNN 60 epoch pad=1 (posisi 1-6) ---")
for sd in (0, 1, 2, 3, 4):
    r = np.random.default_rng(sd)
    layers = [ConvLayer(3, r, lr=0.5, pad=1), ReLU(), Pool2x2(), Flatten(),
              Dense(16, 1, r, lr=0.5)]
    for ep in range(60):
        order = np.random.default_rng(sd * 1000 + ep).permutation(len(Xtr_p))
        for s in range(0, len(Xtr_p), 32):
            idx = order[s:s + 32]
            a = Xtr_p[idx]
            for L in layers:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr_p[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(layers):
                d = L.backward(d)
            for L in layers:
                L.step()
    print(f"seed {sd}: in={akurasi(layers, Xtr_p, ytr_p):.3f} "
          f"va={akurasi(layers, Xva_p, yva_p):.3f} "
          f"| terlarang={akurasi(layers, X_ek, y_ek):.3f}")

print("--- CNN 60 epoch pad=0 (posisi 1-6) ---")
for sd in (0, 1, 2):
    model, hist = latih_cnn2(Xtr_p, ytr_p, Xva_p, yva_p.astype(float), epochs=60, seed=sd)
    tl, ta, vl, va = hist[-1]
    print(f"seed {sd}: tr_loss={tl:.3f} tr_acc={ta:.3f} va_acc={va:.3f} "
          f"| terlarang={akurasi(model, X_ek, y_ek):.3f}")

print("--- MLP 40 epoch (posisi 1-6) ---")
for sd in (0, 1, 2):
    m = latih_mlp(Xtr_p, ytr_p, epochs=40, seed=sd)
    print(f"seed {sd}: in={akurasi(m, Xtr_p, ytr_p):.3f} "
          f"| terlarang={akurasi(m, X_ek, y_ek):.3f}")

print("--- overfitting: 9->128->1, 120 sampel, 300 epoch ---")
X_kecil, y_kecil = X_tr[:120], y_tr[:120].astype(float)


def cnn_besar(seed=3):
    r = np.random.default_rng(seed)
    return [ConvLayer(3, r, lr=0.5), ReLU(), Pool2x2(), Flatten(),
            Dense(9, 128, r, lr=0.5), ReLU(), Dense(128, 1, r, lr=0.5)]


def latih_hist(builder, Xtr, ytr, Xva, yva, epochs=300, batch_size=32, seed=0):
    model = builder(seed)
    tr_h, va_h = [], []
    for ep in range(epochs):
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(Xtr))
        for s in range(0, len(Xtr), batch_size):
            idx = order[s:s + batch_size]
            a = Xtr[idx]
            for L in model:
                a = L.forward(a)
            p = sigmoid(a.ravel())
            d = ((p - ytr[idx]) / len(idx)).reshape(-1, 1)
            for L in reversed(model):
                d = L.backward(d)
            for L in model:
                L.step()
        a = Xtr
        for L in model:
            a = L.forward(a)
        tr_h.append(bce(ytr, sigmoid(a.ravel())))
        a = Xva
        for L in model:
            a = L.forward(a)
        va_h.append(bce(yva, sigmoid(a.ravel())))
    return tr_h, va_h


for sd in (0, 1, 2):
    tr_h, va_h = latih_hist(cnn_besar, X_kecil, y_kecil,
                            X_va, y_va.astype(float), epochs=300, seed=sd)
    best = int(np.argmin(va_h))
    print(f"seed {sd}: tr_akhir={tr_h[-1]:.3f} va_min={min(va_h):.3f}@{best} "
          f"va_akhir={va_h[-1]:.3f} naik={va_h[-1] - min(va_h):+.3f}")
