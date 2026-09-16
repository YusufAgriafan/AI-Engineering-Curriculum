"""Implementasi referensi — untuk memverifikasi tests/ bisa hijau.

Buka HANYA setelah mencoba sendiri atau benar-benar mentok.
"""

import numpy as np


# ---------------- metrik ----------------

def _cek(y):
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y hanya boleh berisi 0 dan 1")
    return y

def confusion_matrix(y_true, y_pred):
    y_true = _cek(y_true)
    y_pred = _cek(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true dan y_pred harus sama bentuk")
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    if total == 0:
        return 0.0
    return (cm["TP"] + cm["TN"]) / total


def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    p = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0.0
    r = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def roc_auc(y_true, y_score):
    y_true = _cek(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true dan y_score harus sama bentuk")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC butuh kedua kelas hadir di y_true")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    s_sorted = y_score[order]
    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------- model ----------------

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


class LogisticRegressionGD:
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
            p = sigmoid(X @ self.w + self.b)
            self.loss_history.append(binary_cross_entropy(y, p))
            dw = X.T @ (p - y) / m
            db = np.sum(p - y) / m
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        if self.w is None:
            raise RuntimeError("model belum di-fit — panggil fit() dulu")
        return sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------- clustering ----------------

def standardize(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    aman = np.where(std == 0, 1.0, std)
    return (X - mean) / aman


def kmeans(X, k, n_init=10, max_iter=300, tol=1e-6, seed=0):
    X = np.asarray(X, dtype=float)
    if k < 1:
        raise ValueError("k harus >= 1")
    if len(X) < k:
        raise ValueError("jumlah titik tidak boleh kurang dari k")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centroids[None]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            new_centroids = np.array([
                X[labels == kk].mean(axis=0) if (labels == kk).any() else centroids[kk]
                for kk in range(k)
            ])
            if np.allclose(new_centroids, centroids, atol=tol):
                centroids = new_centroids
                break
            centroids = new_centroids
        d2 = ((X[:, None, :] - centroids[None]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels, centroids)
    inertia, labels, centroids = best
    return labels, centroids, inertia
