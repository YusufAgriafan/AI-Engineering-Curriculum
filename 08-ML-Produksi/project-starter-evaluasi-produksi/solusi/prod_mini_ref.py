"""Implementasi referensi prod_mini — untuk cek mandiri & verifier.

Buka HANYA setelah mencoba. Setiap fungsi punya padanan 1:1 dengan stub di
prod_mini/ — bandingkan baris demi baris dengan versimu.
"""

import hashlib
from pathlib import Path

import numpy as np

PAD_ID = 0  # tidak dipakai di Bab 8; disediakan agar struktur sama dengan Bab 6/7
OOV_ID = 1

# ============================== data ==============================

N = 800
TRUE_W = np.array([2.0, -1.5])
TRUE_B = 1.0
SIGMA = 1.2
CUT_TRAIN = int(N * 0.8)
CUT_CALIB = int(N * 0.9)
GAN_MEAN, GAN_STD = 2.0, 0.5


def make_dataset(seed=8, n=N):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.5, size=(n, 2))
    noise = rng.normal(0.0, SIGMA, n)
    y = X @ TRUE_W + TRUE_B + noise
    return X, y


def split_tiga_arah(X, y):
    return (X[:CUT_TRAIN], y[:CUT_TRAIN],
            X[CUT_TRAIN:CUT_CALIB], y[CUT_TRAIN:CUT_CALIB],
            X[CUT_CALIB:], y[CUT_CALIB:])


def make_labels(y_train, *parts):
    med = float(np.median(y_train))
    return [(np.asarray(p) > med).astype(int) for p in parts]


def make_gan_data(seed=8, n=512):
    rng = np.random.default_rng(seed)
    return rng.normal(GAN_MEAN, GAN_STD, n)


# ============================== metrics ==============================

def rmse(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def skill_score(rmse_model, rmse_ref):
    return 1.0 - rmse_model / rmse_ref


# ============================== models ==============================

def fit_ols(Xa, ya):
    Xm = np.column_stack([np.asarray(Xa, dtype=float), np.ones(len(Xa))])
    coef, *_ = np.linalg.lstsq(Xm, ya, rcond=None)
    return coef[:-1], float(coef[-1])


def predict(w, b, Xa):
    return np.asarray(Xa, dtype=float) @ np.asarray(w, dtype=float) + b


def fit_ham(ya):
    return float(np.mean(ya))


def predict_ham(mean_y, Xa):
    return np.full(len(Xa), mean_y)


def fit_neg(Xa, ya):
    X1 = np.asarray(Xa, dtype=float)[:, 1:2]
    neg = np.column_stack([-X1, np.ones(len(X1))])
    coef, *_ = np.linalg.lstsq(neg, ya, rcond=None)
    return np.array([-coef[0]]), float(coef[1])


def predict_neg(w, b, Xa):
    X1 = np.asarray(Xa, dtype=float)[:, 1:2]
    return -X1 @ w + b


# ============================== persist ==============================

def save_model(path, w, b):
    np.savez(path, w=np.asarray(w, dtype=float), b=np.array(float(b)))


def load_model(path):
    with np.load(path) as z:
        return z["w"], float(z["b"])


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def roundtrip_ok(w, b, X_uji, prediksi_sebelum):
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "roundtrip.npz")
    save_model(path, w, b)
    w2, b2 = load_model(path)
    pred_baru = predict(w2, b2, X_uji)
    return bool(np.array_equal(pred_baru, prediksi_sebelum))


# ============================== threshold ==============================

def confusion(y_true01, y_score, thr):
    y_true01 = np.asarray(y_true01).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= thr).astype(int)
    return {"tp": int(np.sum((y_true01 == 1) & (pred == 1))),
            "fp": int(np.sum((y_true01 == 0) & (pred == 1))),
            "fn": int(np.sum((y_true01 == 1) & (pred == 0))),
            "tn": int(np.sum((y_true01 == 0) & (pred == 0)))}


def prf(cm):
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pilih_threshold(y_true01, y_score, kandidat):
    best = None
    for thr in kandidat:
        thr = float(thr)
        cm = confusion(y_true01, y_score, thr)
        m = prf(cm)
        if best is None or m["f1"] > best[2]["f1"]:
            best = (thr, cm, m)
    return best


# ============================== calibration ==============================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def binned_calibration(y_true01, p, n_bins=10):
    y_true01 = np.asarray(y_true01).astype(int)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": nb,
                     "conf": float(p[mask].mean()) if nb else 0.0,
                     "acc": float(y_true01[mask].mean()) if nb else 0.0})
    return bins


def ece(y_true01, p, n_bins=10):
    bins = binned_calibration(y_true01, p, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] / n * abs(b["acc"] - b["conf"])
                     for b in bins if b["n"] > 0))


def fit_platt(score, y01, lr=0.1, epochs=3000, seed=8):
    s = np.asarray(score, dtype=float)
    y = np.asarray(y01, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(epochs):
        p = sigmoid(a * s + b)
        ga = float(np.mean((p - y) * s))
        gb = float(np.mean(p - y))
        a -= lr * ga
        b -= lr * gb
    return a, b


def apply_platt(a, b, score):
    return sigmoid(a * np.asarray(score, dtype=float) + b)
