"""Implementasi referensi ts_mini — untuk cek mandiri & verifier.

Buka HANYA setelah mencoba. Setiap fungsi punya padanan 1:1 dengan stub di
ts_mini/ — bandingkan baris demi baris dengan versimu.
"""

import numpy as np

PAD_ID = 0  # tidak dipakai di Bab 7; disediakan agar struktur sama dengan Bab 6
OOV_ID = 1

N_HARI = 365 * 3
SPLIT_IDX = int(N_HARI * 0.8)
HORIZON = N_HARI - SPLIT_IDX


# ============================== data ==============================

def make_series(seed=42, n_days=N_HARI):
    t = np.arange(n_days, dtype=float)
    rng = np.random.default_rng(seed)
    trend = 100.0 + 0.05 * t
    seasonality = 20.0 * np.sin(2 * np.pi * t / 365.0)
    noise = rng.normal(0.0, 5.0, n_days)
    return trend + seasonality + noise, t


def make_mult_series(seed=42, n_days=N_HARI):
    t = np.arange(n_days, dtype=float)
    rng = np.random.default_rng(seed)
    level = 100.0 + 0.05 * t
    seri = level * (1.0 + 0.15 * np.sin(2 * np.pi * t / 365.0))
    seri = seri + rng.normal(0.0, 4.0, n_days)
    return seri, t


# ============================== windowing ==============================

def buat_window(series, W):
    s = np.asarray(series, dtype=float)
    if W < 1 or W >= len(s):
        raise ValueError(f"W harus 1..len(series)-1, dapat W={W}, n={len(s)}")
    X = np.lib.stride_tricks.sliding_window_view(s, W)[:-1]
    return X, s[W:]


def split_waktu(X, y, rasio_train=0.8):
    cut = int(len(X) * rasio_train)
    X, y = np.asarray(X), np.asarray(y)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def buat_window_multi(series, W, h):
    s = np.asarray(series, dtype=float)
    if W < 1 or h < 1 or W + h > len(s):
        raise ValueError(f"W+h harus <= len(series), dapat W={W}, h={h}, n={len(s)}")
    X = np.lib.stride_tricks.sliding_window_view(s, W + h)[:, :W]
    y = np.lib.stride_tricks.sliding_window_view(s, h)[W:]
    return X, y


# ============================== baseline ==============================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2 * np.abs(y_true - y_pred)
                         / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))


def forecast_naive(train, h):
    return np.full(h, float(np.asarray(train)[-1]))


def forecast_seasonal_naive(train, h, period):
    tr = np.asarray(train, dtype=float)
    return np.array([tr[len(tr) - period + j % period] for j in range(h)])


def forecast_ham(train, h):
    return np.full(h, float(np.mean(train)))


# ============================== AR(p) ==============================

def fit_ar(series, p):
    s = np.asarray(series, dtype=float)
    X = np.column_stack([s[p - k - 1:len(s) - k - 1] for k in range(p)])
    y = s[p:]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def ar_forecast(coef, last_values, h):
    hist = list(last_values)[::-1]   # hist[0] = terbaru
    out = []
    for _ in range(h):
        yhat = float(np.asarray(coef) @ np.array(hist[:len(coef)]))
        out.append(yhat)
        hist.insert(0, yhat)
    return np.array(out)


# ============================== Holt-Winters ==============================

def holt_winters_add(tr, period, alpha, beta, gamma, h):
    n, m = len(tr), period
    level0 = float(tr[:m].mean())
    level, trend_b = level0, 0.0
    seas = list(tr[:m] - level0)
    for i in range(m, n):
        lv_old = level
        level = alpha * (tr[i] - seas[i - m]) + (1 - alpha) * (level + trend_b)
        trend_b = beta * (level - lv_old) + (1 - beta) * trend_b
        seas.append(gamma * (tr[i] - level) + (1 - gamma) * seas[i - m])
    f = [level + (j + 1) * trend_b + seas[n - m + (j % m)] for j in range(h)]
    return np.array(f), level, trend_b, np.array(seas[-m:])


def holt_winters_mult(tr, period, alpha, beta, gamma, h):
    n, m = len(tr), period
    eps = 1e-8
    level0 = float(tr[:m].mean())
    level, trend_b = level0, 0.0
    seas = list(tr[:m] / (level0 + eps))
    for i in range(m, n):
        lv_old = level
        level = alpha * (tr[i] / (seas[i - m] + eps)) + (1 - alpha) * (level + trend_b)
        trend_b = beta * (level - lv_old) + (1 - beta) * trend_b
        seas.append(gamma * (tr[i] / (level + eps)) + (1 - gamma) * seas[i - m])
    f = [(level + (j + 1) * trend_b) * seas[n - m + (j % m)] for j in range(h)]
    return np.array(f)


# ============================== backtesting ==============================

def rolling_backtest(tr, h, build_forecast, n_folds=3):
    metrik = []
    n = len(tr)
    for k in range(n_folds):
        cut = n - (k + 1) * h
        trk, tek = tr[:cut], tr[cut:cut + h]
        fc = build_forecast(trk, h)
        metrik.append((mae(tek, fc), rmse(tek, fc)))
    return metrik
