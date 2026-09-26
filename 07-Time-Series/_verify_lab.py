"""Verify 01_lab_time_series.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# console Windows (cp1252) tak bisa mencetak emoji notebook -> paksa UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NB = Path(__file__).resolve().parent / "01_lab_time_series.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "buat_trend_linear": (
        "def buat_trend_linear(series, t):\n"
        "    tc = t - t.mean()\n"
        "    yc = np.asarray(series, dtype=float)\n"
        "    b = float(np.sum(tc * (yc - yc.mean())) / np.sum(tc * tc))\n"
        "    a = float(yc.mean() - b * t.mean())\n"
        "    return a, b\n"
    ),
    "acf": (
        "def acf(x, lag):\n"
        "    xc = np.asarray(x, dtype=float) - np.mean(x)\n"
        "    return float(np.sum(xc[:-lag] * xc[lag:]) / np.sum(xc * xc))\n"
    ),
    "buat_window": (
        "def buat_window(series, W):\n"
        "    s = np.asarray(series, dtype=float)\n"
        "    X = np.lib.stride_tricks.sliding_window_view(s, W)[:-1]\n"
        "    return X, s[W:]\n"
    ),
    "split_waktu": (
        "def split_waktu(X, y, rasio_train=0.8):\n"
        "    cut = int(len(X) * rasio_train)\n"
        "    X, y = np.asarray(X), np.asarray(y)\n"
        "    return X[:cut], y[:cut], X[cut:], y[cut:]\n"
    ),
    "mae": (
        "def mae(y_true, y_pred):\n"
        "    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))\n"
    ),
    "fit_ar": (
        "def fit_ar(series, p):\n"
        "    s = np.asarray(series, dtype=float)\n"
        "    X = np.column_stack([s[p - k - 1:len(s) - k - 1] for k in range(p)])\n"
        "    y = s[p:]\n"
        "    return np.linalg.lstsq(X, y, rcond=None)[0]\n"
    ),
    "ar_forecast": (
        "def ar_forecast(coef, last_values, h):\n"
        "    hist = list(last_values)[::-1]\n"
        "    out = []\n"
        "    for _ in range(h):\n"
        "        yhat = float(np.asarray(coef) @ np.array(hist[:len(coef)]))\n"
        "        out.append(yhat)\n"
        "        hist.insert(0, yhat)\n"
        "    return np.array(out)\n"
    ),
    "holt_winters_add": (
        "def holt_winters_add(tr, period, alpha, beta, gamma, h):\n"
        "    n, m = len(tr), period\n"
        "    level0 = float(tr[:m].mean())\n"
        "    level, trend_b = level0, 0.0\n"
        "    seas = list(tr[:m] - level0)\n"
        "    for i in range(m, n):\n"
        "        lv_old = level\n"
        "        level = alpha * (tr[i] - seas[i - m]) + (1 - alpha) * (level + trend_b)\n"
        "        trend_b = beta * (level - lv_old) + (1 - beta) * trend_b\n"
        "        seas.append(gamma * (tr[i] - level) + (1 - gamma) * seas[i - m])\n"
        "    f = [level + (j + 1) * trend_b + seas[n - m + (j % m)] for j in range(h)]\n"
        "    return np.array(f), level, trend_b, np.array(seas[-m:])\n"
    ),
    "holt_winters_mult": (
        "def holt_winters_mult(tr, period, alpha, beta, gamma, h):\n"
        "    n, m = len(tr), period\n"
        "    eps = 1e-8\n"
        "    level0 = float(tr[:m].mean())\n"
        "    level, trend_b = level0, 0.0\n"
        "    seas = list(tr[:m] / (level0 + eps))\n"
        "    for i in range(m, n):\n"
        "        lv_old = level\n"
        "        level = alpha * (tr[i] / (seas[i - m] + eps)) + (1 - alpha) * (level + trend_b)\n"
        "        trend_b = beta * (level - lv_old) + (1 - beta) * trend_b\n"
        "        seas.append(gamma * (tr[i] / (level + eps)) + (1 - gamma) * seas[i - m])\n"
        "    f = [(level + (j + 1) * trend_b) * seas[n - m + (j % m)] for j in range(h)]\n"
        "    return np.array(f)\n"
    ),
    "pdq_fit": (
        "def pdq_fit(tr, period):\n"
        "    m = period\n"
        "    y = np.asarray(tr, dtype=float)\n"
        "    n = len(y)\n"
        "    X = np.zeros((n, 2 + m))\n"
        "    X[:, 0] = 1.0\n"
        "    X[:, 1] = np.arange(n, dtype=float)\n"
        "    idx = np.arange(n)\n"
        "    X[idx >= m, 2 + (idx[idx >= m] % m)] = 1.0\n"
        "    return np.linalg.lstsq(X, y, rcond=None)[0]\n"
    ),
    "pdq_forecast": (
        "def pdq_forecast(beta, n_train, period, h):\n"
        "    m = period\n"
        "    b0, b1, seas = beta[0], beta[1], beta[2:]\n"
        "    return np.array([b0 + b1 * (n_train + j) + seas[(n_train + j) % m] for j in range(h)])\n"
    ),
    "metrik_semua": (
        "def metrik_semua(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true, dtype=float)\n"
        "    y_pred = np.asarray(y_pred, dtype=float)\n"
        "    err = y_true - y_pred\n"
        "    mae_ = float(np.mean(np.abs(err)))\n"
        "    rmse = float(np.sqrt(np.mean(err ** 2)))\n"
        "    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))\n"
        "    return {'mae': mae_, 'rmse': rmse, 'smape': smape}\n"
    ),
    "rolling_backtest": (
        "def rolling_backtest(tr, h, build_forecast, n_folds=3):\n"
        "    metrik = []\n"
        "    n = len(tr)\n"
        "    for k in range(n_folds):\n"
        "        cut = n - (k + 1) * h\n"
        "        trk, tek = tr[:cut], tr[cut:cut + h]\n"
        "        fc = build_forecast(trk, h)\n"
        "        metrik.append((float(np.mean(np.abs(tek - fc))),\n"
        "                       float(np.sqrt(np.mean((tek - fc) ** 2)))))\n"
        "    return metrik\n"
    ),
}

ctx = {"__name__": "__main__"}
ctx["np"] = np
ctx["plt"] = plt
errors = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src and "raise NotImplementedError" in src:
        injected = False
        for name, refsrc in REF.items():
            if f"def {name}(" in src:
                try:
                    exec(refsrc, ctx)
                    print(f"[cell {i}] injected ref: {name}")
                except Exception:
                    errors.append((i, f"inject {name}", traceback.format_exc()))
                injected = True
        if not injected:
            errors.append((i, "todo-cell-without-ref", src[:80]))
        continue
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ctx)
        print(f"[cell {i}] ok")
    except Exception:
        errors.append((i, "exec", traceback.format_exc()))

print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for i, kind, tb in errors:
        print(f"\n--- cell {i} ({kind}) ---\n{tb}")
    sys.exit(1)
print("LAB OK: semua cell (dengan implementasi referensi) lolos.")
