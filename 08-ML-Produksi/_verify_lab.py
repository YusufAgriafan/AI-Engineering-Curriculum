"""Verify 01_lab_ml_produksi.ipynb: exec every cell; TODO cells get reference impls.

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

NB = Path(__file__).resolve().parent / "01_lab_ml_produksi.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "split_tiga_arah": (
        "def split_tiga_arah(X, y, r1=0.8, r2=0.9):\n"
        "    n = len(X)\n"
        "    c1, c2 = int(n * r1), int(n * r2)\n"
        "    return X[:c1], y[:c1], X[c1:c2], y[c1:c2], X[c2:], y[c2:]\n"
    ),
    "sha256_bytes": (
        "def sha256_bytes(b):\n"
        "    import hashlib\n"
        "    return hashlib.sha256(b).hexdigest()\n"
    ),
    "fit_ols": (
        "def fit_ols(Xa, ya):\n"
        "    Xm = np.column_stack([np.asarray(Xa, dtype=float), np.ones(len(Xa))])\n"
        "    coef, *_ = np.linalg.lstsq(Xm, ya, rcond=None)\n"
        "    return coef[:-1], float(coef[-1])\n"
    ),
    "predict": (
        "def predict(w, b, Xa):\n"
        "    return np.asarray(Xa, dtype=float) @ np.asarray(w, dtype=float) + b\n"
    ),
    "save_model": (
        "def save_model(path, w, b):\n"
        "    np.savez(path, w=np.asarray(w, dtype=float), b=np.array(float(b)))\n"
    ),
    "load_model": (
        "def load_model(path):\n"
        "    with np.load(path) as z:\n"
        "        return z['w'], float(z['b'])\n"
    ),
    "rmse": (
        "def rmse(y_true, y_pred):\n"
        "    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)\n"
        "    return float(np.sqrt(np.mean(err ** 2)))\n"
    ),
    "mae": (
        "def mae(y_true, y_pred):\n"
        "    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))\n"
    ),
    "r2": (
        "def r2(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true, dtype=float)\n"
        "    ss_res = float(np.sum((y_true - y_pred) ** 2))\n"
        "    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))\n"
        "    return 1.0 - ss_res / ss_tot\n"
    ),
    "fit_ham": (
        "def fit_ham(ya):\n"
        "    return float(np.mean(ya))\n"
    ),
    "predict_ham": (
        "def predict_ham(mean_y, Xa):\n"
        "    return np.full(len(Xa), mean_y)\n"
    ),
    "fit_neg": (
        "def fit_neg(Xa, ya):\n"
        "    X1 = np.asarray(Xa, dtype=float)[:, 1:2]\n"
        "    neg = np.column_stack([-X1, np.ones(len(X1))])\n"
        "    coef, *_ = np.linalg.lstsq(neg, ya, rcond=None)\n"
        "    return np.array([-coef[0]]), float(coef[1])\n"
    ),
    "predict_neg": (
        "def predict_neg(w, b, Xa):\n"
        "    X1 = np.asarray(Xa, dtype=float)[:, 1:2]\n"
        "    return -X1 @ w + b\n"
    ),
    "skill_score": (
        "def skill_score(rmse_model, rmse_ref):\n"
        "    return 1.0 - rmse_model / rmse_ref\n"
    ),
    "confusion": (
        "def confusion(y_true01, y_score, thr):\n"
        "    y_true01 = np.asarray(y_true01).astype(int)\n"
        "    y_score = np.asarray(y_score, dtype=float)\n"
        "    pred = (y_score >= thr).astype(int)\n"
        "    return {'tp': int(np.sum((y_true01 == 1) & (pred == 1))),\n"
        "            'fp': int(np.sum((y_true01 == 0) & (pred == 1))),\n"
        "            'fn': int(np.sum((y_true01 == 1) & (pred == 0))),\n"
        "            'tn': int(np.sum((y_true01 == 0) & (pred == 0)))}\n"
    ),
    "prf": (
        "def prf(cm):\n"
        "    tp, fp, fn = cm['tp'], cm['fp'], cm['fn']\n"
        "    p = tp / (tp + fp) if (tp + fp) else 0.0\n"
        "    r = tp / (tp + fn) if (tp + fn) else 0.0\n"
        "    f1 = 2 * p * r / (p + r) if (p + r) else 0.0\n"
        "    return {'precision': p, 'recall': r, 'f1': f1}\n"
    ),
    "pilih_threshold": (
        "def pilih_threshold(y_true01, y_score, kandidat, target='f1'):\n"
        "    best = None\n"
        "    for thr in kandidat:\n"
        "        cm = confusion(y_true01, y_score, float(thr))\n"
        "        m = prf(cm)\n"
        "        if best is None or m['f1'] > best[2]['f1']:\n"
        "            best = (float(thr), cm, m)\n"
        "    return best\n"
    ),
    "sigmoid": (
        "def sigmoid(z):\n"
        "    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))\n"
    ),
    "binned_calibration": (
        "def binned_calibration(y_true01, p, n_bins=10):\n"
        "    y_true01 = np.asarray(y_true01).astype(int)\n"
        "    p = np.asarray(p, dtype=float)\n"
        "    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)\n"
        "    bins = []\n"
        "    for b in range(n_bins):\n"
        "        mask = idx == b\n"
        "        nb = int(mask.sum())\n"
        "        bins.append({'lo': b / n_bins, 'hi': (b + 1) / n_bins, 'n': nb,\n"
        "                     'conf': float(p[mask].mean()) if nb else 0.0,\n"
        "                     'acc': float(y_true01[mask].mean()) if nb else 0.0})\n"
        "    return bins\n"
    ),
    "ece": (
        "def ece(y_true01, p, n_bins=10):\n"
        "    bins = binned_calibration(y_true01, p, n_bins)\n"
        "    n = sum(b['n'] for b in bins)\n"
        "    return float(sum(b['n'] / n * abs(b['acc'] - b['conf']) for b in bins))\n"
    ),
    "fit_platt": (
        "def fit_platt(score, y01, lr=0.1, epochs=3000, seed=8):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    s = np.asarray(score, dtype=float)\n"
        "    y = np.asarray(y01, dtype=float)\n"
        "    a, b = 1.0, 0.0\n"
        "    for _ in range(epochs):\n"
        "        p = sigmoid(a * s + b)\n"
        "        ga = float(np.mean((p - y) * s))\n"
        "        gb = float(np.mean(p - y))\n"
        "        a -= lr * ga\n"
        "        b -= lr * gb\n"
        "    return a, b\n"
    ),
    "apply_platt": (
        "def apply_platt(a, b, score):\n"
        "    return sigmoid(a * np.asarray(score, dtype=float) + b)\n"
    ),
    "gen_sample": (
        "def gen_sample(bs, rng_g):\n"
        "    z = rng_g.normal(0.0, 1.0, bs)\n"
        "    return z * np.exp(G_LOGS) + G_MU, z\n"
    ),
    "disc_forward": (
        "def disc_forward(x):\n"
        "    x = np.asarray(x, dtype=float)\n"
        "    h = np.tanh(x[:, None] * D_W1 + D_B1)\n"
        "    o = h @ D_W2 + D_B2\n"
        "    return o, h\n"
    ),
    "dscore": (
        "def dscore(x):\n"
        "    return sigmoid(disc_forward(x)[0])\n"
    ),
    "d_loss": (
        "def d_loss(x_real, x_fake):\n"
        "    return float(-(np.log(dscore(x_real) + 1e-12).mean()\n"
        "                   + np.log(1.0 - dscore(x_fake) + 1e-12).mean()))\n"
    ),
    "g_loss": (
        "def g_loss(x_fake):\n"
        "    return float(-np.log(dscore(x_fake) + 1e-12).mean())\n"
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
