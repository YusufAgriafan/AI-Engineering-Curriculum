"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_quiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_time_series.ipynb"
EXPECTED = 23

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "c", 2: "b", 3: "b", 4: "b", 5: "d", 6: "a", 7: "b", 8: "b"},
    "buat_window": (
        "def buat_window(series, window_size):\n"
        "    series = np.asarray(series, dtype=float)\n"
        "    n = len(series) - window_size\n"
        "    X = np.empty((n, window_size))\n"
        "    for i in range(n):\n"
        "        X[i] = series[i:i + window_size]\n"
        "    y = series[window_size:]\n"
        "    return X, y\n"
    ),
    "split_waktu": (
        "def split_waktu(X, y, rasio_train=0.8):\n"
        "    n = len(X)\n"
        "    cut = int(n * rasio_train)\n"
        "    X, y = np.asarray(X), np.asarray(y)\n"
        "    return X[:cut], y[:cut], X[cut:], y[cut:]\n"
    ),
    "metrik_semua": (
        "def metrik_semua(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true, dtype=float)\n"
        "    y_pred = np.asarray(y_pred, dtype=float)\n"
        "    err = y_true - y_pred\n"
        "    mae = float(np.mean(np.abs(err)))\n"
        "    rmse = float(np.sqrt(np.mean(err ** 2)))\n"
        "    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))\n"
        "    return {'mae': mae, 'rmse': rmse, 'smape': smape}\n"
    ),
    "fit_ar_ols": (
        "def fit_ar_ols(series, p):\n"
        "    s = np.asarray(series, dtype=float)\n"
        "    X = np.column_stack([s[p - k - 1:len(s) - k - 1] for k in range(p)])\n"
        "    y = s[p:]\n"
        "    return np.linalg.lstsq(X, y, rcond=None)[0]\n"
    ),
}

ctx = {"__name__": "__main__"}
import numpy as np  # noqa: E402  -- referensi impl memakai np.*
ctx["np"] = np

for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "JAWABAN = {" in src and "None" in src:
        exec("JAWABAN = " + repr(ANSWERS["JAWABAN"]), ctx)
        print(f"[cell {i}] injected JAWABAN")
        continue
    if "TODO" in src:
        names = [n for n in ANSWERS if n != "JAWABAN" and f"def {n}(" in src]
        if not names:
            print(f"[cell {i}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        # cell TODO asli TIDAK dieksekusi; timpa dengan implementasi referensi
        for n in names:
            exec(ANSWERS[n], ctx)
            print(f"[cell {i}] injected {n}")
        continue
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, ctx)
    out = buf.getvalue()
    print(f"[cell {i}] executed ({len(out)} chars)")
    if "SKOR AKHIR" in out:
        print(out)
        m = re.search(r"SKOR AKHIR:\s*(\d+)/(\d+)", out)
        skor, total = int(m.group(1)), int(m.group(2))
        status = "PASS" if (skor == total == EXPECTED) else "FAIL"
        print(f">>> {status}: {skor}/{total} (expected {EXPECTED})")
        sys.exit(0 if status == "PASS" else 1)
