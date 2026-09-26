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

NB = Path(__file__).resolve().parent / "02_kuis_ml_produksi.ipynb"
EXPECTED = 23

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "c", 2: "b", 3: "b", 4: "c", 5: "c", 6: "b", 7: "b", 8: "b"},
    "rmse_mae": (
        "def rmse_mae(y_true, y_pred):\n"
        "    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)\n"
        "    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))\n"
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
    "ece_bin": (
        "def ece_bin(y_true01, p, n_bins=10):\n"
        "    y_true01 = np.asarray(y_true01).astype(int)\n"
        "    p = np.asarray(p, dtype=float)\n"
        "    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)\n"
        "    n = len(p)\n"
        "    total = 0.0\n"
        "    for b in range(n_bins):\n"
        "        mask = idx == b\n"
        "        nb = int(mask.sum())\n"
        "        if nb == 0:\n"
        "            continue\n"
        "        conf = float(p[mask].mean())\n"
        "        acc = float(y_true01[mask].mean())\n"
        "        total += (nb / n) * abs(acc - conf)\n"
        "    return float(total)\n"
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
