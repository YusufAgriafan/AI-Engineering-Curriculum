"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage: python _verify_quiz.py <nb.ipynb> <total_score>
"""
import json
import sys
from pathlib import Path

NB = Path(sys.argv[1])
EXPECTED = int(sys.argv[2])
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "c", 2: "b", 3: "b", 4: "c"},
    "relu_forward": "def relu_forward(Z):\n    return np.maximum(np.asarray(Z, dtype=float), 0.0)\n",
    "sigmoid_stabil": (
        "def sigmoid_stabil(z):\n"
        "    z = np.asarray(z, dtype=float)\n"
        "    out = np.empty_like(z)\n"
        "    pos = z >= 0\n"
        "    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n"
        "    ez = np.exp(z[~pos])\n"
        "    out[~pos] = ez / (1.0 + ez)\n"
        "    return out\n"
    ),
    "bce_loss": (
        "def bce_loss(y_true, p_pred, eps=1e-12):\n"
        "    y = np.asarray(y_true, dtype=float)\n"
        "    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)\n"
        "    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))\n"
    ),
    "softmax_stabil": (
        "def softmax_stabil(logits):\n"
        "    logits = np.asarray(logits, dtype=float)\n"
        "    shifted = logits - logits.max(axis=-1, keepdims=True)\n"
        "    e = np.exp(shifted)\n"
        "    return e / e.sum(axis=-1, keepdims=True)\n"
    ),
    "he_init_weights": (
        "def he_init_weights(rng, fan_in, fan_out):\n"
        "    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))\n"
    ),
    "split_batches": (
        "def split_batches(n, batch_size):\n"
        "    ukuran = [batch_size] * (n // batch_size)\n"
        "    sisa = n % batch_size\n"
        "    if sisa:\n"
        "        ukuran.append(sisa)\n"
        "    return ukuran\n"
    ),
}

ctx = {"__name__": "__main__"}
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "JAWABAN" in src and "None" in src:
        exec("JAWABAN = " + repr(ANSWERS["JAWABAN"]), ctx)
        print(f"[cell {i}] injected JAWABAN")
        continue
    if "TODO" in src:
        names = [n for n in ANSWERS if n != "JAWABAN" and f"def {n}(" in src]
        if not names:
            print(f"[cell {i}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        for n in names:
            exec(ANSWERS[n], ctx)
        print(f"[cell {i}] injected: {names}")
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ctx)
    print(f"[cell {i}] ok")

print("\nEXPECTED SCORE:", EXPECTED)
