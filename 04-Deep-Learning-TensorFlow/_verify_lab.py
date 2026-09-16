"""Verify 01_lab_deep_learning.ipynb: exec every cell; TODO cells get reference impls."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

NB = Path(sys.argv[1])
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "softmax_stabil": (
        "def softmax_stabil(logits):\n"
        "    logits = np.asarray(logits, dtype=float)\n"
        "    shifted = logits - logits.max(axis=-1, keepdims=True)\n"
        "    e = np.exp(shifted)\n"
        "    return e / e.sum(axis=-1, keepdims=True)\n"
    ),
    "train_xor": (
        "def train_xor(lr=1.0, epochs=3000, n_hidden=8, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    W1 = rng.normal(0, 1.0, (2, n_hidden)); b1 = np.zeros(n_hidden)\n"
        "    W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1)); b2 = np.zeros(1)\n"
        "    losses = []; m = len(yxor)\n"
        "    for ep in range(epochs):\n"
        "        h = np.maximum(Xxor @ W1 + b1, 0)\n"
        "        o = sigmoid_stabil(h @ W2 + b2)\n"
        "        p = np.clip(o.ravel(), 1e-12, 1 - 1e-12)\n"
        "        losses.append(float(-np.mean(yxor * np.log(p) + (1 - yxor) * np.log(1 - p))))\n"
        "        dz2 = (o - yxor.reshape(-1, 1)) / m\n"
        "        dW2 = h.T @ dz2; db2 = dz2.sum(0)\n"
        "        dz1 = (dz2 @ W2.T) * (h > 0)\n"
        "        dW1 = Xxor.T @ dz1; db1 = dz1.sum(0)\n"
        "        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2\n"
        "    h = np.maximum(Xxor @ W1 + b1, 0)\n"
        "    o = sigmoid_stabil(h @ W2 + b2).ravel()\n"
        "    return float(((o >= 0.5) == yxor).mean()), losses\n"
    ),
    "turunan_numerik": (
        "def turunan_numerik(fn, x, h=1e-6):\n"
        "    return (fn(x + h) - fn(x - h)) / (2 * h)\n"
    ),
    "mlp_forward": (
        "def mlp_forward(X, W1, b1, W2, b2):\n"
        "    h = np.maximum(np.asarray(X, dtype=float) @ W1 + b1, 0)\n"
        "    p = sigmoid_stabil(h @ W2 + b2).ravel()\n"
        "    return h, p\n"
    ),
    "bce_manual": (
        "def bce_manual(y_true, p_pred, eps=1e-12):\n"
        "    y = np.asarray(y_true, dtype=float)\n"
        "    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)\n"
        "    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))\n"
    ),
    "cross_entropy": (
        "def cross_entropy(y_onehot, probs, eps=1e-12):\n"
        "    y = np.asarray(y_onehot, dtype=float)\n"
        "    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)\n"
        "    return float(-np.mean(np.sum(y * np.log(p), axis=1)))\n"
    ),
    "he_std": "def he_std(fan_in):\n    return float(np.sqrt(2.0 / fan_in))\n",
    "xavier_std": "def xavier_std(fan_in):\n    return float(np.sqrt(1.0 / fan_in))\n",
    "langkah_sgd": "def langkah_sgd(w, grad, lr):\n    return np.asarray(w, dtype=float) - lr * np.asarray(grad, dtype=float)\n",
    "split_batch": (
        "def split_batch(n, batch_size):\n"
        "    ukuran = [batch_size] * (n // batch_size)\n"
        "    sisa = n % batch_size\n"
        "    if sisa:\n"
        "        ukuran.append(sisa)\n"
        "    return ukuran\n"
    ),
    "cek_early_stop": (
        "def cek_early_stop(val_loss, patience):\n"
        "    best = np.inf; best_epoch = 0; wait = 0; stop = len(val_loss) - 1\n"
        "    for i, v in enumerate(val_loss):\n"
        "        if v < best - 1e-6:\n"
        "            best = v; best_epoch = i; wait = 0\n"
        "        else:\n"
        "            wait += 1\n"
        "            if wait >= patience:\n"
        "                stop = i\n"
        "                break\n"
        "    return (best_epoch, stop)\n"
    ),
}

ctx = {"__name__": "__main__"}
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
print("ALL CELLS OK")
