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
    "JAWABAN": {1: "b", 2: "b", 3: "b", 4: "a"},
    "confusion_counts": (
        "def confusion_counts(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)\n"
        "    return (int(((y_true==1)&(y_pred==1)).sum()), int(((y_true==0)&(y_pred==0)).sum()),\n"
        "            int(((y_true==0)&(y_pred==1)).sum()), int(((y_true==1)&(y_pred==0)).sum()))\n"
    ),
    "sigmoid": "def sigmoid(z):\n    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))\n",
    "precision_recall": (
        "def precision_recall(y_true, y_pred):\n"
        "    TP, TN, FP, FN = confusion_counts(y_true, y_pred)\n"
        "    return (TP/(TP+FP) if TP+FP else 0.0, TP/(TP+FN) if TP+FN else 0.0)\n"
    ),
    "gini_impurity": (
        "def gini_impurity(labels):\n"
        "    labels = np.asarray(labels)\n"
        "    if len(labels) == 0:\n"
        "        return 0.0\n"
        "    _, c = np.unique(labels, return_counts=True)\n"
        "    p = c / c.sum()\n"
        "    return float(1.0 - (p ** 2).sum())\n"
    ),
    "poly_features": (
        "def poly_features(x, degree):\n"
        "    x = np.asarray(x, dtype=float).reshape(-1)\n"
        "    return np.vander(x, N=degree + 1, increasing=True)[:, 1:]\n"
    ),
    "assign_labels": (
        "def assign_labels(X, centroids):\n"
        "    X = np.asarray(X, dtype=float); c = np.asarray(centroids, dtype=float)\n"
        "    return ((X[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)\n"
    ),
    "langkah_gd": "def langkah_gd(w, grad_fn, lr):\n    return w - lr * grad_fn(w)\n",
    "cos_sim": (
        "def cos_sim(a, b):\n"
        "    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)\n"
        "    na, nb = np.linalg.norm(a), np.linalg.norm(b)\n"
        "    if na == 0 or nb == 0:\n"
        "        return 0.0\n"
        "    return float(np.dot(a, b) / (na * nb))\n"
    ),
    "akurasi": (
        "def akurasi(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)\n"
        "    if len(y_true) == 0:\n"
        "        return 0.0\n"
        "    return float(np.mean(y_true == y_pred))\n"
    ),
    "akurasi_dengan_baseline": (
        "def akurasi_dengan_baseline(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)\n"
        "    if len(y_true) == 0:\n"
        "        return 0.0, 0.0\n"
        "    maj = 1 if y_true.mean() > 0.5 else 0\n"
        "    return float(np.mean(y_true == y_pred)), float(np.mean(y_true == maj))\n"
    ),
    "hitung_metrik": (
        "def hitung_metrik(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)\n"
        "    TP = int(((y_true==1)&(y_pred==1)).sum()); FP = int(((y_true==0)&(y_pred==1)).sum())\n"
        "    FN = int(((y_true==1)&(y_pred==0)).sum())\n"
        "    p = TP/(TP+FP) if TP+FP else 0.0\n"
        "    r = TP/(TP+FN) if TP+FN else 0.0\n"
        "    f1 = 2*p*r/(p+r) if p+r else 0.0\n"
        "    return (float(p), float(r), float(f1))\n"
    ),
    "gradient_descent_gini": (
        "def gradient_descent_gini(grad_fn, w0, lr, n_steps=5):\n"
        "    w = float(w0); riwayat = [w]\n"
        "    for _ in range(n_steps):\n"
        "        w = w - lr * grad_fn(w)\n"
        "        riwayat.append(w)\n"
        "    return w, riwayat\n"
    ),
    "entropi": (
        "def entropi(labels):\n"
        "    labels = np.asarray(labels)\n"
        "    if len(labels) == 0:\n"
        "        return 0.0\n"
        "    _, c = np.unique(labels, return_counts=True)\n"
        "    p = c / c.sum()\n"
        "    return float(-(p * np.log2(p)).sum())\n"
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
    out = (ctx.get("_last_out") or "")
    print(f"[cell {i}] ok")

# print grader output captured? exec prints go to stdout already
print("\nEXPECTED SCORE:", EXPECTED)
