"""Verify 01_lab_ml_klasik.ipynb: exec every cell; TODO cells get reference impls."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

NB = Path(sys.argv[1])
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

# ---- reference implementations for the lab's TODO functions ----
REF = {
    "poly_features": (
        "def poly_features(x, degree):\n"
        "    x = np.asarray(x, dtype=float).reshape(-1)\n"
        "    return np.vander(x, N=degree + 1, increasing=True)[:, 1:]\n"
    ),
    "logistic_predict": (
        "def logistic_predict(X, w, b):\n"
        "    return sigmoid(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b)\n"
    ),
    "confusion_counts": (
        "def confusion_counts(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)\n"
        "    TP = int(((y_true == 1) & (y_pred == 1)).sum())\n"
        "    TN = int(((y_true == 0) & (y_pred == 0)).sum())\n"
        "    FP = int(((y_true == 0) & (y_pred == 1)).sum())\n"
        "    FN = int(((y_true == 1) & (y_pred == 0)).sum())\n"
        "    return (TP, TN, FP, FN)\n"
    ),
    "precision_recall": (
        "def precision_recall(y_true, y_pred):\n"
        "    TP, TN, FP, FN = confusion_counts(y_true, y_pred)\n"
        "    p = TP / (TP + FP) if TP + FP else 0.0\n"
        "    r = TP / (TP + FN) if TP + FN else 0.0\n"
        "    return p, r\n"
    ),
    "ridge_fit": (
        "def ridge_fit(X, y, lam):\n"
        "    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)\n"
        "    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)\n"
    ),
    "gini_impurity": (
        "def gini_impurity(labels):\n"
        "    labels = np.asarray(labels)\n"
        "    if len(labels) == 0:\n"
        "        return 0.0\n"
        "    _, counts = np.unique(labels, return_counts=True)\n"
        "    p = counts / counts.sum()\n"
        "    return float(1.0 - np.sum(p ** 2))\n"
    ),
    "forest_vote": (
        "def forest_vote(preds):\n"
        "    P = np.asarray(preds, dtype=int)\n"
        "    return (P.sum(axis=0) * 2 >= len(preds)).astype(int)\n"
    ),
    "assign_labels": (
        "def assign_labels(X, centroids):\n"
        "    X = np.asarray(X, dtype=float); c = np.asarray(centroids, dtype=float)\n"
        "    return ((X[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)\n"
    ),
}

ctx = {"__name__": "__main__"}
errors = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src and "raise NotImplementedError" in src:
        # student-answer cell: inject matching reference impl(s) instead
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
