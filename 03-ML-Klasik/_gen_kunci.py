"""Generate 03_kunci_jawaban_kuis_ml_klasik.ipynb for Bab 3."""
import json
from pathlib import Path

OUT = Path("AI-Engineering-Curriculum/03-ML-Klasik/03_kunci_jawaban_kuis_ml_klasik.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


cells = [
    md(
        "# 🔑 Kunci Jawaban — Kuis Bab 3 (ML Klasik)",
        "",
        "> **Buka file ini HANYA SETELAH** mencoba kuis sendiri. Membaca jawaban",
        "> sebelum berjuang = nol pembelajaran (ilusi kompetensi).",
        "",
        "Untuk tiap soal: jawaban → **kenapa** → versi idiomatik. Fokus belajar ada di bagian *kenapa*.",
    ),
    md("## Bagian A — Pilihan Ganda"),
    md(
        "### Soal 1 — Jawaban: **b) `0.0`**",
        "",
        "Recall = TP / (TP + FN). Model bodoh tidak PERNAH memprediksi positif →",
        "TP = 0, semua 5% positif jatuh ke FN → recall = 0/0,05n = **0**.",
        "",
        "Jebakannya option a: **accuracy**-nya yang 0.95 (semua negatif ditebak benar),",
        "bukan recall-nya. Inilah kenapa di data imbalanced accuracy boleh tinggi",
        "padahal model tidak berguna — recall-lah yang membongkarnya.",
    ),
    code(
        "import numpy as np\n",
        "\n",
        "y = np.array([0] * 95 + [1] * 5)          # 95% negatif\n",
        "pred_bodoh = np.zeros(100, dtype=int)     # selalu 'negatif'\n",
        "\n",
        "TP = int(((y == 1) & (pred_bodoh == 1)).sum())\n",
        "FN = int(((y == 1) & (pred_bodoh == 0)).sum())\n",
        "print(f'accuracy = {(pred_bodoh == y).mean():.2f}   <- terlihat bagus!')\n",
        "print(f'recall   = {TP}/{TP + FN} = {TP / (TP + FN):.2f}   <- kenyataannya')",
    ),
    md(
        "### Soal 2 — Jawaban: **b) precision**",
        "",
        "Hafalan cepat yang tidak akan pernah lekang:",
        "",
        "- **precision** = dari yang **DIPREDIKSI** positif, berapa benar? → \"seberapa tebal hati menebak positif\"",
        "- **recall** = dari yang **SEBENARNYA** positif, berapa ditemukan? → \"seberapa lengkap menemukan\"",
        "",
        "Kata kunci di soal: \"diprediksi positif\" → penyebutnya TP + **FP** → precision.",
    ),
    md(
        "### Soal 3 — Jawaban: **b) derajat dengan val MSE terendah**",
        "",
        "Train MSE selalu turun saat model dikomplekskan — ia mengukur **hafalan**.",
        "Validation MSE mengukur **generalisasi** — kemampuan menjawab soal baru.",
        "Saat val MSE berbalik naik, model mulai menghafal noise. Model dipilih di",
        "validation; **test hanya untuk laporan akhir sekali**.",
        "",
        "Ini konsekuensi langsung dari kurva bias–variance di lab Bagian 4.",
    ),
    md(
        "### Soal 4 — Jawaban: **a) bobot menyusut, model lebih sederhana**",
        "",
        "Ridge menambahkan hukuman λ‖w‖² ke loss. Solusinya `(XᵀX + λI)⁻¹Xᵀy`: λ",
        "besar membuat matriks \"lebih kaku\" → bobot mengecil → model kurang fleksibel",
        "→ variance turun (bias naik sedikit). Trade-off inilah **regularisasi**.",
        "",
        "λ terlalu besar = underfitting (semua bobot ≈ 0). λ = 0 = OLS biasa.",
    ),
    code(
        "# Bukti kecil untuk soal 4: lambda naik -> norm bobot menyusut\n",
        "def ridge_fit(X, y, lam):\n",
        "    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)\n",
        "\n",
        "rng = np.random.default_rng(0)\n",
        "X = rng.normal(0, 1, size=(40, 5))\n",
        "y = X @ np.array([2., -1., 0.5, 0., 3.]) + rng.normal(0, 0.3, 40)\n",
        "for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:\n",
        "    w = ridge_fit(X, y, lam)\n",
        "    print(f'lam={lam:>6}: |w| = {np.linalg.norm(w):.4f}')",
    ),
    md("## Bagian B — Coding"),
    md(
        "### Soal 5 — Confusion matrix",
        "",
        "Kenapa pakai operasi boolean vektor, bukan loop `for`: satu baris memindai",
        "semua data di C, pola yang sama dipakai di hampir semua kode metrik.",
    ),
    code(
        "def confusion_counts(y_true, y_pred):\n",
        "    y_true = np.asarray(y_true)\n",
        "    y_pred = np.asarray(y_pred)\n",
        "    TP = int(((y_true == 1) & (y_pred == 1)).sum())\n",
        "    TN = int(((y_true == 0) & (y_pred == 0)).sum())\n",
        "    FP = int(((y_true == 0) & (y_pred == 1)).sum())\n",
        "    FN = int(((y_true == 1) & (y_pred == 0)).sum())\n",
        "    return (TP, TN, FP, FN)\n",
        "\n",
        "yt = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]\n",
        "yp = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]\n",
        "print(confusion_counts(yt, yp))     # (4, 4, 1, 1)",
    ),
    md(
        "### Soal 6 — Sigmoid",
        "",
        "`np.exp` bekerja element-wise → fungsi otomatis support array. Versi",
        "produksi sebaiknya stabil-numerik (lihat lab Bagian 2 / project starter),",
        "tapi untuk kuis rumus dasarnya cukup.",
    ),
    code(
        "def sigmoid(z):\n",
        "    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))\n",
        "\n",
        "print(sigmoid(0))                      # 0.5\n",
        "print(sigmoid(np.array([-2., 0., 2.])))   # [0.119 0.5   0.881]",
    ),
    md(
        "### Soal 7 — Precision & recall dengan guard",
        "",
        "Kunci soal ini bukan rumusnya, tapi **guard**-nya: model yang tidak pernah",
        "menebak positif membuat TP+FP = 0 → pembagian nol. Konvensi sklearn: 0.0.",
        "Tanpa kebiasaan guard, error semacam ini muncul terus di data nyata.",
    ),
    code(
        "def precision_recall(y_true, y_pred):\n",
        "    TP, TN, FP, FN = confusion_counts(y_true, y_pred)\n",
        "    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0\n",
        "    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0\n",
        "    return precision, recall\n",
        "\n",
        "yt = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]\n",
        "yp = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]\n",
        "print(precision_recall(yt, yp))                 # (0.8, 0.8)\n",
        "print(precision_recall([0, 1, 1], [0, 0, 0]))   # (0.0, 0.0) — bukan error",
    ),
    md(
        "### Soal 8 — Gini impurity",
        "",
        "`Gini = 1 − Σ pₖ²`. Maknanya: peluang DUA sampel acak dari kelompok itu",
        "beda kelas. Murni → 0. Campuran 50:50 → 0.5 (paling tidak murni, untuk 2 kelas).",
    ),
    code(
        "def gini_impurity(labels):\n",
        "    labels = np.asarray(labels)\n",
        "    if len(labels) == 0:\n",
        "        return 0.0\n",
        "    _, counts = np.unique(labels, return_counts=True)\n",
        "    p = counts / counts.sum()\n",
        "    return float(1.0 - np.sum(p ** 2))\n",
        "\n",
        "print(gini_impurity([0, 0, 1, 1]))   # 0.5\n",
        "print(gini_impurity([0, 1, 1]))      # 1 - (1/9 + 4/9) = 4/9 ≈ 0.444\n",
        "print(gini_impurity([]))             # 0.0 (guard)",
    ),
    md(
        "### Soal 9 — Fitur polinomial",
        "",
        "Versi idiomatik memakai **broadcasting**: `x[:, None]` mengubah `(n,)` jadi",
        "`(n, 1)`, dipangkatkan array `[1..degree]` menjadi `(n, degree)` sekaligus.",
    ),
    code(
        "def poly_features(x, degree):\n",
        "    x = np.asarray(x, dtype=float).reshape(-1)\n",
        "    return x[:, None] ** np.arange(1, degree + 1)\n",
        "\n",
        "print(poly_features(np.array([2.0]), 3))        # [[2., 4., 8.]]\n",
        "print(poly_features(np.array([1., 2., 3.]), 2)) # [[1.,1.],[2.,4.],[3.,9.]]",
    ),
    md(
        "### Soal 10 — Assign centroid terdekat",
        "",
        "Pola broadcasting yang sama dengan soal 9, tapi dua dimensi: `X[:, None, :]",
        "− centroids[None]` menghasilkan tensor `(n, k, d)` berisi selisih tiap titik",
        "ke tiap centroid → jumlah kuadrat → `(n, k)` → `argmin(axis=1)`.",
        "",
        "Ini persis langkah **assign** di k-means — dan fondasi nearest-neighbor",
        "search (dipakai lagi di RAG, Bab 12).",
    ),
    code(
        "def assign_labels(X, centroids):\n",
        "    X = np.asarray(X, dtype=float)\n",
        "    c = np.asarray(centroids, dtype=float)\n",
        "    d2 = ((X[:, None, :] - c[None]) ** 2).sum(axis=2)   # (n, k)\n",
        "    return d2.argmin(axis=1)\n",
        "\n",
        "c = np.array([[0., 0.], [10., 10.]])\n",
        "pts = np.array([[1., 1.], [9., 9.], [0.5, 0.]])\n",
        "print(assign_labels(pts, c))   # [0 1 0]",
    ),
    md(
        "---",
        "",
        "**Polanya jelas:** (1) operasi boolean vektor untuk metrik, (2) guard untuk",
        "edge case, (3) broadcasting + argmin untuk geometri. Tiga pola ini mencakup",
        "mayoritas soal coding ML klasik.",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT, f"({len(cells)} cells)")
