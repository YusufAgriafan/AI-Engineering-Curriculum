"""Generate 02_kuis_ml_klasik.ipynb for Bab 3."""
import json
from pathlib import Path

OUT = Path("AI-Engineering-Curriculum/03-ML-Klasik/02_kuis_ml_klasik.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


cells = [
    md(
        "# 📝 Kuis Bab 3 — Machine Learning Klasik",
        "",
        "10 soal (4 pilihan ganda + 6 coding), total **15 poin**. Penilaian otomatis di cell terakhir.",
        "",
        "**Aturan:** kerjakan tanpa membuka materi. Soal coding: isi di bawah `# TODO`,",
        "lalu jalankan cell kuis di bawah. Boleh pakai NumPy.",
    ),
    md("## Bagian A — Pilihan Ganda (isi `JAWABAN`)"),
    code(
        "# Isi jawaban pilihan ganda di sini, contoh: JAWABAN[1] = 'a'\n",
        "JAWABAN = {\n",
        "    1: None,\n",
        "    2: None,\n",
        "    3: None,\n",
        "    4: None,\n",
        "}",
    ),
    md(
        "### Soal 1 — Model bodoh di data imbalanced",
        "",
        "Data dengan 95% kelas negatif. Sebuah model **selalu** memprediksi \"negatif\".",
        "Berapa **recall**-nya?",
        "",
        "- a) `0.95`",
        "- b) `0.0`",
        "- c) `0.50`",
        "- d) tidak terdefinisi (error)",
    ),
    md(
        "### Soal 2 — Bahasa precision & recall",
        "",
        "Metrik yang menjawab pertanyaan *\"dari semua yang DIPREDIKSI positif, berapa",
        "yang benar-benar positif?\"* adalah…",
        "",
        "- a) recall",
        "- b) precision",
        "- c) accuracy",
        "- d) F1-score",
    ),
    md(
        "### Soal 3 — Memilih kompleksitas model",
        "",
        "Regresi polinomial: saat derajat dinaikkan, **train MSE terus turun**, tapi",
        "**val MSE turun lalu naik lagi**. Derajat mana yang harus dipilih?",
        "",
        "- a) derajat dengan train MSE terendah",
        "- b) derajat dengan val MSE terendah",
        "- c) derajat tertinggi yang bisa dihitung",
        "- d) selalu derajat 1 (linear)",
    ),
    md(
        "### Soal 4 — Efek regularisasi",
        "",
        "Di ridge regression, jika λ (lambda) diperbesar, yang terjadi adalah…",
        "",
        "- a) bobot menyusut, model lebih sederhana, overfitting berkurang",
        "- b) bobot membesar, model makin fleksibel",
        "- c) train MSE pasti turun",
        "- d) tidak ada efek apa pun pada bobot",
    ),
    md("## Bagian B — Coding"),
    md(
        "### Soal 5 — Confusion matrix (2 poin)",
        "",
        "`confusion_counts(y_true, y_pred)` → tuple `(TP, TN, FP, FN)`. y hanya berisi 0/1.",
        "Boleh NumPy. (Sama seperti latihan lab 3.1.)"
    ),
    code(
        "import numpy as np\n",
        "\n",
        "def confusion_counts(y_true, y_pred):\n",
        "    # TODO\n",
        "    pass\n",
        "\n",
        "# yt = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]\n",
        "# yp = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]\n",
        "# print(confusion_counts(yt, yp))   # (4, 4, 1, 1)"
    ),
    md(
        "### Soal 6 — Sigmoid (1 poin)",
        "",
        "`sigmoid(z)` → 1/(1+e^-z), bekerja untuk array maupun skalar.",
        "Input `z = 0` harus menghasilkan tepat `0.5`."
    ),
    code(
        "def sigmoid(z):\n",
        "    # TODO\n",
        "    pass\n",
        "\n",
        "# print(sigmoid(0))          # 0.5\n",
        "# print(sigmoid(np.array([-2., 0., 2.])))"
    ),
    md(
        "### Soal 7 — Precision & recall dengan guard (2 poin)",
        "",
        "`precision_recall(y_true, y_pred)` → tuple `(precision, recall)`.",
        "Kalau model tidak pernah menebak positif → `(0.0, 0.0)`, BUKAN error.",
        "(Pakai `confusion_counts` dari soal 5 kalau mau.)"
    ),
    code(
        "def precision_recall(y_true, y_pred):\n",
        "    # TODO\n",
        "    pass\n",
        "\n",
        "# yt = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]\n",
        "# yp = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]\n",
        "# print(precision_recall(yt, yp))    # (0.8, 0.8)\n",
        "# print(precision_recall([0, 1, 1], [0, 0, 0]))   # (0.0, 0.0)"
    ),
    md(
        "### Soal 8 — Gini impurity (2 poin)",
        "",
        "`gini_impurity(labels)` → `1 - Σ pₖ²`. List kosong → `0.0`."
    ),
    code(
        "def gini_impurity(labels):\n",
        "    # TODO\n",
        "    pass\n",
        "\n",
        "# print(gini_impurity([0, 0, 1, 1]))   # 0.5\n",
        "# print(gini_impurity([0, 1, 1]))      # 4/9\n",
        "# print(gini_impurity([]))             # 0.0"
    ),
    md(
        "### Soal 9 — Fitur polinomial (2 poin)",
        "",
        "`poly_features(x, degree)` → matriks `(len(x), degree)` dengan kolom",
        "`x^1, x^2, …, x^degree`. Tanpa loop per elemen (hint: broadcasting)."
    ),
    code(
        "def poly_features(x, degree):\n",
        "    # TODO\n",
        "    pass\n",
        "\n",
        "# print(poly_features(np.array([2.0]), 3))        # [[2., 4., 8.]]\n",
        "# print(poly_features(np.array([1., 2., 3.]), 2)) # [[1.,1.],[2.,4.],[3.,9.]]"
    ),
    md(
        "### Soal 10 — Assign centroid terdekat (2 poin)",
        "",
        "`assign_labels(X, centroids)` → label (indeks centroid terdekat, jarak kuadrat)",
        "untuk tiap titik. `X: (n, d)`, `centroids: (k, d)` → output `(n,)` int.",
        "Tanpa loop per titik."
    ),
    code(
        "def assign_labels(X, centroids):\n",
        "    # TODO\n",
        "    pass\n",
        "\n",
        "# c = np.array([[0., 0.], [10., 10.]])\n",
        "# pts = np.array([[1., 1.], [9., 9.], [0.5, 0.]])\n",
        "# print(assign_labels(pts, c))   # [0, 1, 0]"
    ),
    md("## 🧮 Penilaian Otomatis"),
    code(
        "import numpy as np\n",
        "\n",
        "total_bobot = 15\n",
        "\n",
        "\n",
        "def jalankan_kuis():\n",
        "    skor = 0\n",
        "    rincian = []\n",
        "\n",
        "    def cek_pilihan(nomor, kunci, bobot=1):\n",
        "        nonlocal skor\n",
        "        benar = JAWABAN.get(nomor) == kunci\n",
        "        skor += bobot if benar else 0\n",
        "        rincian.append((nomor, benar))\n",
        "\n",
        "    def cek_kode(nomor, uji, bobot=1):\n",
        "        nonlocal skor\n",
        "        benar = False\n",
        "        try:\n",
        "            benar = bool(uji())\n",
        "        except Exception as e:\n",
        "            print(f'  soal {nomor}: error -> {type(e).__name__}: {e}')\n",
        "        skor += bobot if benar else 0\n",
        "        rincian.append((nomor, benar))\n",
        "\n",
        "    def approx(a, b, tol=1e-9):\n",
        "        return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol)\n",
        "\n",
        "    cek_pilihan(1, 'b')\n",
        "    cek_pilihan(2, 'b')\n",
        "    cek_pilihan(3, 'b')\n",
        "    cek_pilihan(4, 'a')\n",
        "\n",
        "    yt = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])\n",
        "    yp = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])\n",
        "\n",
        "    cek_kode(5, lambda: confusion_counts(yt, yp) == (4, 4, 1, 1)\n",
        "                      and confusion_counts([1, 1], [1, 1]) == (2, 0, 0, 0)\n",
        "                      and confusion_counts([0, 1, 1], [0, 0, 0]) == (0, 1, 0, 2), bobot=2)\n",
        "    cek_kode(6, lambda: abs(sigmoid(0) - 0.5) < 1e-9\n",
        "                      and approx(sigmoid(np.array([-2., 2.])),\n",
        "                                 [1 / (1 + np.exp(2.)), 1 / (1 + np.exp(-2.))]), bobot=1)\n",
        "    cek_kode(7, lambda: approx(precision_recall(yt, yp), (0.8, 0.8))\n",
        "                      and precision_recall([0, 1, 1], [0, 0, 0]) == (0.0, 0.0), bobot=2)\n",
        "    cek_kode(8, lambda: approx(gini_impurity([0, 0, 1, 1]), 0.5)\n",
        "                      and approx(gini_impurity([0, 1, 1]), 4 / 9)\n",
        "                      and gini_impurity([]) == 0.0\n",
        "                      and gini_impurity([0, 0, 0, 0]) == 0.0, bobot=2)\n",
        "    cek_kode(9, lambda: approx(poly_features(np.array([2.0]), 3), [[2., 4., 8.]])\n",
        "                      and poly_features(np.array([1., 2., 3.]), 2).shape == (3, 2)\n",
        "                      and approx(poly_features(np.array([1., 2., 3.]), 2)[:, 1], [1., 4., 9.]), bobot=2)\n",
        "    cek_kode(10, lambda: assign_labels(np.array([[1., 1.], [9., 9.], [0.5, 0.]]),\n",
        "                                       np.array([[0., 0.], [10., 10.]])).tolist() == [0, 1, 0]\n",
        "                     and assign_labels(np.array([[4.9], [99.], [0.1]]),\n",
        "                                       np.array([[0.], [5.], [100.]])).tolist() == [1, 2, 0], bobot=2)\n",
        "\n",
        "    print('=' * 46)\n",
        "    for nomor, benar in rincian:\n",
        "        print(f'  Soal {nomor:>2}: {\"BENAR\" if benar else \"salah\"}')\n",
        "    print('=' * 46)\n",
        "    print(f'SKOR AKHIR: {skor}/{total_bobot}')\n",
        "    if skor == total_bobot:\n",
        "        print('Sempurna! Lanjut ke project starter.')\n",
        "    elif skor >= 12:\n",
        "        print('Bagus! Review soal yang salah, lalu lanjut.')\n",
        "    else:\n",
        "        print('Ulangi bagian terkait di lab, coba lagi besok.')\n",
        "\n",
        "jalankan_kuis()",
    ),
    md(
        "---",
        "",
        "Sudah mencoba serius? Bandingkan pendekatanmu dengan `03_kunci_jawaban_kuis_ml_klasik.ipynb`",
        "— fokus pada intuisi di balik tiap jawaban."
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
