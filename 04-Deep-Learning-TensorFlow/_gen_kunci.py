"""Generate 03_kunci_jawaban_kuis_deep_learning.ipynb for Bab 4."""
import json
from pathlib import Path

OUT = Path("AI-Engineering-Curriculum/04-Deep-Learning-TensorFlow/03_kunci_jawaban_kuis_deep_learning.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


cells = [
    md(
        "# 🔑 Kunci Jawaban — Kuis Bab 4 (Deep Learning)",
        "",
        "> **Buka file ini HANYA SETELAH** mencoba kuis sendiri. Membaca jawaban",
        "> sebelum berjuang = nol pembelajaran (ilusi kompetensi).",
        "",
        "Untuk tiap soal: jawaban → **kenapa** → versi idiomatik. Fokus belajar ada di bagian *kenapa*.",
    ),
    md("## Bagian A — Pilihan Ganda"),
    md(
        "### Soal 1 — Jawaban: **c) `0`**",
        "",
        "ReLU didefinisikan `max(0, x)`: nilai negatif **dipotong ke nol**, nilai non-negatif",
        "lolos apa adanya. Pilihan b (`3`) adalah jebakan orang yang mengira ReLU = nilai absolut.",
        "",
        "Kenapa ReLU dominan di hidden layer? Turunannya konstan (`1` untuk x>0, `0` untuk x<0)",
        "— tidak menyusut seperti sigmoid/tanh, jadi gradien tetap hidup di jaringan dalam.",
    ),
    code(
        "import numpy as np\n",
        "\n",
        "x = np.array([-3., -0.5, 0., 2.5])\n",
        "print('relu :', np.maximum(x, 0))\n",
        "print('grad :', (x > 0).astype(float), '  <- turunan ReLU: 1 kalau x>0, 0 kalau x<0')",
    ),
    md(
        "### Soal 2 — Jawaban: **b) `[0, 1]`**",
        "",
        "Sigmoid memetakan semua bilangan real ke `(0, 1)` — asimtotik ke 0 dan 1, tidak pernah",
        "menyentuhnya. Karena itu dia cocok jadi output **probabilitas** binary classification.",
        "",
        "Pemetaan cepat biar tak tertukar:",
        "",
        "| Fungsi | Rentang | Dipakai di |",
        "|---|---|---|",
        "| ReLU | `[0, ∞)` | hidden layer |",
        "| Sigmoid | `(0, 1)` | output biner |",
        "| Softmax | `(0, 1)`, Σ=1 | output multikelas |",
        "| Tanh | `(-1, 1)` | jarang; kadang RNN lama |",
        "| Linear | `(-∞, ∞)` | output regresi |",
    ),
    md(
        "### Soal 3 — Jawaban: **b) overfitting → early stopping / regularisasi**",
        "",
        "Kurva khasnya: train loss terus turun (model makin hafal training set), val loss",
        "berbalik naik (model mulai menghafal noise, bukan pola umum). Titik balik val loss",
        "itu momen model terbaik — persis yang dijemput `EarlyStopping(patience=k,",
        "restore_best_weights=True)`.",
        "",
        "Kenapa opsi lain salah: underfitting (a) itu train loss JUGA tinggi/stagnan;",
        "lr terlalu kecil (c) bikin KEDUA kurva turun lambat tapi tetap searah;",
        "menghapus validation set (d) = membutakan diri dari satu-satunya sinyal yang jujur.",
    ),
    code(
        "# Simulasi kurva overfitting — lihat titik balik val loss\n",
        "train = [0.9, 0.6, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09, 0.07]\n",
        "val   = [0.9, 0.65, 0.5, 0.44, 0.43, 0.45, 0.49, 0.55, 0.62]\n",
        "best = int(np.argmin(val))\n",
        "print(f'val loss terbaik di epoch {best}: {val[best]:.2f}')\n",
        "print('EarlyStopping dengan patience cukup akan berhenti ~epoch', best, '+ patience')",
    ),
    md(
        "### Soal 4 — Jawaban: **c) `Dense(10, softmax)` + `sparse_categorical_crossentropy`**",
        "",
        "Tiga keputusan yang harus konsisten:",
        "",
        "1. **10 unit** = satu neuron per kelas.",
        "2. **softmax** = 10 skor mentah (logits) → distribusi probabilitas (Σ=1).",
        "3. **sparse** categorical CE karena label `y` berupa **integer 0..9**.",
        "   Kalau labelmu one-hot `(n, 10)`, baru pakai `categorical_crossentropy` biasa.",
        "",
        "Jebakan b: ReLU bukan output klasifikasi (outputnya bukan probabilitas) dan MSE",
        "bukan loss untuk distribusi kelas. Jebakan d: 10 sigmoid saling lepas — totalnya",
        "tidak 1, bukan distribusi kelas yang valid.",
    ),
    md("## Bagian B — Coding"),
    md(
        "### Soal 5 — `relu_forward` (2 poin)",
        "",
        "`np.maximum(Z, 0)` — elementwise, tanpa loop. Jebakan umum: `max()` bawaan Python",
        "membandingkan SELURUH array dengan 0 (hasilnya skalar), bukan per elemen."
    ),
    code(
        "def relu_forward(Z):\n",
        "    return np.maximum(np.asarray(Z, dtype=float), 0.0)\n",
        "\n",
        "print(relu_forward(np.array([-3., 0., 2.5])))\n",
        "print(relu_forward(np.array([[-1., 4.], [7., -0.5]])))",
    ),
    md(
        "### Soal 6 — `sigmoid_stabil` (3 poin)",
        "",
        "Masalahnya `exp(-z)` untuk `z = -1000` → `exp(1000)` → **overflow**. Solusinya",
        "dua cabang, sehingga `exp` tidak pernah menerima argumen positif besar:",
        "",
        "- `z >= 0` : `1/(1+exp(-z))` — argumen exp selalu ≤ 0",
        "- `z < 0`  : `ez/(1+ez)` dengan `ez = exp(z)` — argumen exp selalu negatif",
        "",
        "Secara aljabar keduanya identik dengan definisi sigmoid (coba turunkan), tapi",
        "salah satunya selalu aman di wilayah z masing-masing. Pola ini persis yang dipakai",
        "di lab Bagian 1 dan di `nn_mini/activations.py` project starter.",
    ),
    code(
        "def sigmoid_stabil(z):\n",
        "    z = np.asarray(z, dtype=float)\n",
        "    out = np.empty_like(z)\n",
        "    pos = z >= 0\n",
        "    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n",
        "    ez = np.exp(z[~pos])\n",
        "    out[~pos] = ez / (1.0 + ez)\n",
        "    return out\n",
        "\n",
        "z = np.array([-1000., 0., 1000.])\n",
        "print(sigmoid_stabil(z))   # [0.  0.5 1. ] — bersih, tanpa RuntimeWarning",
    ),
    md(
        "### Soal 7 — `bce_loss` (3 poin)",
        "",
        "Dua hal yang dinilai: (1) rumus BCE benar, (2) **clipping sebelum log**.",
        "Tanpa clip, `log(0)` = `-inf` dan satu sampel meracuni seluruh rata-rata.",
        "Dengan `eps=1e-12`, kasus \"yakin 100% tapi salah\" menghasilkan nilai besar",
        "≈ `-log(1e-12) ≈ 27.6` — besar tapi **finite**, jadi training tetap jalan.",
        "Perhatikan: p=1 di-clip ke `1-eps`, jadi hasilnya ≈ 0 tapi tidak persis 0.",
    ),
    code(
        "def bce_loss(y_true, p_pred, eps=1e-12):\n",
        "    y = np.asarray(y_true, dtype=float)\n",
        "    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)\n",
        "    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))\n",
        "\n",
        "print(bce_loss([1., 0.], [1., 0.]))            # ~1e-12, praktis nol\n",
        "print(bce_loss([1., 1., 1., 1.], [0.5] * 4))   # ln 2 ≈ 0.6931\n",
        "print(bce_loss([1., 0.], [0., 1.]))            # ≈ 27.6 — besar tapi finite",
    ),
    md(
        "### Soal 8 — `softmax_stabil` (3 poin)",
        "",
        "Trik stabilnya: kurangi **max tiap baris** sebelum exp. Softmax tidak berubah",
        "secara matematis karena pembilang & penyebut sama-sama dibagi `exp(max)`",
        "(konstanta baris itu saling kancel). Manfaatnya: argumen exp terbesar tinggal `0`,",
        "`exp(1000)` mustahil terjadi.",
        "",
        "`keepdims=True` wajib agar `(n,1)` di-broadcast benar terhadap `(n,k)` —",
        "kalau lupa, baris ke-2 dst akan pakai max baris pertama (bug senyap!)."
    ),
    code(
        "def softmax_stabil(logits):\n",
        "    logits = np.asarray(logits, dtype=float)\n",
        "    shifted = logits - logits.max(axis=-1, keepdims=True)\n",
        "    e = np.exp(shifted)\n",
        "    return e / e.sum(axis=-1, keepdims=True)\n",
        "\n",
        "p = softmax_stabil(np.array([[2., 1., 0.1], [-1., 0., 3.]]))\n",
        "print(p)\n",
        "print('jumlah per baris:', p.sum(axis=1))\n",
        "print(softmax_stabil(np.array([[1000., 1000., 999.]])))   # tanpa overflow",
    ),
    md(
        "### Soal 9 — `he_init_weights` (2 poin)",
        "",
        "He initialization: `std = sqrt(2/fan_in)` — dirancang untuk ReLU yang membuang",
        "setengah aktivasi (nilai negatif), sehingga variansi sinyal tetap terjaga antar",
        "layer. Bandingkan: Xavier/Glorot pakai `sqrt(1/fan_in)` — pas untuk sigmoid/tanh.",
        "",
        "Sisa test yang lain mengecek sifat engineering, bukan rumus: **deterministik**",
        "(rng seed sama → bobot sama — penting untuk reproducibility & debugging)."
    ),
    code(
        "def he_init_weights(rng, fan_in, fan_out):\n",
        "    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))\n",
        "\n",
        "w = he_init_weights(np.random.default_rng(0), 128, 64)\n",
        "print(w.shape, 'std ≈', round(w.std(), 4), '| teori:', round(np.sqrt(2 / 128), 4))\n",
        "w2 = he_init_weights(np.random.default_rng(0), 128, 64)\n",
        "print('seed sama -> hasil identik?', np.array_equal(w, w2))",
    ),
    md(
        "### Soal 10 — `split_batches` (2 poin)",
        "",
        "Ini mesin kecil di balik mini-batch SGD: ukuran `[32, 32, 32, 4]` untuk n=100.",
        "Batch terakhir yang lebih kecil itu normal dan diperbolehkan (Keras juga begitu);",
        "yang penting **jumlah total sampel tepat n** dan batch kosong tidak ikut.",
        "Di project starter, logika ini dipakai bersama shuffle per-epoch di `train.py`.",
    ),
    code(
        "def split_batches(n, batch_size):\n",
        "    ukuran = [batch_size] * (n // batch_size)\n",
        "    sisa = n % batch_size\n",
        "    if sisa:\n",
        "        ukuran.append(sisa)\n",
        "    return ukuran\n",
        "\n",
        "print(split_batches(100, 32), '-> total', sum(split_batches(100, 32)))\n",
        "print(split_batches(10, 10), split_batches(10, 7))",
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
