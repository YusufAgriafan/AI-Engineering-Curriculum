"""Generate starter.ipynb & solusi.ipynb for the Ch3 project."""
import json
from pathlib import Path

BASE = Path("AI-Engineering-Curriculum/03-ML-Klasik/project-starter-kampanye-marketing")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def tulis(nama, cells):
    path = BASE / nama
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


SETUP = [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "sys.path.insert(0, str(Path.cwd()))   # biar `ml_sederhana` bisa diimpor dari notebook\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "from ml_sederhana.clustering import kmeans, standardize\n",
    "from ml_sederhana.data import muat_semua\n",
    "from ml_sederhana.metrik import precision_recall_f1, roc_auc\n",
    "from ml_sederhana.model import LogisticRegressionGD\n",
    "\n",
    "X_train, y_train, X_val, y_val, X_test, y_test, NAMA = muat_semua()\n",
    "print('train/val/test:', len(y_train), len(y_val), len(y_test),\n",
    "      '| positif:', y_train.mean().round(3), y_val.mean().round(3), y_test.mean().round(3))\n",
]

# ============================== STARTER ==============================
starter = [
    md(
        "# 🏗️ Starter — Prediksi Respon Kampanye Marketing (Bab 3)",
        "",
        "> Notebook eksperimen untuk project starter. **Prasyarat:** semua test hijau",
        "> (`python -m unittest discover -s tests -v`). Kalau belum, selesaikan dulu",
        "> paket `ml_sederhana/` — notebook ini memakai fungsi-fungsimu.",
        "",
        "**Aturan main:** semua keputusan (threshold, jumlah klaster, interpretasi)",
        "diambil di **validation set**. Test set hanya disentuh di Bagian 6, sekali.",
    ),
    md("## Setup"),
    code(*SETUP),
    md(
        "## Bagian 1 — Baseline Dulu, Baru Model",
        "",
        "Sebelum merayakan accuracy model, hitung dulu baseline bodoh:",
        "**prediksi semua pelanggan tidak akan membeli**. Kelas positif cuma ~11%,",
        "jadi model bodoh ini otomatis dapat accuracy tinggi.",
    ),
    code(
        "# Baseline bodoh: semua prediksi 0\n",
        "y_bodoh = np.zeros_like(y_val)\n",
        "acc_bodoh = float((y_bodoh == y_val).mean())\n",
        "print(f'Accuracy model bodoh di val: {acc_bodoh:.4f}')\n",
        "print('-> Kalau modelmu cuma unggul sedikit dari ini, ia belum belajar apa-apa.')\n",
    ),
    code(
        "# TODO: latih modelmu (dari ml_sederhana.model), prediksi val (threshold 0.5),\n",
        "# lalu hitung precision/recall/F1 pakai ml_sederhana.metrik.\n",
        "model = LogisticRegressionGD(lr=0.1, n_iter=2000).fit(X_train, y_train)\n",
        "\n",
        "p_val = model.predict_proba(X_val)\n",
        "pred_val = model.predict(X_val)\n",
        "\n",
        "# TODO: precision, recall, f1 = precision_recall_f1(...)\n",
        "# print(f'precision={p:.3f} recall={r:.3f} f1={f1:.3f}')\n",
        "# -> Bandingkan precision/recall model vs model bodoh (yang nilainya 0 semua).\n",
    ),
    md(
        "**Refleksi 1:** accuracy 0.91 vs 0.89 — bedanya tipis. Tapi precision/recall",
        "model bodoh = 0/0. Metrik mana yang benar-benar menunjukkan model belajar?",
    ),
    md("## Bagian 2 — Kurva Loss"),
    code(
        "# Loss harus monoton turun kalau GD sehat.\n",
        "plt.figure(figsize=(7, 4))\n",
        "plt.plot(model.loss_history)\n",
        "plt.xlabel('iterasi'); plt.ylabel('BCE (train)'); plt.title('Kurva Loss')\n",
        "plt.grid(alpha=0.3); plt.show()\n",
        "\n",
        "print(f'loss awal  : {model.loss_history[0]:.4f}  (= ln 2, tebakan 50:50)')\n",
        "print(f'loss akhir : {model.loss_history[-1]:.4f}')\n",
    ),
    md(
        "## Bagian 3 — Eksperimen Threshold",
        "",
        "Threshold default 0.5 bukan hukum alam. Geser, catat precision & recall,",
        "dan pikirkan **biaya kesalahan** kampanye ini:",
        "",
        "- **FP** (kirim promo ke yang tak beli) → biaya promo mubazir (murah).",
        "- **FN** (lewatkan pelanggan yang mau beli) → kehilangan pendapatan (mahal).",
    ),
    code(
        "# TODO: loop threshold 0.05..0.95 (step 0.05). Untuk tiap threshold:\n",
        "#   - pred = (p_val >= th).astype(int)\n",
        "#   - hitung precision & recall\n",
        "#   - simpan ke list\n",
        "# Lalu print tabel & jawab: kalau FN jauh lebih mahal dari FP,\n",
        "# threshold berapa yang kamu pilih?\n",
        "ambang = np.round(np.arange(0.05, 0.96, 0.05), 2)\n",
        "# for th in ambang: ...\n",
    ),
    md(
        "**Refleksi 2:** threshold rendah → recall naik, precision turun. Kampanye",
        "ini mau minimal salah melewatkan pembeli. Threshold pilihanmu: ____ karena ____.",
    ),
    md(
        "## Bagian 4 — ROC-AUC: Menilai Ranking, Bukan Keputusan",
        "",
        "AUC menjawab: *kalau diambil satu pelanggan positif dan satu negatif acak,",
        "seberapa sering model memberi skor lebih tinggi ke yang positif?*",
        "AUC tidak terpengaruh threshold — ia menilai urutan, bukan keputusan.",
    ),
    code(
        "# TODO: hitung AUC val pakai roc_auc(y_val, p_val), lalu interpretasikan:\n",
        "#   0.5 = tebak acak, 1.0 = ranking sempurna.\n",
        "# auc = roc_auc(y_val, p_val)\n",
        "# print(f'AUC val: {auc:.4f}')\n",
        "\n",
        "# Sanity check: AUC HARUS sama di threshold mana pun (tidak pakai threshold sama sekali).\n",
    ),
    md("## Bagian 5 — Segmen Pelanggan dengan K-Means"),
    code(
        "# Gabungkan SEMUA pelanggan (clustering = unsupervised, tidak butuh label)\n",
        "X_all = np.vstack([X_train, X_val, X_test])\n",
        "y_all = np.concatenate([y_train, y_val, y_test])\n",
        "\n",
        "# TODO: standardize -> kmeans(k=2, seed=0) -> hitung ukuran & positive-rate tiap klaster\n",
        "Z = standardize(X_all)\n",
        "# labels, centroids, inertia = kmeans(...)\n",
        "# for k in range(2):\n",
        "#     anggota = y_all[labels == k]\n",
        "#     print(f'klaster {k}: {len(anggota)} orang, positive-rate {anggota.mean():.3f}')\n",
    ),
    md(
        "**Refleksi 3:** dua klaster punya positive-rate yang berbeda jauh meski",
        "k-means TIDAK pernah melihat label. Fitur apa yang paling mungkin memisahkan",
        "mereka? (hint: bandingkan centroid dengan NAMA fitur — dan ingat hasil test",
        "`test_kontak_ke_fitur_terkuat`.) Kalau budget promo cuma cukup satu segmen,",
        "mana yang dipilih?",
    ),
    md(
        "## Bagian 6 — Sentuh Test Set (SEKALI!)",
        "",
        "> ⚠ Setelah cell ini dijalankan, eksperimen selesai. Kalau hasil test",
        "> mengecewakan dan kamu mengubah model lalu mengetes lagi — itu artinya test",
        "> set sudah jadi validation set kedua, dan angkanya jadi terlalu optimis.",
    ),
    code(
        "# Pilih threshold TERBAIK dari val (Bagian 3), baru evaluasi sekali di test.\n",
        "TH_PILIHAN = 0.35   # TODO: ganti dengan pilihanmu dari Bagian 3\n",
        "\n",
        "p_test = model.predict_proba(X_test)\n",
        "pred_test = (p_test >= TH_PILIHAN).astype(int)\n",
        "print(f'accuracy test @th={TH_PILIHAN}: {(pred_test == y_test).mean():.4f}')\n",
        "print(f'precision/recall test : {precision_recall_f1(y_test, pred_test)}')\n",
        "print(f'AUC test              : {roc_auc(y_test, p_test):.4f}')\n",
    ),
    md(
        "---",
        "",
        "## 📝 Pertanyaan Analisis (jawab dengan markdown, kata-kata sendiri + angka)",
        "",
        "1. Kenapa accuracy 0.9125 menipu, dan metrik apa yang memperlihatkan kebenarannya?",
        "2. Threshold mana yang kamu pilih untuk kampanye ini? Jelaskan dengan trade-off biaya FP vs FN.",
        "3. Kenapa ROC-AUC tidak berubah saat threshold digeser, padahal F1 berubah?",
        "4. Justifikasi pemilihan segmen promo dengan angka (positive-rate & ukuran klaster).",
        "5. Bandingkan angka val vs test. Seberapa beda? Kapan perbedaan jadi tanda bahaya?",
        "",
        "Setelah semua terjawab → isi `RUBRIK.md`. Baru bandingkan dengan `solusi.ipynb`.",
    ),
]

# ============================== SOLUSI ==============================
solusi = [
    md(
        "# 🔑 Solusi — Prediksi Respon Kampanye Marketing (Bab 3)",
        "",
        "> Buka HANYA setelah mencoba sendiri. Fokus pada **cara membaca angka**,",
        "> bukan sekadar mencocokkan hasil.",
    ),
    code(*SETUP),
    md("## Bagian 1 — Baseline vs Model"),
    code(
        "y_bodoh = np.zeros_like(y_val)\n",
        "print(f'Accuracy model bodoh di val: {(y_bodoh == y_val).mean():.4f}')\n",
        "\n",
        "model = LogisticRegressionGD(lr=0.1, n_iter=2000).fit(X_train, y_train)\n",
        "p_val = model.predict_proba(X_val)\n",
        "pred_val = model.predict(X_val)\n",
        "\n",
        "p, r, f1 = precision_recall_f1(y_val, pred_val)\n",
        "print(f'Model      : accuracy={(pred_val == y_val).mean():.4f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}')\n",
        "print('Model bodoh: precision=0.0 (tidak pernah menebak positif)  recall=0.0')\n",
        "print('-> Selisih accuracy tipis, tapi kemampuan MENEMUKAN pembeli hanya dimiliki model.')\n",
    ),
    md("## Bagian 2 — Kurva Loss"),
    code(
        "plt.figure(figsize=(7, 4))\n",
        "plt.plot(model.loss_history)\n",
        "plt.xlabel('iterasi'); plt.ylabel('BCE (train)'); plt.title('Kurva Loss')\n",
        "plt.grid(alpha=0.3); plt.show()\n",
        "print(f'loss awal {model.loss_history[0]:.4f} (≈ ln 2 = 0.6931, tebakan 50:50) -> akhir {model.loss_history[-1]:.4f}')\n",
    ),
    md("## Bagian 3 — Eksperimen Threshold"),
    code(
        "rows = []\n",
        "for th in np.round(np.arange(0.05, 0.96, 0.05), 2):\n",
        "    pred = (p_val >= th).astype(int)\n",
        "    pr, rc, _ = precision_recall_f1(y_val, pred)\n",
        "    rows.append((th, pr, rc))\n",
        "\n",
        "print(f\"{'th':>5} | {'precision':>9} | {'recall':>7}\")\n",
        "print('-' * 30)\n",
        "for th, pr, rc in rows:\n",
        "    print(f'{th:>5.2f} | {pr:>9.3f} | {rc:>7.3f}')\n",
        "\n",
        "# FN (lewatkan pembeli) jauh lebih mahal -> pilih recall tinggi dengan precision masih layak.\n",
        "# Dari tabel: th ≈ 0.30-0.40 memberi recall ≥ 0.7 dengan precision ≈ 0.4-0.5.\n",
    ),
    md(
        "**Bacaan hasil:** di th=0.5 model mendapat precision 0.65 / recall 0.48 —",
        "hampir setengah pembeli terlewat. Menurunkan threshold ke ~0.35 menukar",
        "sedikit precision dengan recall yang jauh lebih tinggi; untuk kampanye murah",
        "per-kontak, itu pertukaran yang masuk akal.",
    ),
    md("## Bagian 4 — ROC-AUC"),
    code(
        "auc_val = roc_auc(y_val, p_val)\n",
        "print(f'AUC val: {auc_val:.4f}')\n",
        "print(f'-> Dua pelanggan acak (satu pembeli, satu bukan): model mengurutkan benar ~{auc_val:.0%} waktu.')\n",
    ),
    md("## Bagian 5 — Segmen Pelanggan"),
    code(
        "X_all = np.vstack([X_train, X_val, X_test])\n",
        "y_all = np.concatenate([y_train, y_val, y_test])\n",
        "\n",
        "Z = standardize(X_all)\n",
        "labels, centroids, inertia = kmeans(Z, k=2, seed=0)\n",
        "print(f'inertia: {inertia:.1f}\\n')\n",
        "for k in range(2):\n",
        "    anggota = y_all[labels == k]\n",
        "    print(f'klaster {k}: {len(anggota):>4} orang, positive-rate {anggota.mean():.4f}')\n",
        "\n",
        "print('\\ncentroid (fitur 0-2, terskala):')\n",
        "print(centroids[:, :3].round(2))\n",
        "print('fitur paling membedakan klaster:', NAMA[int(np.argmax(np.abs(centroids[0] - centroids[1])))])\n",
    ),
    md(
        "**Bacaan hasil:** klaster terpisah terutama oleh `jam_pakai_per_minggu`",
        "(fitur bimodal yang paling membedakan kelas). Segmen dengan positive-rate",
        "tinggi (~17%) adalah target promo; segmen rendah (~4%) nyaris tidak worth",
        "dikontak — padahal k-means tidak pernah melihat label sedikit pun.",
    ),
    md("## Bagian 6 — Test Set, Sekali"),
    code(
        "TH_PILIHAN = 0.35   # dipilih di validation (Bagian 3), BUKAN dari test\n",
        "\n",
        "p_test = model.predict_proba(X_test)\n",
        "pred_test = (p_test >= TH_PILIHAN).astype(int)\n",
        "print(f'accuracy test @th={TH_PILIHAN}: {(pred_test == y_test).mean():.4f}')\n",
        "print(f'precision/recall test : {precision_recall_f1(y_test, pred_test)}')\n",
        "print(f'AUC test              : {roc_auc(y_test, p_test):.4f}')\n",
        "print(f'AUC val               : {auc_val:.4f}  (bandingkan: beda tipis = sehat)')\n",
    ),
    md(
        "## 📝 Contoh Jawaban Analisis (ringkas)",
        "",
        "1. Accuracy menipu karena 89% kelas negatif: menebak 'tidak semua' sudah 0.89.",
        "   Precision/recall memperlihatkan kemampuan menemukan pembeli — model bodoh 0/0, model kita 0.65/0.48.",
        "2. Threshold ~0.35: FN (kehilangan pembeli) jauh lebih mahal daripada FP (promo mubazir),",
        "   jadi bayar precision sedikit demi recall jauh lebih tinggi.",
        "3. AUC menilai ranking skor (invariant threshold); F1 menilai keputusan biner",
        "   pada satu titik threshold — geser threshold, keputusan berubah, ranking tidak.",
        "4. Segmen positive-rate ~17% vs ~4%: kontak segmen tinggi → ~4x lipat kemungkinan konversi",
        "   per promo; justifikasi pakai positive-rate × ukuran segmen.",
        "5. Val vs test beda ~1-2 poin persen: wajar (kebetulan sampling). Bahaya bila test",
        "   dipakai berulang untuk memilih model → estimasi produksi jadi terlalu optimis.",
    ),
]

tulis("starter.ipynb", starter)
tulis("solusi.ipynb", solusi)
print("done")
