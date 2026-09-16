"""Generate starter.ipynb & solusi.ipynb for the Ch4 numpy-NN project."""
import json
from pathlib import Path

BASE = Path("AI-Engineering-Curriculum/04-Deep-Learning-TensorFlow/project-starter-numpy-nn")


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
    "sys.path.insert(0, str(Path.cwd()))\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "from nn_mini.data import muat_semua\n",
    "from nn_mini.layers import Dense, Sequential\n",
    "from nn_mini.train import train_model\n",
    "\n",
    "X_train, y_train, X_val, y_val, X_test, y_test = muat_semua()\n",
    "print('splits:', len(y_train), len(y_val), len(y_test),\n",
    "      '| positif:', int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))\n",
]

# ============================== STARTER ==============================
starter = [
    md(
        "# 🏗️ Starter — Neural Network Mini dari Nol (Bab 4)",
        "",
        "> Notebook eksperimen. **Prasyarat:** semua 48 test hijau",
        "> (`python -m unittest discover -s tests -v`).",
        "",
        "**Aturan main:** semua keputusan (arsitektur, lr, early stop) diambil di",
        "**validation set**. Test set hanya disentuh di Bagian 6, sekali.",
    ),
    md("## Setup & Peta Data"),
    code(*SETUP),
    code(
        "# Lihat dulu bentuk datanya — non-linear seperti dua bulan yang mengait\n",
        "plt.figure(figsize=(5, 4))\n",
        "plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=15, alpha=0.7)\n",
        "plt.title('Dua bulan (train)'); plt.xlabel('x1'); plt.ylabel('x2'); plt.show()\n",
    ),
    md(
        "## Bagian 1 — Baseline Linear Dulu",
        "",
        "Sebelum MLP, buktikan bahwa model LINEAR mentok. Gunakan `Sequential`",
        "dengan SATU layer output sigmoid tanpa hidden layer — itu logistic regression.",
    ),
    code(
        "linear_model = Sequential([Dense(1, activation='sigmoid', seed=0)])\n",
        "hist_lin = train_model(linear_model, X_train, y_train, X_val, y_val,\n",
        "                       epochs=400, batch_size=32, lr=0.5, seed=0)\n",
        "print(f'Baseline linear: val acc = {hist_lin[\"val_acc\"][-1]:.4f}')\n",
        "print('-> Catat angka ini. Model non-linear nanti HARUS mengalahkannya.')\n",
    ),
    md(
        "## Bagian 2 — MLP Pertamamu",
        "",
        "Arsitektur 16 → 8 → 1 (ReLU + ReLU + sigmoid), lr 0.1, batch 32,",
        "300 epoch tanpa early stopping. Bandingkan kurva loss-nya dengan baseline.",
    ),
    code(
        "mlp = Sequential([Dense(16, activation='relu', seed=7),\n",
        "                  Dense(8, activation='relu', seed=8),\n",
        "                  Dense(1, activation='sigmoid', seed=9)])\n",
        "hist_mlp = train_model(mlp, X_train, y_train, X_val, y_val,\n",
        "                       epochs=300, batch_size=32, lr=0.1, seed=0)\n",
        "print(f'MLP: val acc = {hist_mlp[\"val_acc\"][-1]:.4f}')\n",
        "\n",
        "plt.figure(figsize=(11, 4))\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(hist_lin['train_loss'], label='linear train')\n",
        "plt.plot(hist_lin['val_loss'], '--', label='linear val')\n",
        "plt.plot(hist_mlp['train_loss'], label='MLP train')\n",
        "plt.plot(hist_mlp['val_loss'], '--', label='MLP val')\n",
        "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.title('Loss'); plt.legend()\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot(hist_lin['val_acc'], label='linear val acc')\n",
        "plt.plot(hist_mlp['val_acc'], label='MLP val acc')\n",
        "plt.axhline(0.5, color='gray', ls=':')\n",
        "plt.xlabel('epoch'); plt.ylabel('val accuracy'); plt.title('Akurasi'); plt.legend()\n",
        "plt.tight_layout(); plt.show()\n",
    ),
    md(
        "**Refleksi 1:** berapa % peningkatan val acc MLP atas baseline linear?",
        "Di kurva loss, MLP turun lebih dulu di bagian mana — awal atau akhir?",
    ),
    md("## Bagian 3 — Batas Keputusan (Visualisasi Paling Penting)"),
    code(
        "# Gambar prediksi model di seluruh grid — lihat BATAS KEPUTUSANNYA.\n",
        "def plot_boundary(model, judul):\n",
        "    xs = np.linspace(X_train[:, 0].min() - 0.3, X_train[:, 0].max() + 0.3, 200)\n",
        "    ys = np.linspace(X_train[:, 1].min() - 0.3, X_train[:, 1].max() + 0.3, 200)\n",
        "    xx, yy = np.meshgrid(xs, ys)\n",
        "    grid = np.column_stack([xx.ravel(), yy.ravel()])\n",
        "    pp = model.predict_proba(grid).reshape(xx.shape)\n",
        "    plt.contourf(xx, yy, pp, levels=20, cmap='coolwarm', alpha=0.6)\n",
        "    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='coolwarm', s=12, edgecolors='k', lw=0.3)\n",
        "    plt.title(judul)\n",
        "\n",
        "plt.figure(figsize=(11, 4))\n",
        "plt.subplot(1, 2, 1); plot_boundary(linear_model, 'Baseline linear — batas GARIS')\n",
        "plt.subplot(1, 2, 2); plot_boundary(mlp, 'MLP — batas MELENGKUNG')\n",
        "plt.tight_layout(); plt.show()\n",
    ),
    md(
        "**Refleksi 2:** ini jawaban visual untuk pertanyaan analisis #1.",
        "Garis lurus tidak mungkin memisahkan dua bulan yang mengait; MLP bebas",
        "melengkung karena hidden layer mengubah representasi fiturnya.",
    ),
    md("## Bagian 4 — Eksperimen Arsitektur (satu variabel per percobaan)"),
    code(
        "# TODO: lengkapi eksperimen. Setiap baris = satu konfigurasi.\n",
        "# Catat val acc tiap konfigurasi — cari pola, bukan cuma angka terbaik.\n",
        "konfigurasi = [\n",
        "    ('kecil (8)',        [Dense(8, activation='relu', seed=1),  Dense(1, activation='sigmoid', seed=2)]),\n",
        "    ('sedang (16-8)',    [Dense(16, activation='relu', seed=7), Dense(8, activation='relu', seed=8), Dense(1, activation='sigmoid', seed=9)]),\n",
        "    ('lebar (64)',       [Dense(64, activation='relu', seed=3), Dense(1, activation='sigmoid', seed=4)]),\n",
        "    ('dalam (16-16-8)',  [Dense(16, activation='relu', seed=5), Dense(16, activation='relu', seed=6), Dense(8, activation='relu', seed=10), Dense(1, activation='sigmoid', seed=11)]),\n",
        "]\n",
        "\n",
        "# for nama, layers in konfigurasi:\n",
        "#     model = Sequential(layers)\n",
        "#     h = train_model(model, X_train, y_train, X_val, y_val,\n",
        "#                     epochs=250, batch_size=32, lr=0.1, seed=0)\n",
        "#     print(f'{nama:>16}: val acc = {h[\"val_acc\"][-1]:.4f}  val loss = {h[\"val_loss\"][-1]:.4f}')\n",
    ),
    md(
        "**Refleksi 3:** dataset ini \"mudah\" (600 titik, 2 fitur) — arsitektur besar",
        "tidak selalu menang. Di data apa model besar justru berisiko? (ingat Bab 3: overfitting)",
    ),
    md("## Bagian 5 — Learning Rate & Early Stopping"),
    code(
        "# TODO: jalankan MLP yang sama dengan lr = 50.0 — amati kurva loss.\n",
        "# hist_bad = train_model(..., lr=50.0, epochs=150)\n",
        "# plt.plot(hist_bad['train_loss']); plt.title('lr=50: apa yang terjadi?'); plt.show()\n",
        "\n",
        "# TODO: ulangi dengan early stopping patience=30 (epochs=1000).\n",
        "# Berapa epoch yang benar-benar berjalan? Bandingkan val loss akhirnya\n",
        "# dengan MLP tanpa early stopping (300 epoch) di Bagian 2.\n",
        "# hist_es = train_model(..., epochs=1000, patience=30)\n",
        "# print(len(hist_es['val_loss']), 'epoch berjalan')\n",
    ),
    md("## Bagian 6 — Sentuh Test Set (SEKALI!)"),
    code(
        "# Pakai konfigurasi TERBAIK dari eksperimen val (Bagian 2 & 4).\n",
        "p_test_lin = linear_model.predict_proba(X_test)\n",
        "p_test_mlp = mlp.predict_proba(X_test)\n",
        "print(f'test acc baseline linear : {((p_test_lin >= 0.5) == y_test).mean():.4f}')\n",
        "print(f'test acc MLP             : {((p_test_mlp >= 0.5) == y_test).mean():.4f}')\n",
        "print('-> Selisih ini yang kamu \"jual\" kalau ini proyek produksi.')\n",
    ),
    md(
        "---",
        "",
        "## 📝 Pertanyaan Analisis (jawab dengan markdown, kata-kata sendiri + angka)",
        "",
        "1. Kenapa model linear mentok di 0.889 padahal dilatih 400 epoch?",
        "2. Jelaskan backpropagation dengan kata-katamu sendiri (gradien mengalir mundur).",
        "3. Saat lr=50, apa yang terlihat di kurva loss dan kenapa (mekanisme update)?",
        "4. Kenapa early stopping memantau val loss, bukan train loss? Bahaya patience terlalu kecil?",
        "5. Init semua bobot nol: apa yang terjadi di hidden layer (hint: simetri gradien)?",
        "",
        "Setelah semua terjawab → isi `RUBRIK.md`. Baru bandingkan dengan `solusi.ipynb`.",
    ),
]

# ============================== SOLUSI ==============================
solusi = [
    md(
        "# 🔑 Solusi — Neural Network Mini dari Nol (Bab 4)",
        "",
        "> Buka HANYA setelah mencoba sendiri. Fokus pada **cara membaca kurva & angka**.",
    ),
    code(*SETUP),
    md("## Bagian 1 — Baseline Linear"),
    code(
        "linear_model = Sequential([Dense(1, activation='sigmoid', seed=0)])\n",
        "hist_lin = train_model(linear_model, X_train, y_train, X_val, y_val,\n",
        "                       epochs=400, batch_size=32, lr=0.5, seed=0)\n",
        "print(f'Baseline linear: val acc = {hist_lin[\"val_acc\"][-1]:.4f}')\n",
        "print('-> bertahan di ±0.88-0.89 TIDAK PEDULI berapa lama dilatih: batas keputusannya garis.')\n",
    ),
    md("## Bagian 2 — MLP"),
    code(
        "mlp = Sequential([Dense(16, activation='relu', seed=7),\n",
        "                  Dense(8, activation='relu', seed=8),\n",
        "                  Dense(1, activation='sigmoid', seed=9)])\n",
        "hist_mlp = train_model(mlp, X_train, y_train, X_val, y_val,\n",
        "                       epochs=300, batch_size=32, lr=0.1, seed=0)\n",
        "print(f'MLP: val acc = {hist_mlp[\"val_acc\"][-1]:.4f}  (val loss {hist_mlp[\"val_loss\"][-1]:.4f})')\n",
        "\n",
        "plt.figure(figsize=(11, 4))\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(hist_lin['train_loss'], label='linear train')\n",
        "plt.plot(hist_lin['val_loss'], '--', label='linear val')\n",
        "plt.plot(hist_mlp['train_loss'], label='MLP train')\n",
        "plt.plot(hist_mlp['val_loss'], '--', label='MLP val')\n",
        "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.title('Loss'); plt.legend()\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot(hist_lin['val_acc'], label='linear')\n",
        "plt.plot(hist_mlp['val_acc'], label='MLP')\n",
        "plt.xlabel('epoch'); plt.ylabel('val acc'); plt.legend()\n",
        "plt.tight_layout(); plt.show()\n",
    ),
    md(
        "**Bacaan hasil:** linear datar di 0.889 sejak epoch awal — loss turun sedikit lalu",
        "stuck (konvergen ke solusi linear terbaik). MLP: loss terus turun sampai < 0.1,",
        "val acc 0.97+. Selisih ±9 poin persen = harga dari non-linearitas.",
    ),
    md("## Bagian 3 — Batas Keputusan"),
    code(
        "def plot_boundary(model, judul):\n",
        "    xs = np.linspace(X_train[:, 0].min() - 0.3, X_train[:, 0].max() + 0.3, 200)\n",
        "    ys = np.linspace(X_train[:, 1].min() - 0.3, X_train[:, 1].max() + 0.3, 200)\n",
        "    xx, yy = np.meshgrid(xs, ys)\n",
        "    grid = np.column_stack([xx.ravel(), yy.ravel()])\n",
        "    pp = model.predict_proba(grid).reshape(xx.shape)\n",
        "    plt.contourf(xx, yy, pp, levels=20, cmap='coolwarm', alpha=0.6)\n",
        "    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='coolwarm', s=12, edgecolors='k', lw=0.3)\n",
        "    plt.title(judul)\n",
        "\n",
        "plt.figure(figsize=(11, 4))\n",
        "plt.subplot(1, 2, 1); plot_boundary(linear_model, 'Baseline linear')\n",
        "plt.subplot(1, 2, 2); plot_boundary(mlp, 'MLP')\n",
        "plt.tight_layout(); plt.show()\n",
    ),
    md("## Bagian 4 — Eksperimen Arsitektur"),
    code(
        "konfigurasi = [\n",
        "    ('kecil (8)',        [Dense(8, activation='relu', seed=1),  Dense(1, activation='sigmoid', seed=2)]),\n",
        "    ('sedang (16-8)',    [Dense(16, activation='relu', seed=7), Dense(8, activation='relu', seed=8), Dense(1, activation='sigmoid', seed=9)]),\n",
        "    ('lebar (64)',       [Dense(64, activation='relu', seed=3), Dense(1, activation='sigmoid', seed=4)]),\n",
        "    ('dalam (16-16-8)',  [Dense(16, activation='relu', seed=5), Dense(16, activation='relu', seed=6), Dense(8, activation='relu', seed=10), Dense(1, activation='sigmoid', seed=11)]),\n",
        "]\n",
        "for nama, layers in konfigurasi:\n",
        "    model = Sequential(layers)\n",
        "    h = train_model(model, X_train, y_train, X_val, y_val,\n",
        "                    epochs=250, batch_size=32, lr=0.1, seed=0)\n",
        "    print(f'{nama:>16}: val acc = {h[\"val_acc\"][-1]:.4f}  val loss = {h[\"val_loss\"][-1]:.4f}')\n",
        "print('-> Semua konfigurasi non-linear biasanya 0.96+. Dataset kecil: kapasitas ekstra tidak membantu.')\n",
    ),
    md("## Bagian 5 — Learning Rate & Early Stopping"),
    code(
        "mlp_bad = Sequential([Dense(16, activation='relu', seed=7),\n",
        "                      Dense(8, activation='relu', seed=8),\n",
        "                      Dense(1, activation='sigmoid', seed=9)])\n",
        "hist_bad = train_model(mlp_bad, X_train, y_train, X_val, y_val,\n",
        "                       epochs=150, batch_size=32, lr=50.0, seed=0)\n",
        "plt.plot(hist_bad['train_loss'])\n",
        "plt.title('lr=50 — loss osilasi/menjauh, tidak konvergen'); plt.xlabel('epoch'); plt.show()\n",
        "print(f'lr=50: val loss akhir = {hist_bad[\"val_loss\"][-1]:.4f}  (bandingkan lr=0.1: < 0.15)')\n",
        "\n",
        "mlp_es = Sequential([Dense(16, activation='relu', seed=7),\n",
        "                     Dense(8, activation='relu', seed=8),\n",
        "                     Dense(1, activation='sigmoid', seed=9)])\n",
        "hist_es = train_model(mlp_es, X_train, y_train, X_val, y_val,\n",
        "                      epochs=1000, batch_size=32, lr=0.1, patience=30, seed=0)\n",
        "print(f'early stopping: berhenti setelah {len(hist_es[\"val_loss\"])} epoch (dari 1000)')\n",
        "print(f'val loss akhir {hist_es[\"val_loss\"][-1]:.4f} ≈ versi 300 epoch — hemat waktu, hasil setara')\n",
    ),
    md("## Bagian 6 — Test Set, Sekali"),
    code(
        "p_test_lin = linear_model.predict_proba(X_test)\n",
        "p_test_mlp = mlp.predict_proba(X_test)\n",
        "print(f'test acc baseline linear : {((p_test_lin >= 0.5) == y_test).mean():.4f}')\n",
        "print(f'test acc MLP             : {((p_test_mlp >= 0.5) == y_test).mean():.4f}')\n",
    ),
    md(
        "## 📝 Contoh Jawaban Analisis (ringkas)",
        "",
        "1. Batas keputusan logistic regression adalah garis (hyperplane) di ruang fitur.",
        "   Dua bulan yang mengait TIDAK bisa dipisah garis berapa pun lama training —",
        "   loss konvergen ke solusi linear terbaik (val acc ±0.889).",
        "2. Error di output diubah jadi gradien loss; tiap layer memakai cache forward-nya",
        "   (X, z) untuk menghitung seberapa besar tiap bobot berkontribusi (chain rule),",
        "   lalu meneruskan gradien ke layer sebelumnya lewat Wᵀ — begitu seterusnya sampai input.",
        "3. lr=50: langkah update jauh melebihi lebar lembah loss → parameter melompati",
        "   minimum bolak-balik → loss osilasi/menaik, tidak konvergen.",
        "4. Train loss selalu bisa diturunkan (hafalan); val loss menunjukkan generalisasi.",
        "   Patience terlalu kecil = berhenti saat val loss sedang 'napas' — kena noise epoch.",
        "5. Bobot semua nol → semua neuron di layer yang sama dapat gradien IDENTIK →",
        "   update identik → tetap identik selamanya (simetri tidak pecah) →",
        "   efektifnya cuma 1 neuron per layer. Karena itu init acak (He/Xavier).",
    ),
]

tulis("starter.ipynb", starter)
tulis("solusi.ipynb", solusi)
print("done")
