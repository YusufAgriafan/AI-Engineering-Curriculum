"""Generate 01_lab_deep_learning.ipynb for Bab 4."""
import json
from pathlib import Path

OUT = Path("AI-Engineering-Curriculum/04-Deep-Learning-TensorFlow/01_lab_deep_learning.ipynb")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


cells = []

# ============================ HEADER ============================
cells.append(md(
    "# 🧪 Lab 4 — Praktikum Deep Learning (dari Nol, Menuju TensorFlow)",
    "",
    "> Pendamping materi Bab 4. Prinsip: **NN dibangun dan dilatih dari nol dulu",
    "> (numpy), baru dibandingkan dengan TensorFlow/Keras** — supaya `model.fit()`",
    "> tidak pernah jadi kotak hitam.",
    "",
    "| Bagian | Topik | Waktu (menit) |",
    "|---|---|---|",
    "| 1 | Neuron & fungsi aktivasi (stabil!) | 15 |",
    "| 2 | XOR: kenapa butuh hidden layer | 20 |",
    "| 3 | Backpropagation manual + chain rule | 20 |",
    "| 4 | Melatih MLP dari nol + BCE | 25 |",
    "| 5 | Softmax & cross-entropy multikelas | 15 |",
    "| 6 | Weight init: nol vs He | 15 |",
    "| 7 | Learning rate & langkah SGD | 15 |",
    "| 8 | Mini-batch vs full-batch | 15 |",
    "| 9 | Early stopping | 15 |",
    "| 10 | Skip connection & menuju TensorFlow | 20 |",
    "",
    "Setiap latihan punya cell **✅ Cek** — jalankan, kalau ✅ berarti pemahamanmu benar.",
))

# ============================ BAGIAN 1 ============================
cells.append(md(
    "## Bagian 1 — Neuron & Fungsi Aktivasi (Stabil-Numerik!)",
    "",
    "Neuron = `z = w·x + b` lalu aktivasi. Dua aktivasi paling penting: ReLU",
    "(hidden) dan sigmoid (output biner). *Stabil-numerik* bukan bonus — ini wajib:",
    "`exp(1000)` = overflow.",
))

cells.append(code(
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def sigmoid_stabil(z):\n",
    "    \"\"\"Dua cabang: exp tidak pernah menerima argumen besar positif.\"\"\"\n",
    "    z = np.asarray(z, dtype=float)\n",
    "    out = np.empty_like(z)\n",
    "    pos = z >= 0\n",
    "    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n",
    "    ez = np.exp(z[~pos])\n",
    "    out[~pos] = ez / (1.0 + ez)\n",
    "    return out\n",
    "\n",
    "def sigmoid_naive(z):\n",
    "    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))\n",
    "\n",
    "z_ekstrem = np.array([-1000.0, 0.0, 1000.0])\n",
    "print('naive :', sigmoid_naive(z_ekstrem))    # RuntimeWarning overflow!\n",
    "print('stabil:', sigmoid_stabil(z_ekstrem))   # bersih: [0, 0.5, 1]\n",
    "\n",
    "zz = np.linspace(-8, 8, 200)\n",
    "plt.figure(figsize=(9, 3.5))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(zz, sigmoid_stabil(zz)); plt.title('Sigmoid'); plt.axhline(0.5, ls=':', c='gray')\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(zz, np.maximum(zz, 0)); plt.title('ReLU');\n",
    "plt.tight_layout(); plt.show()\n",
))

cells.append(md(
    "### 🎯 Latihan 1.1 — Softmax Stabil",
    "",
    "Softmax mengubah logits jadi distribusi probabilitas (jumlah = 1). Trik",
    "stabil: kurangi dengan max per baris — hasilnya MATEMATIS SAMA tapi tanpa overflow.",
))

cells.append(code(
    "def softmax_stabil(logits):\n",
    "    \"\"\"Softmax per baris (1D juga boleh).\n",
    "    langkah: shifted = logits - max ; e = exp(shifted) ; e / e.sum()\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(softmax_stabil(np.array([2.0, 1.0, 0.1])))          # [0.659, 0.242, 0.099]\n",
    "# print(softmax_stabil(np.array([1000., 0.])))              # [1., 0.] tanpa NaN!\n",
))

cells.append(code(
    "# ✅ Cek latihan 1.1\n",
    "p = softmax_stabil(np.array([2.0, 1.0, 0.1]))\n",
    "assert abs(p.sum() - 1.0) < 1e-12 and np.all(p > 0)\n",
    "assert np.allclose(p, [0.659, 0.242, 0.099], atol=1e-3)\n",
    "# invarian shift: +1000 pada semua logits TIDAK mengubah hasil\n",
    "assert np.allclose(softmax_stabil(np.array([1., 2., 3.])),\n",
    "                   softmax_stabil(np.array([1001., 1002., 1003.]))), 'shift invariance'\n",
    "P = softmax_stabil(np.array([[1., 2., 3.], [1., 2., 3.]]))\n",
    "assert P.shape == (2, 3) and np.allclose(P.sum(axis=1), 1.0), 'per baris untuk 2D'\n",
    "print('✅ Latihan 1.1 benar!')\n",
))

# ============================ BAGIAN 2 ============================
cells.append(md(
    "## Bagian 2 — XOR: Kenapa Butuh Hidden Layer",
    "",
    "XOR tidak bisa dipisahkan garis lurus (bukan *linearly separable*). Logistic",
    "regression = model linear → PASTI gagal, berapa pun epoch. MLP 1 hidden layer",
    "→ berhasil. Ini demonstrasi terbaik \"kenapa deep\".",
))

cells.append(code(
    "Xxor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])\n",
    "yxor = np.array([0., 1., 1., 0.])\n",
    "\n",
    "# --- model linear (logistic regression, full-batch GD) ---\n",
    "w = np.zeros(2); b = 0.0\n",
    "for _ in range(2000):\n",
    "    p = sigmoid_stabil(Xxor @ w + b)\n",
    "    w -= 1.0 * Xxor.T @ (p - yxor) / 4\n",
    "    b -= 1.0 * np.sum(p - yxor) / 4\n",
    "pred = (sigmoid_stabil(Xxor @ w + b) >= 0.5).astype(int)\n",
    "print('logistic regression prediksi:', pred, '| target:', yxor.astype(int))\n",
    "print(f'accuracy: {(pred == yxor).mean():.2f}  <- TEPUK TANGAN gagal, selamanya.')\n",
))

cells.append(code(
    "def latih_xor(lr=1.0, epochs=3000, n_hidden=8, seed=0):\n",
    "    \"\"\"MLP 2->hidden->1 (ReLU + sigmoid), full-batch, manual backprop.\"\"\"\n",
    "    rng = np.random.default_rng(seed)\n",
    "    W1 = rng.normal(0, 1.0, (2, n_hidden)); b1 = np.zeros(n_hidden)\n",
    "    W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1)); b2 = np.zeros(1)\n",
    "    losses = []\n",
    "    m = len(yxor)\n",
    "    for ep in range(epochs):\n",
    "        h = np.maximum(Xxor @ W1 + b1, 0)          # hidden ReLU\n",
    "        o = sigmoid_stabil(h @ W2 + b2)            # output sigmoid\n",
    "        p = np.clip(o.ravel(), 1e-12, 1 - 1e-12)\n",
    "        losses.append(float(-np.mean(yxor * np.log(p) + (1 - yxor) * np.log(1 - p))))\n",
    "        # --- backprop (chain rule, turunan BCE: (o - y)) ---\n",
    "        dz2 = (o - yxor.reshape(-1, 1)) / m\n",
    "        dW2 = h.T @ dz2; db2 = dz2.sum(0)\n",
    "        dh = dz2 @ W2.T\n",
    "        dz1 = dh * (h > 0)                         # turunan ReLU\n",
    "        dW1 = Xxor.T @ dz1; db1 = dz1.sum(0)\n",
    "        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2\n",
    "    h = np.maximum(Xxor @ W1 + b1, 0)\n",
    "    o = sigmoid_stabil(h @ W2 + b2).ravel()\n",
    "    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,\n",
    "            'probs': o, 'losses': losses}\n",
    "\n",
    "hasil = latih_xor()\n",
    "print('probabilitas MLP:', hasil['probs'].round(3))\n",
    "print(f'accuracy: {((hasil[\"probs\"] >= 0.5) == yxor).mean():.2f}')\n",
    "print(f'loss: {hasil[\"losses\"][0]:.4f} -> {hasil[\"losses\"][-1]:.5f}')\n",
))

cells.append(md(
    "### 🎯 Latihan 2.1 — Generalisasi ke Fungsi `train_xor`",
    "",
    "Bungkus loop di atas jadi fungsi yang bisa dieksperimenkan: ubah lr, epochs,",
    "jumlah neuron, seed — SATU variabel per percobaan (disiplin Bab 3).",
))

cells.append(code(
    "def train_xor(lr=1.0, epochs=3000, n_hidden=8, seed=0):\n",
    "    \"\"\"Sama seperti latih_xor tapi parameter masuk lewat argumen.\n",
    "    Return (accuracy, list_loss).\"\"\"\n",
    "    # TODO: salin + adaptasi badan latih_xor (jangan panggil latih_xor)\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# acc, losses = train_xor(lr=1.0, epochs=3000, seed=0)\n",
    "# print(acc, losses[-1])\n",
))

cells.append(code(
    "# ✅ Cek latihan 2.1\n",
    "acc, losses = train_xor(lr=1.0, epochs=3000, seed=0)\n",
    "assert acc == 1.0, f'XOR harus bisa 100%, dapat {acc}'\n",
    "assert losses[-1] < 0.01, 'loss harus konvergen jauh'\n",
    "assert len(losses) == 3000\n",
    "# robust: seed berbeda tetap berhasil (masalahnya bukan kebetulan init)\n",
    "for seed in (1, 2, 3):\n",
    "    a2, _ = train_xor(lr=1.0, epochs=3000, seed=seed)\n",
    "    assert a2 == 1.0, f'seed {seed} gagal: {a2}'\n",
    "print('✅ Latihan 2.1 benar!')\n",
))

# ============================ BAGIAN 3 ============================
cells.append(md(
    "## Bagian 3 — Backpropagation Manual & Chain Rule",
    "",
    "Backprop = chain rule yang diterapkan sistematis: gradien loss mengalir mundur,",
    "tiap layer memakai cache forward-nya. Cek dulu pemahaman chain rule dengan",
    "turunan numerik (selisih-hingga) — alat debugging NN yang paling jujur.",
))

cells.append(code(
    "# Chain rule pada satu neuron: f(x) = (2x + 3)^2\n",
    "f = lambda x: (2 * x + 3) ** 2\n",
    "df_analitik = lambda x: 2 * (2 * x + 3) * 2        # d/dx (2x+3)^2 = 2(2x+3)*2\n",
    "\n",
    "def turunan_numerik(fn, x, h=1e-6):\n",
    "    \"\"\"Selisih-hingga tengah: (f(x+h) - f(x-h)) / 2h.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# for x in [-2, 0, 1.5]:\n",
    "#     print(x, df_analitik(x), turunan_numerik(f, x))\n",
))

cells.append(code(
    "# ✅ Cek latihan 3.1\n",
    "assert abs(turunan_numerik(f, 1.5) - df_analitik(1.5)) < 1e-4\n",
    "assert abs(turunan_numerik(f, -2.0) - df_analitik(-2.0)) < 1e-4\n",
    "assert abs(turunan_numerik(np.sin, 0.7) - np.cos(0.7)) < 1e-6, 'harus generic!'\n",
    "assert abs(turunan_numerik(np.exp, 1.1) - np.exp(1.1)) < 1e-5\n",
    "print('✅ Latihan 3.1 benar! Alat ini akan mem-bukti-kan backprop-mu di bawah.')\n",
))

cells.append(md(
    "### 🎯 Latihan 3.2 — Forward Pass MLP 2-Layer",
    "",
    "Sebelum backward, kuasai forward: `h = ReLU(X@W1 + b1)` lalu `p = sigmoid(h@W2 + b2)`.",
))

cells.append(code(
    "def mlp_forward(X, W1, b1, W2, b2):\n",
    "    \"\"\"Forward 2-layer. Return (h, p): hidden aktivasi & probabilitas output.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# h, p = mlp_forward(Xxor, hasil['W1'], hasil['b1'], hasil['W2'], hasil['b2'])\n",
    "# print(p.round(3))   # harus sama dengan hasil['probs']\n",
))

cells.append(code(
    "# ✅ Cek latihan 3.2\n",
    "h, p = mlp_forward(Xxor, hasil['W1'], hasil['b1'], hasil['W2'], hasil['b2'])\n",
    "assert h.shape == (4, 8) and p.shape == (4,)\n",
    "assert np.allclose(p, hasil['probs']), 'harus mereproduksi output model terlatih'\n",
    "assert np.all(h >= 0), 'ReLU output tak negatif'\n",
    "h2, p2 = mlp_forward(np.zeros((1, 2)), np.ones((2, 3)), np.zeros(3),\n",
    "                     np.array([[1.0], [-1.0], [0.5]]), np.zeros(1))\n",
    "assert abs(p2[0] - 0.5) < 1e-12, 'z=0 -> p=0.5'\n",
    "print('✅ Latihan 3.2 benar!')\n",
))

# ============================ BAGIAN 4 ============================
cells.append(md(
    "## Bagian 4 — Melatih MLP dari Nol pada Data Nyata",
    "",
    "Dua blob (kelas di dua sisi) + sedikit overlap — versi \"gampang\" untuk",
    "melihat kurva loss sehat. Loop training-nya persis pola XOR, hanya datanya lebih besar.",
))

cells.append(code(
    "rng = np.random.default_rng(3)\n",
    "n_per = 150\n",
    "blob0 = rng.normal(-1, 0.6, (n_per, 2))\n",
    "blob1 = rng.normal(+1, 0.6, (n_per, 2))\n",
    "Xb = np.vstack([blob0, blob1])\n",
    "yb = np.array([0] * n_per + [1] * n_per, dtype=float)\n",
    "\n",
    "def latih_mlp(X, y, n_hidden=8, lr=0.5, epochs=500, seed=0):\n",
    "    \"\"\"MLP 2->n_hidden->1, full-batch GD. Return (params, losses).\"\"\"\n",
    "    r = np.random.default_rng(seed)\n",
    "    W1 = r.normal(0, 1.0, (X.shape[1], n_hidden)); b1 = np.zeros(n_hidden)\n",
    "    W2 = r.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1)); b2 = np.zeros(1)\n",
    "    losses = []\n",
    "    m = len(y)\n",
    "    for ep in range(epochs):\n",
    "        h = np.maximum(X @ W1 + b1, 0)\n",
    "        o = sigmoid_stabil(h @ W2 + b2)\n",
    "        pc = np.clip(o.ravel(), 1e-12, 1 - 1e-12)\n",
    "        losses.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))\n",
    "        dz2 = (o - y.reshape(-1, 1)) / m\n",
    "        dW2 = h.T @ dz2; db2 = dz2.sum(0)\n",
    "        dz1 = (dz2 @ W2.T) * (h > 0)\n",
    "        dW1 = X.T @ dz1; db1 = dz1.sum(0)\n",
    "        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2\n",
    "    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, losses\n",
    "\n",
    "params, losses = latih_mlp(Xb, yb)\n",
    "h, p = mlp_forward(Xb, **params)\n",
    "print(f'accuracy train: {((p >= 0.5) == yb).mean():.3f}  loss: {losses[0]:.4f} -> {losses[-1]:.4f}')\n",
    "plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('BCE');\n",
    "plt.title('Kurva loss sehat: turun cepat lalu melandai'); plt.show()\n",
))

cells.append(md(
    "### 🎯 Latihan 4.1 — BCE dari Nol (dengan Guard)",
    "",
    "Loss yang tadi dipakai diam-diam — sekarang tulis sendiri, dengan guard",
    "clip supaya `p = 0` atau `p = 1` tidak menghasilkan `log(0)`.",
))

cells.append(code(
    "def bce_manual(y_true, p_pred, eps=1e-12):\n",
    "    \"\"\"BCE rata-rata: -mean(y log p + (1-y) log(1-p)), p di-clip [eps, 1-eps].\n",
    "    Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(bce_manual([1.0], [0.9]))     # ~0.105\n",
    "# print(bce_manual([1.0], [0.5]))     # ~0.693 (= ln 2)\n",
    "# print(bce_manual([1.0], [0.0]))     # ~27.6 — bukan inf, terima kasih clip\n",
))

cells.append(code(
    "# ✅ Cek latihan 4.1\n",
    "assert abs(bce_manual([1.0], [0.9]) - (-np.log(0.9))) < 1e-9\n",
    "assert abs(bce_manual([0.0], [0.1]) - (-np.log(0.9))) < 1e-9, 'simetris untuk y=0'\n",
    "assert abs(bce_manual([1.0], [0.5]) - np.log(2)) < 1e-9\n",
    "assert abs(bce_manual([1.0], [0.1]) - (-np.log(0.1))) < 1e-9, 'yakin pd diri salah = mahal'\n",
    "assert np.isfinite(bce_manual([1.0], [0.0])) and np.isfinite(bce_manual([0.0], [1.0])), 'guard clip'\n",
    "assert isinstance(bce_manual([1.0], [0.9]), float)\n",
    "print('✅ Latihan 4.1 benar!')\n",
))

# ============================ BAGIAN 5 ============================
cells.append(md(
    "## Bagian 5 — Softmax & Cross-Entropy Multikelas",
    "",
    "Klasifikasi 3 kelas: output = softmax (probabilitas per kelas), loss =",
    "cross-entropy `-log p[kelas_benar]`. Pola yang sama dipakai classifier",
    "intent, dan... output layer LLM (Bab 9).",
))

cells.append(code(
    "# Data 3 kelas sintetis\n",
    "rng = np.random.default_rng(9)\n",
    "pusat = np.array([[0., 0.], [4., 0.], [0., 4.]])\n",
    "X3 = np.vstack([r.normal(c, 0.7, (60, 2)) for c in pusat])\n",
    "y3 = np.repeat([0, 1, 2], 60)\n",
    "\n",
    "# Model linear multikelas: logits = X @ W (3 kolom = 3 kelas), softmax\n",
    "W3 = np.zeros((2, 3))\n",
    "logits = X3 @ W3\n",
    "P = softmax_stabil(logits)\n",
    "print('baris probabilitas (model belum dilatih, semua 1/3):')\n",
    "print(P[:2].round(3))\n",
    "\n",
    "# Cross-entropy\n",
    "def cross_entropy_manual(y_onehot, probs, eps=1e-12):\n",
    "    \"\"\"-mean(sum(y * log(probs))). Return float.\"\"\"\n",
    "    # TODO (untuk latihan 5.1)\n",
    "    raise NotImplementedError\n",
    "\n",
    "Y3 = np.eye(3)[y3]                          # one-hot\n",
    "ce = -np.mean(np.log(np.clip(P[np.arange(len(y3)), y3], 1e-12, 1)))\n",
    "print(f'CE model acak: {ce:.4f}  (= ln 3: tebakan seragam 3 kelas)')\n",
))

cells.append(md(
    "### 🎯 Latihan 5.1 — Cross-Entropy dari Nol",
))

cells.append(code(
    "def cross_entropy(y_onehot, probs, eps=1e-12):\n",
    "    \"\"\"-mean( Σ_k y[k] * log(p[k]) ). Return float. y_onehot: (n, k), probs: (n, k).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# Y = np.eye(3)[[0, 1, 2]]\n",
    "# P = np.full((3, 3), 1/3)\n",
    "# print(cross_entropy(Y, P))    # ln 3 ≈ 1.0986\n",
))

cells.append(code(
    "# ✅ Cek latihan 5.1\n",
    "Y = np.eye(3)[[0, 1, 2]]\n",
    "P_uniform = np.full((3, 3), 1 / 3)\n",
    "assert abs(cross_entropy(Y, P_uniform) - np.log(3)) < 1e-9, 'seragam = ln k'\n",
    "# prediksi yakin & BENAR (tiap baris cocok dengan kelas aslinya)\n",
    "P_perfect = np.array([[0.999, 0.0005, 0.0005],\n",
    "                      [0.0005, 0.999, 0.0005],\n",
    "                      [0.0005, 0.0005, 0.999]])\n",
    "assert cross_entropy(Y, P_perfect) < 0.01, 'prediksi yakin & benar = CE kecil'\n",
    "# prediksi yakin & SALAH (kelas asli hanya dapat 0.001)\n",
    "P_wrong = np.array([[0.001, 0.999, 0.0],\n",
    "                    [0.999, 0.001, 0.0],\n",
    "                    [0.001, 0.0, 0.999]])\n",
    "assert cross_entropy(Y, P_wrong) > 5, 'yakin & salah = CE MAHAL'\n",
    "assert isinstance(cross_entropy(Y, P_uniform), float)\n",
    "print('✅ Latihan 5.1 benar!')\n",
))

# ============================ BAGIAN 6 ============================
cells.append(md(
    "## Bagian 6 — Weight Init: Nol vs He",
    "",
    "Init semua bobot NOL terdengar rapi — dan fatal: semua neuron di layer yang",
    "sama mendapat gradien identik → update identik → tetap identik selamanya",
    "(**simetri tidak pecah**). Init He (`std = √(2/fan_in)`) memecah simetri dengan",
    "skala yang tepat untuk ReLU.",
))

cells.append(code(
    "def latih_mlp_init(X, y, init='he', n_hidden=8, lr=0.5, epochs=800, seed=0):\n",
    "    \"\"\"Sama seperti latih_mlp tapi init bisa 'he' atau 'nol'.\"\"\"\n",
    "    r = np.random.default_rng(seed)\n",
    "    if init == 'he':\n",
    "        W1 = r.normal(0, np.sqrt(2.0 / X.shape[1]), (X.shape[1], n_hidden))\n",
    "        W2 = r.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))\n",
    "    else:\n",
    "        W1 = np.zeros((X.shape[1], n_hidden)); W2 = np.zeros((n_hidden, 1))\n",
    "    b1 = np.zeros(n_hidden); b2 = np.zeros(1)\n",
    "    losses = []\n",
    "    m = len(y)\n",
    "    for ep in range(epochs):\n",
    "        h = np.maximum(X @ W1 + b1, 0)\n",
    "        o = sigmoid_stabil(h @ W2 + b2)\n",
    "        pc = np.clip(o.ravel(), 1e-12, 1 - 1e-12)\n",
    "        losses.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))\n",
    "        dz2 = (o - y.reshape(-1, 1)) / m\n",
    "        dW2 = h.T @ dz2; db2 = dz2.sum(0)\n",
    "        dz1 = (dz2 @ W2.T) * (h > 0)\n",
    "        dW1 = X.T @ dz1; db1 = dz1.sum(0)\n",
    "        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2\n",
    "    h = np.maximum(X @ W1 + b1, 0)\n",
    "    p = sigmoid_stabil(h @ W2 + b2).ravel()\n",
    "    return float(((p >= 0.5) == y).mean()), losses\n",
    "\n",
    "acc_he, loss_he = latih_mlp_init(Xb, yb, init='he')\n",
    "acc_nol, loss_nol = latih_mlp_init(Xb, yb, init='nol')\n",
    "print(f'init He  : acc={acc_he:.3f}  loss akhir={loss_he[-1]:.4f}')\n",
    "print(f'init nol : acc={acc_nol:.3f}  loss akhir={loss_nol[-1]:.4f}  <- MACET TOTAL')\n",
    "plt.plot(loss_he, label='init He')\n",
    "plt.plot(loss_nol, label='init nol')\n",
    "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.legend(); plt.title('Simetri membunuh learning')\n",
    "plt.show()\n",
))

cells.append(md(
    "### 🎯 Latihan 6.1 — He & Xavier",
))

cells.append(code(
    "def he_std(fan_in):\n",
    "    \"\"\"Std init He: sqrt(2 / fan_in).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def xavier_std(fan_in):\n",
    "    \"\"\"Std init Xavier (versi sederhana): sqrt(1 / fan_in).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
))

cells.append(code(
    "# ✅ Cek latihan 6.1\n",
    "assert abs(he_std(64) - np.sqrt(2 / 64)) < 1e-12\n",
    "assert abs(xavier_std(64) - np.sqrt(1 / 64)) < 1e-12\n",
    "assert he_std(10) > xavier_std(10), 'He lebih lebar (dibuat utk ReLU)'\n",
    "assert abs(he_std(256) - np.sqrt(2) / 16) < 1e-12\n",
    "print('✅ Latihan 6.1 benar!')\n",
))

# ============================ BAGIAN 7 ============================
cells.append(md(
    "## Bagian 7 — Learning Rate & Langkah SGD",
    "",
    "`w ← w − lr · grad`. Terlalu kecil = sipir; terlalu besar = melompati lembah.",
    "Lihat efeknya pada kurva loss MLP (lr 0.05 / 0.5 / 5.0).",
))

cells.append(code(
    "for lr in [0.05, 0.5, 5.0]:\n",
    "    _, ls = latih_mlp(Xb, yb, lr=lr, epochs=400)\n",
    "    plt.plot(ls, label=f'lr={lr}')\n",
    "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.yscale('log')\n",
    "plt.title('Efek learning rate (skala log)'); plt.legend(); plt.show()\n",
))

cells.append(md(
    "### 🎯 Latihan 7.1 — Satu Langkah SGD",
))

cells.append(code(
    "def langkah_sgd(w, grad, lr):\n",
    "    \"\"\"Return w baru setelah satu update SGD.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
))

cells.append(code(
    "# ✅ Cek latihan 7.1\n",
    "w0 = np.array([1.0, -2.0])\n",
    "g  = np.array([0.5, -1.0])\n",
    "assert np.allclose(langkah_sgd(w0, g, 0.1), [0.95, -1.9])\n",
    "assert np.allclose(langkah_sgd(5.0, 2.0, 0.5), 4.0)\n",
    "assert np.allclose(langkah_sgd(w0, np.zeros(2), 1.0), w0), 'grad nol -> diam'\n",
    "print('✅ Latihan 7.1 benar!')\n",
))

# ============================ BAGIAN 8 ============================
cells.append(md(
    "## Bagian 8 — Mini-Batch vs Full-Batch",
    "",
    "Full-batch: gradien bersih tapi 1 update per epoch. Mini-batch: gradien",
    "\"berisik\" tapi BANYAK update per epoch — biasanya lebih cepat konvergen",
    "per epoch, dan noise-nya membantu keluar dari titik datar.",
))

cells.append(code(
    "def iter_batch(X, y, batch_size, seed=0):\n",
    "    \"\"\"Yield (Xb, yb) dengan urutan acak per epoch.\"\"\"\n",
    "    r = np.random.default_rng(seed)\n",
    "    order = r.permutation(len(y))\n",
    "    for s in range(0, len(y), batch_size):\n",
    "        idx = order[s:s + batch_size]\n",
    "        yield X[idx], y[idx]\n",
    "\n",
    "def latih_mlp_sgd(X, y, n_hidden=8, lr=0.1, epochs=60, batch_size=16, seed=0):\n",
    "    r = np.random.default_rng(seed)\n",
    "    W1 = r.normal(0, np.sqrt(2.0 / X.shape[1]), (X.shape[1], n_hidden))\n",
    "    W2 = r.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))\n",
    "    b1 = np.zeros(n_hidden); b2 = np.zeros(1)\n",
    "    losses = []\n",
    "    for ep in range(epochs):\n",
    "        for Xb_, yb_ in iter_batch(X, y, batch_size, seed=seed * 1000 + ep):\n",
    "            m = len(yb_)\n",
    "            h = np.maximum(Xb_ @ W1 + b1, 0)\n",
    "            o = sigmoid_stabil(h @ W2 + b2)\n",
    "            dz2 = (o - yb_.reshape(-1, 1)) / m\n",
    "            dW2 = h.T @ dz2; db2 = dz2.sum(0)\n",
    "            dz1 = (dz2 @ W2.T) * (h > 0)\n",
    "            dW1 = Xb_.T @ dz1; db1 = dz1.sum(0)\n",
    "            W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2\n",
    "        h = np.maximum(X @ W1 + b1, 0)\n",
    "        pc = np.clip(sigmoid_stabil(h @ W2 + b2).ravel(), 1e-12, 1 - 1e-12)\n",
    "        losses.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))\n",
    "    return losses\n",
    "\n",
    "loss_full = latih_mlp_sgd(Xb, yb, lr=0.1, epochs=60, batch_size=len(yb))\n",
    "loss_mini = latih_mlp_sgd(Xb, yb, lr=0.1, epochs=60, batch_size=16)\n",
    "plt.plot(loss_full, label='full-batch (300 update/epoch? tidak — 1 update/epoch)')\n",
    "plt.plot(loss_mini, label='mini-batch 16 (19 update/epoch)')\n",
    "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.title('Mini-batch biasanya turun lebih cepat')\n",
    "plt.legend(); plt.show()\n",
    "print(f'loss akhir: full={loss_full[-1]:.4f}  mini={loss_mini[-1]:.4f}')\n",
))

cells.append(md(
    "### 🎯 Latihan 8.1 — Pemecahan Batch",
))

cells.append(code(
    "def split_batch(n, batch_size):\n",
    "    \"\"\"Return list ukuran batch untuk n sampel.\n",
    "    split_batch(7, 3) -> [3, 3, 1] ; split_batch(0, 3) -> [].\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
))

cells.append(code(
    "# ✅ Cek latihan 8.1\n",
    "assert split_batch(7, 3) == [3, 3, 1]\n",
    "assert split_batch(9, 3) == [3, 3, 3]\n",
    "assert split_batch(2, 5) == [2]\n",
    "assert split_batch(0, 3) == []\n",
    "assert sum(split_batch(123, 32)) == 123\n",
    "print('✅ Latihan 8.1 benar!')\n",
))

# ============================ BAGIAN 9 ============================
cells.append(md(
    "## Bagian 9 — Early Stopping",
    "",
    "Pantau **val** loss; berhenti kalau tak membaik selama `patience` epoch.",
    "Ini regularisasi gratis — dan versi Keras-nya: `EarlyStopping(monitor='val_loss', patience=…)`.",
))

cells.append(code(
    "# Split train/val dari blob\n",
    "rng = np.random.default_rng(21)\n",
    "idx = rng.permutation(len(yb))\n",
    "n_va = 80\n",
    "iva, itr = idx[:n_va], idx[n_va:]\n",
    "Xtr, ytr = Xb[itr], yb[itr]\n",
    "Xva, yva = Xb[iva], yb[iva]\n",
    "\n",
    "def latih_dengan_val(Xtr, ytr, Xva, yva, n_hidden=8, lr=0.2, epochs=600, seed=0):\n",
    "    r = np.random.default_rng(seed)\n",
    "    W1 = r.normal(0, np.sqrt(2.0 / 2), (2, n_hidden))\n",
    "    W2 = r.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))\n",
    "    b1 = np.zeros(n_hidden); b2 = np.zeros(1)\n",
    "    tr_hist, va_hist = [], []\n",
    "    m = len(ytr)\n",
    "    for ep in range(epochs):\n",
    "        h = np.maximum(Xtr @ W1 + b1, 0)\n",
    "        o = sigmoid_stabil(h @ W2 + b2)\n",
    "        dz2 = (o - ytr.reshape(-1, 1)) / m\n",
    "        dW2 = h.T @ dz2; db2 = dz2.sum(0)\n",
    "        dz1 = (dz2 @ W2.T) * (h > 0)\n",
    "        dW1 = Xtr.T @ dz1; db1 = dz1.sum(0)\n",
    "        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2\n",
    "        for XX, yy, hist in ((Xtr, ytr, tr_hist), (Xva, yva, va_hist)):\n",
    "            hh = np.maximum(XX @ W1 + b1, 0)\n",
    "            pp = np.clip(sigmoid_stabil(hh @ W2 + b2).ravel(), 1e-12, 1 - 1e-12)\n",
    "            hist.append(float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))))\n",
    "    return tr_hist, va_hist\n",
    "\n",
    "tr_hist, va_hist = latih_dengan_val(Xtr, ytr, Xva, yva)\n",
    "plt.plot(tr_hist, label='train')\n",
    "plt.plot(va_hist, label='val')\n",
    "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.legend()\n",
    "plt.title('Val loss berhenti membaik = momen early stopping'); plt.show()\n",
    "best_ep = int(np.argmin(va_hist))\n",
    "print(f'val loss terbaik di epoch {best_ep} (dari {len(va_hist)})')\n",
    "print(f'loss terbaik {min(va_hist):.4f} vs akhir {va_hist[-1]:.4f}')\n",
))

cells.append(md(
    "### 🎯 Latihan 9.1 — Logika Early Stopping",
    "",
    "Pisahkan LOGIKA dari training: fungsi murni yang bisa di-test tanpa melatih model.",
))

cells.append(code(
    "def cek_early_stop(val_loss, patience):\n",
    "    \"\"\"Return (best_epoch, stop_epoch).\n",
    "    best_epoch = index val_loss TERENDAH (yang pertama kalau seri).\n",
    "    stop_epoch = epoch pertama di mana 'tidak membaik' beruntun mencapai\n",
    "                 patience; kalau tidak pernah -> len(val_loss) - 1.\n",
    "    'Membaik' = val_loss[i] < best_so_far - 1e-6 (best_so_far mulai inf).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(cek_early_stop([1.0, 0.8, 0.7, 0.71, 0.72, 0.73], patience=2))\n",
))

cells.append(code(
    "# ✅ Cek latihan 9.1\n",
    "assert cek_early_stop([1.0, 0.8, 0.7, 0.71, 0.72, 0.73], 2) == (2, 4)\n",
    "assert cek_early_stop([0.5, 0.4, 0.3, 0.2], 3) == (3, 3), 'selalu membaik -> tak berhenti'\n",
    "assert cek_early_stop([0.3, 0.3, 0.3, 0.3], 2) == (0, 2), 'stagnan = tidak membaik'\n",
    "assert cek_early_stop([0.9, 0.5, 0.6, 0.4, 0.55, 0.35], 1) == (1, 2), 'patience=1: stop di pelanggaran pertama'\n",
    "assert cek_early_stop([0.7], 5) == (0, 0)\n",
    "print('✅ Latihan 9.1 benar!')\n",
))

# ============================ BAGIAN 10 ============================
cells.append(md(
    "## Bagian 10 — Skip Connection & Menuju TensorFlow",
    "",
    "MLP dalam tanpa skip connection: gradien menempuh banyak lapisan multiply →",
    "makin dalam makin sulit (*degradation*). Dengan **residual** `a ← a + F(a)`,",
    "gradien punya \"jalan tol\" — pola yang dipakai ResNet dan SEMUA Transformer.",
))

cells.append(code(
    "def latih_dalam(X, y, n_layers=6, n_hidden=16, lr=0.05, epochs=400, skip=False, seed=0):\n",
    "    \"\"\"MLP dalam full-batch. skip=True -> a = a + Dense(a) tiap hidden layer.\"\"\"\n",
    "    r = np.random.default_rng(seed)\n",
    "    L = n_layers\n",
    "    Ws = [r.normal(0, np.sqrt(2.0 / (X.shape[1] if i == 0 else n_hidden)),\n",
    "                   ((X.shape[1] if i == 0 else n_hidden), n_hidden)) for i in range(L)]\n",
    "    bs = [np.zeros(n_hidden) for _ in range(L)]\n",
    "    W_out = r.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1)); b_out = np.zeros(1)\n",
    "    losses = []\n",
    "    m = len(y)\n",
    "    for ep in range(epochs):\n",
    "        # forward\n",
    "        acts = [X]\n",
    "        a = X\n",
    "        for i in range(L):\n",
    "            z = a @ Ws[i] + bs[i]\n",
    "            h = np.maximum(z, 0)\n",
    "            a = h + a if (skip and i > 0) else h      # residual (dim sama)\n",
    "            acts.append(a)\n",
    "        o = sigmoid_stabil(a @ W_out + b_out)\n",
    "        pc = np.clip(o.ravel(), 1e-12, 1 - 1e-12)\n",
    "        losses.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))\n",
    "        # backward\n",
    "        dz = (o - y.reshape(-1, 1)) / m\n",
    "        dW_out = a.T @ dz; db_out = dz.sum(0)\n",
    "        da = dz @ W_out.T\n",
    "        for i in range(L - 1, -1, -1):\n",
    "            a_prev = acts[i]\n",
    "            h = acts[i + 1] - (a_prev if (skip and i > 0) else 0)\n",
    "            dz_i = da * (h > 0)\n",
    "            dWs = a_prev.T @ dz_i; dbs = dz_i.sum(0)\n",
    "            da = dz_i @ Ws[i].T\n",
    "            if skip and i > 0:\n",
    "                da = da + da  # residual: gradien lewat 2 jalur (a + F(a))\n",
    "            Ws[i] -= lr * dWs; bs[i] -= lr * dbs\n",
    "    return losses\n",
    "\n",
    "loss_biasa = latih_dalam(Xb, yb, n_layers=6, skip=False)\n",
    "loss_skip  = latih_dalam(Xb, yb, n_layers=6, skip=True)\n",
    "plt.plot(loss_biasa, label='6 layer biasa')\n",
    "plt.plot(loss_skip, label='6 layer + skip')\n",
    "plt.xlabel('epoch'); plt.ylabel('BCE'); plt.legend()\n",
    "plt.title('Skip connection membantu gradien menempuh jarak jauh'); plt.show()\n",
    "print(f'loss akhir: biasa={loss_biasa[-1]:.4f}  skip={loss_skip[-1]:.4f}')\n",
))

cells.append(md(
    "**Observasi:** pada jaringan dalam, jalur residual membuat loss turun lebih",
    "lancar. Persis alasan ResNet bisa 100+ layer — dan Transformer (Bab 9) punya",
    "`x + attention(x)` di tiap blok.",
))

cells.append(code(
    "# Sekarang hal yang sama dalam Keras — 10 baris (jalankan kalau TF terpasang)\n",
    "try:\n",
    "    import tensorflow as tf\n",
    "    from tensorflow import keras\n",
    "    from tensorflow.keras import layers\n",
    "\n",
    "    model = keras.Sequential([\n",
    "        layers.Input(shape=(2,)),\n",
    "        layers.Dense(16, activation='relu'),\n",
    "        layers.Dense(8, activation='relu'),\n",
    "        layers.Dense(1, activation='sigmoid'),\n",
    "    ])\n",
    "    model.compile(optimizer=keras.optimizers.Adam(0.01),\n",
    "                  loss='binary_crossentropy', metrics=['accuracy'])\n",
    "    hist = model.fit(Xb, yb, epochs=50, batch_size=32, verbose=0)\n",
    "    print(f'Keras: acc akhir = {hist.history[\"accuracy\"][-1]:.3f}')\n",
    "    print('-> Loop manual-mu di Bagian 4 melakukan HAL YANG SAMA (SGD vs Adam).')\n",
    "except ImportError:\n",
    "    print('TensorFlow belum terpasang — install dengan: pip install tensorflow')\n",
    "    print('Konsepnya sama persis dengan yang sudah kamu bangun dari nol di lab ini.')\n",
))

# ============================ PENUTUP ============================
cells.append(md(
    "## 🏁 Penutup Lab",
    "",
    "**Refleksi:**",
    "1. Jelaskan ke teman: kenapa XOR tidak mungkin diselesaikan model linear,",
    "   dan apa yang hidden layer lakukan supaya mungkin?",
    "2. Backprop = chain rule + cache. Bagian mana yang masih samar — dz, dW, atau",
    "   urutan backward? Ulangi Bagian 3–4 sampai lancar MENGALIRKAN gradien dengan tangan.",
    "3. Daftar 3 hal di lab ini yang otomatis di Keras (init, batching, early stopping)",
    "   — itu yang kamu dapat gratis nanti.",
    "",
    "**Lanjut:** `02_kuis_deep_learning.ipynb` → `project-starter-numpy-nn/` → `cheatsheet-deep-learning.md`.",
))

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
