"""Generate 01_lab_ml_produksi.ipynb for Bab 8."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_lab_ml_produksi.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


cells = []

# ============================ HEADER ============================
cells.append(md(
    "# 🧪 Lab 8 — Praktikum ML Produksi (dari Nol)",
    "",
    "> Pendamping materi Bab 8. Prinsip sama dengan Lab 4–7: **semua dibangun dari nol",
    "> dengan numpy** — reproduksibilitas, persistensi model, metrik & skill score,",
    "> threshold tuning, kalibrasi probability, sampai GAN 1D sederhana — supaya",
    "> `MLflow`, `Pipeline`, `classification_report`, dan konsep adversarial tidak pernah",
    "> jadi kotak hitam. Dataset regresi sintetis dengan resep yang KITA TENTUKAN,",
    "> seed terkunci, jadi kebenaran model bisa diukur persis.",
    "",
    "| Bagian | Topik | Waktu (menit) |",
    "|---|---|---|",
    "| 1 | Reproduksibilitas: seed, split tiga arah, hash deterministik | 15 |",
    "| 2 | Persistensi model & round-trip test | 15 |",
    "| 3 | RMSE, MAE, R² — dan baseline HAM yang wajib dikalahkan | 20 |",
    "| 4 | Skill score: relatif terhadap baseline, bukan angka absolut | 15 |",
    "| 5 | Confusion matrix, precision/recall/F1 dari nol | 20 |",
    "| 6 | Threshold tuning di CALIB — evaluasi di TEST | 20 |",
    "| 7 | Kalibrasi probability: bins, ECE, reliability curve | 20 |",
    "| 8 | GAN 1D: Generator vs Discriminator dari nol | 30 |",
    "| 9 | Ringkasan: angka-angka yang kamu pegang | 10 |",
    "",
    "Setiap latihan punya cell **✅ Cek** — jalankan, kalau ✅ berarti pemahamanmu benar.",
))

# ============================ BAGIAN 1 ============================
cells.append(md(
    "## Bagian 1 — Reproduksibilitas: Split Tiga Arah & Hash Deterministik",
    "",
    "Model 'jalan di laptop' ≠ model siap produksi. Fondasi pertama: **hasil yang sama",
    "persis** saat dijalankan ulang. Resep data (seed 8, dikunci di seluruh bab):",
    "",
    "```",
    "y = 2.0·x1 − 1.5·x2 + 1.0 + N(0, 1.2²)     n = 800",
    "split: train 80% (640) | calib 10% (80) | test 10% (80)",
    "```",
    "",
    "Tiga split, bukan dua: **calib** dipakai untuk memutuskan threshold (Bagian 6).",
    "Kalau threshold dituning di test, test berhenti jadi pengukur jujur.",
))

cells.append(code(
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "rng = np.random.default_rng(8)   # seed terkunci — seluruh lab reproducible\n",
    "\n",
    "N = 800\n",
    "TRUE_W = np.array([2.0, -1.5])\n",
    "TRUE_B = 1.0\n",
    "SIGMA = 1.2\n",
    "\n",
    "X = rng.normal(0.0, 1.5, size=(N, 2))\n",
    "noise = rng.normal(0.0, SIGMA, N)\n",
    "y = X @ TRUE_W + TRUE_B + noise\n",
    "\n",
    "cut1 = int(N * 0.8)   # 640\n",
    "cut2 = int(N * 0.9)   # 720\n",
    "Xtr, ytr = X[:cut1], y[:cut1]\n",
    "Xca, yca = X[cut1:cut2], y[cut1:cut2]\n",
    "Xte, yte = X[cut2:], y[cut2:]\n",
    "print(f'n={N}  train={len(Xtr)}  calib={len(Xca)}  test={len(Xte)}')\n",
    "print(f'correlation fitur: x1={np.corrcoef(X[:, 0], y)[0, 1]:+.3f}  x2={np.corrcoef(X[:, 1], y)[0, 1]:+.3f}')\n",
))

cells.append(code(
    "def split_tiga_arah(X, y, r1=0.8, r2=0.9):\n",
    "    \"\"\"Split BERURUTAN (indeks kecil = awal data) menjadi train/calib/test.\n",
    "    Return (Xtr, ytr, Xca, yca, Xte, yte).\n",
    "    r1 = proporsi batas train, r2 = proporsi batas calib (int(n*r)).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# Xa, ya, Xb, yb, Xc, yc = split_tiga_arah(X, y)\n",
    "# print(Xa.shape, Xb.shape, Xc.shape)\n",
))

cells.append(code(
    "# ✅ Cek latihan 1.1\n",
    "Xa, ya, Xb, yb, Xc, yc = split_tiga_arah(X, y)\n",
    "assert len(Xa) == 640 and len(Xb) == 80 and len(Xc) == 80\n",
    "assert np.array_equal(Xa, X[:640]) and np.array_equal(Xb, X[640:720])\n",
    "assert np.array_equal(Xc, X[720:]) and np.array_equal(yc, y[720:])\n",
    "# tak ada baris yang bocor antar split\n",
    "assert not set(map(tuple, Xa)) & set(map(tuple, Xb))\n",
    "print('✅ Latihan 1.1 benar! train 640 | calib 80 | test 80, urutan terjaga.')\n",
    "print('   Calib ada supaya keputusan (threshold) TIDAK pernah menyentuh test.')\n",
))

cells.append(md(
    "### Hash deterministik: bukti artefak identik\n",
    "",
    "Dua file model yang diklaim 'sama' bisa saja beda byte. Produksi menguji klaim itu",
    "dengan **hash** — bukan dengan 'kayaknya sama'.",
))

cells.append(code(
    "def sha256_bytes(b):\n",
    "    \"\"\"SHA-256 dari bytes. Return string hex.\n",
    "    Petunjuk: hashlib.sha256(b).hexdigest().\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(sha256_bytes(b'prod'))\n",
    "# print(sha256_bytes(b'prod'))  # harus sama persis\n",
))

cells.append(code(
    "# ✅ Cek latihan 1.2\n",
    "import hashlib\n",
    "assert sha256_bytes(b'prod') == hashlib.sha256(b'prod').hexdigest()\n",
    "assert sha256_bytes(b'a') != sha256_bytes(b'b'), 'input beda 1 byte, hash beda total'\n",
    "assert sha256_bytes(b'prod') == sha256_bytes(b'prod'), 'input sama, hash HARUS sama'\n",
    "print('✅ Latihan 1.2 benar! hash deterministik: input sama -> output sama.')\n",
    "print('   Di produksi: hash artifact + hash data = bukti reproducibility yang bisa diaudit.')\n",
))

# ============================ BAGIAN 2 ============================
cells.append(md(
    "## Bagian 2 — Persistensi Model: Save → Load → Prediksi Sama",
    "",
    "Training di notebook A, serving di proses B. Satu-satunya jembatan yang sah:",
    "**artefak di disk** yang round-trip-nya terbukti identik. Formatnya `np.savez` —",
    "kamu yang menulis, kamu yang membaca, tidak ada kejutan.",
))

cells.append(code(
    "def fit_ols(Xa, ya):\n",
    "    \"\"\"OLS dengan kolom bias: [x1, x2, 1]. Return (w, b).\n",
    "    Petunjuk: lstsq di matriks kolom [x, 1]; w = coef[:-1], b = coef[-1].\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def predict(w, b, Xa):\n",
    "    \"\"\"Prediksi = X @ w + b. Return array (n,).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# w, b = fit_ols(Xtr, ytr)\n",
    "# print(w, b)   # target: [2.0, -1.5] dan 1.0\n",
))

cells.append(code(
    "# ✅ Cek latihan 2.1\n",
    "w_ols, b_ols = fit_ols(Xtr, ytr)\n",
    "assert w_ols.shape == (2,) and np.allclose(w_ols, TRUE_W, atol=0.15), f'{w_ols}'\n",
    "assert abs(b_ols - TRUE_B) < 0.2, f'{b_ols}'\n",
    "p1, p2 = predict(w_ols, b_ols, Xte[:3]), Xte[:3] @ TRUE_W + TRUE_B\n",
    "assert np.allclose(p1, p2, atol=1.5), 'prediksi harus dekat resep asli (sigma=1.2)'\n",
    "print(f'✅ Latihan 2.1 benar! OLS memulihkan resep: w={np.round(w_ols, 3).tolist()}, b={b_ols:.3f}')\n",
    "print('   Model sederhana yang benar > model kompleks yang tak bisa dijelaskan.')\n",
))

cells.append(code(
    "def save_model(path, w, b):\n",
    "    \"\"\"Simpan (w, b) ke file .npz secara DETERMINISTIK (hasil byte selalu sama).\"\"\"\n",
    "    # TODO: np.savez(path, w=np.asarray(w, dtype=float), b=np.array(float(b)))\n",
    "    raise NotImplementedError\n",
    "\n",
    "def load_model(path):\n",
    "    \"\"\"Baca kembali (w, b) dari file .npz. Return (w, b).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# save_model('/tmp/m.npz', w_ols, b_ols)\n",
    "# w2, b2 = load_model('/tmp/m.npz')\n",
    "# print(w2, b2)\n",
))

cells.append(code(
    "# ✅ Cek latihan 2.2 — round-trip penuh\n",
    "import tempfile, os\n",
    "tmpdir = tempfile.mkdtemp()\n",
    "p1 = os.path.join(tmpdir, 'model_a.npz')\n",
    "p2 = os.path.join(tmpdir, 'model_b.npz')\n",
    "save_model(p1, w_ols, b_ols)\n",
    "save_model(p2, w_ols, b_ols)   # simpan DUA KALI dari objek yang sama\n",
    "w2, b2 = load_model(p1)\n",
    "assert np.array_equal(w2, w_ols) and b2 == b_ols\n",
    "assert np.array_equal(predict(w2, b2, Xte), predict(w_ols, b_ols, Xte)), \\\n",
    "    'prediksi sebelum & sesudah round-trip harus identik bit-per-bit'\n",
    "print('✅ Latihan 2.2 benar! save -> load -> prediksi identik.')\n",
    "print('   Ini \"round-trip test\" wajib sebelum artefak dipakai di serving.')\n",
))

cells.append(md(
    "**Pertanyaan pemahaman:** kenapa round-trip test wajib, padahal `save_model`-nya",
    "benar? (hint: apa yang bisa berubah antara proses training dan proses serving —",
    "versi library, dtype, urutan kolom?)",
))

# ============================ BAGIAN 3 ============================
cells.append(md(
    "## Bagian 3 — Metrik Regresi & Baseline HAM\n",
    "",
    "Tiga metrik yang WAJIB bisa dihitung dari nol:\n",
    "",
    "```",
    "RMSE = sqrt(mean((y − ŷ)²))     ← menghukum error besar",
    "MAE  = mean(|y − ŷ|)            ← satuan asli, robust",
    "R²   = 1 − SS_res/SS_tot        ← vs baseline rata-rata (bisa negatif!)",
    "```",
    "",
    "Baseline: **HAM** = prediksi konstan `mean(train)`. Model apa pun yang kalah",
    "dari HAM belum menambah apa-apa.",
))

cells.append(code(
    "def rmse(y_true, y_pred):\n",
    "    \"\"\"Root Mean Squared Error. Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def mae(y_true, y_pred):\n",
    "    \"\"\"Mean Absolute Error. Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def r2(y_true, y_pred):\n",
    "    \"\"\"Koefisien determinasi R². Return float. Boleh negatif!\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(rmse([1., 2.], [2., 4.]))    # 1.5\n",
    "# print(r2([1., 2., 3.], [1., 2., 3.]))  # 1.0\n",
))

cells.append(code(
    "# ✅ Cek latihan 3.1\n",
    "assert mae(np.array([1., 2.]), np.array([2., 4.])) == 1.5                 # mean([1, 2])\n",
    "assert abs(rmse(np.array([1., 2.]), np.array([2., 4.])) - np.sqrt(2.5)) < 1e-12  # mean([1, 4])\n",
    "assert abs(rmse(np.array([0., 0.]), np.array([3., 4.])) - np.sqrt(12.5)) < 1e-12\n",
    "assert abs(mae(np.array([0., 0.]), np.array([3., 4.])) - 3.5) < 1e-9     # di sini RMSE > MAE\n",
    "assert r2(np.array([1., 2., 3.]), np.array([1., 2., 3.])) == 1.0\n",
    "assert r2(np.array([1., 2., 3.]), np.array([3., 2., 1.])) < 0.0, 'lebih buruk dari mean -> negatif'\n",
    "print('✅ Latihan 3.1 benar! RMSE >= MAE selalu (Cauchy-Schwarz) — coba error (1, 9):')\n",
    "e = np.array([1., 9.]); p = np.zeros(2)\n",
    "print(f'   error (1,9): RMSE={rmse(e, p):.2f} vs MAE={mae(e, p):.2f} — RMSE menghukum outlier')\n",
))

cells.append(code(
    "def fit_ham(ya):\n",
    "    \"\"\"Baseline HAM: mean(ya). Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def predict_ham(mean_y, Xa):\n",
    "    \"\"\"Prediksi konstan mean_y untuk semua baris. Return array (n,).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# mu = fit_ham(ytr)\n",
    "# print(predict_ham(mu, Xte)[:5])\n",
))

cells.append(code(
    "# ✅ Cek latihan 3.2\n",
    "mu = fit_ham(ytr)\n",
    "assert abs(mu - float(ytr.mean())) < 1e-12\n",
    "assert predict_ham(mu, Xte).shape == (len(Xte),)\n",
    "\n",
    "rmse_ols = rmse(yte, predict(w_ols, b_ols, Xte))\n",
    "rmse_ham = rmse(yte, predict_ham(mu, Xte))\n",
    "r2_ols = r2(yte, predict(w_ols, b_ols, Xte))\n",
    "print(f'RMSE OLS = {rmse_ols:.4f}   (sigma noise 1.2 — OLS nyaris sempurna)')\n",
    "print(f'RMSE HAM = {rmse_ham:.4f}   (baseline: hanya mean train)')\n",
    "print(f'R²   OLS = {r2_ols:.4f}')\n",
    "assert 1.0 < rmse_ols < 1.6, f'RMSE OLS harus ~1.25 (noise sigma=1.2): {rmse_ols}'\n",
    "assert 4.0 < rmse_ham < 5.8, f'HAM tidak melihat fitur: {rmse_ham}'\n",
    "assert 0.9 < r2_ols < 1.0\n",
    "assert rmse_ols < rmse_ham, 'OLS harus mengalahkan HAM — kalau tidak, ada bug'\n",
    "print('✅ Latihan 3.2 benar! RMSE OLS ≈ sigma noise: model menangkap SEMUA sinyal.')\n",
))

# ============================ BAGIAN 4 ============================
cells.append(md(
    "## Bagian 4 — Skill Score: \"Bagus Dibanding Apa?\"\n",
    "",
    "```",
    "skill = 1 − RMSE_model / RMSE_baseline",
    "skill = 1.0  → sempurna;  0 → sama dengan baseline;  negatif → LEBIH BURUK dari baseline",
    "```",
    "",
    "RMSE 1.25 terdengar bagus sampai kamu tahu baseline-nya RMSE 0.9 — skill-nya",
    "**negatif**. Angka absolut tidak berarti apa-apa tanpa pembanding. Untuk melihat",
    "ini, kita buat model \"sengaja salah\": fit HANYA fitur kedua dengan koefisien",
    "**dibalik** — simulasi eksperimen yang salah arah tapi tetap dilaporkan.",
))

cells.append(code(
    "def fit_neg(Xa, ya):\n",
    "    \"\"\"Model 'negatif': fit hanya fitur kedua, koefisien DIBALIK tanda (overfit\n",
    "    eksperimen yang salah arah). Return (w berbentuk (1,), b).\n",
    "    Petunjuk: OLS pada [-x2, 1]; kembalikan w = [-coef[0]], b = coef[1].\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def predict_neg(w, b, Xa):\n",
    "    \"\"\"Prediksi = -x2·w + b (w sudah dibalik di fit; arah fitur tetap x2).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# w_n, b_n = fit_neg(Xtr, ytr)\n",
    "# print(w_n, b_n)\n",
))

cells.append(code(
    "def skill_score(rmse_model, rmse_ref):\n",
    "    \"\"\"Skill score relatif baseline: 1 - rmse_model / rmse_ref. Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(skill_score(1.0, 2.0))   # 0.5\n",
    "# print(skill_score(3.0, 2.0))   # negatif!\n",
))

cells.append(code(
    "# ✅ Cek latihan 4.1\n",
    "assert abs(skill_score(1.0, 2.0) - 0.5) < 1e-12\n",
    "assert abs(skill_score(2.0, 2.0) - 0.0) < 1e-12\n",
    "assert skill_score(3.0, 2.0) < 0, 'lebih buruk dari baseline -> negatif'\n",
    "assert skill_score(0.0, 2.0) == 1.0, 'sempurna'\n",
    "\n",
    "w_neg, b_neg = fit_neg(Xtr, ytr)\n",
    "rmse_neg = rmse(yte, predict_neg(w_neg, b_neg, Xte))\n",
    "skill_ols = skill_score(rmse_ols, rmse_ham)\n",
    "skill_neg = skill_score(rmse_neg, rmse_ham)\n",
    "print(f'RMSE neg = {rmse_neg:.4f}  (lebih buruk dari HAM {rmse_ham:.4f}!)')\n",
    "print(f'skill OLS = {skill_ols:.4f}   skill neg = {skill_neg:.4f}')\n",
    "assert 0.65 < skill_ols < 0.85, f'skill OLS ~0.74: {skill_ols}'\n",
    "assert rmse_neg > rmse_ham and skill_neg < 0, 'model negatif harus kalah dari HAM'\n",
    "print('✅ Latihan 4.1 benar! Skill negatif = model TOLAK, seberapa pun angka RMSE-nya.')\n",
    "print('   Di sini koefisien x2 dibalik: korelasi x2,y = -0.59 tapi dipakai sebagai +.')\n",
))

cells.append(md(
    "**Pertanyaan pemahaman:** model dengan RMSE 6.68 vs baseline 4.83 — kenapa 'model",
    "dilatih dengan ML' tidak otomatis lebih baik dari 'ambil rata-rata'? Kapan tim harus",
    "TOLAK model dan pakai baseline?",
))

# ============================ BAGIAN 5 ============================
cells.append(md(
    "## Bagian 5 — Confusion Matrix, Precision, Recall, F1 dari Nol\n",
    "",
    "Untuk keputusan (klasifikasi), RMSE tidak lagi cukup. Kita ubah masalah regresi jadi",
    "keputusan: **label = 1 jika y > median(train)**. Prediksi skor = ŷ model.\n",
    "",
    "```",
    "                aktual 1     aktual 0",
    "pred ≥ thr   |    TP     |     FP    |   precision = TP/(TP+FP)",
    "pred < thr   |    FN     |     TN    |   recall    = TP/(TP+FN)",
    "                                      F1 = 2PR/(P+R)",
    "```",
    "",
    "Prevalence (proporsi label 1) ~0.5 di sini, jadi accuracy menipu kurang jelas —",
    "tapi coba bayangkan prevalence 1%: model 'selalu negatif' akurasinya 99% dan",
    "useless-nya 100%. Precision/recall tak bisa dibohongi begini.",
))

cells.append(code(
    "def confusion(y_true01, y_score, thr):\n",
    "    \"\"\"Confusion matrix dari skor kontinu + threshold.\n",
    "    y_true01: array 0/1. y_score: array kontinu. Prediksi positif jika y_score >= thr.\n",
    "    Return dict {'tp', 'fp', 'fn', 'tn'} berisi int.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(confusion([1, 0], [0.9, 0.1], 0.5))\n",
))

cells.append(code(
    "# ✅ Cek latihan 5.1\n",
    "cm_tes = confusion(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.3, 0.1]), 0.5)\n",
    "assert cm_tes == {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1}, f'{cm_tes}'\n",
    "cm_edge = confusion(np.array([0]), np.array([0.5]), 0.5)   # tepat di threshold = positif\n",
    "assert cm_edge == {'tp': 0, 'fp': 1, 'fn': 0, 'tn': 0}\n",
    "med = float(np.median(ytr))\n",
    "yte01 = (yte > med).astype(int)\n",
    "yca01 = (yca > med).astype(int)\n",
    "print('✅ Latihan 5.1 benar! Aturan >= thr = positif konsisten.')\n",
    "print(f'   median(train) = {med:.4f}  prevalence test = {yte01.mean():.3f}')\n",
))

cells.append(code(
    "def prf(cm):\n",
    "    \"\"\"Precision, recall, F1 dari confusion matrix. Return dict.\n",
    "    Konvensi aman: pembagi nol -> metrik 0.0 (bukan error).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(prf({'tp': 3, 'fp': 1, 'fn': 2, 'tn': 4}))\n",
))

cells.append(code(
    "# ✅ Cek latihan 5.2\n",
    "m = prf({'tp': 3, 'fp': 1, 'fn': 2, 'tn': 4})\n",
    "assert abs(m['precision'] - 0.75) < 1e-9\n",
    "assert abs(m['recall'] - 0.6) < 1e-9\n",
    "assert abs(m['f1'] - 2 * 0.75 * 0.6 / 1.35) < 1e-9, 'F1 = harmonic mean P & R'\n",
    "m0 = prf({'tp': 0, 'fp': 0, 'fn': 5, 'tn': 5})   # tak pernah prediksi positif\n",
    "assert m0['precision'] == 0.0 and m0['recall'] == 0.0 and m0['f1'] == 0.0\n",
    "print('✅ Latihan 5.2 benar! Konvensi pembagi-nol -> 0.0 (bukan crash di produksi).')\n",
))

cells.append(code(
    "# Terapkan ke model sungguhan: skor OLS vs skor HAM\n",
    "score_ols_te = predict(w_ols, b_ols, Xte)\n",
    "score_ham_te = predict_ham(mu, Xte)\n",
    "\n",
    "cm_ols = confusion(yte01, score_ols_te, med)\n",
    "cm_ham = confusion(yte01, score_ham_te, med)\n",
    "m_ols, m_ham = prf(cm_ols), prf(cm_ham)\n",
    "print(f'OLS @thr={med:.2f}: cm={cm_ols}')\n",
    "print(f'      P={m_ols[\"precision\"]:.3f}  R={m_ols[\"recall\"]:.3f}  F1={m_ols[\"f1\"]:.3f}')\n",
    "print(f'HAM @thr={med:.2f}: cm={cm_ham}  F1={m_ham[\"f1\"]:.3f}   <- baseline gagal total')\n",
    "assert m_ham['f1'] == 0.0, 'HAM konstan = prediktor tanpa daya beda'\n",
    "assert m_ols['f1'] > 0.8, f'F1 OLS harus tinggi: {m_ols[\"f1\"]}'\n",
    "print('✅ Baseline gagal total di metrik keputusan — F1-nya nol persis.')\n",
))

# ============================ BAGIAN 6 ============================
cells.append(md(
    "## Bagian 6 — Threshold Tuning di CALIB, Evaluasi di TEST\n",
    "",
    "Threshold adalah **keputusan bisnis**, bukan angka model:\n",
    "- Screening medis → recall tinggi (satupasien terlewat mahal)\n",
    "- Filter spam → precision tinggi (email penting masuk spam mahal)\n",
    "",
    "Protokol yang benar: pilih threshold di **CALIB**, evaluasi SEKALI di **TEST**.",
    "Menyetel threshold di test = test bocor jadi calib kedua — angka test tidak lagi",
    "bisa dipercaya.",
))

cells.append(code(
    "def pilih_threshold(y_true01, y_score, kandidat, target='f1'):\n",
    "    \"\"\"Pilih threshold dengan F1 tertinggi pada data yang diberikan (SEHARUSNYA calib).\n",
    "    Return (thr, cm, metrik) pada pemenang. Ties: ambil yang pertama.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# thr, cm, m = pilih_threshold(yca01, predict(w_ols, b_ols, Xca), np.arange(0.5, 3.01, 0.1))\n",
    "# print(thr, m)\n",
))

cells.append(code(
    "# ✅ Cek latihan 6.1 — tuning di CALIB\n",
    "kandidat = np.arange(0.5, 3.01, 0.1)\n",
    "thr_star, cm_c, m_c = pilih_threshold(yca01, predict(w_ols, b_ols, Xca), kandidat)\n",
    "print(f'thr* = {float(thr_star):.1f}   (dari CALIB, bukan test!)')\n",
    "print(f'calib: cm={cm_c}')\n",
    "print(f'       P={m_c[\"precision\"]:.3f}  R={m_c[\"recall\"]:.3f}  F1={m_c[\"f1\"]:.3f}')\n",
    "assert 0.8 <= thr_star <= 1.6, f'thr* harus sekitar 1.0-1.1: {thr_star}'\n",
    "assert 0.85 < m_c['f1'] < 0.95, f'F1 calib ~0.90: {m_c[\"f1\"]}'\n",
    "print('✅ Latihan 6.1 benar! Threshold dipilih dari calib — test masih \"perawan\".')\n",
))

cells.append(code(
    "# Evaluasi SEKALI di test dengan threshold dari calib\n",
    "cm_t = confusion(yte01, score_ols_te, thr_star)\n",
    "m_t = prf(cm_t)\n",
    "print(f'test @thr*={float(thr_star):.1f}: cm={cm_t}')\n",
    "print(f'       P={m_t[\"precision\"]:.3f}  R={m_t[\"recall\"]:.3f}  F1={m_t[\"f1\"]:.3f}')\n",
    "assert abs(m_t['f1'] - 0.9318) < 0.02, f'F1 test @thr* harus ~0.932: {m_t[\"f1\"]}'\n",
    "assert m_t['precision'] >= 0.95, 'P tinggi: hanya 1 FP'\n",
    "\n",
    "# Bahaya: kalau threshold disetel LANGSUNG di test (kebocoran)\n",
    "thr_leak, _, m_leak = pilih_threshold(yte01, score_ols_te, kandidat)\n",
    "print(f'\\nBANDINGKAN: F1 test di-tune di test (bocor) = {m_leak[\"f1\"]:.3f} @thr={thr_leak}')\n",
    "print(f'            F1 test dengan thr dari calib (jujur)  = {m_t[\"f1\"]:.3f} @thr={thr_star}')\n",
    "assert m_leak['f1'] >= m_t['f1'], 'angka bocor harus terlihat lebih bagus (itu jebakannya!)'\n",
    "print('✅ Angka \"bocor\" lebih tinggi — dan itulah kenapa dia TIDAK SAH. Test yang')\n",
    "print('   menyetel modelnya sendiri = calib kedua, bukan pengukur lagi.')\n",
))

cells.append(code(
    "# Sweep F1 vs threshold di TEST — hanya untuk VISUALISASI trade-off (bukan memilih!)\n",
    "ths = np.arange(0.0, 3.51, 0.05)\n",
    "f1s = [prf(confusion(yte01, score_ols_te, t))['f1'] for t in ths]\n",
    "prec = [prf(confusion(yte01, score_ols_te, t))['precision'] for t in ths]\n",
    "rec = [prf(confusion(yte01, score_ols_te, t))['recall'] for t in ths]\n",
    "\n",
    "plt.figure(figsize=(12, 4))\n",
    "plt.plot(ths, f1s, label='F1')\n",
    "plt.plot(ths, prec, '--', label='precision')\n",
    "plt.plot(ths, rec, ':', label='recall')\n",
    "plt.axvline(float(thr_star), color='r', ls='--', alpha=0.6, label=f'thr* dari calib = {float(thr_star):.1f}')\n",
    "plt.xlabel('threshold'); plt.legend(); plt.grid(alpha=0.3)\n",
    "plt.title('Trade-off threshold: naikkan thr -> precision naik, recall turun')\n",
    "plt.show()\n",
    "\n",
    "print('Interpretasi: threshold = kenop bisnis. Geser kenopnya sesuai biaya FP vs FN,')\n",
    "print('tapi putuskan nilainya di CALIB, bukan di TEST.')\n",
))

# ============================ BAGIAN 7 ============================
cells.append(md(
    "## Bagian 7 — Kalibrasi Probability: ECE & Reliability Curve\n",
    "",
    "Model bilang \"80% yakin\" — apakah benar 80%-nya benar? **Kalibrasi** menjawabnya:",
    "",
    "```",
    "kelompokkan prediksi per bin probabilitas  [0.0,0.1) ... [0.9,1.0]",
    "per bin: confidence = mean(p)   |   accuracy = mean(label)",
    "ECE = Σ (n_bin / n) · |accuracy − confidence|",
    "```",
    "",
    "Kita pakai skor OLS disigmoid-kan sebagai probabilitas tiruan, lalu perbaiki dengan",
    "**Platt scaling**: regresi logistik 1-parameter pada CALIB (belajar di calib lagi!).",
))

cells.append(code(
    "def sigmoid(z):\n",
    "    \"\"\"Numerik-stabil: 1/(1+exp(-z)). Clip z ke [-60, 60].\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(sigmoid(0.0), sigmoid(100.0), sigmoid(-100.0))\n",
))

cells.append(code(
    "# ✅ Cek latihan 7.1\n",
    "assert sigmoid(0.0) == 0.5\n",
    "assert 0.0 < sigmoid(-1000.0) < 1e-20, 'tidak boleh overflow warning'\n",
    "assert 0.9999999999 < sigmoid(1000.0) <= 1.0\n",
    "p = sigmoid(np.array([-2.0, 0.0, 3.0]))\n",
    "assert np.allclose(p, [0.1192029, 0.5, 0.9525741], atol=1e-6)\n",
    "print('✅ Latihan 7.1 benar! sigmoid stabil: nilai ekstrem tidak menghasilkan NaN/inf.')\n",
))

cells.append(code(
    "def binned_calibration(y_true01, p, n_bins=10):\n",
    "    \"\"\"Kelompokkan prediksi per bin. Return list dict per bin:\n",
    "    {'lo', 'hi', 'n', 'conf', 'acc'} — conf=mean(p), acc=mean(label).\n",
    "    Bin terakhir inklusif atas (p == 1.0 masuk bin terakhir).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def ece(y_true01, p, n_bins=10):\n",
    "    \"\"\"Expected Calibration Error = Σ (n_bin/n)·|acc − conf|. Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# bins = binned_calibration(yte01, sigmoid(score_ols_te))\n",
    "# print(ece(yte01, sigmoid(score_ols_te)))\n",
))

cells.append(code(
    "# ✅ Cek latihan 7.2\n",
    "yt = np.array([1, 0, 1, 0])\n",
    "pp = np.array([0.9, 0.1, 0.8, 0.2])\n",
    "bins = binned_calibration(yt, pp, 10)\n",
    "# konvensi idx = min(int(p*10), 9): p=0.1 -> bin [0.1,0.2) idx 1, p=0.9 -> bin idx 9\n",
    "assert bins[0]['n'] == 0, 'bin [0.0,0.1) kosong di contoh ini'\n",
    "assert bins[1]['n'] == 1 and bins[1]['acc'] == 0.0 and abs(bins[1]['conf'] - 0.1) < 1e-9\n",
    "assert bins[8]['n'] == 1 and bins[8]['acc'] == 1.0   # p=0.8, label 1\n",
    "assert bins[9]['n'] == 1 and bins[9]['acc'] == 1.0   # p=0.9 -> bin terakhir\n",
    "assert abs(ece(yt, pp, 10) - 0.15) < 1e-9, 'ECE = rata-rata tertimbang |acc - conf|'\n",
    "print('✅ Latihan 7.2 benar!')\n",
))

cells.append(code(
    "def fit_platt(score, y01, lr=0.1, epochs=3000, seed=8):\n",
    "    \"\"\"Platt scaling: regresi logistik 1D p = sigmoid(a·s + b), fit gradient descent\n",
    "    pada data yang diberikan (SEHARUSNYA calib). Return (a, b).\n",
    "    Gradien MSE-free: loss = cross-entropy -> dL/da = mean((p - y)·s), dL/db = mean(p - y).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def apply_platt(a, b, score):\n",
    "    return sigmoid(a * np.asarray(score, dtype=float) + b)\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# a_p, b_p = fit_platt(score_ols_ca, yca01)\n",
    "# print(a_p, b_p)\n",
))

cells.append(code(
    "# ✅ Cek latihan 7.3 — Platt scaling belajar dari CALIB, evaluasi di TEST\n",
    "score_ols_ca = predict(w_ols, b_ols, Xca)\n",
    "p_raw_te = sigmoid(score_ols_te)\n",
    "a_p, b_p = fit_platt(score_ols_ca, yca01)\n",
    "p_cal_te = apply_platt(a_p, b_p, score_ols_te)\n",
    "ece_raw = ece(yte01, p_raw_te, 10)\n",
    "ece_cal = ece(yte01, p_cal_te, 10)\n",
    "print(f'ECE raw  = {ece_raw:.4f}')\n",
    "print(f'ECE platt = {ece_cal:.4f}   (a={a_p:.3f}, b={b_p:.3f})')\n",
    "assert ece_cal <= ece_raw + 0.01, 'Platt scaling harus memperbaiki (atau setara) ECE'\n",
    "assert np.all((p_cal_te > 0) & (p_cal_te < 1))\n",
    "print('✅ Latihan 7.3 benar! Platt scaling belajar DARI CALIB — sekali lagi: test tidak boleh dilatih.')\n",
))

cells.append(code(
    "# Reliability diagram\n",
    "bins_raw = binned_calibration(yte01, p_raw_te, 10)\n",
    "bins_cal = binned_calibration(yte01, p_cal_te, 10)\n",
    "xs = np.arange(10) / 10 + 0.05\n",
    "\n",
    "plt.figure(figsize=(6.5, 5))\n",
    "plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='sempurna')\n",
    "plt.bar(xs - 0.18, [b['acc'] for b in bins_raw], width=0.34, alpha=0.6, label='raw')\n",
    "plt.bar(xs + 0.18, [b['acc'] for b in bins_cal], width=0.34, alpha=0.6, label='platt')\n",
    "plt.xlabel('confidence bin'); plt.ylabel('accuracy')\n",
    "plt.legend(); plt.grid(alpha=0.3)\n",
    "plt.title(f'Reliability diagram — ECE raw {ece_raw:.3f} vs platt {ece_cal:.3f}')\n",
    "plt.show()\n",
    "\n",
    "print('Bin yang bar-nya di bawah diagonal = overconfident (banyak bilang yakin, kurang benar).')\n",
))

# ============================ BAGIAN 8 ============================
cells.append(md(
    "## Bagian 8 — GAN 1D: Generator vs Discriminator dari Nol\n",
    "",
    "GAN = dua jaringan berduel:\n",
    "",
    "```",
    "G: noise z ~ N(0,1)  →  x_fake = mu + sigma·z          (belajar meniru data)",
    "D: x                →  P(x asli)                       (belajar membedakan)",
    "",
    "loss D  = −mean[log D(x_real)] − mean[log(1 − D(x_fake))]",
    "loss G  = −mean[log D(x_fake)]        (non-saturating; gradien MELEWATI D ke G)",
    "```",
    "",
    "Data nyata: `N(2.0, 0.5²)`. Target: G belajar `mu→2.0`. Ini GAN **paling kecil yang",
    "masih mengajarkan semua konsep**: equilibrium, mode collapse, non-saturating loss.",
    "Kita tulis backprop-nya manual — dan disitulah 'dua jaringan' terlihat sebagai dua",
    "set gradien yang saling menentukan.",
))

cells.append(code(
    "# Inisialisasi pipeline GAN (seed terkunci 8)\n",
    "N_GAN, MEAN_G, STD_G = 512, 2.0, 0.5\n",
    "rg = np.random.default_rng(8)\n",
    "real_data = rg.normal(MEAN_G, STD_G, N_GAN)\n",
    "\n",
    "G_MU, G_LOGS = -2.0, np.log(0.5)          # G: x = mu + exp(logs)·z\n",
    "# D di-init ACAK kecil — init nol = saddle point: semua gradien nol, D tak pernah belajar\n",
    "D_W1, D_B1 = rg.normal(0.0, 0.5, 8), np.zeros(8)\n",
    "D_W2, D_B2 = rg.normal(0.0, 0.5, 8), 0.0\n",
    "LR = 0.05\n",
    "print(f'real: mean={real_data.mean():.3f} std={real_data.std():.3f}')\n",
    "print(f'G awal: mu={G_MU:.2f}  sigma={np.exp(G_LOGS):.3f}')\n",
))

cells.append(code(
    "def gen_sample(bs, rng_g):\n",
    "    \"\"\"Sample G: z ~ N(0,1) -> x = G_MU + exp(G_LOGS)*z. Return (x, z).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def disc_forward(x):\n",
    "    \"\"\"D: x -> hidden (8 unit tanh) -> logit (1 nilai). Return (logit, hidden).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def dscore(x):\n",
    "    \"\"\"Skor D(x) = sigmoid(logit). Return array (n,).\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# xr, zr = gen_sample(4, rg)\n",
    "# o, h = disc_forward(real_data[:4])\n",
    "# print(dscore(real_data[:4]))\n",
))

cells.append(code(
    "# ✅ Cek latihan 8.1\n",
    "xr, zr = gen_sample(4, rg)\n",
    "assert xr.shape == (4,) and zr.shape == (4,)\n",
    "assert np.allclose(xr, G_MU + np.exp(G_LOGS) * zr), 'x = mu + sigma*z'\n",
    "o, h = disc_forward(real_data[:6])\n",
    "assert o.shape == (6,) and h.shape == (6, 8)\n",
    "assert np.all(h >= -1) and np.all(h <= 1), 'tanh harus di [-1, 1]'\n",
    "s = dscore(real_data[:6])\n",
    "assert np.all((s > 0) & (s < 1)), 'sigmoid harus di (0, 1)'\n",
    "print('✅ Latihan 8.1 benar! G dari noise, D dari data ke skor (0,1).')\n",
))

cells.append(code(
    "def d_loss(x_real, x_fake):\n",
    "    \"\"\"Loss D: -mean log D(real) - mean log(1-D(fake)). Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def g_loss(x_fake):\n",
    "    \"\"\"Loss G non-saturating: -mean log D(fake). Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(d_loss(real_data[:8], xr[:8]), g_loss(xr[:8]))\n",
))

cells.append(code(
    "# ✅ Cek latihan 8.2\n",
    "x_real_k = real_data[:8]\n",
    "x_fake_k = gen_sample(8, rg)[0]\n",
    "dl = d_loss(x_real_k, x_fake_k)\n",
    "gl = g_loss(x_fake_k)\n",
    "assert dl > 0 and gl > 0, 'loss log selalu positif'\n",
    "# maksimum loss D = 2·log(2) saat D = 0.5 untuk semua input (equilibrium teoretis)\n",
    "print(f'D loss={dl:.3f} (max teoretis {2 * np.log(2):.3f})   G loss={gl:.3f}')\n",
    "assert dl <= 2 * np.log(2) + 0.1, 'D loss tidak boleh jauh melebihi 2·log2 di awal'\n",
    "print('✅ Latihan 8.2 benar! Equilibrium teoretis: D=0.5 di semua x, kedua loss = log(2).')\n",
))

cells.append(code(
    "# TRAINING LOOP — baca dulu, lalu jalankan. Dua set gradien yang TIDAK boleh tertukar:\n",
    "rng_t = np.random.default_rng(8)\n",
    "BATCH, EPOCHS = 128, 300\n",
    "hist = []\n",
    "for ep in range(EPOCHS):\n",
    "    xr = rng_t.choice(real_data, BATCH, replace=False)\n",
    "    xf, _ = gen_sample(BATCH, rng_t)\n",
    "\n",
    "    # ---- update D (dua langkah: real dan fake) ----\n",
    "    for x, label in ((xr, 1.0), (xf, 0.0)):\n",
    "        o, h = disc_forward(x)\n",
    "        p = sigmoid(o)\n",
    "        dO = (p - label) / len(x)              # d loss/d logit\n",
    "        dW2 = h.T @ dO; dB2 = dO.sum()\n",
    "        dH = dO[:, None] * D_W2[None, :] * (1.0 - h ** 2)\n",
    "        dW1 = (x[:, None] * dH).sum(0); dB1 = dH.sum(0)\n",
    "        D_W1 -= LR * dW1; D_B1 -= LR * dB1\n",
    "        D_W2 -= LR * dW2; D_B2 -= LR * dB2\n",
    "\n",
    "    # ---- update G: gradien loss G MELEWATI D sampai ke parameter G ----\n",
    "    xf, _ = gen_sample(BATCH, rng_t)\n",
    "    o_g, h_g = disc_forward(xf)                # D dibekukan: hanya hitung, tidak update\n",
    "    p_g = sigmoid(o_g)\n",
    "    dO_g = (p_g - 1.0) / len(xf)               # d(-log p)/d logit = p - 1\n",
    "    dx_g = (D_W2[None, :] * (1.0 - h_g ** 2)) @ D_W1   # d logit/d x, jumlah over hidden\n",
    "    dmu = float(np.sum(dO_g * dx_g))           # d x/d mu = 1\n",
    "    dlogs = float(np.sum(dO_g * dx_g * (xf - G_MU)))   # d x/d logs = sigma\n",
    "    G_MU -= LR * dmu\n",
    "    G_LOGS -= LR * dlogs\n",
    "\n",
    "    if ep % 50 == 0 or ep == EPOCHS - 1:\n",
    "        xs_, _ = gen_sample(2000, rng_t)\n",
    "        hist.append((ep, d_loss(xr, xs_), g_loss(xs_),\n",
    "                     float(xs_.mean()), float(xs_.std())))\n",
    "        print(f'ep {ep:4d}: D={hist[-1][1]:.3f} G={hist[-1][2]:.3f} '\n",
    "              f'fake mean={hist[-1][3]:+.3f} std={hist[-1][4]:.3f}')\n",
))

cells.append(code(
    "# ✅ Cek latihan 8.3 — tanda-tanda konvergensi ke equilibrium\n",
    "fake_final, _ = gen_sample(5000, rng_t)\n",
    "d_real_final = float(dscore(real_data).mean())\n",
    "d_fake_final = float(dscore(fake_final).mean())\n",
    "print(f'G akhir: mu={G_MU:.4f} (target {MEAN_G}), sigma={np.exp(G_LOGS):.4f} (target {STD_G})')\n",
    "print(f'D(real)={d_real_final:.3f}  D(fake)={d_fake_final:.3f}  (equilibrium: keduanya ~0.5)')\n",
    "assert 1.5 < G_MU < 2.8, f'G harus menemukan mu ~2.0: {G_MU}'\n",
    "assert d_fake_final > 0.2, 'D(fake) harus jauh dari 0 (G berhasil menipu)'\n",
    "assert abs(d_real_final - d_fake_final) < 0.25, 'D tak lagi bisa membedakan: equilibrium'\n",
    "\n",
    "plt.figure(figsize=(12, 4))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.hist(real_data, bins=40, alpha=0.6, label='real')\n",
    "plt.hist(fake_final, bins=40, alpha=0.6, label='fake (G)')\n",
    "plt.legend(); plt.title('Distribusi real vs generated')\n",
    "plt.subplot(1, 2, 2)\n",
    "eps_ = [h_[0] for h_ in hist]\n",
    "plt.plot(eps_, [h_[3] for h_ in hist], label='fake mean')\n",
    "plt.axhline(MEAN_G, color='r', ls='--', label='target mean 2.0')\n",
    "plt.xlabel('epoch'); plt.legend(); plt.title('G belajar mean')\n",
    "plt.tight_layout(); plt.show()\n",
    "print('✅ Latihan 8.3 benar! D dan G berakhir di equilibrium — tak ada yang \"menang\".')\n",
    "print('   Inilah beda GAN dengan supervised learning: tidak ada loss yang \"monoton turun\".')\n",
))

cells.append(md(
    "**Pertanyaan pemahaman:**\n",
    "",
    "1. Kenapa init D nol membuat training macet total? (hint: hitung gradien di titik itu)",
    "2. Kita hanya melatih `mu` dan `sigma` G. Mode collapse itu apa dalam bahasa",
    "   distribusi ini — dan kapan terjadi kalau G punya kapasitas lebih besar dari",
    "   variasi data?",
    "3. `d_loss` memakai `log(1−D(fake))` (saturating) tapi update G memakai gradien",
    "   `p − 1` dari `−log D(fake)` (non-saturating). Kenapa paper GAN menyarankan",
    "   versi non-saturating untuk G?",
))

# ============================ BAGIAN 9 ============================
cells.append(md(
    "## Bagian 9 — Ringkasan: Angka yang Kamu Pegang",
    "",
    "Semua angka di bawah terkunci seed. Cocokkan dengan hasil epromu — kalau beda,",
    "kembali ke bagian terkait.",
))

cells.append(code(
    "print('=' * 58)\n",
    "print('RINGKASAN LAB 8 — semua seed terkunci, harus cocok persis')\n",
    "print('=' * 58)\n",
    "print(f'1. split                : train 640 | calib 80 | test 80')\n",
    "print(f'2. OLS                  : w={np.round(w_ols, 3).tolist()} b={b_ols:.3f}  (true [2.0, -1.5], 1.0)')\n",
    "print(f'3. RMSE (test)          : OLS {rmse_ols:.3f} | neg {rmse_neg:.3f} | HAM {rmse_ham:.3f}')\n",
    "print(f'4. skill score vs HAM   : OLS {skill_ols:.3f} | neg {skill_neg:.3f}')\n",
    "print(f'5. F1 test @thr*        : OLS {m_t[\"f1\"]:.3f} (thr dari calib = {float(thr_star):.1f})')\n",
    "print(f'6. ECE                  : raw {ece_raw:.3f} -> platt {ece_cal:.3f}')\n",
    "print(f'7. GAN                  : G mu {G_MU:.3f} (target 2.0) | D(fake) {d_fake_final:.3f}')\n",
    "print('\\nKalau semua angka cocok: kamu memegang pipeline evaluasi produksi lengkap')\n",
    "print('yang bisa dijelaskan baris demi baris — tanpa library evaluasi apa pun.')\n",
))

cells.append(md(
    "---",
    "",
    "## 🏁 Selesai — Lanjut ke Kuis",
    "",
    "**Yang sudah kamu bangun dari nol:** split tiga arah, hash deterministik,",
    "persistensi + round-trip test, RMSE/MAE/R², baseline HAM, skill score,",
    "confusion matrix + P/R/F1, threshold tuning (calib vs test, plus demo bocor),",
    "kalibrasi probability + ECE + Platt scaling, dan GAN 1D dengan dua set gradien.",
    "",
    "**Lanjut:** `02_kuis_ml_produksi.ipynb` (23 poin, skor otomatis) →",
    "`project-starter-evaluasi-produksi/` → baca kembali `cheatsheet-ml-produksi.md`",
    "sebelum mengerjakan project.",
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
