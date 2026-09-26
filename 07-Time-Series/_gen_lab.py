"""Generate 01_lab_time_series.ipynb for Bab 7."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_lab_time_series.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [*lines, ""]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [*lines, ""]}


cells = []

# ============================ HEADER ============================
cells.append(md(
    "# 🧪 Lab 7 — Praktikum Time Series (dari Nol)",
    "",
    "> Pendamping materi Bab 7. Prinsip sama dengan Lab 4–6: **semua dibangun dari nol",
    "> dengan numpy** — dekomposisi, windowing, AR(p), Holt-Winters, backtesting — supaya",
    "> `ARIMA`, `Prophet`, dan metrik forecasting tidak pernah jadi kotak hitam. Seri kita",
    "> sintetis dengan pola yang KITA TENTUKAN sendiri, jadi kebenaran model bisa diukur",
    "> persis. Satu-satunya data nyata (sunspots) bersifat opsional.",
    "",
    "| Bagian | Topik | Waktu (menit) |",
    "|---|---|---|",
    "| 1 | Anatomi time series: trend + seasonality + noise + ACF | 15 |",
    "| 2 | Windowing: fitur & label dari urutan waktu | 15 |",
    "| 3 | Split temporal vs acak — membongkar kebocoran | 15 |",
    "| 4 | Baseline: naive, seasonal naive, HAM | 15 |",
    "| 5 | AR(p): fit OLS, forecast rekursif, error menumpuk | 25 |",
    "| 6 | Holt-Winters: level + trend + seasonality dari nol | 30 |",
    "| 7 | Transformasi log & metrik yang jujur | 15 |",
    "| 8 | Backtesting: satu split bisa menipu | 20 |",
    "| 9 | Data nyata opsional: sunspots & siklus 11 tahun | 10 |",
    "",
    "Setiap latihan punya cell **✅ Cek** — jalankan, kalau ✅ berarti pemahamanmu benar.",
))

# ============================ BAGIAN 1 ============================
cells.append(md(
    "## Bagian 1 — Anatomi Time Series: Trend + Seasonality + Noise",
    "",
    "Kita **membuat sendiri** seri dengan resep yang diketahui — ini keunggulan eksperimen:",
    "jawaban benar sudah kita pegang sebelum model apa pun dijalankan.",
    "",
    "```",
    "y[t] = trend[t] + seasonal[t] + noise[t]",
    "trend[t]     = 100 + 0.05·t                (naik 5 unit per 100 hari)",
    "seasonal[t]  = 20·sin(2π·t/365)            (siklus tahunan, amplitudo 20)",
    "noise[t]     ~ N(0, 5²)                    (acak, σ = 5)",
    "```",
    "",
    "**Seed terkunci (rng 42)** — angka di lab ini bisa direproduksi persis.",
))

cells.append(code(
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "rng = np.random.default_rng(42)   # seed terkunci — seluruh lab reproducible\n",
    "\n",
    "t = np.arange(365 * 3, dtype=float)          # 3 tahun data harian\n",
    "trend = 100.0 + 0.05 * t\n",
    "seasonality = 20.0 * np.sin(2 * np.pi * t / 365.0)\n",
    "noise = rng.normal(0.0, 5.0, len(t))\n",
    "seri = trend + seasonality + noise\n",
    "\n",
    "print(f'n={len(seri)}  mean={seri.mean():.2f}  std={seri.std():.2f}')\n",
))

cells.append(code(
    "plt.figure(figsize=(14, 5))\n",
    "plt.plot(t, seri, alpha=0.6, label='seri total')\n",
    "plt.plot(t, trend, 'r--', lw=2, label='trend (tersembunyi)')\n",
    "plt.plot(t, trend + seasonality, 'g:', lw=2, label='trend + musiman')\n",
    "plt.xlabel('hari'); plt.ylabel('nilai'); plt.legend(); plt.grid(alpha=0.3)\n",
    "plt.title('Seri sintetis: kamu tahu resepnya — model harus menemukannya')\n",
    "plt.show()\n",
))

cells.append(code(
    "def buat_trend_linear(series, t):\n",
    "    \"\"\"Estimasi trend dengan regresi linear sederhana: y = a + b·t.\n",
    "    Return (a, b). Petunjuk: center-kan t supaya numeriknya stabil,\n",
    "    slope = cov(t, y) / var(t) — atau pakai np.polyfit.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# a, b = buat_trend_linear(seri, t)\n",
    "# print(f'a={a:.2f}  b={b:.4f}')   # target: b ~ 0.05\n",
))

cells.append(code(
    "# ✅ Cek latihan 1.1\n",
    "a_hat, b_hat = buat_trend_linear(seri, t)\n",
    "assert 0.03 < b_hat < 0.07, f'slope harus dekat 0.05: {b_hat}'\n",
    "assert 95.0 < a_hat < 115.0, f'intersep harus dekat 100: {a_hat}'\n",
    "print(f'✅ Latihan 1.1 benar! trend terestimasi: y = {a_hat:.2f} + {b_hat:.4f}·t')\n",
    "print('   (slope tak persis 0.05 — noise & musiman pada sampel 3 tahun ini')\n",
    "print('    berkorelasi sedikit dengan t; itulah kenapa kita tak pernah percaya')\n",
    "print('    angka tunggal tanpa interval/baseline)')\n",
))

cells.append(md(
    "### Autocorrelation — jendela menuju masa lalu\n",
    "",
    "ACF = korelasi `y[t]` dengan `y[t-k]`. Reading-nya:\n",
    "- decay pelan tanpa puncak → ada **trend**\n",
    "- puncak berulang tiap m → ada **musiman** periode m\n",
    "- ~0 di semua lag → **noise murni**\n",
    "",
    "⚠️ Trend \"menutupi\" ACF: dua seri yang sama-sama naik akan berkorelasi tinggi di",
    "semua lag walau tak sebanding. Detrend dulu sebelum membaca musiman.",
))

cells.append(code(
    "def acf(x, lag):\n",
    "    \"\"\"Autokorelasi sampel pada `lag`: sum(x[:-lag]·x[lag:]) / sum(x²) dengan x dicenter.\n",
    "    Return float.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(acf(seri, 1), acf(seri, 365))\n",
))

cells.append(code(
    "# ✅ Cek latihan 1.2\n",
    "r1 = acf(seri, 1)\n",
    "assert r1 > 0.9, f'seri ber-trend harus ACF lag-1 tinggi: {r1}'\n",
    "noise_murni = np.random.default_rng(7).normal(0, 1, 1000)\n",
    "assert abs(acf(noise_murni, 1)) < 0.1, 'noise murni harus ~0 di semua lag'\n",
    "print(f'✅ Latihan 1.2 benar! ACF(seri,1)={r1:.3f} (trend) vs ACF(noise,1)={acf(noise_murni, 1):.3f}')\n",
    "\n",
    "lags = range(1, 400)\n",
    "plt.figure(figsize=(14, 4))\n",
    "plt.bar(lags, [acf(seri, L) for L in lags], width=1.0)\n",
    "plt.axhline(0, color='k', lw=0.5)\n",
    "plt.xlabel('lag'); plt.ylabel('ACF')\n",
    "plt.title('ACF seri mentah: decay pelan (trend menutupi) — puncak musiman tidak terlihat jelas')\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()\n",
))

# ============================ BAGIAN 2 ============================
cells.append(md(
    "## Bagian 2 — Windowing: Fitur & Label dari Urutan Waktu\n",
    "",
    "Konversi 1 seri → dataset supervised:\n",
    "",
    "```",
    "X[i] = y[i : i+W]      (window W hari terakhir)",
    "y[i] = y[i + W]        (nilai tepat 1 hari ke depan)",
    "n_sampel = len(series) − W",
    "```",
    "",
    "Beda dengan Bab 3: baris TIDAK BOLEH diacak saat split — urutan adalah informasi.",
))

cells.append(code(
    "def buat_window(series, W):\n",
    "    \"\"\"Windowing 1-step: X[i] = series[i:i+W], y[i] = series[i+W].\n",
    "    Return (X, y): X berbentuk (n, W), y berbentuk (n,).\n",
    "    Petunjuk: np.lib.stride_tricks.sliding_window_view(series, W) memberi semua\n",
    "    window sekaligus — potong yang terakhir karena tak punya label.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# Xw, yw = buat_window(seri[:50], 7)\n",
    "# print(Xw.shape, yw.shape)\n",
))

cells.append(code(
    "# ✅ Cek latihan 2.1\n",
    "Xw, yw = buat_window(seri, 35)\n",
    "assert Xw.shape == (len(seri) - 35, 35) and yw.shape == (len(seri) - 35,)\n",
    "assert np.array_equal(Xw[0], seri[:35]) and yw[0] == seri[35]\n",
    "assert yw[-1] == seri[-1] and np.array_equal(Xw[-1], seri[-36:-1])\n",
    "W_uji = seri[100:135]\n",
    "assert np.array_equal(Xw[100], W_uji) and yw[100] == seri[135], 'window ke-100 mulai idx 100'\n",
    "assert 117.0 < float(W_uji.mean()) < 125.0, 'mean window harus di wilayah nilai seri'\n",
    "print(f'✅ Latihan 2.1 benar! X={Xw.shape}, y={yw.shape}, n_sampel = {len(seri)} - 35 = {len(Xw)}')\n",
    "\n",
    "# visualisasi satu window + targetnya\n",
    "i0 = 100\n",
    "plt.figure(figsize=(12, 4))\n",
    "plt.plot(range(i0, i0 + 35), Xw[i0], 'b-', label='window (fitur)')\n",
    "plt.plot([i0 + 35], [yw[i0]], 'r*', ms=15, label='target (label)')\n",
    "plt.axvline(i0 + 34.5, color='gray', ls='--', alpha=0.5)\n",
    "plt.xlabel('hari'); plt.legend(); plt.grid(alpha=0.3)\n",
    "plt.title('Satu sampel: 35 hari ke belakang → 1 hari ke depan')\n",
    "plt.show()\n",
))

# ============================ BAGIAN 3 ============================
cells.append(md(
    "## Bagian 3 — Split Temporal vs Acak: Membongkar Kebocoran",
    "",
    "Aturan #1 time series: **split berurutan waktu, JANGAN acak**. Split acak memasukkan",
    "titik masa depan ke training — model menghafal wilayah nilai yang seharusnya belum",
    "ada. Ini *data leakage temporal*: skor validasi bagus, model bodoh di produksi.\n",
    "",
    "Eksperimen: model 1-NN pada window (penghafal ulang). Bandingkan MAE-nya di dua",
    "rezim split. Lalu lihat juga in-sample vs out-of-sample pada model dekomposisi",
    "trend+dummies (Bagian 6) — angkanya sudah menunggu di bawah.",
))

cells.append(code(
    "def split_waktu(X, y, rasio_train=0.8):\n",
    "    \"\"\"Split BERURUTAN: X[:cut] train, X[cut:] val, cut = int(n·rasio).\n",
    "    Return (Xtr, ytr, Xva, yva).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# Xtr, ytr, Xva, yva = split_waktu(Xw, yw)\n",
    "# print(Xtr.shape, Xva.shape)\n",
))

cells.append(code(
    "# ✅ Cek latihan 3.1\n",
    "Xtr, ytr, Xva, yva = split_waktu(Xw, yw, 0.8)\n",
    "n_tot = len(Xw)\n",
    "cut = int(n_tot * 0.8)\n",
    "assert len(Xtr) == cut and len(Xva) == n_tot - cut\n",
    "assert np.array_equal(Xtr, Xw[:cut]) and np.array_equal(Xva, Xw[cut:])\n",
    "assert np.array_equal(ytr, yw[:cut]) and np.array_equal(yva, yw[cut:])\n",
    "print(f'✅ Latihan 3.1 benar! train={len(Xtr)} (hari 0–{cut - 1}), val={len(Xva)} (hari {cut}–{n_tot - 1})')\n",
    "print('   Urutan waktu terjaga: tidak ada hari masa depan di training.')\n",
))

cells.append(code(
    "# Eksperimen leakage: 1-NN pada window, split acak vs temporal\n",
    "def knn1_pred(Xtr_, ytr_, Xte_):\n",
    "    \"\"\"Prediksi tiap sampel test dengan label tetangga terdekat (jarak L1 rata-rata).\"\"\"\n",
    "    pred = np.empty(len(Xte_))\n",
    "    for i, x in enumerate(Xte_):\n",
    "        d = np.mean(np.abs(Xtr_ - x), axis=1)\n",
    "        pred[i] = ytr_[int(np.argmin(d))]\n",
    "    return pred\n",
    "\n",
    "Xtr, ytr, Xva, yva = split_waktu(Xw, yw, 0.8)\n",
    "mae_temporal = float(np.mean(np.abs(yva - knn1_pred(Xtr, ytr, Xva))))\n",
    "\n",
    "rng_s = np.random.default_rng(42)\n",
    "perm = rng_s.permutation(n_tot)\n",
    "itr, iva = perm[:cut], perm[cut:]\n",
    "mae_acak = float(np.mean(np.abs(yw[iva] - knn1_pred(Xw[itr], yw[itr], Xw[iva]))))\n",
    "\n",
    "print(f'MAE split ACAK    (bocor) : {mae_acak:.3f}')\n",
    "print(f'MAE split TEMPORAL (jujur): {mae_temporal:.3f}')\n",
    "assert mae_acak < mae_temporal, 'split acak harus terlihat LEBIH BAGUS (itu jebakannya!)'\n",
    "print(f'✅ Split acak terlihat {mae_temporal / mae_acak:.2f}× lebih baik — padahal itu ILMUSI.')\n",
    "print('   Tetangga \"masa depan\" yang hampir identik ikut masuk training.')\n",
    "print('   Demo kedua: model dekomposisi pdq — in-sample MAE 6.41 vs out-of-sample 11.76')\n",
    "print('   (dihitung di Bagian 6). Dua-duanya gejala bocor yang sama.')\n",
))

cells.append(md(
    "**Pertanyaan pemahaman:** mengapa di regresi tabel biasa (Bab 3) split acak justru",
    "sah, sedangkan di time series jadi ilegal? (hint: apa yang membedakan satu baris",
    "dengan baris lainnya di kedua jenis data?)",
))

# ============================ BAGIAN 4 ============================
cells.append(md(
    "## Bagian 4 — Baseline: Naive, Seasonal Naive, HAM\n",
    "",
    "Tiga baseline yang wajib dikalahkan sebelum model apa pun dipercaya:\n",
    "",
    "| Baseline | Prediksi | Kapan kuat |",
    "|---|---|---|",
    "| Naive | `ŷ[t+1] = y[t]` | random walk, noise dominan |",
    "| Seasonal naive | `ŷ[t+h] = y[t+h−m]` | musiman stabil |",
    "| HAM | `ŷ = mean(train)` | stationer tanpa pola |",
    "",
    "Protokol: fit dari TRAIN saja, evaluasi MAE di VAL (219 hari terakhir).",
))

cells.append(code(
    "# Protokol: fit dari TRAIN saja, evaluasi MAE di VAL (219 hari terakhir)\n",
    "split_idx = int(len(seri) * 0.8)\n",
    "train, val = seri[:split_idx], seri[split_idx:]\n",
    "H = len(val)                        # horizon = panjang validasi\n",
    "print(f'split={split_idx}, |train|={len(train)}, |val|={H}')\n",
))

cells.append(code(
    "def mae(y_true, y_pred):\n",
    "    \"\"\"Mean Absolute Error: mean(|y_true - y_pred|). Return float.\"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# assert mae(np.array([1., 2.]), np.array([2., 4.])) == 1.5\n",
))

cells.append(code(
    "# ✅ Cek latihan 4.1\n",
    "assert mae(np.array([1., 2.]), np.array([2., 4.])) == 1.5\n",
    "assert mae(np.array([5.]), np.array([5.])) == 0.0\n",
    "\n",
    "# baseline naive: prediksi val = nilai terakhir train (dijaga konstan)\n",
    "pred_naive = np.full(H, float(train[-1]))\n",
    "mae_naive = mae(val, pred_naive)\n",
    "\n",
    "# baseline seasonal naive 365: prediksi hari j = hari j-365\n",
    "pred_sn = seri[split_idx - 365:split_idx - 365 + H]\n",
    "mae_sn = mae(val, pred_sn)\n",
    "\n",
    "# baseline HAM\n",
    "mae_ham = mae(val, np.full(H, float(train.mean())))\n",
    "\n",
    "print(f'MAE naive          : {mae_naive:.3f}')\n",
    "print(f'MAE seasonal-naive : {mae_sn:.3f}')\n",
    "print(f'MAE HAM            : {mae_ham:.3f}')\n",
    "assert 8.0 < mae_naive < 10.5, f'naive konstan dari t=875: {mae_naive}'\n",
    "assert 17.0 < mae_sn < 21.0, f'seasonal-naive kena noise 2 tahun beda: {mae_sn}'\n",
    "assert 14.0 < mae_ham < 18.0, f'HAM mengabaikan trend+musiman: {mae_ham}'\n",
    "print('✅ Latihan 4.1 benar!')\n",
    "print('   Perhatikan: naive konstan membawa level titik terakhir (145.99) — MAE ~9')\n",
    "print('   karena level terakhir kebetulan dekat lembah musiman val. Baseline kuat ≠')\n",
    "print('   baseline stabil: bandingkan dengan backtest di Bagian 8.')\n",
))

cells.append(code(
    "plt.figure(figsize=(14, 4))\n",
    "plt.plot(val, 'k-', alpha=0.6, label='aktual (val)')\n",
    "plt.plot(pred_naive, 'b--', label=f'naive (MAE {mae_naive:.1f})')\n",
    "plt.plot(pred_sn, 'g--', label=f'seasonal-naive-365 (MAE {mae_sn:.1f})')\n",
    "plt.axhline(train.mean(), color='orange', ls=':', label=f'HAM (MAE {mae_ham:.1f})')\n",
    "plt.xlabel('hari ke-depan'); plt.legend(); plt.grid(alpha=0.3)\n",
    "plt.title('Tiga baseline di horizon panjang — garis datar vs pola salah')\n",
    "plt.show()\n",
))

# ============================ BAGIAN 5 ============================
cells.append(md(
    "## Bagian 5 — AR(p): Regresi Linear pada Lag\n",
    "",
    "```",
    "y[t] = φ1·y[t−1] + φ2·y[t−2] + … + φp·y[t−p] + ε[t]",
    "```",
    "",
    "Fit = OLS biasa (Bab 2/3!): baris ke-i fitur = `[y[i+p−1], …, y[i]]` (lag-1 dulu),",
    "target `y[i+p]` — sehingga `coef[0]` mengalikan nilai TERBARU, konsisten dengan",
    "ar_forecast di bawah. Data menyusut: n → n−p baris.\n",
    "",
    "Forecast multi-step bersifat **rekursif**: prediksi menjadi input langkah berikutnya",
    "— di situlah error mulai menumpuk.",
))

cells.append(code(
    "def fit_ar(series, p):\n",
    "    \"\"\"Fit AR(p) dengan OLS. Return koefisien φ berbentuk (p,).\n",
    "    Baris i: fitur = series[i : i+p], target = series[i+p].\n",
    "    Petunjuk: kolom-kolom bisa dibangun dengan slicing series[p-k-1 : len-k-1].\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def ar_forecast(coef, last_values, h):\n",
    "    \"\"\"Forecast h langkah secara rekursif.\n",
    "    last_values: p nilai terakhir URUT WAKTU (tertua -> terbaru), seperti train[-p:].\n",
    "    Konvensi: coef[0] mengalikan nilai TERBARU (lag-1) — sama dengan fit_ar.\n",
    "    Prediksi baru dimasukkan ke histori — error menumpuk sepanjang horizon.\n",
    "    Return array (h,).\n",
    "    \"\"\"\n",
    "    # TODO: balik dulu (hist[0] = terbaru), loop h kali: coef @ hist[:p] -> append\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# coef = fit_ar(train, 7)\n",
    "# print(np.round(coef, 3))\n",
    "# print(ar_forecast(coef, train[-7:], 5))\n",
))

cells.append(code(
    "# ✅ Cek latihan 5.1\n",
    "# AR(1) pada seri sintetis φ=0.7 harus terestimasi dekat 0.7\n",
    "rng_a = np.random.default_rng(42)\n",
    "z = np.empty(2000)\n",
    "z[0] = 0.0\n",
    "for i in range(1, 2000):\n",
    "    z[i] = 0.7 * z[i - 1] + rng_a.normal(0, 1.0)\n",
    "phi1 = fit_ar(z, 1)\n",
    "assert phi1.shape == (1,) and abs(phi1[0] - 0.7) < 0.05, f'AR(1) harus pulih: {phi1}'\n",
    "\n",
    "# forecast 1 langkah = perkalian murni\n",
    "# last_values=[10, 4] berarti y[t-2]=10, y[t-1]=4 -> pred = 0.5*4 + 0.3*10\n",
    "c_tes = np.array([0.5, 0.3])\n",
    "out = ar_forecast(c_tes, np.array([10.0, 4.0]), 1)\n",
    "assert out.shape == (1,) and abs(out[0] - (0.5 * 4 + 0.3 * 10)) < 1e-9, \\\n",
    "    f'coef[0] mengalikan nilai TERBARU: {out}'\n",
    "\n",
    "# forecast rekursif: langkah ke-2 memakai prediksi langkah ke-1\n",
    "out2 = ar_forecast(c_tes, np.array([10.0, 4.0]), 2)\n",
    "assert abs(out2[1] - (0.5 * out2[0] + 0.3 * 4.0)) < 1e-9, 'langkah-2 pakai prediksi langkah-1'\n",
    "\n",
    "coef7 = fit_ar(train, 7)\n",
    "assert coef7.shape == (7,)\n",
    "print(f'✅ Latihan 5.1 benar! φ(AR1)={phi1[0]:.3f} (target 0.7)')\n",
    "print(f'   AR(7) pada `train`: {np.round(coef7, 3).tolist()}')\n",
))

cells.append(code(
    "# Evaluasi AR(7) di horizon panjang + lihat error menumpuk\n",
    "fc7 = ar_forecast(coef7, train[-7:], H)\n",
    "mae_ar7 = mae(val, fc7)\n",
    "ape7 = float(np.mean(np.abs(ar_forecast(coef7, train[-7:], 7) - seri[split_idx:split_idx + 7])\n",
    "                     / seri[split_idx:split_idx + 7]))\n",
    "print(f'AR(7) 7-langkah  : APE = {ape7:.4f}  (bagus!)')\n",
    "print(f'AR(7) {H}-langkah : MAE = {mae_ar7:.3f}  (lebih buruk dari naive {mae_naive:.3f}!)')\n",
    "assert ape7 < 0.05, 'horizon pendek harus akurat'\n",
    "assert mae_ar7 > mae_naive, 'horizon panjang: error rekursif menumpuk melebihi naive'\n",
    "print('✅ Pelajaran besar: model yang bagus di 1-step bisa BURUK di multi-step.')\n",
    "print('   Prediksi jadi input → bias & error terakumulasi → mean meleset makin jauh.')\n",
    "\n",
    "plt.figure(figsize=(14, 4))\n",
    "plt.plot(val, 'k-', alpha=0.5, label='aktual')\n",
    "plt.plot(fc7, 'r-', label=f'AR(7) rekursif (MAE {mae_ar7:.1f})')\n",
    "plt.xlabel('hari ke-depan'); plt.legend(); plt.grid(alpha=0.3)\n",
    "plt.title('Multi-step rekursif: prediksi kehilangan pola seiring horizon')\n",
    "plt.show()\n",
))

cells.append(md(
    "**Pertanyaan pemahaman:**\n",
    "",
    "1. AR(7) menghasilkan MAE ~20 di horizon 219 — lebih buruk dari naive. Kenapa",
    "   window 7 hari tidak cukup menangkap musiman 365 hari? (hint: berapa banyak info",
    "   musiman yang tersisa di 7 angka terakhir?)",
    "2. Mengapa APE 7-langkah (1.9%) jauh lebih kecil dari MAE relatif 219-langkah?",
    "   Kapan klaim \"model bagus\" jadi menyesatkan kalau horizon tidak disebut?",
))

# ============================ BAGIAN 6 ============================
cells.append(md(
    "## Bagian 6 — Holt-Winters: Level + Trend + Seasonality dari Nol\n",
    "",
    "Tiga smoothing eksponensial yang berlapis — semuanya update per langkah:\n",
    "",
    "```",
    "level  = α·(y[t] − s[t−m]) + (1−α)·(level + trend)      # aditif",
    "trend  = β·(level_baru − level_lama) + (1−β)·trend",
    "s[t]   = γ·(y[t] − level_baru) + (1−γ)·s[t−m]",
    "ŷ[t+h] = level + h·trend + s[t+h−m]",
    "```",
    "",
    "Inisialisasi: `level₀ = mean(y[:m])`, `season₀ = y[:m] − level₀`.",
))

cells.append(code(
    "def holt_winters_add(tr, period, alpha, beta, gamma, h):\n",
    "    \"\"\"Holt-Winters aditif dari nol.\n",
    "    Return (forecast (h,), level akhir, trend akhir, season (m,)).\n",
    "    Ikuti persamaan update di atas, urutan: level -> trend -> season.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# f, lv, tb, ss = holt_winters_add(train, 365, 0.3, 0.05, 0.3, H)\n",
    "# print(f'{lv:.2f} {tb:.4f}')\n",
))

cells.append(code(
    "# ✅ Cek latihan 6.1\n",
    "fc_add, lv, tb, ss = holt_winters_add(train, 365, 0.3, 0.05, 0.3, H)\n",
    "assert fc_add.shape == (H,) and ss.shape == (365,)\n",
    "assert abs(np.sum(ss)) < 3.0, f'season aditif harus berjumlah ~0: {np.sum(ss):.3f}'\n",
    "assert 130.0 < lv < 155.0, f'level akhir harus dekat level data akhir ~142: {lv}'\n",
    "assert abs(tb) < 0.5, f'trend harus kecil (0.05/hari diredam smoothing): {tb}'\n",
    "assert 5.0 < float(np.max(np.abs(ss))) < 28.0, 'amplitudo season ~ 20'\n",
    "mae_add = mae(val, fc_add)\n",
    "print(f'✅ Latihan 6.1 benar! level={lv:.2f}, trend={tb:.4f}')\n",
    "print(f'   MAE HW aditif = {mae_add:.3f}')\n",
))

cells.append(code(
    "# Versi multiplikatif: seasonal = RASIO (amplitudo membesar bersama level)\n",
    "def holt_winters_mult(tr, period, alpha, beta, gamma, h):\n",
    "    \"\"\"Holt-Winters multiplikatif. ŷ[t+h] = (level + h·trend) · s[t+h−m].\n",
    "    Gunakan eps=1e-8 di pembagi. Inisialisasi season: y[:m]/level0.\n",
    "    Return forecast (h,).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# fc_mul = holt_winters_mult(train, 365, 0.3, 0.05, 0.3, H)\n",
    "# print(mae(val, fc_mul))\n",
))

cells.append(code(
    "fc_mul = holt_winters_mult(train, 365, 0.3, 0.05, 0.3, H)\n",
    "mae_mul = mae(val, fc_mul)\n",
    "print(f'HW aditif      MAE: {mae_add:.3f}')\n",
    "print(f'HW multiplikatif MAE: {mae_mul:.3f}')\n",
    "assert 20.7 < mae_add < 21.3, f'mae_add harus di rentang kalibrasi: {mae_add}'\n",
    "assert 16.1 < mae_mul < 16.7, f'mae_mul harus di rentang kalibrasi: {mae_mul}'\n",
    "print('✅ Di seri ADITIF ini multiplikatif justru unggul (16.4 vs 21.0)!')\n",
    "print('   Penjelasan: smoothing trend β=0.05 menghasilkan drift kecil; di model aditif')\n",
    "print('   drift itu terakumulasi penuh sepanjang horizon, di multiplikatif terredam rasio.')\n",
    "print('   Pelajaran: aditif vs multiplikatif bukan formalitas — uji KEDUANYA lewat')\n",
    "print('   backtest, dan lihat plot amplitudo terhadap level.')\n",
))

cells.append(code(
    "# Seri multiplikatif: amplitudo musiman membesar bersama level\n",
    "rng_m = np.random.default_rng(42)\n",
    "seri_mult = (100.0 + 0.05 * t) * (1.0 + 0.15 * np.sin(2 * np.pi * t / 365.0)) \\\n",
    "    + rng_m.normal(0, 4.0, len(t))\n",
    "sp2 = int(len(seri_mult) * 0.8)\n",
    "tr2, va2 = seri_mult[:sp2], seri_mult[sp2:]\n",
    "fc_a2 = holt_winters_add(tr2, 365, 0.3, 0.05, 0.3, len(va2))[0]\n",
    "fc_m2 = holt_winters_mult(tr2, 365, 0.3, 0.05, 0.3, len(va2))\n",
    "mae_a2 = mae(va2, fc_a2)\n",
    "mae_m2 = mae(va2, fc_m2)\n",
    "print(f'Seri MULTIPLIKATIF: HW add MAE = {mae_a2:.3f} | HW mult MAE = {mae_m2:.3f}')\n",
    "assert mae_m2 < mae_a2, 'multiplikatif harus menang di seri multiplikatif'\n",
    "print(f'✅ Di seri MULTIPLIKATIF: multiplikatif menang {mae_a2:.1f} vs {mae_m2:.1f}.')\n",
    "print('   Aturan praktis: amplitudo ikut naik-turun level → multiplikatif (atau log dulu).')\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)\n",
    "axes[0].plot(va2, 'k-', alpha=0.5, label='aktual')\n",
    "axes[0].plot(fc_a2, 'b--', label=f'add (MAE {mae_a2:.1f})')\n",
    "axes[0].plot(fc_m2, 'r-', label=f'mult (MAE {mae_m2:.1f})')\n",
    "axes[0].set_title('Seri multiplikatif'); axes[0].legend(); axes[0].grid(alpha=0.3)\n",
    "axes[1].plot(val, 'k-', alpha=0.5, label='aktual')\n",
    "axes[1].plot(fc_add, 'b--', label=f'add (MAE {mae_add:.1f})')\n",
    "axes[1].plot(fc_mul, 'r-', label=f'mult (MAE {mae_mul:.1f})')\n",
    "axes[1].set_title('Seri aditif'); axes[1].legend(); axes[1].grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
))

cells.append(md(
    "### Bonus: trend + dummies musiman (regresi \"pdq\")\n",
    "",
    "AR(p) tak praktis untuk musiman 365 (butuh 365 lag). Alternatif sederhana:",
    "regresi `[1, t, D_1..D_365]` dengan `D_j = 1` jika `t % 365 == j`. In-sample dia",
    "sangat bagus — dan out-of-sample? Ini demo kedua soal in-sample vs out-of-sample.",
))

cells.append(code(
    "def pdq_fit(tr, period):\n",
    "    \"\"\"Regresi trend + dummies musiman. Return beta (2+period,).\n",
    "    Kolom: [1, t, D_1..D_m] dengan D_j = 1 iff t % m == j (t >= m).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "def pdq_forecast(beta, n_train, period, h):\n",
    "    \"\"\"Forecast h langkah: b0 + b1·(n_train+j) + season[(n_train+j) % m].\n",
    "    Return array (h,).\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# beta = pdq_fit(train, 365)\n",
    "# print(beta.shape)\n",
))

cells.append(code(
    "# ✅ Cek latihan 6.2\n",
    "beta = pdq_fit(train, 365)\n",
    "assert beta.shape == (367,)\n",
    "fc_pdq = pdq_forecast(beta, len(train), 365, H)\n",
    "mae_pdq = mae(val, fc_pdq)\n",
    "assert 11.5 < mae_pdq < 12.1, f'mae_pdq harus di rentang kalibrasi: {mae_pdq}'\n",
    "assert abs(beta[1] - 0.0204) < 0.005, f'slope terestimasi ~0.0204: {beta[1]:.4f}'\n",
    "\n",
    "# in-sample vs out-of-sample — demo leakage kedua\n",
    "pred_in = np.array([beta[0] + beta[1] * i + (beta[2 + (i % 365)] if i >= 365 else 0.0)\n",
    "                    for i in range(len(train))])\n",
    "mae_pdq_in = mae(train, pred_in)\n",
    "print(f'pdq in-sample  MAE : {mae_pdq_in:.3f}')\n",
    "print(f'pdq out-sample MAE : {mae_pdq:.3f}   (hampir 2× lebih buruk!)')\n",
    "assert mae_pdq > mae_pdq_in * 1.5\n",
    "print('✅ Latihan 6.2 benar! 365 parameter \"mengunci\" noise train — keluar train, rusak.')\n",
    "print(f'   (untuk pembanding: 1-NN split acak tadi {mae_acak:.2f} vs temporal {mae_temporal:.2f})')\n",
    "\n",
    "# rangkuman semua model sejauh ini\n",
    "print('\\n=== PAPAN SKOR (val 219 hari) ===')\n",
    "papan = [('naive', mae_naive), ('seasonal-naive-365', mae_sn), ('HAM', mae_ham),\n",
    "         ('AR(7) rekursif', mae_ar7), ('HW aditif', mae_add),\n",
    "         ('HW multiplikatif', mae_mul), ('trend+dummies', mae_pdq)]\n",
    "for nama_, m_ in sorted(papan, key=lambda x: x[1]):\n",
    "    print(f'  {nama_:20s}: {m_:.3f}')\n",
))

# ============================ BAGIAN 7 ============================
cells.append(md(
    "## Bagian 7 — Transformasi Log & Metrik yang Jujur\n",
    "",
    "`log1p` mengompres skala & menstabilkan variance (multiplikatif → aditif). Tapi",
    "metrik **tidak invariant** terhadap transformasi: MAE di ruang log bukan MAE di",
    "ruang asli. Evaluasi akhir **selalu back-transform dulu** (`expm1` = invers eksak).",
))

cells.append(code(
    "y_log = np.log1p(seri)\n",
    "sp = int(len(y_log) * 0.8)\n",
    "tr_log, va_log = y_log[:sp], y_log[sp:]\n",
    "\n",
    "# naive di ruang log\n",
    "pred_log = np.concatenate([[tr_log[-1]], va_log[:-1]])   # y_hat[t] = y[t-1] di val\n",
    "mae_log = float(np.mean(np.abs(va_log - pred_log)))\n",
    "\n",
    "# back-transform SEBELUM menghitung MAE ruang asli\n",
    "mae_back = float(np.mean(np.abs(np.expm1(va_log) - np.expm1(pred_log))))\n",
    "\n",
    "print(f'MAE naive ruang log      : {mae_log:.4f}   <- JANGAN dilaporkan sebagai MAE asli!')\n",
    "print(f'MAE naive back-transform : {mae_back:.4f}   <- angka yang jujur')\n",
    "assert 0.03 < mae_log < 0.05\n",
    "assert 4.5 < mae_back < 6.0\n",
    "print('✅ Kedua angka valid tapi menjawab pertanyaan BERBEDA.')\n",
    "print('   Laporkan metrik di ruang yang sama dengan keputusan bisnis-mu.')\n",
))

cells.append(code(
    "def metrik_semua(y_true, y_pred):\n",
    "    \"\"\"MAE, RMSE, sMAPE sekaligus (dipakai lagi di kuis Soal 11).\n",
    "    smape = mean(2·|e| / (|y| + |p| + 1e-8)) — epsilon menjaga dari y=0.\n",
    "    Return dict {'mae', 'rmse', 'smape'}.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# m = metrik_semua(np.array([1., 3., 5., 7.]), np.array([2., 2., 4., 4.]))\n",
    "# print(m)  # mae=1.5, rmse≈1.732, smape≈0.4586\n",
))

cells.append(code(
    "# ✅ Cek latihan 7.1\n",
    "m = metrik_semua(np.array([1., 3., 5., 7.]), np.array([2., 2., 4., 4.]))\n",
    "assert abs(m['mae'] - 1.5) < 1e-9 and abs(m['rmse'] - np.sqrt(3.0)) < 1e-9, \\\n",
    "    f'rmse = sqrt(mean(e^2)) = sqrt(3): {m[\"rmse\"]}'\n",
    "assert abs(m['smape'] - 0.458586) < 1e-6\n",
    "m2 = metrik_semua(np.array([0., 1.]), np.array([0., 0.]))\n",
    "assert np.isfinite(m2['smape']) and abs(m2['smape'] - 1.0) < 1e-6, 'epsilon menjaga'\n",
    "m3 = metrik_semua(np.array([2., 2.]), np.array([2., 2.]))\n",
    "assert m3['mae'] == 0 and m3['rmse'] == 0 and m3['smape'] == 0\n",
    "print('✅ Latihan 7.1 benar! (fungsi yang sama diuji di kuis — copy dengan bangga)')\n",
    "print('\\nMetrik papan skor lengkap (HW aditif):')\n",
    "print(metrik_semua(val, fc_add))\n",
))

# ============================ BAGIAN 8 ============================
cells.append(md(
    "## Bagian 8 — Backtesting: Satu Split Bisa Menipu\n",
    "",
    "MAE val 219 hari tadi adalah ANGKA DARI SATU POTONGAN. Bisa jadi potongan itu",
    "\"gampang\". *Rolling backtest* menggeser titik potong mundur k kali, forecast h",
    "langkah tiap fold, lalu merata-ratakan — satu angka yang lebih stabil.",
    "",
    "```",
    "fold k: train = data[:n−(k+1)·h], test = data[n−(k+1)·h : n−k·h]",
    "```",
    "",
    "⚠️ `build_forecast` hanya boleh melihat TRAIN fold — bukan seluruh data.",
))

cells.append(code(
    "def rolling_backtest(tr, h, build_forecast, n_folds=3):\n",
    "    \"\"\"Rolling backtest: k fold, tiap fold forecast h langkah dari train fold saja.\n",
    "    Return list (mae, rmse) per fold.\n",
    "    \"\"\"\n",
    "    # TODO\n",
    "    raise NotImplementedError\n",
    "\n",
    "# --- uji sendiri ---\n",
    "# print(rolling_backtest(train, 90, buat_naive, 2))\n",
))

cells.append(code(
    "# Tiga \"pabrik forecast\" untuk backtest — menerima TRAIN fold saja\n",
    "def buat_naive(trk, h):\n",
    "    return np.full(h, float(trk[-1]))\n",
    "\n",
    "def buat_sn365(trk, h):\n",
    "    return trk[len(trk) - 365:len(trk) - 365 + h]\n",
    "\n",
    "def buat_hw(trk, h):\n",
    "    f, *_ = holt_winters_add(trk, 365, 0.3, 0.05, 0.3, h)\n",
    "    return f\n",
    "\n",
    "print('pabrik forecast siap.')\n",
))

cells.append(code(
    "# ✅ Cek latihan 8.1\n",
    "hasil = rolling_backtest(train, 90, buat_naive, 3)\n",
    "assert len(hasil) == 3\n",
    "for mae_f, rmse_f in hasil:\n",
    "    assert rmse_f >= mae_f - 1e-9, 'RMSE >= MAE (Cauchy-Schwarz)'\n",
    "hasil_sn = rolling_backtest(train, 90, buat_sn365, 3)\n",
    "hasil_hw = rolling_backtest(train, 90, buat_hw, 3)\n",
    "print('fold (mae, rmse):')\n",
    "for nama_, hs in [('naive', hasil), ('seas-naive365', hasil_sn), ('HW-add', hasil_hw)]:\n",
    "    print(f'  {nama_:14s}: {[(round(a, 2), round(b, 2)) for a, b in hs]}')\n",
    "\n",
    "mean_naive = float(np.mean([a for a, _ in hasil]))\n",
    "mean_sn = float(np.mean([a for a, _ in hasil_sn]))\n",
    "mean_hw = float(np.mean([a for a, _ in hasil_hw]))\n",
    "assert 3.0 < mean_naive < 25.0, 'naive bagus di rata-rata tapi fold-2 kena musiman'\n",
    "assert 17.0 < mean_sn < 19.0, f'seasonal-naive stabil ~18: {mean_sn}'\n",
    "assert 14.0 < mean_hw < 19.0, f'HW-add rata-rata ~16.4: {mean_hw}'\n",
    "assert mean_hw < mean_sn, 'HW-add harus mengalahkan seasonal-naive di rata-rata'\n",
    "print(f'✅ Latihan 8.1 benar! rata-rata MAE: naive {mean_naive:.2f} | seas-naive {mean_sn:.2f} | HW {mean_hw:.2f}')\n",
    "print('   Naive terlihat \"terbaik\" di rata-rata — TAPI fold-2 MAE ~21 vs ~5 di fold lain:')\n",
    "print('   satu fold yang jatuh di lembah musiman membalikkan peringkat. Keputusan butuh')\n",
    "print('   SEBARAN fold, bukan hanya mean. HW-add konsisten menang dari seasonal-naive.')\n",
))

cells.append(md(
    "**Pertanyaan pemahaman:**\n",
    "",
    "1. Kenapa `build_forecast(trk, h)` yang menerima seluruh `seri` (bukan fold train)",
    "   menghasilkan angka yang terlalu optimis? Di dunia nyata, bentuk leakage ini",
    "   biasanya masuk lewat apa? (hint: preprocessing/normalisasi)",
    "2. Model mana yang kamu pilih untuk produksi di seri ini, dan mengapa pilihan itu",
    "   berubah setelah melihat backtest dibanding setelah melihat satu split?",
))

# ============================ BAGIAN 9 ============================
cells.append(md(
    "## Bagian 9 — Data Nyata (Opsional): Sunspots & Siklus 11 Tahun",
    "",
    "Butuh internet. Dataset sunspots bulanan (1749–1983) — klasik untuk menguji",
    "pemahaman ACF: siklus aktivitas matahari ~11 tahun = 132 bulan.\n",
    "",
    "Tantangan: detrend dulu (sunspots tidak punya trend linear tapi ACF mentahnya",
    "decay karena level berubah antar siklus), lalu cari puncak ACF.",
))

cells.append(code(
    "try:\n",
    "    sunspots = np.loadtxt(\n",
    "        'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv',\n",
    "        delimiter=',', skiprows=1, usecols=1)\n",
    "    print(f'sunspots: n={len(sunspots)} bulan, min={sunspots.min():.0f}, max={sunspots.max():.0f}')\n",
    "\n",
    "    # ACF langsung\n",
    "    print(f'ACF lag 132: {acf(sunspots, 132):.3f} | ACF lag 12: {acf(sunspots, 12):.3f}')\n",
    "\n",
    "    # detrend kasar: kurangkan rata-rata bergulir 11 tahun\n",
    "    w = 132\n",
    "    kernel = np.ones(w) / w\n",
    "    ma = np.convolve(sunspots, kernel, mode='same')\n",
    "    detrended = sunspots - ma\n",
    "    acf_all = {L: acf(detrended[w:-w], L) for L in range(1, 300)}\n",
    "    kandidat = {L: v for L, v in acf_all.items() if 60 <= L <= 300}\n",
    "    puncak = max(kandidat, key=kandidat.get)\n",
    "    print(f'Puncak ACF (lag 60-300) setelah detrend: lag {puncak} (ACF {kandidat[puncak]:.3f})')\n",
    "    print(f'   ACF di lag 132 (teori 11 tahun): {acf_all[132]:.3f}')\n",
    "    assert 110 <= puncak <= 140, 'puncak harus di sekitar 123-132 (~10-11 tahun)'\n",
    "    print('✅ Siklus ~10-11 tahun menemukan dirinya sendiri lewat ACF — tanpa model ML apa pun.')\n",
    "except Exception as e:\n",
    "    print(f'⚠️ Bagian ini butuh internet — dilewati ({type(e).__name__}).')\n",
    "    print('   Angka rujukan: sunspots n=2820; ACF raw lag 132 ~0.53; setelah detrend')\n",
    "    print('   puncak ACF di lag ~123 (ACF ~0.57) — siklus ~10.3 tahun.')\n",
))

cells.append(md(
    "---",
    "",
    "## 🏁 Selesai — Lanjut ke Kuis",
    "",
    "**Yang sudah kamu bangun dari nol:** dekomposisi & ACF, windowing, split temporal,",
    "tiga baseline, AR(p) OLS + forecast rekursif, Holt-Winters aditif & multiplikatif,",
    "trend+dummies, log-transform + tiga metrik, dan rolling backtest.\n",
    "",
    "**Lanjut:** `02_kuis_time_series.ipynb` (23 poin, skor otomatis) →",
    "`project-starter-forecasting-mini/` → baca kembali `cheatsheet-time-series.md`",
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
