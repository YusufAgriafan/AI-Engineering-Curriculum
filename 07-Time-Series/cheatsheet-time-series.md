# 📄 Cheatsheet - Time Series (Bab 7)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Anatomi Time Series - Dekomposisi

```
y[t] = trend[t] + seasonal[t] + noise[t]

trend     : pergerakan jangka panjang (slope, perubahan level)
seasonal  : pola berulang periode tetap m (harian 24, mingguan 7, tahunan 365/12)
noise     : sisa acak - yang tidak bisa dijelaskan model
```

- Detrend cepat: fit polinomial pada waktu, kurangkan; atau differencing `z[t] = y[t] - y[t-1]`.
- ACF = korelasi `y[t]` vs `y[t-k]`:

```python
def acf(x, lag):
    x = x - x.mean()
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x))
```

- **Membaca ACF:** decay pelan tanpa puncak = trend; puncak berulang tiap m = musiman;
  ~0 di semua lag = noise murni. Trend "menutupi" ACF - **detrend dulu sebelum membaca musiman**
  (bukti: sunspots - ACF mentah decay, setelah detrend puncak di lag ~125 = siklus 11 tahun).

---

## 2. Windowing - Fitur & Label dari Urutan Waktu

```
X[i] = y[i : i+W]        (window W ke belakang)
y[i] = y[i + W]          (target tepat 1 langkah ke depan)
n_sampel = len(series) - W
```

```python
def buat_window(series, W):
    s = np.asarray(series, dtype=float)
    X = np.lib.stride_tricks.sliding_window_view(s, W)[:-1]
    return X, s[W:]
```

- **Split WAJIB berurutan**: `cut = int(n * 0.8)` → train `[:cut]`, val `[cut:]`.
  Split acak = titik masa depan bocor ke training = **data leakage temporal** - skor val
  bagus tapi model bodoh di produksi.
- Horizon: `y[i] = y[i+W : i+W+h]` untuk h langkah ke depan; multi-step rekursif
  mengakumulasi error (lihat §6).

---

## 3. Baseline - Wajib Dikalahkan

| Baseline | Prediksi | Kapan kuat |
|---|---|---|
| Naive | `y_hat[t+1] = y[t]` (nilai terakhir) | random walk, noise dominan |
| Seasonal naive | `y_hat[t+h] = y[t + h - m]` (satu siklus lalu) | musiman dominan & stabil |
| HAM (rata-rata) | `y_hat = mean(train)` | stationer tanpa pola |

- Model yang tidak mengalahkan baseline **belum layak dipakai** - ia menambah noise.
- MAE naive ≠ 0 walau noise σ kecil: MAE_naive ≈ 0.8·σ_noise untuk lag-1.

---

## 4. AR(p) - Regresi Linear pada Lag

```
y[t] = φ1·y[t-1] + φ2·y[t-2] + ... + φp·y[t-p] + ε[t]
```

```python
def fit_ar_ols(s, p):
    X = np.column_stack([s[p-k-1 : len(s)-k-1] for k in range(p)])  # kolom k = lag k+1
    y = s[p:]
    return np.linalg.lstsq(X, y, rcond=None)[0]

def ar_forecast(coef, last_values, h):        # rekursif multi-step
    hist = list(last_values)[::-1]            # hist[0] = y[t-1], hist[1] = y[t-2], ...
    out = []
    for _ in range(h):
        yhat = float(coef @ np.array(hist[:len(coef)]))
        out.append(yhat)
        hist.insert(0, yhat)                  # prediksi jadi input berikutnya
    return np.array(out)
```

- Fitur = lag (masa lalu), target = nilai sekarang. Data menyusut: n → n-p baris.
- **Multi-step rekursif menumpuk error** - MAE 1-step kecil ≠ MAE h-step kecil.
- AR pada seasonality panjang (365) tidak praktis → pakai dekomposisi (§5) atau
  dummies musiman: regresi `[1, t, D_1..D_m]` dengan `D_j = 1` jika `t % m == j`.

---

## 5. Holt-Winters (Triple Exponential Smoothing)

```python
# update per langkah (versi aditif):
level   = α·(y[t] - s[t-m]) + (1-α)·(level + trend)
trend   = β·(level_baru - level_lama) + (1-β)·trend
s[t]    = γ·(y[t] - level_baru) + (1-γ)·s[t-m]
# forecast h langkah:
y_hat[t+h] = level + h·trend + s[t+h-m]        # aditif
y_hat[t+h] = (level + h·trend) · s[t+h-m]      # multiplikatif
```

- **Aditif** (`+ seasonal`): amplitudo fluktuasi KONSTAN di semua level.
- **Multiplikatif** (`× seasonal`): amplitudo membesar bersama level (rasio, ±10%).
  Setara: log dulu (`log y`), lalu model aditif.
- Inisialisasi: `level₀ = mean(y[:m])`, `season₀ = y[:m] - level₀`.
- α/β/γ ∈ [0,1]: tinggi = responsif tapi berisik; rendah = halus tapi tertinggal.

---

## 6. Transformasi & Metrik

```python
z = np.log1p(y)      # kompres skala; stabilkan variance; multiplikatif -> aditif
y = np.expm1(z)      # invers EKSAK - selalu back-transform SEBELUM menghitung MAE
```

| Metrik | Rumus | Catatan |
|---|---|---|
| MAE | `mean(|e|)` | satuan asli, robust |
| RMSE | `sqrt(mean(e²))` | menghukum error besar |
| MAPE | `mean(|e/y|)` | MELEDAK saat y_true→0 |
| sMAPE | `mean(2|e| / (|y|+|p|+ε))` | aman dari nol, hasil di [0, 2] |
| MASE | MAE / MAE_naive(train) | < 1 = kalahkan naive |

- Karena metrik TIDAK invariant terhadap transformasi: evaluasi di ruang asli.

---

## 7. Backtesting - Menilai Model Sebagai Sistem

```
split tunggal  : satu potongan val - bisa kebetulan (untung/naas)
rolling backtest: geser cut mundur k kali, forecast h langkah tiap fold,
                  rata-ratakan metrik - satu angka yang stabil
```

```python
def rolling_backtest(train, h, build_forecast, n_folds=3):
    metrik = []
    for k in range(n_folds):
        cut = len(train) - (k + 1) * h
        tr, te = train[:cut], train[cut:cut + h]
        fc = build_forecast(tr, h)            # HANYA boleh lihat tr
        metrik.append((np.mean(np.abs(te - fc)), np.sqrt(np.mean((te - fc)**2))))
    return metrik
```

- `build_forecast` menerima TRAIN fold saja - leakage di sini menodai semua angka.
- Pilih model dari rata-rata backtest, bukan dari satu split keberuntungan.

---

## 8. Koneksi ke Bab Lain

- **Bab 3 (ML Klasik):** "bandingkan dengan baseline" - di time series baseline-nya naive.
- **Bab 4/6 (Sequence):** windowing time series = windowing teks; RNN/LSTM yang sama,
  bedanya fiturnya angka terukur, bukan id kata.
- **Bab 9 (LLM):** forecasting rekursif = autoregressive generation - prediksi jadi input.
- **Bab 16 (MLOps):** backtesting = protokol evaluasi model di produksi; drift data =
  pola berubah setelah model dipasang - ulangi backtest berkala.

---

## 9. Kesalahan Umum (90% Bug)

1. **Split acak** di time series - leakage temporal.
2. **Membangun fitur normalisasi/scaler dari seluruh data** - harus dari train saja.
3. **MAE dihitung di ruang log** (lupa back-transform) - angka tidak sebanding satuan asli.
4. **AR forecast lupa urutan hist** - `hist[0]` harus selalu nilai TERBARU.
5. **Membandingkan model di horizon berbeda** - 1-step vs 90-step itu buah yang berbeda.
6. **Sumber data diubah setelah angka dicatat** - kunci seed, kunci split, kunci data.
