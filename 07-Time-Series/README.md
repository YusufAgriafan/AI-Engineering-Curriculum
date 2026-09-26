# Bab 7 — Time Series

> Prediksi masa depan dari data historis — keahlian yang juga melatih intuisi *windowing*, *labeling*, dan *forecasting*, relevan untuk analitik dan monitoring sistem AI.

## 🎯 Tujuan Belajar
- Statistik time series: tren, musiman, noise.
- Windowing: fitur & label dari data berurutan.
- DNN, RNN, LSTM untuk forecasting.
- Metrik evaluasi forecasting.

## 1. Materi Inti

> Prediksi masa depan dari data historis — keahlian yang juga melatih intuisi *windowing*, *labeling*, dan *forecasting*, relevan untuk analitik bisnis dan monitoring sistem AI.

---

### 1.1 Anatomi Time Series

#### Komponen Utama

```
Time Series = Trend + Seasonality + Noise

♦ Trend: pola jangka panjang (naik/turun/stabil)
  Contoh: populasi Indonesia terus naik

♦ Seasonality: pola berulang dalam periode tertentu
  Contoh: penjualan ritel naik tiap Natal, turun tiap pasca-lebaran
  └─ Periode: harian (traffic website), mingguan, bulanan, tahunan

♦ Noise: variasi acak yang tidak bisa dijelaskan model
  Contoh: gangguan acak, error pengukuran, kejadian tak terduga
```

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Buat time series sintetis
days = np.arange(365 * 3)  # 3 tahun
trend = 100 + 0.05 * days          # trend naik pelan
seasonality = 20 * np.sin(2 * np.pi * days / 365)  # musiman tahunan
noise = np.random.normal(0, 5, len(days))           # noise acak

series = trend + seasonality + noise

plt.figure(figsize=(14, 6))
plt.plot(days, series, alpha=0.7, label='Time Series Total')
plt.plot(days, trend, 'r--', linewidth=2, label='Trend')
plt.plot(days, trend + seasonality, 'g:', linewidth=2, label='Trend + Seasonality')
plt.xlabel('Hari')
plt.ylabel('Nilai')
plt.title('Anatomi Time Series: Trend + Seasonality + Noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### Autocorrelation — Hubungan dengan Masa Lalu

```
Autocorrelation = korelasi antara series dengan versi tertunda (lag)

Lag 0: correlation(series[t], series[t]) = 1.0 (sama persis)
Lag 1: correlation(series[t], series[t-1]) → seberapa mirip hari ini dengan kemarin
Lag 7: correlation(series[t], series[t-7]) → korelasi mingguan

Grafik ACF (Autocorrelation Function) menunjukkan pola:
  - Puncak di lag tertentu → ada seasonality
  - Decaying pelan → ada trend
  - Nol di semua lag → noise murni
```

---

### 1.2 Windowing — Kunci Utama Time Series ML

#### Konsep Window

```
❌ Salah (time series):
  X = [sembarang data points], y = [sembarang labels]
  → Split acak → data masa depan masuk training → hasil tidak berarti

✓ Benar (time series):
  X = [t-59, t-58, ..., t]   (window 60 hari terakhir)
  y = [t+1]                  (prediksi hari berikutnya)
  → Gunakan urutan waktu secara benar
```

```python
import numpy as np

def create_windowed_dataset(series, window_size=60, horizon=1):
    """
    Buat dataset windowed dari time series.
    
    Args:
        series: 1D array time series
        window_size: berapa langkah ke belakang sebagai fitur
        horizon: berapa langkah ke depan sebagai target
    
    Returns:
        X: array shape (n_samples, window_size)
        y: array shape (n_samples, horizon)
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        window = series[i:i+window_size]
        target = series[i+window_size:i+window_size+horizon]
        X.append(window)
        y.append(target)
    
    return np.array(X), np.array(y)

# Contoh penggunaan
series = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)

X, y = create_windowed_dataset(series, window_size=50, horizon=1)

print(f"Original series length: {len(series)}")
print(f"Number of windows: {len(X)}")
print(f"X shape: {X.shape}  ← (samples, window_size)")
print(f"y shape: {y.shape}  ← (samples, horizon)")

# Visualisasi windowing
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(series, alpha=0.7)
plt.axvspan(X[100, 0], X[100, -1], alpha=0.3, color='blue', label='Window 100')
plt.axvline(x=X[100, -1], color='red', linestyle='--', label='End of window')
plt.title('Window yang digunakan sebagai X')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(series, alpha=0.7)
plt.axvspan(X[100, 0], X[100, -1], alpha=0.3, color='blue')
plt.scatter([X[100, -1] + 1], [y[100, 0]], color='red', s=100, zorder=5, label='Target (y)')
plt.title('Window + Target (horizon=1)')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Train/Val Split — Urut Waktu, Bukan Acak!

```python
# ❌ SALAH: split acak
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)  # TIDAK DIPAKAI!

# ✓ BENAR: split berbasis waktu
split_idx = int(len(X) * 0.8)  # 80% untuk training
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
print(f"Train time range: 0 → {split_idx}")
print(f"Val time range: {split_idx} → {len(X)}")
```

---

### 1.3 Model Forecasting

#### Baseline Naive — Wajib Dikalahkan

```python
# Baseline 1: "Naive" — prediksi = nilai terakhir yang diketahui
def naive_forecast(series, window_size, horizon):
    X, y_true = create_windowed_dataset(series, window_size, horizon)
    y_pred = np.array([series[i+window_size-1] for i in range(len(X))])
    return y_true, y_pred

# Baseline 2: "Seasonal naive" — prediksi = nilai 1 siklus lalu
def seasonal_naive_forecast(series, season_period, horizon=1):
    X, y_true = create_windowed_dataset(series, season_period, horizon)
    y_pred = np.array([series[i] for i in range(len(X))])
    return y_true, y_pred

# Evaluasi baseline
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_true, y_pred_naive = naive_forecast(series, window_size=50, horizon=1)
mae_naive = mean_absolute_error(y_true, y_pred_naive)

print(f"Naive forecast MAE: {mae_naive:.4f}")
print(f"Ini adalah baseline yang harus dikalahkan oleh model manapun")
```

#### DNN untuk Forecasting

```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Siapin data untuk neural network
X_train_nn = X_train.reshape(-1, 50, 1).astype('float32')
X_val_nn = X_val.reshape(-1, 50, 1).astype('float32')

# Model 1: Dense sederhana (flatten window → predict)
model_dense = keras.Sequential([
    layers.Flatten(input_shape=(50, 1)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Model 2: RNN/LSTM
model_lstm = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(50, 1)),
    layers.LSTM(32),
    layers.Dense(1)
])

for name, model in [('Dense', model_dense), ('LSTM', model_lstm)]:
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print(f"\nTraining {name} model...")
    history = model.fit(
        X_train_nn, y_train,
        epochs=20,
        validation_data=(X_val_nn, y_val),
        verbose=0
    )
    print(f"{name} - Val MAE: {history.history['val_mae'][-1]:.4f}")
```

#### Perbandingan Semua Model

```python
# Prediksi semua model
y_pred_dense = model_dense.predict(X_val_nn, verbose=0).flatten()
y_pred_lstm = model_lstm.predict(X_val_nn, verbose=0).flatten()

# Evaluasi
results = {
    'Naive Baseline': mean_absolute_error(y_val, y_pred_naive),
    'Dense NN': mean_absolute_error(y_val, y_pred_dense),
    'LSTM': mean_absolute_error(y_val, y_pred_lstm),
}

print("\n" + "="*50)
print("PERBANDINGAN MODEL FORECASTING")
print("="*50)
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:20s}: MAE = {mae:.4f}")

best_model = min(results, key=results.get)
print(f"\nModel terbaik: {best_model} dengan MAE {results[best_model]:.4f}")
```

---

### 1.4 Pilihan Model & Kapan Pakai

| Model | Kelebihan | Kekurangan | Kapan |
|---|---|---|---|
| Naive / Seasonal Naive | Sangat sederhana, cepat | Tidak belajar pola kompleks | Baseline wajib; jika tidak ada pola kompleks |
| ARIMA / SARIMA | Statistik solid, interpretable | Tidak scalable ke data besar, perlu stationary | Series pendek, analisis statistik |
| DNN Dense | Belajar pola non-linear, cepat training | Tidak menangani sequence dependency dengan baik | Series dengan pola non-linear tapi konteks pendek |
| RNN / LSTM | Contextual memory, sequence dependency | Lambat training, vanishing gradient (LSTM kurang) | Series panjang dengan dependensi temporal |
| Transformer (Time Series) | State-of-the-art di banyak task | Kompleks, butuh data lebih banyak | Task kompleks dengan data besar |
| Prophet / Holt-Winters | Musiman & trend, mudah digunakan | Kaku, tidak fleksibel untuk pola kompleks | Metrik bisnis dengan seasonality kuat |

---

### 1.5 Metrik Evaluasi Forecasting

| Metrik | Rumus | Kapan Dipakai |
|---|---|---|
| **MAE** (Mean Absolute Error) | Σ|y_pred - y_true| / n | Interpretable: "rata-rata error dalam satuan asli" |
| **RMSE** (Root Mean Squared Error) | √(Σ(y_pred - y_true)² / n) | Penalize error besar lebih parah |
| **MAPE** (Mean Absolute % Error) | Σ|((y_pred - y_true) / y_true)| / n × 100% | Interpretasi persen, tapi bermasalah kalau y_true=0 |
| **sMAPE** | Symmetric version of MAPE | Lebih fair, tapi masih ada kelemahan |

> **Koneksi ke Bab 3:** Sama seperti klasifikasi/regression di Bab 3, selalu bandingkan dengan baseline yang masuk akal.

---

### 1.6 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 7:
  1. Anatomi time series: trend, seasonality, noise
  2. Autocorrelation: lihat hubungan dengan masa lalu
  3. Windowing: fitur = window ke belakang, label = nilai ke depan
  4. Split WAJIB urut waktu, bukan acak!
  5. Baseline: naive (nilai terakhir) & seasonal naive
  6. Model: dari Dense → RNN/LSTM → Transformer
  7. Metrik: MAE, RMSE, MAPE — selalu bandingkan dengan baseline
  8. Koneksi ke Bab lain: windowing mirip sequence model di Bab 6
```

---

## 2. Latihan Praktis

---

### Latihan 1: Analisis & Visualisasi Time Series

**Tujuan:** Pahami komponen time series dan cara mendeteksinya.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Gunakan dataset Sunspots (klasik di time series)
sunspots = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv', 
                      delimiter=',', skiprows=1)[:, 1]

print(f"Sunspots dataset: {len(sunspots)} bulan")
print(f"Range: {sunspots.min():.0f} ― {sunspots.max():.0f} sunspots")

# Visualisasi time series
plt.figure(figsize=(14, 6))
plt.plot(sunspots, linewidth=0.8, alpha=0.8)
plt.xlabel('Bulan')
plt.ylabel('Jumlah Sunspots')
plt.title('Monthly Sunspots (1749-1983)')
plt.grid(True, alpha=0.3)
plt.show()

# Decomposisi manual (perkiraan)
from scipy.signal import detrend

# 1. Detrend
trend_component = np.poly1d(np.polyfit(range(len(sunspots)), sunspots, 2))(range(len(sunspots)))
detrended = sunspots - trend_component

# 2. Estimasi seasonality (period ~11 tahun = 132 bulan)
season_period = 132
seasonal_component = np.zeros(len(sunspots))
for i in range(season_period):
    mask = np.arange(i, len(sunspots), season_period)
    seasonal_component[mask] = np.mean(detrended[mask])

# 3. Noise
noise_component = detrended - seasonal_component[:len(detrended)]

# Plot decomposisi
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(sunspots, 'b-', alpha=0.7)
plt.title('Original Series')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(trend_component, 'r-', linewidth=2)
plt.title('Trend Component')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 3)
plt.plot(seasonal_component[:len(sunspots)], 'g-', alpha=0.7)
plt.title(f'Seasonal Component (period={season_period} bulan)')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(noise_component, 'gray', alpha=0.5, linewidth=0.5)
plt.title('Noise Component')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Autocorrelation Plot
plt.figure(figsize=(12, 4))
plot_acf(sunspots, lags=50, title='Autocorrelation Function (ACF)')
plt.show()
```

---

### Latihan 2: All Models Forecasting Competition

**Tujuan:** Latih dan bandingkan berbagai model untuk task forecasting yang sama.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Gunakan dataset Sunspots
sunspots_raw = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv', 
                          delimiter=',', skiprows=1)[:, 1]

# Normalisasi
sunspots = sunspots_raw / sunspots_raw.max()

# Windowing
WINDOW_SIZE = 24   # 2 tahun terakhir
HORIZON = 1       # prediksi 1 bulan ke depan

def create_windows(series, window_size, horizon):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size:i+window_size+horizon])
    return np.array(X), np.array(y)

X, y = create_windows(sunspots, WINDOW_SIZE, HORIZON)

# Split time-based
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Baseline: Naive (prediksi nilai terakhir di window)
y_pred_naive = np.array([X_val[i, -1] for i in range(len(X_val))])

# Model 1: Simple Dense
model_dense = keras.Sequential([
    layers.Flatten(input_shape=(WINDOW_SIZE,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_dense.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model 2: Deep Dense
model_deep = keras.Sequential([
    layers.Flatten(input_shape=(WINDOW_SIZE,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_deep.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model 3: LSTM
model_lstm = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    layers.LSTM(32),
    layers.Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

history_dense = model_dense.fit(X_train, y_train, epochs=30, 
                                  validation_data=(X_val, y_val),
                                  callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                                  verbose=0)

history_deep = model_deep.fit(X_train, y_train, epochs=30,
                               validation_data=(X_val, y_val),
                               callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                               verbose=0)

# LSTM butuh reshape
X_train_lstm = X_train.reshape(-1, WINDOW_SIZE, 1)
X_val_lstm = X_val.reshape(-1, WINDOW_SIZE, 1)

history_lstm = model_lstm.fit(X_train_lstm, y_train, epochs=30,
                               validation_data=(X_val_lstm, y_val),
                               callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                               verbose=0)

# Prediksi & Evaluasi
y_pred_dense = model_dense.predict(X_val, verbose=0).flatten()
y_pred_deep = model_deep.predict(X_val, verbose=0).flatten()
y_pred_lstm = model_lstm.predict(X_val_lstm, verbose=0).flatten()

results = {
    'Naive': mean_absolute_error(y_val, y_pred_naive),
    'Dense': mean_absolute_error(y_val, y_pred_dense),
    'Deep Dense': mean_absolute_error(y_val, y_pred_deep),
    'LSTM': mean_absolute_error(y_val, y_pred_lstm),
}

print("\n" + "="*50)
print("HASIL FORECASTING (MAE)")
print("="*50)
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:15s}: MAE = {mae:.4f}")

# Visualisasi perbandingan prediksi
plt.figure(figsize=(14, 6))

# Plot sebagian validation
n_plot = min(200, len(X_val))
indices = np.arange(len(y_val))[-n_plot:]

plt.subplot(1, 2, 1)
plt.plot(indices, y_val[-n_plot:], 'b-', label='Actual', linewidth=1.5)
plt.plot(indices, y_pred_naive[-n_plot:], 'gray', label='Naive', linewidth=1)
plt.plot(indices, y_pred_dense[-n_plot:], 'g--', label='Dense', linewidth=1)
plt.plot(indices, y_pred_lstm[-n_plot:], 'r-', label='LSTM', linewidth=1)
plt.xlabel('Sample index')
plt.ylabel('Normalized Sunspots')
plt.legend()
plt.title(f'Forecast Comparison (last {n_plot} samples)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
models = list(results.keys())
maes = [results[m] for m in models]
colors = ['gray', 'green', 'orange', 'red']
bars = plt.bar(models, maes, color=colors)
plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('Model Comparison (MAE)')
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{mae:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\nModel terbaik: {min(results, key=results.get)}")
```

---

### Latihan 3: Pipeline Windowing untuk Data Nyata

**Tujuan:** Terapkan windowing ke data yang kamu punya atau data publik.

Ide data yang bisa dipakai:
- Data harga saham/harian (Yahoo Finance)
- Data lalu lintas harian
- Data penjualan bulanan
- Data suhu harian
- Data COVID case harian

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download data harga saham (contoh: Apple)
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
close_prices = data['Close'].values

print(f"Data: {len(close_prices)} hari harga close Apple")
print(f"Range: ${close_prices.min():.2f} ― ${close_prices.max():.2f}")

# Visualisasi
plt.figure(figsize=(14, 6))
plt.plot(close_prices, linewidth=1)
plt.xlabel('Hari')
plt.ylabel('Harga Close ($)')
plt.title(f'{ticker} Daily Closing Price (2020-2023)')
plt.grid(True, alpha=0.3)
plt.show()

# Windowing custom
def custom_windowing(series, window_size, horizon):
    """Custom windowing dengan flexibility lebih."""
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        # Fitur: window harga
        window = series[i:i+window_size]
        
        # Target: harga H hari ke depan
        target = series[i+window_size:i+window_size+horizon]
        
        # Opsional: tambah fitur tambahan (mis. return, moving average)
        return_window = (window[-1] - window[0]) / window[0]  # return selama window
        
        X.append([*window, return_window])  # harga + return sebagai fitur
        y.append(target[0] if horizon == 1 else target)
    
    return np.array(X), np.array(y)

WINDOW = 60  # 60 hari terakhir
HORIZON = 5  # prediksi 5 hari ke depan

X, y = custom_windowing(close_prices, WINDOW, HORIZON)

print(f"\nWindowing result:")
print(f"  X shape: {X.shape}  ← (samples, window_size+1 fitur)")
print(f"  y shape: {y.shape}  ← (samples, horizon)")

# Split time-based
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
```

> **Tugas lanjutan:**
> 1. Bangun model untuk memprediksi harga 5 hari ke depan
> 2. Bandingkan dengan baseline "prediksi = harga hari ini"
> 3. Evaluasi dengan MAE dan RMSE
> 4. Visualisasikan prediksi vs actual untuk 30 hari terakhir

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_time_series.ipynb` | Lab praktikum: 9 bagian — dekomposisi + ACF, windowing, split temporal vs acak (demo leakage), baseline, AR(p) rekursif, Holt-Winters aditif/multiplikatif, log-transform + metrik, rolling backtest, sunspots opsional | Kerjakan setelah baca materi inti |
| `02_kuis_time_series.ipynb` | 12 soal (PG + coding), 23 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_time_series.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-time-series.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-forecasting-mini/` | Proyek end-to-end: pipeline forecasting dari nol — windowing + split temporal + baseline + AR(p) + Holt-Winters + backtest (TDD, 40 test) + starter/solusi + RUBRIK | Setelah kuis ≥ 18/23 |

---

## 📚 Referensi Online

| Sumber | Topik | Keterangan |
|---|---|---|
| [TensorFlow Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series) | Tutorial resmi | Panduan step-by-step dengan TensorFlow |
| [Forecasting: Principles and Practice (Hyndman)](https://otexts.com/fpp3/) | Buku gratis | Rujukan terbaik untuk forecasting; sangat komprehensif |
| [statsmodels — Time Series Analysis](https://www.statsmodels.org/stable/index.html) | Library Python | ARIMA, ACF/PACF, dekomposisi, dan tools statistik lain |
| [Kaggle Time Series Course](https://www.kaggle.com/learn/time-series) | Kursus gratis | Singkat dan praktis |

---

## ✅ Checklist Kompetensi

- [ ] Membuat windowed dataset dari raw time series tanpa melihat lab
- [ ] Menjelaskan kenapa split acak ilegal di time series
- [ ] Selalu membandingkan model dengan naive baseline
- [ ] Visualisasikan time series dan dekomposisinya
- [ ] Implementasi minimal 2 arsitektur model (Dense, LSTM, dll)
- [ ] Evaluasi dengan MAE/RMSE dan interpretasi hasil
- [ ] Tulis pipeline windowing dari nol untuk data baru
