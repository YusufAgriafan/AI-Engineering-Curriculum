# 📄 Cheatsheet — Machine Learning Klasik (Bab 3)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Regresi Linear & Logistic — Dua Saudara

| | Regresi Linear | Logistic Regression |
|---|---|---|
| Output | kontinu (harga, suhu) | probabilitas 0–1 |
| Forward | `ŷ = Xw + b` | `p = σ(Xw + b)` |
| Loss | MSE: `mean((ŷ−y)²)` | BCE: `−mean(y·log p + (1−y)·log(1−p))` |
| Gradien | `Xᵀ(ŷ−y)·2/m` | `Xᵀ(p−y)/m` |
| Dipakai | prediksi nilai | klasifikasi biner |

```python
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))   # cabang aman
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)                # hindari overflow exp(700+)
    return out
```

**Loop training universal** (linear, logistic, NN — semua sama):

```python
for _ in range(n_iter):
    pred = forward(X)                       # linear / sigmoid / dst.
    loss.append(loss_fn(y, pred))           # catat — harus monoton turun
    grad = X.T @ (pred - y) / m             # "prediksi minus kenyataan"
    w -= lr * grad                          # langkah melawan gradien
```

- Loss naik → `lr` terlalu besar. Loss turun sangat lambat → `lr` terlalu kecil.
- BCE butuh clipping: `p = np.clip(p, 1e-15, 1 - 1e-15)` sebelum `log`.

---

## 2. Metrik — Hafalan Satu Menit

```
TP prediksi 1, benar  |  FP prediksi 1, salah  (false alarm)
FN prediksi 0, salah  |  TN prediksi 0, benar

accuracy  = (TP+TN)/total          → menipu di data imbalanced!
precision = TP/(TP+FP)             → "dari yang kuprediksi positif, benar?"
recall    = TP/(TP+FN)             → "dari yang benar-benar positif, ketemu?"
F1        = 2PR/(P+R)              → harmonik: rendah salah satunya = F1 rendah
ROC-AUC   → P(skor(pos) > skor(neg)) → menilai RANKING, tak terpengaruh threshold
```

| Situasi | Utamakan |
|---|---|
| Deteksi penyakit / fraud | **recall** (jangan lewatkan kasus) |
| Spam filter / autonomination kandidat | **precision** (jangan salah menuduh) |
| Biaya FP ≪ biaya FN | threshold **rendah** (recall naik) |
| Kelas sangat imbalanced | Lupakan accuracy; lihat P/R/F1 + AUC |

**Guard wajib:** `TP+FP == 0 → precision 0.0`, `TP+FN == 0 → recall 0.0` (konvensi sklearn — bukan error).

**ROC-AUC via rank (Mann-Whitney):** rank semua skor (tie → rata-rata rank), lalu
`AUC = (Σ rank(pos) − n_pos(n_pos+1)/2) / (n_pos·n_neg)`.

---

## 3. Split Data — Hukum yang Tidak Bisa Dilanggar

```
TRAIN → belajar parameter (fit bobot)
VAL   → pilih hyperparameter / threshold / arsitektur / early stop
TEST  → SEKALI di paling akhir, seperti ujian negara

Melanggar = data leakage = metrik yang TIDAK BERARTI apa-apa.
Gejala leakage: val/test "terlalu bagus", turun drastis di produksi.
```

- Selalu bandingkan dengan **baseline bodoh** dulu: prediksi kelas mayoritas.
  Accuracy 0.91 vs baseline 0.89 = selisih tipis; precision/recall-lah yang membuktikan model belajar.
- Split stratified (jaga proporsi kelas) untuk klasifikasi imbalanced.

---

## 4. Bias–Variance & Regularisasi

| Gejala | Diagnosis | Obat |
|---|---|---|
| train & val loss sama-sama tinggi, gap kecil | **underfit** (high bias) | fitur lebih, model lebih kompleks, train lebih lama |
| train loss rendah, val loss tinggi, **gap besar** | **overfit** (high variance) | lebih banyak data, regularisasi, model sederhana, early stop |
| train turun terus, val berbalik naik | titik overfit mulai | berhenti di minimum val — pilih model di VAL |

```python
# Ridge (L2): w = (XᵀX + λI)⁻¹ Xᵀy  — λ=0 → OLS biasa; λ↑ → ‖w‖ menyusut monoton
w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
```

- Fitur polinomial: `x[:, None] ** np.arange(1, d+1)` — model tetap "linear terhadap parameter".
- Derajat naik → train MSE **monoton turun**; val MSE bentuk U → pilih derajat di lembah (val, bukan train).

---

## 5. Tree & Ensemble

```
Gini(node) = 1 − Σ pₖ²          # pₖ = proporsi kelas k
  murni → 0 ; campur 50:50 → 0.5

Split terbaik = weighted Gini terkecil:
  G = (n_kiri/n)·Gini(kiri) + (n_kanan/n)·Gini(kanan)
```

| | Random Forest | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| Cara | banyak tree paralel, bootstrap + fitur acak, hasil **vote/rata-rata** | tree berurutan, tiap tree memperbaiki **residual** sebelumnya |
| Kuat | stabil, anti-overfit, mudah di-tune | sering paling akurat di data tabular |
| Hati-hati | besar & lambat saat predict | overfit cepat kalau lr/depth tidak dijaga |

- Single tree depth besar = overfit mesin; forest/boosting menalarisasi via rata-rata.
- Ensemble tabular masih menjadi baseline kuat bahkan di era LLM — murah, cepat, deterministic.

---

## 6. K-Means — Algoritma 4 Baris

```python
cent = X[rng.choice(len(X), k, replace=False)]          # init acak
for _ in range(n_iter):
    d2 = ((X[:, None, :] - cent[None]) ** 2).sum(-1)    # (n, k) jarak²
    lab = d2.argmin(-1)                                 # ASSIGN
    cent = np.array([X[lab == j].mean(0) if (lab == j).any() else cent[j]
                     for j in range(k)])                # UPDATE (guard klaster kosong)
```

- **Inertia** = Σ jarak² titik ke centroid-nya → selalu pakai `n_init` percobaan, ambil terkecil.
- Standardisasi fitur (`(x−mean)/std`) SEBELUM k-means — jarak tak boleh didominasi satu skala.
- K-means butuh label 0..k-1 tapi urutannya acak → kalau dibanding ground truth, perlu matching.
- Interpretasi klaster = melihat centroid + rata-rata target per klaster (unsupervised tetap bisa dapat insight supervised).

---

## 7. Pola Kode yang Berulang

```python
# Metrik via boolean masking — tanpa loop:
TP = int(((y_true == 1) & (y_pred == 1)).sum())

# Guard clause (muncul terus):
if TP + FP == 0: return 0.0
if len(labels) == 0: return 0.0
if n_pos == 0 or n_neg == 0: raise ValueError(...)

# Broadcasting geometri: (n,1,d) − (1,k,d) → (n,k,d) → sum → (n,k) → argmin
d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)

# np.asarray di awal fungsi = terima list maupun array:
x = np.asarray(x, dtype=float)
```

---

## 8. Koneksi ke Bab Lain

| Konsep Bab 3 | Dipakai di |
|---|---|
| Sigmoid + BCE | Output layer NN (4), klasifikasi CV (5) |
| Gradient descent penuh | Training TF (4), fine-tuning (13) |
| Train/val/test + threshold | Evaluasi SEMUA model, termasuk LLM (15) |
| Precision/recall & imbalanced | Deteksi anomali (7), evaluasi retrieval (12) |
| Gini / information gain | Feature importance, klasifikasi intent (14) |
| K-means + cosine | Clustering embedding, RAG (12) |
| Regularisasi / early stop | Melatih NN (4) — callback `EarlyStopping` |

> Uji diri terakhir: tulis rumus precision, recall, dan gradien logistic regression dari ingatan. Berhasil tiga-tiganya? Siap kuis.
