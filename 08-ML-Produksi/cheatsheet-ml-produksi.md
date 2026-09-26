# 📄 Cheatsheet - ML Produksi (Bab 8)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Struktur Project & Reproduksibilitas

```
proyek/            data/raw|processed | notebooks/ | src/ | tests/ | reports/
keputusan kunci    seed, split, versi data, versi library — SEMUA dicatat
```

- Notebook = eksplorasi. Produksi = module importable + test.
- **Split tiga arah:** train (fit parameter) | calib (ambil keputusan: threshold,
  Platt) | test (lapor, SEKALI). Menyetel apa pun di test = calib kedua.

```python
def split_tiga_arah(X, y, r1=0.8, r2=0.9):
    c1, c2 = int(len(X) * r1), int(len(X) * r2)
    return X[:c1], y[:c1], X[c1:c2], y[c1:c2], X[c2:], y[c2:]
```

- **Deterministik:** input sama → artefak byte sama. Bukti: `sha256_file(a) == sha256_file(b)`.
- **Persistensi + round-trip test** (wajib sebelum serving):

```python
save_model(path, w, b)                       # np.savez, dtype eksplisit
w2, b2 = load_model(path)
assert np.array_equal(predict(w2, b2, X), prediksi_sebelum)   # bit-per-bit!
```

- Normalisasi/scaler: `fit` di train saja, `transform` ke calib/test — bukan sebaliknya.

---

## 2. Metrik Regresi & Skill Score

```python
RMSE = sqrt(mean(e²))       # menghukum error besar; RMSE >= MAE selalu
MAE  = mean(|e|)            # satuan asli, robust
R²   = 1 - SS_res/SS_tot    # vs mean; BOLEH NEGATIF
```

```
skill = 1 − RMSE_model / RMSE_baseline      # baseline = HAM (mean train)

skill > 0  → ADOPT (layak dipertimbangkan)
skill = 0  → setara baseline (tidak menambah apa-apa)
skill < 0  → TOLAK (model menambah noise!)
```

- RMSE absolut tanpa pembanding = iklan. "RMSE 1.25" bermakna hanya setelah
  tahu baseline-nya (mis. HAM 4.83 → skill 0.74).
- Model yang kalah baseline: pakai baseline dulu, debug model di samping.

---

## 3. Metrik Keputusan: Confusion, P/R/F1

```
                aktual 1    aktual 0
pred ≥ thr   |   TP     |    FP    |    precision = TP/(TP+FP)   "dari yang kuprediksi+, berapa benar"
pred < thr   |   FN     |    TN    |    recall    = TP/(TP+FN)   "dari yang asli+, berapa ketemu"
                                            F1 = 2PR/(P+R)      harmonic mean
```

- Konvensi wajib: **`>= thr` = positif**; pembagi nol → metrik = 0.0 (bukan crash).
- Accuracy menipu saat kelas timpang (prevalence 1%: "selalu negatif" = akurasi 99%,
  recall 0%).

```python
def prf(cm):
    tp, fp, fn = cm['tp'], cm['fp'], cm['fn']
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'precision': p, 'recall': r, 'f1': f1}
```

- **Kenop threshold = biaya bisnis:**
  - FN mahal (screening medis, fraud) → turunkan thr → recall naik.
  - FP mahal (spam filter, auto-reject) → naikkan thr → precision naik.
  - Nilai thr dipilih di CALIB (mis. maksimasi F1), dievaluasi SEKALI di TEST.

---

## 4. Kalibrasi Probability: ECE & Platt

```
bin:  idx = min(int(p · n_bins), n_bins − 1)      # p=1.0 masuk bin terakhir
ECE = Σ (n_bin / n) · |acc − conf|                # bin kosong dilewati
```

- ECE tinggi = "90% yakin" ternyata benar 70% → keputusan berbasis risiko salah.
- **Platt scaling** = regresi logistik 1D `p = sigmoid(a·s + b)`, belajar dari
  CALIB (gradien `(p − y)`), bukan test. Memperbaiki over/under-confidence.

```python
def fit_platt(score, y01, lr=0.1, epochs=3000):
    a, b = 1.0, 0.0
    for _ in range(epochs):
        p = sigmoid(a * score + b)
        a -= lr * np.mean((p - y01) * score)
        b -= lr * np.mean(p - y01)
    return a, b
```

- Kapan kalibrasi penting: harga risiko, triase, alokasi, gabung ke keputusan manusia.
  Kapan cukup tanpa kalibrasi: ranking murni (top-k) — transformasi monoton tak
  mengubah urutan.

---

## 5. GAN — Dua Pemain, Satu Equilibrium

```
G: z ~ N(0,1) → x_fake               (pemalsu; belajar meniru)
D: x → P(x asli)                     (detektif; belajar membedakan)

loss D = −mean log D(real) − mean log(1−D(fake))      max teoretis = 2·log 2
loss G = −mean log D(fake)                            NON-saturating (gradien kuat saat D kecil)
```

- **Update bergantian:** D dulu (dua langkah: real, fake), lalu G.
- Gradien G **melewati D yang dibekukan**: logit D → hidden D → x_fake → param G.
  Di GAN penuh, pipa yang sama menembus seluruh jaringan.
- **Equilibrium:** D(real) ≈ D(fake) ≈ 0.5; kedua loss ≈ log 2; TIDAK ada loss yang
  monoton turun — dua pemain saling menjaga.
- **Init D nol = saddle point** (semua gradien nol) → init acak kecil.
- **Mode collapse:** G berkapasitas besar memetakan banyak z ke output sama →
  histogram fake jadi spike; deteksi: diversity sampel, std fake ≪ std real.
- Bukti konvergensi kecil: G menemukan mean data (lab: mu −2 → 2.1, target 2.0).

---

## 6. Kapan Pakai Apa (Decision Guide)

| Situasi | Pilihan | Alasan |
|---|---|---|
| Tabular, rows > 10rb, butuh feature importance | XGBoost/LightGBM | performa + interpretable |
| Tabular, butuh probabilitas terkalibrasi | Logistic Regression (+ Platt) | kalibrasi bagus |
| Gambar, data sedikit | Transfer learning CNN | mulai dari fitur terlatih |
| Teks pendek, latency ketat | Embedding + LR | inference cepat |
| Teks kompleks/budget ada | LLM (Bab 9+) | kualitas terbaik |
| Privasi/offline wajib | Open-weight (Bab 13) | data tak keluar |
| Tanpa label, anomali | Isolation Forest / Autoencoder | tak butuh label |

---

## 7. Mentalitas Production

```
researcher : "akurasi 98%!"
engineer   : "98% di test yang jujur, inference <50ms, memori <100MB,
              termonitor, punya fallback, artefak ber-hash, dokumentasi ada."
```

- Akurasi = SATU dari banyak metrik. Skill vs baseline, kalibrasi, latensi,
  biaya, reliabilitas sama pentingnya.
- Angka tanpa protokol = klaim. Protokol: keputusan di calib, lapor di test
  sekali, bukti hash + round-trip, backtest/monitoring berkala (Bab 16: drift).

---

## 8. Koneksi ke Bab Lain

- **Bab 3:** baseline & split jujur — di produksi baseline jadi keputusan ADOPT/TOLAK.
- **Bab 4:** backprop GAN = rantai gradien yang sama, hanya dibelah dua jaringan.
- **Bab 5 (CV):** GAN penuh = GAN 1D ini dengan Conv2DTranspose (lihat materi inti).
- **Bab 9+:** sigmoid skor → kalibrasi probability sama seperti LLM confidence.
- **Bab 16 (MLOps):** hash artefak = model registry; drift = ulangi evaluasi berkala.

---

## 9. Kesalahan Umum (90% Bug)

1. **Menyetel threshold di test** — test berubah jadi calib kedua; angka optimis palsu.
2. **Scaler di-fit di seluruh data** — statistik test bocor ke training.
3. **Membaca akurasi di kelas timpang** — pakai P/R/F1; accuracy menipu.
4. **Pembagi nol melempar exception** di serving — konvensi → 0.0.
5. **`>` bukan `>=`** di threshold — dua orang, dua angka, dari model yang sama.
6. **Binning p=1.0** keluar indeks — pakai `min(int(p·bins), bins−1)`.
7. **Init D nol** di GAN — training macet total (saddle point).
8. **Update D dan G dari loss yang sama** — dua pemain punya dua loss berbeda.
9. **Klaim "artefak sama" tanpa hash** — verifikasi byte, bukan kepercayaan.
