# 📊 Rubrik Penilaian Mandiri — Forecasting Mini dari Nol

> Isi dengan **jujur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `windowing.py` — window 1-step & multi-step benar, split temporal tanpa shuffle | | | |
| `baseline.py` — tiga metrik (incl. sMAPE epsilon-aman) + tiga baseline | | | |
| `ar.py` — AR(p) OLS konvensi lag benar + forecast rekursif | | | |
| `holtwinters.py` — aditif & multiplikatif, urutan update level→trend→season | | | |
| Angka di starter masuk rentang kalibrasi (naive ~9.2, HW ~16–21, pdq ~11.8) | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Baseline dicatat SEBELUM klaim model lebih baik | | | |
| APE 7-langkah vs MAE 219-langkah AR(7) dibandingkan & dijelaskan | | | |
| HW aditif vs multiplikatif diuji di KEDUA seri (aditif & multiplikatif) | | | |
| Backtest dilaporkan per-fold (bukan cuma mean) | | | |
| Data val disentuh PERSIS sekali di akhir | | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Mekanisme leakage split acak (kenapa angka "bagus" itu jebakan) | |
| 2. Error rekursif menumpuk: dari APE 1.9% menjadi MAE ~19.7 | |
| 3. AR vs musiman panjang; biaya 365 parameter dummies (in- vs out-sample) | |
| 4. Keputusan aditif vs multiplikatif pada data nyata | |
| 5. Mean backtest menipu; sebaran fold yang harus dilaporkan | |
| 6. Drift & re-backtesting berkala (koneksi MLOps) | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/kurva eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Guard clause (W ≥ 1, W+h ≤ n, periode > 0) | | |
| Konvensi koefisien AR konsisten (coef[0] = lag-1) di fit & forecast | | |
| Baseline hanya melihat train fold pada backtest | | |
| Tidak mengubah test/data untuk "menang" | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Kesalahan apa yang paling lama kamu debug (split? konvensi lag? urutan update?)?
- Apa yang berubah dari cara kamu membaca angka evaluasi setelah melihat
  in-sample vs out-of-sample dan backtest per-fold?
- Satu hal yang mau kamu coba berikutnya (log-transform di pipeline penuh?
  cross-validation time series dari sklearn? model Keras dari materi Bab 7?)
