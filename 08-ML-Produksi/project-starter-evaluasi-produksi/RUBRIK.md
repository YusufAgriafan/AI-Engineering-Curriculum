# 📊 Rubrik Penilaian Mandiri — Evaluasi Model Produksi dari Nol

> Isi dengan **jujur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `metrics.py` — RMSE/MAE/R² benar (R² boleh negatif) + skill score | | | |
| `models.py` — OLS pulihkan resep; fit/predict negatif konsisten (dua pembalikan) | | | |
| `persist.py` — save deterministik (hash sama), round-trip bit-per-bit | | | |
| `threshold.py` — konvensi `>=`, pembagi nol → 0.0, ties ambil pertama | | | |
| `calibration.py` — sigmoid stabil, binning (p=1.0 → bin terakhir), ECE, Platt | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Baseline HAM dicatat SEBELUM klaim model lebih baik | | | |
| Keputusan ADOPT/TOLAK ditulis eksplisit dari skill score | | | |
| Threshold dipilih di CALIB; test disentuh SEKALI untuk lapor | | | |
| Demo bocor (tune di test) dijalankan & dibandingkan dengan jujur | | | |
| ECE raw vs platt dilaporkan + reliability diagram diinterpretasi | | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Skill score sebagai syarat klaim; TOLAK untuk model salah arah | |
| 2. Mekanisme kebocoran threshold (bukan cuma hasilnya) | |
| 3. Skenario kegagalan tanpa hash; hash + round-trip menutup celah | |
| 4. Dua peran HAM; kapan baseline menang = metrik yang salah | |
| 5. Kapan kalibrasi penting vs ranking murni cukup | |
| 6. Equilibrium GAN, mode collapse, deteksi dari diagnostik | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/kurva eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Konvensi `>=` konsisten di semua fungsi keputusan | | |
| Pembagi nol → 0.0 (produksi: jangan crash), bukan exception | | |
| `roundtrip_ok` memakai `array_equal` (bukan allclose) | | |
| Tidak mengubah test/data untuk "menang" | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Kesalahan apa yang paling lama kamu debug (konvensi `>=`? binning p=1.0?
  urutan gradien GAN? deterministik save?)?
- Apa yang berubah dari cara kamu membaca angka evaluasi setelah melihat
  skill score, demo bocor, dan ECE raw-vs-platt?
- Satu hal yang mau kamu coba berikutnya (ECE dengan 15 bins? temperature
  scaling? GAN 2D dengan dua cluster — mode collapse sungguhan?)
