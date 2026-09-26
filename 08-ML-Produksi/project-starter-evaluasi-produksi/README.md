# 🏗️ Project Starter — Evaluasi Model Produksi dari Nol (Bab 8)

> Kamu membangun **pipeline evaluasi model produksi lengkap** dari nol: metrik
> (RMSE/MAE/R²), skill score untuk keputusan ADOPT/TOLAK, OLS + baseline HAM +
> model "negatif" (simulasi eksperimen salah arah), persistensi deterministik
> dengan hash & round-trip test, confusion matrix + precision/recall/F1,
> threshold tuning di calib, kalibrasi probability (ECE + Platt scaling), dan
> GAN 1D dengan dua pemain menuju equilibrium. Semua numpy murni, TDD 52 test.

## Struktur

```
project-starter-evaluasi-produksi/
├── README.md              ← kamu di sini
├── RUBRIK.md              ← penilaian mandiri
├── starter.ipynb          ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb           ← notebook SOLUSI (buka setelah selesai / mentok)
├── prod_mini/             ← paket yang kamu isi (numpy murni)
│   ├── data.py            ← dataset terkunci (JANGAN DIUBAH)
│   ├── metrics.py         ← TODO: RMSE, MAE, R², skill score
│   ├── models.py          ← TODO: OLS, HAM, model 'negatif'
│   ├── persist.py         ← TODO: save/load deterministik + hash + round-trip
│   ├── threshold.py       ← TODO: confusion, P/R/F1, threshold tuning
│   └── calibration.py     ← TODO: sigmoid, binning, ECE, Platt scaling
├── tests/                 ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_persist.py
│   ├── test_threshold.py
│   └── test_calibration.py
├── _verify_project.py     ← verifier otomatis (patch referensi + jalankan test)
└── solusi/
    └── prod_mini_ref.py   ← implementasi referensi (untuk cek mandiri)
```

## Dataset: Regresi Sintetis Terkunci

`y = 2.0·x1 − 1.5·x2 + 1.0 + N(0, 1.2²)`, n=800, seed 8 (identik dengan lab).

- Split: **train 640 | calib 80 | test 80** (berurutan, tanpa shuffle).
- Label keputusan (bagian threshold/kalibrasi): `1` jika `y > median(train)`.
- Data GAN: `N(2.0, 0.5²)`, n=512.

Kenapa sintetis? **Resep kebenarannya diketahui** — kita tahu persis OLS harus
memulihkan `w=[2.0, -1.5]`, model salah arah harus kalah dari HAM, dan G harus
menemukan `mu=2.0`. Di data nyata, kebenaran itu tak pernah ada; protokol
evaluasi yang kamu bangun di sini yang menggantikannya.

## Apa yang Kamu Bangun

1. **Metrik & skill score** — RMSE/MAE/R² dari nol, lalu keputusan ADOPT/TOLAK:
   model "negatif" (koefisien dibalik) punya RMSE 6.68 vs HAM 4.83 → skill −0.38
   → TOLAK. Model yang kalah baseline menambah noise, bukan nilai.
2. **Persistensi terbukti** — save deterministik (hash identik untuk model sama)
   + round-trip test: prediksi sebelum/sesudah save-load identik bit-per-bit.
   Klaim "artefak siap serving" kini punya bukti, bukan harapan.
3. **Metrik keputusan** — confusion matrix, precision/recall/F1 dengan konvensi
   pembagi-nol → 0.0; threshold dipilih di CALIB, test disentuh SEKALI. Termasuk
   demo angka bocor yang lebih tinggi — dan kenapa ia tidak sah.
4. **Kalibrasi probability** — binning, ECE, reliability curve, Platt scaling
   yang belajar dari calib: ECE turun, probabilitas jadi bisa dipercaya.
5. **GAN 1D dengan backprop manual** — dua set gradien yang saling menentukan,
   non-saturating loss untuk G, dan bukti equilibrium: D(real) ≈ D(fake) ≈ 0.5,
   tanpa loss yang monoton turun.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_metrics.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `metrics.py` → `models.py` → `persist.py` →
   `threshold.py` → `calibration.py` sampai **semua 52 test hijau**.
4. Buka `starter.ipynb` — kerjakan eksperimen & jawab **Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/prod_mini_ref.py` / `solusi.ipynb`.

> 💡 Kalau test macet: cek (1) konvensi `>=` di confusion (tepat threshold =
> positif), (2) pembagi nol → 0.0 (bukan exception), (3) `min(int(p·n_bins),
> n_bins−1)` di binning (p=1.0 masuk bin terakhir), (4) urutan update GAN:
> D dulu (real, fake), baru G melewati D yang dibekukan. Empat itu penyebab 90%
> bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. RMSE OLS ~1.25 terdengar bagus — tunjukkan dengan skill score kenapa klaim
   itu baru bermakna SETELAH dibandingkan dengan HAM. Model "negatif" (RMSE 6.7):
   mengapa keputusan yang benar TOLAK, bukan "perbaiki dikit"?
2. Threshold bocor (tune di test) menghasilkan F1 lebih tinggi daripada jujur.
   Jelaskan MEKANISME kebocorannya, dan kenapa selisih kecil di dataset ini
   tidak membuatnya sah untuk produksi.
3. Round-trip test lolos, tapi tim masih meng-claim "model sudah di-update"
   tanpa hash. Skenario kegagalan apa yang terlewat, dan bagaimana hash menutupnya?
4. HAM gagal total sebagai klasifikator (F1=0) tapi berguna sebagai baseline
   regresi. Jelaskan dua peran itu — dan kapan baseline "tidak berdaya" justru
   menunjukkan metrikmu yang salah?
5. ECE turun setelah Platt scaling. Dalam keputusan bisnis apa penurunan itu
   benar-benar penting, dan kapan F1 tinggi TANPA kalibrasi justru cukup?
6. GAN berakhir di equilibrium, bukan "menang". Kaitkan dengan mode collapse:
   kalau G punya kapasitas jauh lebih besar dari variasi data, apa yang terjadi
   dan bagaimana mendeteksinya dari histogram/diagnostik?

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini memakai kembali pola dari Bab 3 (regresi OLS, split yang jujur) dan
> Bab 4 (backprop, loop training). Yang diuji bukan kecanggihan model tapi
> **kejujuran evaluasi dan bukti reproduksibilitas** — dua skill yang membedakan
> notebook dari sistem produksi.
