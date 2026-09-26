# 🏗️ Project Starter — Forecasting Mini dari Nol (Bab 7)

> Kamu membangun **pipeline forecasting lengkap** dari nol: windowing + split
> temporal, metrik (MAE/RMSE/sMAPE), tiga baseline, AR(p) dengan forecast
> rekursif, Holt-Winters aditif & multiplikatif, dan rolling backtest — lalu
> memilih model terbaik dengan bukti, bukan perasaan. Semua numpy murni.

## Struktur

```
project-starter-forecasting-mini/
├── README.md              ← kamu di sini
├── RUBRIK.md              ← penilaian mandiri
├── starter.ipynb          ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb           ← notebook SOLUSI (buka setelah selesai / mentok)
├── ts_mini/               ← paket yang kamu isi (numpy murni)
│   ├── data.py            ← seri sintetis terkunci (JANGAN DIUBAH)
│   ├── windowing.py       ← TODO: windowing 1-step & multi-step + split temporal
│   ├── baseline.py        ← TODO: metrik (MAE/RMSE/sMAPE) + tiga baseline
│   ├── ar.py              ← TODO: AR(p) OLS + forecast rekursif
│   └── holtwinters.py     ← TODO: Holt-Winters aditif & multiplikatif
├── tests/                 ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_windowing.py
│   ├── test_baseline.py
│   ├── test_ar.py
│   └── test_holtwinters.py
└── solusi/
    └── ts_mini_ref.py     ← implementasi referensi (untuk cek mandiri)
```

## Dataset: Seri Sintetis Terkunci

Dua seri sintetis (`data.py`, seed terkunci 42 — resep identik dengan lab):

1. **Seri aditif** — `y[t] = (100 + 0.05·t) + 20·sin(2π·t/365) + N(0, 5²)`,
   1095 hari. Split standar: train 876 hari, val 219 hari.
2. **Seri multiplikatif** — `y[t] = (100 + 0.05·t)·(1 + 0.15·sin(2π·t/365)) + N(0, 4²)`,
   amplitudo musiman membesar bersama level.

Kenapa seri sintetis? Karena **resep kebenarannya diketahui** — kita bisa mengukur
persis apakah model menemukan trend & musiman yang ditanam, sesuatu yang mustahil
di data nyata. Skill windowing/split/evaluasi yang dilatih di sini berlaku persis
saat kamu pindah ke data produksi.

## Apa yang Kamu Bangun

1. **Windowing + split temporal** — konversi 1 seri → dataset supervised
   (1-step dan multi-step), split BERURUTAN waktu tanpa shuffle. Fondasi semua
   forecasting; salah di sini = semua angka setelahnya bohong.
2. **Metrik + baseline** — MAE, RMSE, sMAPE (epsilon-aman), lalu naive,
   seasonal-naive, dan HAM. Model apa pun yang tak mengalahkan baseline ini
   belum layak dipakai.
3. **AR(p) + forecast rekursif** — regresi OLS pada lag; lalu bukti penting
   bahwa error menumpuk di multi-step: APE 7-langkah ~1.9% tapi MAE 219-langkah
   lebih buruk dari naive.
4. **Holt-Winters** — level + trend + seasonality dari nol, versi aditif dan
   multiplikatif; di seri multiplikatif pilihan versi menentukan (MAE ~12.9 vs
   ~16.5).
5. **Rolling backtest** — satu split bisa menipu; rata-rata k fold memberi
   keputusan yang lebih stabil.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_windowing.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `windowing.py` → `baseline.py` → `ar.py` →
   `holtwinters.py` sampai **semua 40 test hijau**.
4. Buka `starter.ipynb` — kerjakan eksperimen & jawab **Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/ts_mini_ref.py` / `solusi.ipynb`.

> 💡 Kalau angka tak cocok dengan rentang di test: cek (1) split TANPA shuffle
> (`int(n·0.8)`), (2) konvensi koefisien AR — `coef[0]` mengalikan nilai TERBARU
> (lag-1), (3) urutan update Holt-Winters level → trend → season, dan loop mulai
> dari `i = m`. Tiga itu penyebab 90% bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. Split acak menampilkan MAE ~6.4 vs temporal ~7.3 pada 1-NN window —
   **kenapa angka yang lebih rendah itu justru tanda model lebih buruk?**
   Jelaskan mekanisme kebocorannya, bukan cuma hasilnya.
2. AR(7) punya APE 7-langkah ~1.9% tapi MAE 219-langkah ~19.7 (kalah dari naive).
   Apa yang terjadi di antara keduanya, dan kapan klaim "model bagus" menyesatkan?
3. Kenapa AR pada musiman 365 hari tidak praktis, dan bagaimana trend+dummies
   menyelesaikannya? Apa biayanya (bandingkan in-sample 6.41 vs out-of-sample
   11.76)?
4. Di eksperimen ini HW-multiplikatif menang di KEDUA seri — tapi karena alasan
   berbeda (redaman drift vs bentuk amplitudo). Bagaimana kamu MEMUTUSKAN aditif
   vs multiplikatif di data nyata tanpa tahu resepnya? (hint: plot amplitudo vs
   level, atau backtest keduanya)
5. Naive menang rata-rata backtest (10.33) tapi MAE fold-2-nya ~21 vs ~5 di fold
   lain. Kenapa mean saja tidak cukup untuk memilih model, dan metrik sebaran
   apa yang akan kamu laporkan?
6. Koneksi ke Bab 16 (MLOps): pola berubah setelah model dipasang (drift).
   Bagian mana dari pipeline-mu yang harus dijalankan ulang berkala, dan angka
   apa yang kamu pantau di produksi?

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini memakai kembali pola dari Bab 3 & 6: regresi OLS, split yang jujur,
> dan evaluasi dengan baseline. Konvolusi/attention diganti smoothing
> eksponensial; yang diuji bukan kecanggihan model tapi **kejujuran evaluasi**.
