# 🏗️ Project Starter — Gradient Descent Studio (Bab 2)

> Proyek mini **eksperimen**: kamu membangun "studio" untuk mengamati gradient descent bekerja pada beberapa fungsi loss — lalu menjawab pertanyaan analisis seperti peneliti. Notebook template + notebook solusi tersedia.

## Struktur

```
project-starter-gradient-descent/
├── README.md            ← kamu di sini
├── RUBRIK.md            ← penilaian mandiri
├── gd_starter.ipynb     ← notebook TUGASMU (ada TODO)
└── gd_solusi.ipynb      ← notebook SOLUSI (buka setelah selesai / mentok)
```

## Apa yang Kamu Bangun

1. **Fungsi GD generik** untuk loss 1D & 2D (dari latihan lab, di-generalisasi).
2. **Eksperimen learning rate**: cari `lr` yang bikin divergen untuk `J(w) = (w-5)²` — dan jelaskan kenapa.
3. **Perbandingan titik awal**: jalankan GD dari `w = -10` dan `w = 10`; apakah sama-sama sampai?
4. **Loss 2D**: `J(x, y) = (x-2)² + 3(y+1)²` — visualisasikan *contour plot* + jalur GD.
5. **Bonus**: loss non-konveks `J(w) = w⁴ - 3w² + w` — temukan 2 titik awal yang berakhir di *local minimum berbeda*.

## Cara Kerja

1. Buka `gd_starter.ipynb` (Jupyter / VS Code).
2. Isi semua `# TODO` — jangan buka `gd_solusi.ipynb` dulu.
3. Jalankan Restart & Run All — semua cell `✅ cek` harus hijau.
4. Jawab **Pertanyaan Analisis** (markdown) dengan kata-katamu sendiri.
5. Isi `RUBRIK.md`.

## Pertanyaan Analisis (bagian penilaian!)

1. Kenapa `lr` terlalu besar membuat GD divergen? Jelaskan dengan bentuk kurva loss, bukan cuma "jadi error".
2. Untuk loss 2D di atas, kenapa langkah di arah `y` terasa "lebih agresif" daripada `x`? (hint: koefisien 3)
3. Loss non-konveks: apa artinya bagi training neural network sungguhan?

## Prasyarat

```bash
pip install numpy matplotlib
```

Bab ini adalah fondasi Bab 4 (Deep Learning) — pahami sampai **bisa jelaskan ke orang lain**, bukan cuma kode jalan.
