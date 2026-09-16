# 📊 Rubrik Penilaian Mandiri — CNN Mini dari Nol

> Isi dengan **jujur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `conv.py` — pad_zero, conv2d_forward, maxpool forward | | | |
| `conv.py` — backward lolos gradient check numerik (dK & dX) | | | |
| `conv.py` — Flatten & Conv2D (lazy init, pad dibuang di backward) | | | |
| `augment.py` — flip, noise, shift, pipeline (label tidak berubah) | | | |
| `model.py` — CNNMini forward/backward lolos gradient check end-to-end | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Baseline MLP (piksel mentah) DILATIH dan dicatat sebelum CNN dibandingkan | | | |
| Kurva loss (train & val) digambar dan dibaca, bukan sekadar ditonton | | | |
| Eksperimen kernel/augmentasi diubah SATU variabel per percobaan | | | |
| Angka val acc dilaporkan untuk tiap konfigurasi | | | |
| Test set disentuh PERSIS sekali di akhir | | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–6) |
|---|---|
| 1. Kenapa CNN lebih tahan terhadap garis yang digeser daripada MLP? | |
| 2. Penjelasan weight sharing + invarian translasi dengan kata-kata sendiri | |
| 3. Kenapa gradien max-pool hanya mengalir ke satu posisi per jendela? | |
| 4. Kenapa flip vertikal MERUSAK label di dataset ini padahal flip horizontal aman? | |
| 5. Trade-off ukuran kernel: informasi lokal vs parameter & tepi yang hilang | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/kurva/peta aktivasi eksperimenmu. Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Guard clause (bentuk input, stride ≥ 1, dimensi genap untuk pool) | | |
| Backward tidak meng-assign gradien yang harusnya di-akumulasi (`+=`) | | |
| Fungsi augmentasi tidak mengubah input asli (copy dulu) | | |
| Tidak mengubah test/data untuk "menang" | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Bagian mana yang paling membuatmu "akhirnya paham" cara CNN bekerja?
- Kesalahan apa yang paling lama kamu debug, dan apa pelajarannya?
- Satu hal yang mau kamu coba berikutnya (multi-channel? stride? pooling lain?)
