# 📊 Rubrik Penilaian Mandiri — Neural Network Mini dari Nol

> Isi dengan **juhur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `activations.py` — relu, relu_grad, sigmoid stabil, softmax stabil | | | |
| `losses.py` — BCE/MSE + gradien lolos gradient check numerik | | | |
| `layers.py` — Dense forward/backward lolos gradient check | | | |
| `train.py` — SGD + batching + training loop | | | |
| `train.py` — early stopping berperilaku sesuai kontrak | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Baseline linear DILATIH dan dicatat sebelum MLP dibandingkan | | |
| Kurva loss (train & val) digambar dan dibaca, bukan sekadar ditonton | | |
| Eksperimen lr/arsitektur diubah SATU variabel per percobaan | | |
| Angka val acc dilaporkan untuk tiap konfigurasi | | |
| Test set disentuh PERSIS sekali di akhir | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–6) |
|---|---|
| 1. Kenapa model linear gagal di data non-linear | |
| 2. Penjelasan backprop dengan kata-kata sendiri | |
| 3. Mekanisme lr besar → loss meledak | |
| 4. Kenapa early stopping memantau val loss | |
| 5. Init nol → masalah simetri | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/kurva eksperimenmu. Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Aktivasi stabil-numerik (bukan `1/(1+exp(-z))` polos) | | |
| Guard clause (bentuk input, params vs grads, kelas validasi) | | |
| Cache forward dibersihkan/dipakai sadar (tidak hitung ulang yang tak perlu) | | |
| Tidak mengubah test/data untuk "menang" | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

1. Di titik mana backprop akhirnya "klik" — apa yang memicunya?
2. Konsep mana yang masih ingin kamu buktikan ulang dengan eksperimen lain?

**Skor total: ____ / 100** · Tanggal selesai: ____________
