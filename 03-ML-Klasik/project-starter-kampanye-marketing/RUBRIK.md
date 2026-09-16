# 📊 Rubrik Penilaian Mandiri — Prediksi Respon Kampanye Marketing

> Isi dengan **jujur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `metrik.py` — confusion matrix, P/R/F1 | | | |
| `metrik.py` — ROC-AUC dengan tie handling | | | |
| `model.py` — sigmoid stabil + BCE + gradient descent | | | |
| `model.py` — loss monoton turun, val accuracy ≥ 0.89 | | | |
| `clustering.py` — standardize + kmeans (n_init, inertia) | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Baseline bodoh ("semua nol") dihitung SEBELUM merayakan akurasi model | | |
| Threshold digeser dengan pertanyaan jelas, bukan asal geser | | |
| Trade-off precision–recall dicatat dalam angka, bukan cuma plot | | |
| Test set disentuh PERSIS sekali, di akhir | | |
| Klaster diinterpretasikan (bukan cuma dihitung inertia-nya) | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–6) |
|---|---|
| 1. Accuracy menipu — jelaskan dengan angka dari eksperimenmu | |
| 2. Pemilihan threshold berbasis biaya (bukan selalu 0.5) | |
| 3. AUC = ranking vs threshold = keputusan | |
| 4. Justifikasi pemilihan segmen promo dengan angka | |
| 5. Val vs test — kapan perbedaan jadi tanda bahaya | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka eksperimenmu. Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Guard clause sadar (pembagian nol, kelas tak lengkap, k >= 1) | | |
| Docstring + nama variabel yang menjelaskan maksud | | |
| Tidak mengubah test/data untuk "menang" | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

1. Konsep mana yang baru benar-benar "klik" — dan apa yang memicunya?
2. Bagian mana yang masih terasa seperti "ikuti langkah tanpa paham"?

**Skor total: ____ / 100** · Tanggal selesai: ____________
