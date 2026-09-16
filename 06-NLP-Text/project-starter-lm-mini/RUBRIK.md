# 📊 Rubrik Penilaian Mandiri — LM Mini dari Nol

> Isi dengan **jujur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `text.py` — tokenize, vocab terurut, encode OOV-aman, pad/truncate post | | | |
| `embeddings.py` — pasangan dua arah + skip-gram lolos gradient check | | | |
| `embeddings.py` — purity ≥ 0.9 dan loss skip-gram turun | | | |
| `languagemodel.py` — bigram smoothing + EE + temperature benar perilakunya | | | |
| `rnn.py` — forward + BPTT lolos gradient check, val acc > bigram | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Baseline (random embedding / unigram / bigram) dicatat SEBELUM klaim | | | |
| Kurva loss (skip-gram & RNN) digambar dan dibaca, bukan sekadar ditonton | | | |
| Eksperimen (dim, window, k_neg, temperature, hidden) diubah SATU variabel | | | |
| Angka metrik (purity, acc, EE) dilaporkan untuk tiap konfigurasi | | | |
| Data uji/val disentuh PERSIS sekali di akhir | | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Mekanisme ko-okurensi yang membuat kata satu topik berdekatan | |
| 2. Apa yang hilang tanpa negative sampling | |
| 3. Kenapa embedding generalisasi tapi one-hot menghafal | |
| 4. Temperature: bukti sampel + keputusan pemakaian | |
| 5. Apa yang benar-benar dipelajari hidden state RNN | |
| 6. Vanishing gradient + dua solusi arsitektural | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/kurva/sampel eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Guard clause (bentuk input, window ≥ 1, vocab_size > 0) | | |
| Gradien dihitung pada nilai ASLI sebelum update; akumulasi pakai `+=` | | |
| Matriks bigram dinormalisasi per baris (tiap baris distribusi) | | |
| Tidak mengubah test/corpus untuk "menang" | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Bagian mana yang paling membuatmu "akhirnya paham" dari mana makna kata datang?
- Kesalahan apa yang paling lama kamu debug, dan apa pelajarannya?
- Satu hal yang mau kamu coba berikutnya (subword/BPE? LSTM gate? attention?)
