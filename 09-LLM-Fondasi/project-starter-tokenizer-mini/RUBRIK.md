# 📊 Rubrik Penilaian Mandiri — Tokenizer & Mini-LM dari Nol

> Isi dengan **jujur** setelah semua test hijau + pertanyaan analisis terjawab. Total 100 poin.
> Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `bpe.py` — pair counting, tie-break leksikografis, merge non-destruktif | | | |
| `tokenizer.py` — vocab tanpa celah, round-trip teks benar | | | |
| `embedding.py` — cosine aman nol, cari_mirip terurut | | | |
| `position.py` — sin/cos per kolom, dot = f(p−q) | | | |
| `attention.py` — softmax stabil, `sqrt(d_k)`, mask segitiga atas k=1 | | | |
| `lm.py` — loss shift target (logits[:-1] vs ids[1:]), PE tidak dilatih | | | |
| `sampling.py` — nucleus min. 1 token, softmax LOKAL di kandidat | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Token dihitung per kata & dibandingkan antar kata (bukan cuma total) | | |
| Kurva merge dilaporkan di ≥5 titik n_merge (0, 1, 2, 4, 8) | | |
| Matrix cosine divisualisasikan & blok kelompok dibaca | | |
| Encoder vs decoder dibandingkan PER BARIS (bukan sekadar heatmap) | | |
| Sampling diuji dengan ≥300 sample + temperature sweep | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Anatomi token: kata sering vs kata baru, implikasi biaya API | |
| 2. Bentuk kurva merge & kenapa tokenizer tidak merge 1 juta kali | |
| 3. Struktur tanpa label; std cosine vektor acak d=8 vs d=64 | |
| 4. Simetri encoder/decoder; gejala mask hilang saat training | |
| 5. Kepribadian tiap strategi sampling; konfigurasi chatbot + risikonya | |
| 6. Relasi vs posisi — kenapa `[1] -> 0` bisa benar | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/kurva eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Konvensi wajib diikuti (`>=`/leksikografis/k=1/shift target) tanpa "mencurangi" test | | |
| Fungsi murni — input tidak dimodifikasi (merge_pasangan, apply_causal_mask) | | |
| Guard: pembagi nol → 0.0, nucleus clamp ke len(order) | | |
| Seed dicatat di setiap eksperimen | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Kesalahan apa yang paling lama kamu debug (tie-break? shift target? nucleus index)?
- Setelah membangun attention sendiri, apa yang berubah dari cara kamu membaca
  parameter `temperature`/`top_p` di API LLM?
- Satu hal yang mau kamu coba berikutnya (multi-head attention? tokenizer
  bahasa Indonesia? byte-level BPE seperti GPT-2?)
