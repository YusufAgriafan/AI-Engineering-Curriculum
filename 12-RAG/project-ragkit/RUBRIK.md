# 📊 Rubrik Penilaian Mandiri — ragkit (Bab 12)

> Isi dengan **jujur** setelah semua 66 test hijau + pertanyaan analisis terjawab.
> Total 100 poin. Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait
> di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `vectorstore.py` — cosine eksplisit, tie → index kecil, store kosong `[]` | | | |
| `retriever.py` — index sekali, hit@k + posisi 1-based + kueri_gagal | | | |
| `abstain.py` — dict BARU, tidak menimpa abstain, `abstaikan_oleh` tercatat | | | |
| `cache.py` — kunci normalisasi, TTL tick logis, `expired` ≠ `miss` | | | |
| `hybrid.py` — BM25 (k1/b/idf), RRF `1/(k+rank+1)`, tie deterministik | | | |
| `rerank.py` — angka ×3, tie → index kecil, `skorer` bisa disuntik | | | |
| `pipeline.py` + `evaluate.py` — biaya & latensi per tahap, gate lewat modul | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Delapan blok dijalankan dan angka terkunci **dilaporkan** (hit@1 0.8, dst.) | | |
| Blok 2: analisis mengapa q1 gagal top-1 (FAQ vs kebijakan) | | |
| Blok 3: ambang 0.15 / 0.45 dicoba dan trade-off-nya dijelaskan | | |
| Blok 4: pola cache + jumlah pencarian vs jumlah permintaan dilaporkan | | |
| Blok 7–8: biaya per permintaan + akurasi e2e + abstain dilaporkan | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. q1 gagal top-1 — mekanisme + konsekuensi ke `top_k` | | |
| 2. Ambang abstain terlalu rendah vs terlalu tinggi | | |
| 3. Kunci cache: kueri mirip-mirip & risikonya | | |
| 4. Kapan hybrid layak vs vector saja | | |
| 5. Pemangkasan konteks oleh reranker | | |
| 6. Lapisan yang tetap wajib saat generator jadi LLM nyata | | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/aksi eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Fungsi murni — `abstaikan` mengembalikan dict baru, input tidak diubah | | |
| Tidak ada state global yang mengintai (cache/store dibuat per eksperimen) | | |
| Tidak "mencurangi" test (hardcode angka terkunci di dalam fungsi) | | |
| Tie-break deterministik di semua pengurutan (skor, RRF, rerank) | | |
| Input kosong (korpus kosong, kueri kosong, kandidat kosong) tidak crash | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Test mana yang paling lama kamu perbaiki (tie-break cosine? `expired` vs `miss`?
  offset peringkat RRF?), dan apa yang akhirnya kamu pahami?
- Setelah melihat hit@1 = 0.8 padahal hit@2 = 1.0, bagaimana itu mengubah cara
  kamu memilih `top_k` dan menyusun konteks?
- Satu hal yang mau kamu bawa ke Bab 13 (Fine-tuning & open-weight models):
  embedding model sendiri, evaluasi retrieval di CI, atau abstain gate?
