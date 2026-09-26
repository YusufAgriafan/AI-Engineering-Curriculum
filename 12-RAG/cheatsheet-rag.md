# 📄 Cheatsheet — RAG: Retrieval Augmented Generation (Bab 12)

> Review cepat. Uji diri: tutup file ini, tulis ulang pipeline end-to-end +
> rumus RRF + kontrak `abstaikan` dari ingatan, baru cek.

---

## 1. Pipeline & Kapan RAG

```
INGESTION (sekali/periodic)   QUERY TIME (tiap permintaan)
load → chunk → embed → index  |  embed kueri → retrieve top-k → prompt → generate
                              |  + abstain gate + sitasi
```

| Masalah | RAG selesaikan dengan | Alternatif yang salah |
|---|---|---|
| Data privat / fakta berubah | retrieval saat query time — data cukup diperbarui di index | fine-tuning fakta (mahal, basi, tetap bisa halusinasi) |
| Dokumen terlalu besar untuk konteks | chunk + ambil yang relevan saja | memaksakan semua ke prompt |
| Jawaban harus bisa diaudit | sitasi [n] → chunk → sumber & halaman | "percayalah ke model" |

**Aturan:** RAG dulu untuk fakta; fine-tuning untuk perilaku/format (Bab 13).

---

## 2. Chunking: Keputusan dengan Trade-off Terukur

- **Ukuran:** 300–800 token (di lab: 320 karakter). Kecil = fokus tapi kehilangan
  konteks & gagal lintas-kalimat; besar = utuh tapi bising.
- **Overlap:** 10–20% (di lab: 64) — menjaga kalimat yang melintasi batas chunk.
- **Potong di batas natural:** kalimat/paragraf/heading; jangan memotong kata.
- Metadata wajib menempel di chunk: `source`, `page`, `chunk_index` — sitasi
  hanyalah metadata yang dibawa sampai akhir.
- **Mengubah chunking = mengubah embedding = index dibangun ulang.**

---

## 3. Embedding & Vector Search

```
cosine(a, b) = a·b / (|a||b|)          # vektor ternormalisasi L2 → dot = cosine
vektor nol (teks tanpa token dikenal)  → return 0.0, JANGAN biarkan NaN/ZeroDivision
```

- Skor selalu ada ("paling mirip") **tidak berarti jawaban ada** — q6 skor 0.065
  (project) / 0.052 (kuis) vs kueri dalam korpus 0.45–0.67.
- Pilih model multilingual untuk bahasa Indonesia: `bge-m3`, `multilingual-e5`,
  `text-embedding-3-small/large`. Ganti model embedding = embed ulang semuanya.
- Vector DB: FAISS/sqlite-vec untuk mulai; Qdrant/pgvector untuk produksi.
  Brute-force cosine O(N) cukup sampai ribuan chunk (ANN bisa menyusul).

---

## 4. Retrieval + hit@k: Ukur Sebelum Bangga

```
hit@k: kueri "kena" bila sumber emas ada di antara sumber top-k hasil
hit_rate = kueri_kena / total_kueri
```

Angka terkunci (project, SEED 12): **hit@1 = 0.8** (q1 gagal — chunk FAQ menang
top-1 karena mengulang kata kueri), **hit@2 = hit@3 = 1.0**.

Pelajaran: top-1 bisa gagal karena FAQ/parafrase menang peringkat →
`top_k` ≥ 2–3 + reranker; dan **evaluasi retrieval dulu, kualitas jawaban belakangan**.

---

## 5. Abstain Gate: Berani Bilang "Tidak Ada di Dokumen"

```
skor_terbaik >= ambang  -> biarkan jawaban lewat (batas inklusif!)
skor_terbaik <  ambang  -> timpa: jawaban = "tidak ada di dokumen",
                           kosongkan sitasi & sumber, catat abstaikan_oleh = skor
sudah abstain           -> DIBIARKAN (jangan pernah turunkan abstain → jawaban)
```

- Kembalikan **dict baru** — input utuh untuk audit.
- `skor_terbaik` hilang = dianggap 0.0 → abstain (aman).
- Ambang: terlalu rendah = jawaban salah lolos; terlalu tinggi = abstain saat
  jawaban ada. Ukur trade-off-nya (0.15 / 0.30 / 0.45 di lab).

---

## 6. Hybrid Search (BM25 + Vektor) + RRF

```
skor_rrf(chunk) = Σ_daftar 1/(k + peringkat + 1)     # peringkat 0-BASED, k=60 lazim
```

- Vektor menangkap makna; BM25 menangkap kata eksak (nama produk, kode, angka).
- Chunk muncul di banyak daftar **menjumlah** kontribusi — konsisten di dua
  daftar mengalahkan peringkat-atas di satu daftar (contoh kuis: c0 = 2/62 menang).
- Hybrid = **dua pencarian + fusi** → lebih mahal per permintaan; beli kualitas
  peringkat, bukan kecepatan. Angka terkunci: hybrid hit@3 = 1.0 (project).
- Korpus kosong → `[]`; kata tak dikenal → skor BM25 0, bukan crash.

---

## 7. Reranker: Pola Two-Stage

```
retriever murah    : top-k besar (50)   ← cepat, kurang presisi
reranker per-pasangan: skor kueri×chunk satu-satu → ambil top-k kecil (5) ← presisi
```

- Reranker selalu **memangkas**: informasi di kandidat yang dibuang hilang dari
  konteks — layak bila presisi konteks lebih penting daripada cakupan.
- Di project: skorer kata (angka berbobot 3.0) menggantikan cross-encoder
  (bge-reranker / Cohere Rerank) — **polanya sama**, hanya skornya.

---

## 8. Prompt RAG + Cache Konteks

Prompt wajib: jawab HANYA dari KONTEKS · tidak ada → "tidak ada di dokumen" ·
sebut sumber [n] · jangan arang angka. Sitasi = metadata chunk [n] → (source, page).

```
cache retrieval : kunci = kueri.strip().lower() + (produksi: versi index, top_k, filter)
                  hit ≠ expired (dihitung TERPISAH) · TTL tick logis agar bisa diuji
```

Angka terkunci (project): pola `miss, hit, hit, miss` → **2 pencarian untuk 4
permintaan**, hit-rate 0.5; latensi hit 2 ms vs miss 16 ms.
Bedanya dengan cache LLM (Bab 11): yang di-cache **KONTEKS**, jawaban tetap
digenerate segar. Lupa versi index di kunci = konteks dari korpus lama.

---

## 9. Biaya & Anti-pattern

```
biaya ≈ token_in/1e6 × harga_in + token_out/1e6 × harga_out   # konteks = token input!
```

Konteks top-k adalah komponen token input terbesar → top_k & chunk size = tuas biaya.

| Anti-pattern | Gejala | Solusi |
|---|---|---|
| Skor rendah tetap dijawab | halusinasi percaya diri di luar korpus | abstain gate + ambang terukur |
| Chunking tanpa evaluasi | hit@k turun diam-diam | hit@k dulu, ganti konfigurasi, bandingkan angka |
| Tie-break tak deterministik | retrieval "bergoyang" antar run | urut (-skor, index) |
| RRF 1-based / tanpa +1 | urutan hybrid berubah / div-nol | 1/(k + rank + 1), rank 0-based |
| Cache tanpa versi index | konteks basi pasca update korpus | kunci + TTL + versi index |
| Sitasi tanpa metadata | pengguna tidak bisa memverifikasi | metadata ikut dari chunking |
| Evaluasi "rasanya oke" | tidak bisa dibandingkan antar versi | akurasi emas + abstain_benar (Bab 15) |
| Generator tanpa aturan | LLM mengarang dari konteks kurang | prompt ketat + gate + sitasi |

---

## 10. Checklist Review 30 Detik

- [ ] Pipeline utuh: load → chunk → embed → index → retrieve → prompt → generate → gate.
- [ ] Metadata (source, page, chunk_index) menempel sejak chunking.
- [ ] hit@k dievaluasi sebelum bicara kualitas jawaban.
- [ ] Abstain gate: dict baru, batas inklusif, abstain tidak diturunkan.
- [ ] RRF 0-based dengan `+1`; tie-break deterministik di semua pengurutan.
- [ ] Cache: kunci ternormalisasi + versi index; hit/miss/expired terpisah.
- [ ] Prompt ketat + sitasi; ambang abstain punya angka, bukan perasaan.
- [ ] Semua angka diuji tanpa jaringan (encoder & generator palsu deterministik).
