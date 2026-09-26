# 🏗️ Project Starter — ragkit: Sistem RAG dari Nol (Bab 12)

> Kamu membangun **seluruh pipeline RAG tanpa framework**: vector store cosine,
> retriever + evaluasi hit@k, abstain gate yang berani bilang "tidak ada di
> dokumen", cache konteks retrieval, hybrid search (BM25 + RRF), reranker
> two-stage, pipeline end-to-end dengan biaya & latensi per tahap, dan evaluasi
> ujung-ke-ujung. Stdlib murni, TDD **66 test**, tanpa API key dan tanpa biaya token.

## Struktur

```
project-ragkit/
├── README.md               ← kamu di sini
├── RUBRIK.md               ← penilaian mandiri
├── starter.ipynb           ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb            ← notebook SOLUSI (buka setelah selesai / mentok)
├── ragkit/                 ← paket yang kamu isi (stdlib murni)
│   ├── data.py             ← korpus, kueri emas, tolok ukur (JANGAN DIUBAH)
│   ├── embed.py            ← GIVEN: encoder embedding palsu deterministik
│   ├── chunking.py         ← GIVEN: chunking per kalimat + overlap
│   ├── generator.py        ← GIVEN: generator ekstraktif + abstain + sitasi
│   ├── vectorstore.py      ← TODO: vector store cosine similarity
│   ├── retriever.py        ← TODO: retriever + evaluasi hit@k
│   ├── abstain.py          ← TODO: gate "tidak ada di dokumen"
│   ├── cache.py            ← TODO: cache konteks retrieval + tolok ukur
│   ├── hybrid.py           ← TODO: BM25 mini + Reciprocal Rank Fusion
│   ├── rerank.py           ← TODO: reranker skor kata (pola cross-encoder)
│   ├── pipeline.py         ← TODO: pipeline RAG end-to-end + biaya
│   └── evaluate.py         ← TODO: evaluasi ujung-ke-ujung
├── tests/                  ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_vectorstore.py ( 9 test)
│   ├── test_retriever.py   ( 8 test)
│   ├── test_abstain.py     ( 7 test)
│   ├── test_cache.py       (10 test)
│   ├── test_hybrid.py      ( 9 test)
│   ├── test_rerank.py      ( 8 test)
│   ├── test_pipeline.py    (10 test)
│   └── test_evaluate.py    ( 5 test)
├── _calib.py               ← skrip audit: mencetak angka terkunci (bukan test)
├── _gen_notebooks.py
├── _verify_project.py      ← verifier otomatis (patch referensi + test + notebook)
└── solusi/
    └── ragkit_ref.py       ← implementasi referensi (untuk cek mandiri)
```

## Kenapa Embedding & Generatornya Palsu, Bukan API?

Pola yang sama dengan `transport.py` di Bab 11 — tiga alasan:

1. **Reproducible.** Embedding API asli berubah diam-diam. Hashing n-gram
   deterministik: teks sama → vektor sama, selamanya, di komputer siapa pun.
   Evaluasi RAG tanpa determinisme = angka yang tidak bisa dibandingkan.
2. **Gratis & tanpa jaringan.** Kamu boleh meng-index ulang seribu kali, menguji
   tiga konfigurasi, dan me-retrieval seratus kueri — tanpa biaya token.
3. **Semantiknya cukup untuk didik.** Hashing trick (teknik nyata di
   scikit-learn `HashingVectorizer`) membuat teks yang berbagi kata berbagi
   vektor → cosine similarity tinggi. Keterbatasannya (sinonim, urutan makna)
   justru pelajaran: itu sebabnya ada hybrid search.

"LLM"-nya adalah **generator ekstraktif** yang hanya mengutip kalimat konteks.
Ia tidak bisa mengarang — jadi kalau jawaban salah, penyebabnya retrieval atau
prompt, bukan "modelnya". Gate abstain + prompt ber-aturan tetap dilatih karena
itu yang wajib ada saat kamu ganti ke LLM nyata.

## Angka Terkunci (dari `_calib.py`, SEED 12)

```
CHUNKING (GIVEN)
  3 dokumen → 6 halaman → 7 chunk (ukuran 320 / overlap 64 karakter)
  [0] kebijakan-ketenagakerjaan.pdf p1   [3] kebijakan-cuti-kesehatan.pdf p1
  [1] kebijakan-ketenagakerjaan.pdf p1   [4] kebijakan-cuti-kesehatan.pdf p2
  [2] kebijakan-ketenagakerjaan.pdf p2   [5] faq-karyawan.pdf p1
                                         [6] faq-karyawan.pdf p2

VECTORSTORE + RETRIEVER
  q1 top-3  : [(5, 0.587), (0, 0.5134), (4, 0.3522)]
  hit@1 0.8 (q1 gagal — FAQ menang top-1) | hit@2 1.0 | hit@3 1.0

ABSTAIN GATE
  q6 (luar korpus) skor 0.0652 → abstain "tidak ada di dokumen"

PIPELINE (q1)
  jawaban "…12 hari cuti tahunan…" | sumber kebijakan-ketenagakerjaan.pdf p1
  biaya 5.91e-05 | retrieve_ms 47 | generate_ms 50
  e2e: akurasi 1.0 | abstain_benar True

HYBRID + RERANK
  hybrid q1 top-3 : [5, 0, 4] | hit@3 hybrid 1.0
  rerank q2       : [3, 5, 4] → [3, 4]

CACHE + TOLOK UKUR (4 permintaan, 2 kueri unik)
  pola miss, hit, hit, miss → 2 pencarian | hit_rate 0.5
  latensi [66, 52, 52, 66] ms | total biaya 2e-05
```

## Apa yang Kamu Bangun

1. **VectorStore** — cosine similarity brute-force dengan tie-break
   deterministik; paham kapan ANN dibutuhkan.
2. **Retriever + hit@k** — index sekali, evaluasi jujur sebelum bicara kualitas
   jawaban.
3. **Abstain gate** — timpa jawaban bila skor retrieval rendah; return dict BARU,
   tidak mengganti abstain → jawaban.
4. **Cache konteks** — kunci = kueri ternormalisasi; TTL tick logis;
   hit/miss/expired terpisah; tolok ukur latensi & hit-rate.
5. **Hybrid search** — BM25 mini (stdlib) + RRF; kata kunci eksak menemani makna.
6. **Reranker** — skor murah per kandidat, ambil top-k terbaik untuk konteks LLM.
7. **Pipeline end-to-end** — retrieve → prompt → generate → gate → biaya +
   latensi per tahap.
8. **Evaluasi ujung-ke-ujung** — akurasi terhadap jawaban emas + kewajiban abstain.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_vectorstore.py`.
2. Isi satu modul, jalankan:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan: `vectorstore.py` → `retriever.py` → `abstain.py` → `cache.py` →
   `hybrid.py` → `rerank.py` → `pipeline.py` → `evaluate.py` sampai **66 test hijau**.
4. Buka `starter.ipynb` — jalankan delapan blok eksperimen (angka terkunci), lalu
   jawab **6 Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/ragkit_ref.py` / `solusi.ipynb`.

Ingin melihat angka terkunci tanpa menjalankan notebook?

```bash
python _calib.py
```

> 💡 Tiga jebakan yang menyumbang kebanyakan bug di project ini:
> (1) `abstaikan` harus return **dict baru** dan tidak menimpa abstain yang sudah
> ada; (2) kunci cache = `strip().lower()` — kueri "Berapa" dan "berapa" adalah
> permintaan yang sama; (3) skor RRF = `1/(k + peringkat + 1)` dengan peringkat
> 0-based — salah offset satu saja, urutan hybrid berubah.

## Pertanyaan Analisis (bagian penilaian!)

1. hit@1 = 0.8 tapi hit@2 = 1.0 — kueri mana yang gagal dan mengapa? Apa
   konsekuensinya untuk pilihan `top_k`?
2. Apa yang terjadi bila ambang abstain terlalu rendah / terlalu tinggi? Uji
   dengan 0.15 dan 0.45 di Blok 3.
3. Kunci cache menyatukan kueri yang hanya beda huruf besar-kecil. Kueri yang
   beda kata tapi semantik sama — kena cache atau tidak? Apa risikonya?
4. Hybrid menambah biaya pencarian per permintaan. Kapan layak, kapan tidak?
5. Rerank memangkas konteks 3 → 2 chunk. Informasi apa yang bisa hilang, dan
   kapan pemangkasan itu lebih baik?
6. Generator ekstraktif tidak bisa mengarang; LLM nyata bisa. Lapisan mana yang
   tetap wajib saat generator diganti model nyata?

## Prasyarat

```bash
# tidak ada!
```

> Stdlib murni (`math`, `re`, `hashlib`). Tidak ada numpy, tidak ada faiss, tidak
> ada langchain — supaya *mekanismenya* terlihat. Setelah paham, memakai library
> produksi terasa seperti memakai ulang kode yang kamu tulis sendiri.

> Bab ini memakai kembali pola dari Bab 11 (cache, determinisme, TDD, biaya) dan
> persiapan Bab 15 (evaluasi terukur, bukan "rasanya oke"). Bab 14 memakai
> retrieval ini sebagai tool di tangan agent.
