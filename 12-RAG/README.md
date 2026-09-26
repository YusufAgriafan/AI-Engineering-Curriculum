# Bab 12 — RAG: Retrieval Augmented Generation

> Teknik paling banyak dipakai di industri: beri model "buku catatan terbuka"
> sehingga jawabannya berbasis dokumen Anda, bukan ingatan yang bisa mengarang.

## 🎯 Tujuan Belajar

- Pipeline lengkap RAG: load → chunk → embed → index → retrieve → generate.
- Memilih embedding model & memahami vector search (dot product dari Bab 2!).
- Hybrid search + reranking untuk kualitas retrieval.
- Evaluasi RAG (faithfulness, relevance) → lanjut ke Bab 15.

## 🗂️ Materi Pendukung Bab Ini

| Materi | File | Isi |
|---|---|---|
| 🧪 Lab | [`01_lab_rag.ipynb`](01_lab_rag.ipynb) | 8 bagian: chunking → embedding & cosine → vector store → hit@k → prompt + generator + abstain → hybrid & rerank → pipeline + biaya → ringkasan & analisis |
| 📝 Kuis | [`02_kuis_rag.ipynb`](02_kuis_rag.ipynb) | 10 soal (6 PG + 4 coding), 22 poin, dinilai otomatis, **mandiri** (encoder mini deterministik + korpus mini) |
| 🔑 Kunci | [`03_kunci_jawaban_kuis_rag.ipynb`](03_kunci_jawaban_kuis_rag.ipynb) | Jawaban + **kenapa**, kode referensi tiap soal coding |
| 🏗️ Project | [`project-ragkit/`](project-ragkit/README.md) | Paket `ragkit` TDD **66 test**: vectorstore, retriever + hit@k, abstain gate, cache konteks, hybrid BM25+RRF, reranker, pipeline e2e, evaluasi |
| 📄 Cheatsheet | [`cheatsheet-rag.md`](cheatsheet-rag.md) | Ringkas 1 halaman untuk review cepat |

Materi lab & project **tanpa API key**: embedding diganti
`project-ragkit/ragkit/embed.py` — encoder hashing n-gram yang
**deterministik** (teks sama → vektor sama, di komputer siapa pun) — dan
generatornya **ekstraktif**: hanya mengutip kalimat konteks, tidak bisa
mengarang. Jadi kalau jawaban salah, penyebabnya retrieval atau prompt, bukan
"modelnya". Angka terkunci SEED 12 dan diverifikasi otomatis
(`project-ragkit/_verify_project.py`, `_verify_quiz.py`, `_verify_lab.py`).

---

## 1. Materi Inti

### 1.1 Kenapa RAG?

```
Masalah yang RAG selesaikan:

❌ Tanpa RAG:
  User: "Apa kebijakan cuti tahunan di perusahaan kita?"
  LLM: "Hmm... saya tidak tahu karena tidak dilatih dengan data perusahaan Anda."
  atau lebih buruk:
  LLM: "Kebijakan cuti tahunan biasanya 12 hari..." ← HALUSINASI!

✓ Dengan RAG:
  User: "Apa kebijakan cuti tahunan di perusahaan kita?"
  System → retrieval dokumen kebijakan → menemukan "Kebijakan CUTI"
  LLM: "Berdasarkan Kebijakan Ketenagakerjaan (2024), cuti tahunan adalah 12 hari
        untuk karyawan tetap. [Sumber: kebijakan-ketenagakerjaan.pdf, halaman 5]"
```

| Pendekatan | Kapan | Kelemahan |
|---|---|---|
| **Prompt dengan data** | Data sedikit, statis | Boros token, konteks terbatas, data kadaluarsa |
| **Fine-tuning** | Perilaku/format khusus, data stabil | Mahal, tidak cocok untuk fakta yang sering berubah, risiko halusinasi jika tidak done right |
| **RAG** | Data privat, fakta berubah, dokumen besar | Kompleksitas pipeline; **kualitas retrieval menentukan kualitas jawaban** |

> **Aturan:** RAG adalah solusi pertama untuk data privat/fakta. Fine-tuning
> untuk perilaku/format (Bab 13).

---

### 1.2 Pipeline Lengkap RAG

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE RAG END-TO-END                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   INGESTION (sekali atau periodic)                                        │
│   ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌───────────┐        │
│   │   LOAD   │ →  │  CHUNK   │ →  │  EMBED  │ →  │   INDEX   │        │
│   │ PDF/HTML │    │ 300-800t │    │  VECTOR │    │  VECTORDB │        │
│   │ /docx   │    │ overlap  │    │  MODEL   │    │  + META   │        │
│   └──────────┘    └───────────┘    └─────────┘    └───────────┘        │
│                                                                           │
│   QUERY TIME                                                              │
│   ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌───────────┐        │
│   │   USER   │ →  │  EMBED   │ →  │  RETRIE │ →  │  GENERA-  │        │
│   │  QUERY   │    │  QUERY   │    │   VE +  │    │   TION    │        │
│   │          │    │          │    │ RECALL  │    │  + SITA-  │        │
│   │          │    │          │    │ (top-k) │    │   SI      │        │
│   └──────────┘    └───────────┘    └───────────┘    └───────────┘        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

Dua fase, dua anggaran:

- **Ingestion** dijalankan sekali / periodic — boleh lambat dan mahal.
- **Query time** dijalankan tiap permintaan — harus cepat dan murah; di sinilah
  cache konteks (§1.4–1.5 project) dan top-k yang tepat bekerja.

---

### 1.3 Pipeline Ingestion — Detail

#### Step 1: Load (Ekstraksi Teks dari Dokumen)

```python
# Format dokumen yang umum & cara ekstrak

from pathlib import Path

def load_document(file_path: str) -> str:
    """Ekstrak teks dari berbagai format dokumen."""
    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages])

    elif ext in ['.docx', '.doc']:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    elif ext in ['.html', '.htm']:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        text = soup.get_text(separator='\n')

    else:
        raise ValueError(f"Format tidak didukung: {ext}")

    return text

# Contoh penggunaan
# text = load_document('dokumen.pdf')
```

#### Step 2: Chunking — Strategi & Trade-off

```
Chunking = memotong dokumen menjadi potongan-potongan yang:
  1. Cukup kecil untuk dimasukkan ke context window LLM
  2. Cukup besar untuk mengandung informasi yang bermakna

Trade-off:
  ┌─────────────────────┬─────────────────────┐
  │  Chunk KECIL        │  Chunk BESAR         │
  ├─────────────────────┼─────────────────────┤
  │ ✓ Fokus, kurang noise│ ✓ Konteks lengkap    │
  │ ✓ Hemat token        │ ✗ Lebih banyak token │
  │ ✗ Kehilangan konteks │ ✗ Bisa ada info tidak relevan │
  │ ✗ Sulit jawab pertanyaan kompleks │ ✓ Jawab pertanyaan kompleks lebih mudah │
  └─────────────────────┴─────────────────────┘
```

Aturan praktis: **300–800 token** per chunk, **overlap 10–20%**, potong di
batas natural (kalimat/paragraf/heading). Dan ingat: mengubah chunking =
mengubah embedding = **index dibangun ulang**.

```python
import re

def chunk_by_paragraph(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list:
    """Chunk dokumen berdasarkan paragraf."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_size = len(para)

        # Kalau paragraf sendiri sudah lebih besar dari max_chunk_size,
        # potong dengan sentence boundary
        if para_size > max_chunk_size:
            # Potong per kalimat
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_size + len(sentence) > max_chunk_size and current_chunk:
                    # Simpan chunk saat ini + overlap
                    chunks.append(' '.join(current_chunk))
                    # Overlap: ambil beberapa kalimat terakhir
                    current_chunk = current_chunk[-max(1, len(current_chunk) - overlap // 50):]
                    current_size = sum(len(s) for s in current_chunk)

                current_chunk.append(sentence)
                current_size += len(sentence)

        elif current_size + para_size > max_chunk_size and current_chunk:
            # Chunk penuh, simpan dan mulai baru
            chunks.append(' '.join(current_chunk))
            # Overlap
            current_chunk = current_chunk[-max(1, len(current_chunk) - 1):]
            current_size = sum(len(s) for s in current_chunk)
            current_chunk.append(para)
            current_size += para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    # Chunk terakhir
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Metadata untuk tiap chunk — menempel SEJAK chunking sampai sitasi akhir
class ChunkMetadata:
    def __init__(self, source: str, page: int = None, section: str = None):
        self.source = source
        self.page = page
        self.section = section
        self.chunk_index = None  # diisi saat chunking
```

#### Step 3: Embedding

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_text(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed teks ke vektor numerik."""
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    embedding = response.data[0].embedding
    return np.array(embedding)

# Contoh
# text = "Machine learning adalah bidang studi yang memberikan komputer kemampuan untuk belajar tanpa diprogram secara eksplisit."
# embedding = embed_text(text)
# print(f"Embedding shape: {embedding.shape}")  # (1536,) untuk text-embedding-3-small
# print(f"Embedding[:5]: {embedding[:5]}")

# Pilihan embedding model untuk bahasa Indonesia:
EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "provider": "OpenAI",
        "dimensions": 1536,
        "multilingual": True,
        "cost_per_1m": 0.02,  # USD
        "cocok untuk": "Umum, cepat, murah"
    },
    "text-embedding-3-large": {
        "provider": "OpenAI",
        "dimensions": 3072,
        "multilingual": True,
        "cost_per_1m": 0.13,
        "cocok untuk": "Quality lebih tinggi, lebih mahal"
    },
    "gemini-embedding": {
        "provider": "Google",
        "dimensions": 768,
        "multilingual": True,
        "cocok untuk": "Jika pakai ekosistem Google"
    },
    "bge-m3": {
        "provider": "Open-source (BAAI)",
        "dimensions": 1024,
        "multilingual": True,
        "cocok untuk": "Offline, gratis, multilingual bagus"
    },
    "multilingual-e5": {
        "provider": "Microsoft (open-source)",
        "dimensions": 1024,
        "multilingual": True,
        "cocok untuk": "Alternatif open-source multilingual"
    },
}

print("\nPilihan embedding model untuk bahasa Indonesia:")
for name, info in EMBEDDING_MODELS.items():
    print(f"  - {name}: {info['cocok untuk']} (dim={info['dimensions']})")
```

> Ganti model embedding = **embed ulang semua chunk**. Untuk latihan, lab &
> project memakai encoder hashing deterministik (`ragkit/embed.py`) supaya
> angka bisa dibandingkan antar hari — antarmukanya sama persis dengan API nyata.

#### Step 4: Index ke Vector Database

```python
# Pilihan vector database

VECTOR_DB_OPTIONS = {
    "FAISS (Facebook AI Similarity Search)": {
        "tipe": "Local library",
        "cocok untuk": "Prototipe, small-medium scale, gratis",
        "install": "pip install faiss-cpu",
        "keterangan": "Tidak ada server, langsung di-memory"
    },
    "SQLite-Vec": {
        "tipe": "Local SQLite extension",
        "cocok untuk": "Small scale, embedded, gratis",
        "install": "pip install sqlite-vec",
        "keterangan": "SQLite dengan vector search capability"
    },
    "pgvector (PostgreSQL)": {
        "tipe": "Database extension",
        "cocok untuk": "Production, sudah pakai Postgres",
        "install": "Install extension di Postgres",
        "keterangan": "Vector search + relational queries"
    },
    "Qdrant": {
        "tipe": "Dedicated vector DB",
        "cocok untuk": "Production, scalable, open-source",
        "install": "Docker atau cloud",
        "keterangan": "Fitur lengkap: filter, payload, hybrid search"
    },
    "Pinecone": {
        "tipe": "Managed cloud",
        "cocok untuk": "Production, tanpa infrastruktur",
        "install": "Cloud service",
        "keterangan": "Managed, scalable, tapi berbayar"
    },
    "Weaviate": {
        "tipe": "Dedicated vector DB",
        "cocok untuk": "Production, GraphQL interface",
        "install": "Docker atau cloud",
        "keterangan": "Object storage + vector search"
    },
}

print("Pilihan Vector Database:")
for name, info in VECTOR_DB_OPTIONS.items():
    print(f"  - {name}")
    print(f"    {info['cocok untuk']}")

# Contoh FAISS sederhana
import faiss

class SimpleVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (dot product)
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadata_list: list):
        """Tambah embeddings ke index."""
        # FAISS butuh float32 dan proper shape
        embeddings = embeddings.astype('float32')
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """Cari chunk terkait berdasarkan cosine similarity."""
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # FAISS inner product = dot product
        # Kalau mau cosine similarity, normalize dulu
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "metadata": self.metadata[idx]
                })

        return results

# Contoh penggunaan
dimension = 1536  # text-embedding-3-small
vector_store = SimpleVectorStore(dimension)

# Tambah beberapa chunk
embeddings = np.random.randn(10, dimension).astype('float32')
metadata = [{"source": f"doc_{i}.pdf", "page": i+1} for i in range(10)]
vector_store.add(embeddings, metadata)

# Search
query_embedding = np.random.randn(dimension).astype('float32')
results = vector_store.search(query_embedding, top_k=3)

print("\nHasil search (simulasi):")
for r in results:
    print(f"  Rank {r['rank']}: score={r['score']:.4f}, sumber={r['metadata']['source']}")
```

> Brute-force cosine O(N) cukup sampai ribuan chunk. ANN (HNSW, IVF) baru
> diperlukan saat korpus jutaan — dan ia menukar sedikit recall untuk kecepatan.

---

### 1.4 Hybrid Search & Reranking

#### Kenapa Hybrid?

```
❌ Masalah pure vector search:
  Query: "Berapa harga iPhone 15 di toko kami?"
  → Embedding mungkin mirip dengan dokumen yang bahas "harga smartphone"
  → Tapi nggak tepat: dokumen yang benar sebaiknya ada kata "iPhone 15" secara eksplisit

✓ Hybrid search:
  - Vector search: semantic similarity (makna)
  - Keyword search (BM25): exact match kata kunci
  → Gabung hasil keduanya → lebih robust
```

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, chunks: list, embeddings: np.ndarray, vector_store, rrf_k: int = 60):
        """
        Fusi dengan Reciprocal Rank Fusion (RRF); rrf_k = konstanta peredam
        (60 lazim). Metadata tiap chunk WAJIB punya 'chunk_index'.
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        # Buat BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 10) -> list:
        """Hybrid search: gabungkan vector + BM25."""
        # 1. Vector search
        query_embedding = embed_text(query)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k*2)

        # 2. BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = np.argsort(bm25_scores)[::-1][:top_k*2]

        # 3. Fusi Reciprocal Rank Fusion (RRF):
        #    skor = Σ 1/(k + peringkat + 1)  — peringkat 0-BASED.
        #    Chunk yang muncul di KEDUA daftar menjumlahkan dua kontribusi —
        #    konsistensi di banyak daftar mengalahkan peringkat-atas di satu daftar.
        rrf_scores = {}
        for rank, r in enumerate(vector_results):
            idx = r["metadata"]["chunk_index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # 4. Urutkan menurun + tie-break deterministik (index kecil dulu), ambil top-k
        ranked = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [self.chunks[idx] for idx, _ in ranked]
```

> Dua jebakan RRF: memakai peringkat **1-based** saat rumus mengharapkan
> 0-based (semua skor bergeser, urutan hybrid berubah), dan melupakan `+1`
> (peringkat pertama mendapat pembagian nol). Diuji di kuis soal 9.

#### Reranker

```python
# Reranker = model yang lebih "paham" konteks untuk ranking ulang

# Flow (pola two-stage):
# 1. Retriever murah mengambil top-50 (cepat tapi kurang presisi)
# 2. Reranker scoring ulang tiap pasangan (query, chunk) → top-5 presisi

# Contoh dengan cross-encoder
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-large')

def rerank(query: str, candidates: list, top_k: int = 5) -> list:
    """Reranking dengan cross-encoder."""
    pairs = [[query, candidate] for candidate in candidates]
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

Trade-off yang harus sadar: reranker selalu **memangkas** — informasi pada
kandidat yang dibuang hilang dari konteks. Layak bila presisi konteks lebih
penting daripada cakupan.

---

### 1.5 Generation + Sitasi

```python
PROMPT_TEMPLATE = """
Anda adalah asisten yang menjawab pertanyaan berdasarkan dokumen yang diberikan.

INSTRUKSI:
- Jawab pertanyaan berdasarkan KONTEKS yang diberikan saja
- Jika informasi tidak ada di KONTEKS, jawab "Maaf, saya tidak memiliki informasi tentang hal tersebut"
- JANGAN mengarang atau menebak
- Sertakan sitasi sumber untuk setiap klaim [nomor]

KONTEKS:
{context}

PERTANYAAN:
{question}

FORMAT OUTPUT:
Jawaban: [jawaban]
Sumber: [daftar sumber dengan nomor]
"""

def build_rag_prompt(question: str, retrieved_chunks: list, metadata_list: list) -> str:
    """Build prompt untuk RAG."""
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata_list), 1):
        context_parts.append(
            f"[{i}] (sumber: {meta.get('source', 'dokumen')}, halaman {meta.get('page', 'N/A')})\n"
            f"    {chunk}"
        )

    context = "\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)
```

Empat aturan prompt RAG yang tidak bisa ditawar: jawab **hanya** dari konteks ·
tidak ada → frasa abstain yang konsisten · sebut sumber `[n]` · jangan mengarang
angka. Ditambah **abstain gate** di kode (skor retrieval rendah → timpa jawaban),
bukan hanya di prompt — prompt bisa diabaikan model, gate tidak.

---

### 1.6 Troubleshooting RAG

| Masalah | Kemungkinan Penyebab | Solusi |
|---|---|---|
| Jawaban tidak relevan | Chunk tidak relevan, embedding model kurang bagus | Cek retrieval quality (hit@k), coba embedding model lain, tambah reranker |
| Jawaban terlalu umum | Konteks kurang spesifik | Kurangi chunk size, tambah filtering metadata |
| Jawaban too long | Chunk terlalu besar atau terlalu banyak | Kurangi top-k, kurangi chunk size |
| Tidak menemukan jawaban | Dokumen tidak ada di index, chunking salah, embedding gagal | Cek ingestion pipeline, coba query berbeda |
| Halusinasi | Prompt kurang ketat; model mengabaikan konteks | Perketat aturan prompt + abstain gate di kode (skor < ambang → "tidak ada di dokumen") |
| Sumber tidak akurat | Metadata tidak disimpan dengan benar saat chunking | Pastikan metadata (source, page, chunk_index) konsisten sejak chunking |

---

### 1.7 Decision Tree: Chunking Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│  METODE CHUNKING YANG TEPAT                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Dokumen jenis apa?                                                 │
│  ├── 📄 Dokumen terstruktur (PDF, DOCX, HTML)                      │
│  │   ├── Ada heading/bab? → Chunk per bab/subsection              │
│  │   └── Nggak ada? → Chunk per paragraf                          │
│  │                                                                 │
│  ├── 💬 Chat/transkrip                                             │
│  │   └── Chunk per pesan/utterance, simpan speaker metadata        │
│  │                                                                 │
│  ├── 📊 Data tabular (CSV, database)                               │
│  │   └── Row-based chunking, atau embed baris yang relevan         │
│  │                                                                 │
│  └── 📝 Kode sumber                                                │
│      └── Chunk per fungsi/kelas, simpan nama fungsi & file         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.8 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 12:
  1. RAG = retrieval + generation, solusi untuk data privat & fakta
  2. Pipeline: Load → Chunk → Embed → Index → Retrieve → Generate
  3. Chunking: 300-800 token, overlap 10-20%, potong di batas natural
  4. Embedding: pilih model multilingual untuk bahasa Indonesia
  5. Vector DB: mulai FAISS/SQLite-vec, skalakan ke Qdrant/Pinecone
  6. Hybrid search: vector + BM25, fusi RRF — Σ 1/(k + peringkat + 1)
  7. Reranker: cross-encoder untuk ranking ulang top-k
  8. Prompt: strict constraint, sitasi, abstain gate DI KODE
  9. Framework: paham pipeline manual dulu, baru pakai LangChain/LlamaIndex
  10. Evaluasi: hit@k + faithfulness → pakai RAGAS (Bab 15)
```

---

## 2. Latihan Praktis

Materi latihan bab ini sudah lengkap sebagai file terpisah (lihat tabel
**Materi Pendukung** di atas). Urutan yang disarankan:

1. **Lab** — [`01_lab_rag.ipynb`](01_lab_rag.ipynb): bangun pipeline RAG dari
   nol (chunking → cosine → vector store → hit@k → prompt + generator + abstain
   → hybrid & rerank → pipeline + biaya). Selesaikan semua cell **✅ Cek**
   (angka terkunci SEED 12), lalu jawab pertanyaan analisis di Bagian 8.
2. **Kuis** — [`02_kuis_rag.ipynb`](02_kuis_rag.ipynb): 10 soal, 22 poin,
   dinilai otomatis, **mandiri** (encoder mini deterministik + korpus mini
   sudah disediakan di cell setup). Baru setelah mencoba serius: buka
   [`03_kunci_jawaban_kuis_rag.ipynb`](03_kunci_jawaban_kuis_rag.ipynb) dan
   baca bagian **kenapa**-nya, bukan sekadar kuncinya.
3. **Project (TDD)** — [`project-ragkit/`](project-ragkit/README.md): lengkapi
   8 modul TODO (`vectorstore` → `retriever` → `abstain` → `cache` → `hybrid`
   → `rerank` → `pipeline` → `evaluate`) sampai **66 test hijau**:

   ```bash
   cd project-ragkit
   python -m unittest discover -s tests -v
   ```

   Lalu kerjakan `starter.ipynb` (8 blok eksperimen + 6 pertanyaan analisis),
   isi `RUBRIK.md`, dan baru bandingkan dengan `solusi/ragkit_ref.py`.
   Ingin melihat angka terkunci tanpa menjalankan notebook? `python _calib.py`.
4. **Cheatsheet** — [`cheatsheet-rag.md`](cheatsheet-rag.md): review cepat
   sebelum lanjut ke Bab 13.

> Semua angka (skor cosine, hit@k, hit-rate cache, biaya per tahap) terkunci
> SEED 12 — hasil eksekusimu harus sama persis. Kalau beda, cari dulu
> penyebabnya: itulah kegunaan determinisme.

---

## 📚 Referensi Online

| Sumber | Tipe | Keterangan |
|---|---|---|
| [LlamaIndex — Building RAG](https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/) | Dokumentasi | Tutorial RAG dari nol dengan LlamaIndex |
| [Pinecone Learning Center — RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/) | Tutorial | Panduan visual dan mudah dipahami |
| [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) | Dokumentasi | Info embedding model OpenAI |
| [RAGAS Evaluation Framework](https://docs.ragas.io/) | Library | Evaluasi RAG yang terukur |
| [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | Blog | Teknik meningkatkan kualitas chunk |
| [Retrieval Augmented Generation (paper)](https://arxiv.org/abs/2005.11401) | Paper | Paper asli RAG |

---

## ✅ Checklist Kompetensi

- [ ] Membangun RAG end-to-end tanpa framework (paham tiap langkah)
- [ ] Menjelaskan chunking strategy pilihanmu beserta trade-off-nya
- [ ] Punya angka evaluasi retrieval & faithfulness, bukan "rasanya oke"
- [ ] Implementasi chunking per dokumen jenis berbeda
- [ ] Paham perbedaan embedding models dan implikasi multilingual
- [ ] Setup vector store dan retrieval sederhana
- [ ] Evaluasi retrieval quality dengan metrik sederhana
- [ ] Kuis bab ini selesai (22 poin) — atau review kunci sampai paham **kenapa**
- [ ] Project `ragkit` 66 test hijau + 6 pertanyaan analisis terjawab
