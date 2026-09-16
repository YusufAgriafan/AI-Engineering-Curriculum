# Bab 12 — RAG: Retrieval Augmented Generation

> Teknik paling banyak dipakai di industri: beri model "buku catatan terbuka" sehingga jawabannya berbasis dokumen Anda, bukan ingatan yang bisa mengarang.

## 🎯 Tujuan Belajar
- Pipeline lengkap RAG: load → chunk → embed → index → retrieve → generate.
- Memilih embedding model & memahami vector search (dot product dari Bab 2!).
- Hybrid search + reranking untuk kualitas retrieval.
- Evaluasi RAG (faithfulness, relevance) → lanjut ke Bab 15.

## 1. Materi Inti

> Teknik paling banyak dipakai di industri: beri model "buku catatan terbuka" sehingga jawabannya berbasis dokumen Anda, bukan ingatan yang bisa mengarang.

---

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
  LLM: "Berdasarkan Kebijakan Ketenagakerja (2024), cuti tahunan adalah 12 hari 
        untuk karyawan tetap. [Sumber: kebijakan-ketenagakerja.pdf, halaman 5]"
```

| Pendekatan | Kapan | Kelemahan |
|---|---|---|
| **Prompt dengan data** | Data sedikit, statis | Boros token, konteks terbatas, data kadaluarsa |
| **Fine-tuning** | Perilaku/format khusus, data stabil | Mahal, tidak cocok untuk fakta yang sering berubah, risiko halusinasi jika tidak done right |
| **RAG** | Data privat, fakta berubah, dokumen besar | Kompleksitas pipeline, retrieval qualitypengaruhi jawaban |

> **Aturan:** RAG adalah solusi pertama untuk data privat/fakta. Fine-tuning untuk perilaku/format.

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
│   │ /docx   │    │ overlap  │    │ MODEL   │    │  + META   │        │
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
# text = load_document(' dokumen.pdf')
```

#### Step 2: Chunking — Strategi & Trade-off

```
Chunking = memotong dokumen menjadi potongan-pontongan yang:
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

# Metadata untuk tiap chunk
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
    def __init__(self, chunks: list, embeddings: np.ndarray, vector_store, alpha: float = 0.5):
        """
        alpha = bobot vector search vs BM25
        alpha=1 → pure vector, alpha=0 → pure BM25
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.alpha = alpha
        
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
        
        # 3. Gabungkan dengan Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(vector_results):
            doc_id = idx.get("metadata", {}).get("chunk_index", rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/(alpha * (rank + 60) + (1-alpha) * 60)
        
        for rank, idx in enumerate(bm25_ranked):
            doc_id = idx
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/((1-alpha) * (rank + 60) + alpha * 60)
        
        # 4. Sort danambil top-k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [self.chunks[doc_id] for doc_id, _ in ranked]
```

#### Reranker

```python
# Reranker = model yang lebih "paham" konteks untuk ranking ulang

# Flow:
# 1. Biarai retriever chunk top-50 (cepat tapi kurang akurat)
# 2. Reranker scoring ulang tiap chunk vs query → top-5 lebih akurat

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

---

### 1.6 Troubleshooting RAG

| Masalah | Kemungkinan Penyebab | Solusi |
|---|---|---|
| Jawaban tidak relevan | Chunk tidak relevan, embedding model kurang bagus | Cek retrieval quality, coba embedding model lain, tambah reranker |
| Jawaban terlalu umum | Konteks kurang spesifik | Kurangi chunk size, tambah filtering metadata |
| Jawaban too long | Chunk terlalu besar atau terlalu banyak | Kurangi top-k, kurangi chunk size |
| Tidak menemukan jawaban | Dokumen tidak ada di index, chunking salah, embedding gagal | Cek ingestion pipeline, coba query berbeda |
| Halusinasi | Prompt tidak cukup_constraints_, model ignored context | Tambah constraint di prompt, lebih strict: "JANGAN jawab kalau tidak ada di konteks" |
| Sumber tidak akurat | Metadata tidak disimpan dengan benar saat chunking | Pastikan metadata konsisten, perbaiki chunking logic |

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
  6. Hybrid search: vector + BM25, fusi RRF
  7. Reranker: cross-encoder untuk ranking ulang top-k
  8. Prompt: strict constraint, sitasi, "TIDAK ADA = jawab tidak tahu"
  9. Framework: paham pipeline manual dulu, baru pakai LangChain/LlamaIndex
  10. Evaluasi: faithfulness, answer relevancy → pakai RAGAS
```

---

## 2. Latihan Praktis

---

### Proyek Mini: RAG End-to-End Tanpa Framework

**Tujuan:** Bangun RAG pipeline dari nol sampai bisa tanya-jawab dokumen.

```python
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Simulasi dataset dokumen
documents = [
    {
        "id": "doc1",
        "title": "Kebijakan Ketenagakerja",
        "source": "kebijakan-ketenagakerja.pdf",
        "page": 1,
        "content": """
CUTI TAHUNAN
Kebijakan cuti tahunan untuk karyawan tetap adalah sebagai berikut:
1. Karyawan tetap mendapatkan 12 hari cuti tahunan per tahun kalender.
2. Cuti tahunan harus digunakan dalam periode yang sama, tidak bisa ditumpuk.
3. Permohonan cuti harus diajukan minimal 14 hari sebelum tanggal cuti.
4. Cuti tahunan tidak including weekend dan hari libur nasional.
"""
    },
    {
        "id": "doc2",
        "title": "Kebijakan Cuti Kesehatan",
        "source": "kebijakan-cuti-kesehatan.pdf",
        "page": 3,
        "content": """
CUTI SAKIT
1. Karyawan dapat mengajukan cuti sakit dengan membawa surat dokter.
2. Cuti sakit pertama tidak menggunakan SiC, cuti sakit kedua dan seterusnya menggunakan SiC.
3. Maksimum cuti sakit 5 hari per kejadian tanpa dokumen medis.
4. Untuk epilepsi, jangka panjang, dll, jadi kasus khusus yang harus dikonsultasi HR.
"""
    },
    {
        "id": "doc3",
        "title": "FAQ Karyawan",
        "source": "faq-karyawan.pdf",
        "page": 5,
        "content": """
FAQS KEBIJAKAN PERUSAHAAN
Q: Bagaimana cara mengajukan cuti?
A: Submit form cuti via HR portal minimal 14 hari sebelumnya.

Q: Berapa hari cuti tahunan?
A: 12 hari per tahun kalender untuk karyawan tetap.

Q: Bisakah cuti tahunan ditunda?
A: Tidak bisa ditunda. Jika tidak digunakan dalam tahun yang sama, akan hangus.

Q: Apakah ada cuti khusus?
A: Ya, ada cuti hamil, cuti pria untuk melahirkan anak (3 hari), cuti sakit, dll.
"""
    },
]

# Ingestion Pipeline
print("="*70)
print("LANGKAH 1: INGESTION PIPELINE")
print("="*70)

class Chunk:
    def __init__(self, text: str, doc_id: str, source: str, page: int, index: int):
        self.text = text
        self.doc_id = doc_id
        self.source = source
        self.page = page
        self.index = index
    
    def to_dict(self):
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.index
        }

def chunk_document(doc: dict, chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    """Potong dokumen menjadi chunks."""
    text = doc["content"]
    sentences = text.replace('\n', ' ').split('. ')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Simpan chunk saat ini
            chunk_text = ". ".join(current_chunk) + "."
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc["id"],
                source=doc["source"],
                page=doc["page"],
                index=len(chunks)
            ))
            
            # Overlap: ambil beberapa kalimat terakhir
            overlap_count = max(1, len(current_chunk) - 2)
            current_chunk = current_chunk[-overlap_count:]
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Chunk terakhir
    if current_chunk:
        chunk_text = ". ".join(current_chunk) + "."
        chunks.append(Chunk(
            text=chunk_text,
            doc_id=doc["id"],
            source=doc["source"],
            page=doc["page"],
            index=len(chunks)
        ))
    
    return chunks

# Chunk semua dokumen
all_chunks = []
for doc in documents:
    doc_chunks = chunk_document(doc)
    all_chunks.extend(doc_chunks)
    print(f"\nDokumen: {doc['title']}")
    print(f"  Chunks created: {len(doc_chunks)}")
    for i, chunk in enumerate(doc_chunks, 1):
        print(f"  Chunk {i}: {len(chunk.text)} chars")

print(f"\nTotal chunks: {len(all_chunks)}")

# Embedding (simulasi karena butuh API key)
print("\n" + "="*70)
print("LANGKAH 2: EMBEDDING (simulasi dengan random untuk demo)")
print("="*70)

class MockEmbeddingModel:
    """Mock embedding model untuk demonstrasi."""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return random embeddings untuk demonstrasi."""
        n = len(texts)
        embeddings = np.random.randn(n, self.dimension).astype('float32')
        # Normalize untuk cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

embedding_model = MockEmbeddingModel(dimension=768)

# Embed semua chunks
chunk_texts = [chunk.text for chunk in all_chunks]
chunk_embeddings = embedding_model.embed(chunk_texts)

print(f"Total chunks: {len(all_chunks)}")
print(f"Embedding dimension: {chunk_embeddings.shape[1]}")
print(f"Embedding matrix shape: {chunk_embeddings.shape}")

# Vector Store
print("\n" + "="*70)
print("LANGKAH 3: VECTOR STORE (FAISS-like)")
print("="*70)

class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.chunks = []
        self.metadata = []
    
    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.metadata = [chunk.to_dict() for chunk in chunks]
        self.embeddings = embeddings
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Cari chunk terdekat menggunakan cosine similarity."""
        # Cosine similarity = dot product (karena sudah normalized)
        similarities = np.dot(query_embedding, self.embeddings.T)
        
        # Sort danambil top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "chunk": self.chunks[idx],
                "similarity": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        
        return results

vector_store = VectorStore()
vector_store.add(all_chunks, chunk_embeddings)

print(f"Vector store ready: {len(all_chunks)} chunks indexed")

# Retrieval
print("\n" + "="*70)
print("LANGKAH 4: RETRIEVAL (query → chunks)")
print("="*70)

test_queries = [
    "Berapa hari cuti tahunan untuk karyawan tetap?",
    "Berapa lama cuti sakit tanpa dokumen medis?",
    "Bagaimana cara mengajukan cuti?",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    query_embedding = embedding_model.embed_query(query)
    results = vector_store.search(query_embedding, top_k=2)
    
    print(f"  Top {len(results)} chunks:")
    for r in results:
        print(f"    [{r['rank']}] Similarity: {r['similarity']:.4f}")
        print(f"        Sumber: {r['metadata']['source']}, halaman {r['metadata']['page']}")
        print(f"        Chunk: {r['chunk'].text[:80]}...")

# Generation
print("\n" + "="*70)
print("LANGKAH 5: GENERATION + SITSIN")
print("="*70)

# Simulate LLM response
def generate_rag_answer(question: str, retrieved_chunks: List[Chunk], metadata_list: List[Dict]) -> str:
    """Generate jawaban berdasarkan retrieved chunks."""
    
    # Build context
    context = []
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata_list), 1):
        context.append(f"[{i}] ({meta['source']}, halaman {meta['page']}): {chunk.text}")
    
    # Simulated LLM response (karena kita nggak punya API key)
    answer = f"""
Berdasarkan informasi yang ditemukan:

"""
    
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata_list), 1):
        # Sederhana: ambil informasi yang relevan dari chunk
        sentences = chunk.text.split('. ')
        answer += f"[{i}] {sentences[0]}.\n"
    
    answer += f"\nSumber: "
    answer += ", ".join([f"[{i}] {meta['source']}" for i, meta in enumerate(metadata_list, 1)])
    
    return answer

for query in test_queries:
    print(f"\nQuery: '{query}'")
    query_embedding = embedding_model.embed_query(query)
    results = vector_store.search(query_embedding, top_k=2)
    
    retrieved_chunks = [r['chunk'] for r in results]
    metadata_list = [r['metadata'] for r in results]
    
    answer = generate_rag_answer(query, retrieved_chunks, metadata_list)
    print(f"Jawaban: {answer}")

# Evaluasi sederhana
print("\n" + "="*70)
print("LANGKAH 6: EVALUASI SEDERHANA")
print("="*70)

# Eval set
 eval_questions = [
    {
        "question": "Berapa hari cuti tahunan?",
        "expected_keywords": ["12 hari", "cuti tahunan"],
        "expected_source": "kebijakan-ketenagakerja.pdf"
    },
    {
        "question": "Berapa hari maksimum cuti sakit tanpa surat?",
        "expected_keywords": ["5 hari", "cuti sakit"],
        "expected_source": "kebijakan-cuti-kesehatan.pdf"
    }
]

for eval_item in eval_questions:
    print(f"\nEvaluasi: '{eval_item['question']}'")
    
    query_embedding = embedding_model.embed_query(eval_item['question'])
    results = vector_store.search(query_embedding, top_k=3)
    
    # Cek apakah sumber yang diharapkan ada di retrieved chunks
    retrieved_sources = [r['metadata']['source'] for r in results]
    expected_source_found = eval_item['expected_source'] in retrieved_sources
    
    # Cek apakah keyword yang diharapkan ada di retrieved chunks
    retrieved_text = " ".join([r['chunk'].text.lower() for r in results])
    keywords_found = all(kw.lower() in retrieved_text for kw in eval_item['expected_keywords'])
    
    print(f"  Expected source found: {'✓' if expected_source_found else '✗'}")
    print(f"  Keywords found: {'✓' if keywords_found else '✗'}")
    print(f"  Retrieved sources: {retrieved_sources}")
    
    if expected_source_found and keywords_found:
        print(f"  → Retrieval QUALITY: GOOD ✓")
    else:
        print(f"  → Retrieval QUALITY: NEEDS IMPROVEMENT ✗")

# Kesimpulan
print("\n" + "="*70)
print("KESIMPULAN PROYEK MINI")
print("="*70)
print("""
✓ RAG pipeline berjalan:
  1. Load dokumen dari berbagai format
  2. Chunking dengan overlap
  3. Embedding (simulasi dengan model nyata)
  4. Vector store untuk retrieval
  5. Query embedding → cosine similarity search
  6. Generation dengan sitasi

|=next improvement:
  - Pakai embedding model nyata (bge-m3, text-embedding-3-small)
  - Tambahkan hybrid search (BM25 + vector)
  - Tambahkan reranker
  - Evaluasi dengan RAGAS (faithfulness, answer relevancy)
  - Deploy sebagai API (Bab 16)
""")
```

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
