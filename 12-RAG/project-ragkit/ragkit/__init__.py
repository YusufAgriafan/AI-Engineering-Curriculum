"""ragkit — paket stdlib murni untuk project Bab 12 (RAG).

Modul:
    data.py        : korpus, kueri uji, jawaban emas, tolok ukur (JANGAN DIUBAH)
    embed.py       : encoder embedding palsu deterministik (hashing n-gram) — GIVEN
    chunking.py    : chunking per kalimat + overlap, deterministik      — GIVEN
    generator.py   : generator ekstraktif + abstain + sitasi            — GIVEN
    vectorstore.py : TODO: vector store cosine similarity
    retriever.py   : TODO: retriever + evaluasi hit@k
    abstain.py     : TODO: abstain gate dari skor retrieval
    cache.py       : TODO: cache konteks retrieval + tolok ukur
    hybrid.py      : TODO: BM25 mini + Reciprocal Rank Fusion
    rerank.py      : TODO: reranker skor kata (pola cross-encoder)
    pipeline.py    : TODO: pipeline RAG end-to-end + laporan per tahap
    evaluate.py    : TODO: evaluasi ujung-ke-ujung (akurasi + abstain)

Prasyarat: stdlib saja (math, re, hashlib). Semua kontrak ada di tests/.
Kunci angka: SEED 12 — korpus, chunk, skor, hit-rate semuanya deterministik.
"""
__all__ = ["data", "embed", "chunking", "generator", "vectorstore",
           "retriever", "abstain", "cache", "hybrid", "rerank",
           "pipeline", "evaluate"]
