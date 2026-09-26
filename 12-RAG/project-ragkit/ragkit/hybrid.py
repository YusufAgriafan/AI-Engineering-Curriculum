"""Bagian 5 — Hybrid search: BM25 mini + RRF.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `cari_bm25(kueri, chunks, top_k=3)` — BM25 ringan tanpa library:
      k1=1.2, b=0.75; idf = ln(1 + (N - df + 0.5)/(df + 0.5));
      skor dokumen = Σ_idf(tf * (k1+1) * tf / (tf + k1*(1-b+b*len/rata_len)));
      tokenisasi pakai `embed.tokenisasi`; tie → index kecil dulu.
      Return list {"chunk", "score", "index"}.
  - `gabung_rrf(daftar_dapat, k=60)` — skor RRF per chunk_index:
      Σ 1/(k + peringkat + 1) untuk tiap daftar; return dict {index: skor}.
  - `cari_hybrid(retrieve_vektor, chunks, kueri, top_k=3, rrf_k=60)`:
      vektor top-k + BM25 top-k → RRF → urut (skor turun, tie → index kecil)
      → top_k dict {"chunk", "score", "index"} (score = skor RRF).
"""
import math

from ragkit.embed import tokenisasi


def cari_bm25(kueri, chunks, top_k=3):
    """BM25 mini (stdlib): k=1.2, b=0.75 — tanpa library eksternal."""
    # TODO
    raise NotImplementedError


def gabung_rrf(daftar_dapat, k=60):
    """Reciprocal Rank Fusion: skor = Σ 1/(k + peringkat + 1)."""
    # TODO
    raise NotImplementedError


def cari_hybrid(retrieve_vektor, chunks, kueri, top_k=3, rrf_k=60):
    """Vektor + BM25 → gabung RRF → urutkan → ambil top_k."""
    # TODO
    raise NotImplementedError
