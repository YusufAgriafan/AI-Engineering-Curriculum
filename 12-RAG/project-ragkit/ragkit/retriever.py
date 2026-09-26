"""Bagian 2 — Retriever + evaluasi hit@k.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `buat_retriever` me-index korpus SEKALI (chunking GIVEN → embed → store),
    lalu mengembalikan fungsi `retrieve(kueri, top_k=3)` yang:
      embed kueri → store.cari → list {"chunk", "score", "index"};
  - `evaluasi_retrieval` menghitung hit@k: kueri "kena" bila `source_emas`
    ada di antara sumber hasil top-k.
      Return dict: {"per_kueri": [{"id", "hit", "posisi"}...],
                    "hit_rate": float, "kueri_gagal": [id]}
      hit_rate = jumlah hit / jumlah kueri; posisi = 1-based (None bila miss).
"""
from ragkit.chunking import buat_semua_chunks
from ragkit.data import KUERI_UJI
from ragkit.embed import EncoderEmbedding
from ragkit.vectorstore import VectorStore


def buat_retriever(encoder=None, chunks=None):
    """Index korpus sekali → fungsi retrieve(kueri, top_k) siap pakai.

    encoder None → EncoderEmbedding(); chunks None → buat_semua_chunks().
    """
    # TODO
    raise NotImplementedError


def evaluasi_retrieval(retrieve, kueri_uji=None, top_k=3):
    """hit@k per kueri + agregat hit_rate + daftar id kueri yang gagal."""
    # TODO
    raise NotImplementedError
