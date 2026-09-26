"""Bagian 1 — Vector store cosine similarity.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - vektor DIASUMSIKAN sudah ternormalisasi L2 (dari `embed.EncoderEmbedding`),
    jadi cosine similarity = dot product biasa — tapi tulis cosine eksplisit
    (dengan pembagi norm) supaya tetap benar untuk vektor tak ternormalisasi;
  - `cari` mengembalikan DAFTAR dict berurutan dari skor TERTINGGI:
        {"chunk": chunk, "score": float, "index": int}
    tie skor dipecah ke index KECIL dulu (deterministik!);
  - `top_k` lebih besar dari jumlah chunk → kembalikan semuanya;
  - store kosong → `cari` mengembalikan [] (jangan crash).
"""
from ragkit.embed import cosine


class VectorStore:
    """Vector store in-memory paling sederhana yang masih jujur:

    - index = daftar vektor (dan chunk-nya);
    - pencarian = cosine terhadap SEMUA vektor (brute force).
      Di sini memang cukup (7 chunk). Di produksi, 1 juta chunk × 1024 dim
      dihitung dengan ANN (HNSW/IVF) — konsep "indeks sekali, cari cepat"
      tetap sama, hanya strukturnya yang berganti.
    """

    def __init__(self):
        # TODO
        raise NotImplementedError

    def add(self, chunks, matriks):
        """Simpan daftar chunk + matriks embedding (list of vektor)."""
        # TODO
        raise NotImplementedError

    def cari(self, vektor_kueri, top_k=3):
        """Cosine vs semua vektor → top_k terbaik (skor turun, tie → index kecil)."""
        # TODO
        raise NotImplementedError
