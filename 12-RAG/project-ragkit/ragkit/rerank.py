"""Bagian 6 — Reranker skor kata.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `skorer_kata(kueri, teks)` — skor leksikal murah (deterministik):
      buang stopword dari daftar di bawah; tiap kata kueri yang MUNCUL di teks
      (substring lowercased) menambah: 3.0 bila kata = ANGKA (isdigit), else 1.0;
      kata kueri kosong setelah stopword → skor 0.0.
  - `rerank(kueri, hasil, top_k=2, skorer=None)`:
      urutkan hasil dict-retrieval dengan (-skor, chunk_index) lalu ambil
      top_k; skorer None → skorer_kata.
"""
STOPWORD = {"berapa", "bagaimana", "apa", "yang", "untuk", "dengan", "adalah",
            "tanpa", "dari", "ke", "di", "dan", "atau", "hari"}


def skorer_kata(kueri, teks):
    """Skor leksikal murah: Σ bobot kata kueri yang ada di teks (angka ×3)."""
    # TODO
    raise NotImplementedError


def rerank(kueri, hasil, top_k=2, skorer=None):
    """Urutkan ulang kandidat dengan skorer murah, ambil top_k terbaik."""
    # TODO
    raise NotImplementedError
