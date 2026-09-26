"""Encoder embedding PALSU & deterministik — GIVEN, JANGAN DIUBAH.

Kenapa palsu? Tiga alasan (pola yang sama dengan `transport.py` di Bab 11):

1. **Reproducible.** Embedding API asli berubah diam-diam (deprecation, update
   model). Hashing n-gram deterministik: teks yang sama → vektor yang sama,
   selamanya, di komputer siapa pun. Evaluasi RAG tanpa determinisme = angka
   yang tidak bisa dibandingkan antar hari.

2. **Gratis & tanpa jaringan.** Kamu boleh meng-index 1000 chunk, menguji tiga
   konfigurasi chunking, dan me-retrieval seratus kueri — tanpa biaya token.

3. **Semantiknya cukup baik untuk didik.** Hashing trick (teknik nyata di
   scikit-learn `HashingVectorizer`) memetakan n-gram ke indeks bucket lewat
   hash; teks yang berbagi kata/n-gram berbagi komponen vektor → cosine
   similarity tinggi. Kata "cuti" di kueri dan di chunk yang sama menyalakan
   bucket yang sama — itulah inti vector search.

Yang TIDAK bisa dilakukan encoder ini (dan memang itu bagian pelajarannya):
memahami sinonim ("mengajukan cuti" ↔ "permohonan cuti" mirip, tapi "liburan"
tidak), menangkap urutan makna, atau bahasa yang tidak berbagi kata. Di titik
itu di produksi kamu ganti ke model multilingual nyata (bge-m3, e5,
text-embedding-3) — antarmukanya persis sama: `embed([teks...]) → matriks.
"""
import hashlib
import math
import re

from .data import SEED

_DIM = 256
_TOKEN_PAT = re.compile(r"[a-z0-9]+")

# anti-offset per dimensi agar distribusi vektor tidak condong (deterministik)
_OFFSET = [
    int.from_bytes(hashlib.sha256(f"ragkit-offset-{SEED}-{i}".encode()).digest()[:4], "big") / 2**32
    for i in range(_DIM)
]


def tokenisasi(teks):
    """Tokenisasi sederhana: huruf kecil, ambil [a-z0-9]+. Cukup untuk ID/EN."""
    return _TOKEN_PAT.findall(str(teks).lower())


class EncoderEmbedding:
    """Encoder hashing n-gram: teks → vektor float32 ternormalisasi L2.

    - unigram berat 1.0, bigram berat 0.5 (bigram menangkap sedikit urutan);
    - tanda hash bernilai +/-1 dari bit hash → bucket bisa "mengurangi"
      (seperti HashingVectorizer dengan alternate_sign);
    - vektor diberi offset deterministik kecil lalu dinormalisasi L2, supaya
      dua teks TANPA kata sama punya cosine ≈ 0 (bukan memang acak besar).
    """

    def __init__(self, dim=_DIM, seed=SEED):
        self.dim = dim
        self.seed = seed

    # -------------------- inti encoder --------------------
    def _indeks_bucket(self, gram, salt=""):
        bahan = f"{self.seed}|{salt}|{gram}".encode("utf-8")
        digest = hashlib.sha256(bahan).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dim
        tanda = 1.0 if digest[4] % 2 == 0 else -1.0
        return idx, tanda

    def _vektor_mentah(self, teks):
        vec = [0.0] * self.dim
        tokens = tokenisasi(teks)
        for tok in tokens:
            idx, tanda = self._indeks_bucket(tok)
            vec[idx] += tanda
            if len(tok) > 3:
                idx5, t5 = self._indeks_bucket(tok[:5], salt="pre")
                vec[idx5] += 0.3 * t5
        for a, b in zip(tokens, tokens[1:]):
            idx, tanda = self._indeks_bucket(a + "_" + b, salt="bi")
            vec[idx] += 0.5 * tanda
        return vec

    def _vektor(self, teks):
        mentah = self._vektor_mentah(teks)
        bercampur = [v + 0.01 * _OFFSET[i] for i, v in enumerate(mentah)]
        norm = math.sqrt(sum(v * v for v in bercampur))
        if norm == 0.0:
            return [0.0] * self.dim
        return [v / norm for v in bercampur]

    # -------------------- API publik (bentuk API nyata) --------------------
    def embed(self, teks_list):
        """List[str] → list[list[float]] (sudah ternormalisasi L2)."""
        return [self._vektor(t) for t in teks_list]

    def embed_query(self, teks):
        return self._vektor(teks)

    def dimensi(self):
        return self.dim


def cosine(a, b):
    """Cosine similarity dua vektor (list). Vektor nol → 0.0."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
