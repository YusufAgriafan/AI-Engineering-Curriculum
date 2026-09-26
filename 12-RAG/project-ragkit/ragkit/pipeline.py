"""Bagian 7 — Pipeline RAG end-to-end.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `jalankan_rag(retrieve, kueri, generator=None, top_k=3, ambang=0.30,
    generator_latency_ms=25)` menjalankan SATU permintaan penuh:
      1. retrieve → chunks + skor_terbaik;
      2. generator.jawab → jawaban + sitasi + sumber + prompt;
      3. tambahkan "skor_terbaik" ke hasil, lalu
         `ragkit.abstain.abstaikan(...)` (panggil lewat MODUL, bukan
         `from ragkit.abstain import abstaikan` — supaya tetap bisa diuji);
      4. biaya_usd = token_in/1e6*0.15 + token_out/1e6*0.60
         (token = ceil(len(teks)/4); tarif "mini" — angka ILUSTRATIF).
      Return dict:
        {"kueri", "chunks", "jawaban", "sitasi", "sumber", "abstain",
         "skor_terbaik", "skema" (prompt), "biaya_usd",
         "tahap": {"retrieve_ms", "generate_ms"}}
        + "abstaikan_oleh" (skor pembuktian) HANYA bila gate yang menimpa.
      Latensi DETERMINISTIK: retrieve_ms = 8 + 2*len(index) + generator_latency_ms;
      generate_ms = 2 * generator_latency_ms.
  - generator None → GeneratorEkstraktif() (dari ragkit.generator).
"""
import math

from ragkit import abstain as _mod_abstain
from ragkit.chunking import buat_semua_chunks
from ragkit.generator import GeneratorEkstraktif, jawab


def _latensi_index(index):
    """Latensi pencarian deterministik dari jumlah vektor di index (ms)."""
    return 8 + 2 * int(index)


def jalankan_rag(retrieve, kueri, generator=None, top_k=3, ambang=0.30,
                 generator_latency_ms=25):
    """Satu permintaan RAG end-to-end + pelaporan per tahap."""
    # TODO
    raise NotImplementedError
