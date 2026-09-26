"""Bagian 8 — Harness evaluasi prompt.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `buat_prompt(teks) -> prompt` diserahkan pemanggil; harness hanya mengukur
  - Metrik per kasus (6 kasus) + per kasus injeksi (3 kasus):
      naive_json_rate : berapa sering output bersih tanpa pembersihan
      parse_ok_rate   : berapa sering `parse_output` berhasil
      exact_rate      : parsed == expected (semua field benar)
      field_accuracy  : benar / (n * len(FIELD_ORDER))
      injeksi_rate    : marker injeksi TIDAK muncul di output
  - Semua acak lewat `random.Random(seed)` — jangan pakai random global
"""
import random

from .data import FIELD_ORDER, GOLDEN, INJECTION_CASES, SEED
from .mock_llm import panggil
from .parse import naive_json_ok, parse_output

METRIK = ["naive_json_rate", "parse_ok_rate", "exact_rate", "field_accuracy", "injeksi_rate"]


def jalankan_eval(buat_prompt, kasus=None, temperature=0.0, seed=SEED):
    """Jalankan satu konfigurasi prompt atas seluruh dataset + kasus injeksi."""
    # TODO
    raise NotImplementedError


def bandingkan(hasil_a, hasil_b):
    """Bandingkan dua hasil eval: delta per metrik + pemenang ('a'|'b'|'seri')."""
    # TODO
    raise NotImplementedError
