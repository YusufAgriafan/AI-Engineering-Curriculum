"""Bagian 3 — Few-shot: format, pemilihan, pengurutan contoh.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Format satu contoh: 'Input: "<teks>"\\nOutput: <json>' (unicode TIDAK di-escape)
  - `pilih_contoh` menyebar merata: idx ke-i = round(i * (n-1) / (k-1))
  - `urutkan_contoh` STABIL: contoh paling sedikit-terisi dulu, paling kaya TERAKHIR
"""
import json


def format_contoh(teks, expected):
    """Satu pasangan contoh: Input: "..." / Output: {...}."""
    # TODO
    raise NotImplementedError


def pilih_contoh(contoh, k):
    """Pilih k contoh yang tersebar MERATA (deterministik, tanpa duplikat)."""
    # TODO
    raise NotImplementedError


def urutkan_contoh(contoh):
    """Urutkan berdasar jumlah field non-null, menaik (paling kaya terakhir)."""
    # TODO
    raise NotImplementedError


def bangun_fewshot(contoh, k):
    """Blok [CONTOH] siap tempel: pilih k -> urutkan -> format, dipisah baris kosong."""
    # TODO
    raise NotImplementedError
