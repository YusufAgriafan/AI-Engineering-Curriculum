"""Bagian 5 — Parse output berisik & perbaikan JSON.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `ekstrak_json` mengambil objek JSON PERTAMA; pemindai brace-matching yang
    sadar string & escape (jadi `{` di dalam string tidak merusak kedalaman)
  - Bila JSON mentah gagal, coba `perbaiki_json` dulu sebelum menyerah
  - `perbaiki_json` menangani: komentar `//`, kutip tunggal, True/False/None,
    dan koma menggantung (trailing comma)
  - Gagal total -> raise ValueError
"""
import json
import re

from .data import SKEMA_PENUH


def perbaiki_json(teks):
    """Betulkan pelanggaran JSON umum, lalu return objek hasil json.loads."""
    # TODO
    raise NotImplementedError


def ekstrak_json(teks):
    """Ambil objek JSON pertama dari teks berisik. Raise ValueError bila tidak ada."""
    # TODO
    raise NotImplementedError


def parse_output(teks, skema=None):
    """Output mentah model -> dict tervalidasi. Raise ValueError bila gagal."""
    # TODO
    raise NotImplementedError


def naive_json_ok(teks):
    """Apakah `json.loads` MENTAH berhasil (output bersih tanpa basa-basi)?"""
    # TODO
    raise NotImplementedError
