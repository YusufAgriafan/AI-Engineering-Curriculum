"""Bagian 6 — Guard: deteksi injeksi & pembungkusan data.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Pola injeksi diambil dari tabel `data.POLA_INJEKSI` (satu sumber kebenaran
    dengan mock) — jangan menulis pola sendiri
  - `deteksi_injection` mengembalikan nama pola unik & TERURUT
  - `bungkus_data` meng-escape tag lebih dulu, baru membungkusnya
"""
from .data import LEASH_MARKER_TERLARANG, POLA_INJEKSI


def deteksi_injection(teks):
    """Nama-nama pola injeksi yang cocok (terurut, unik)."""
    # TODO
    raise NotImplementedError


def bungkus_data(teks, tag="dokumen"):
    """Bungkus data tak tepercaya dengan delimiter yang sudah di-escape."""
    # TODO
    raise NotImplementedError


def langgar_leash(output, marker=None):
    """True bila output keluar dari tugas (mengandung marker terlarang)."""
    # TODO
    raise NotImplementedError
