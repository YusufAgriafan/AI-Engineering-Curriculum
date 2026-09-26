"""Bagian 1 — Token, biaya, penjaga anggaran.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `token_estimasi(teks) = ceil(len(teks) / 4)`
  - `hitung_biaya` memakai tarif per 1 JUTA token dari `data.HARGA`
    (model tak dikenal -> ValueError)
  - `jalankan_anggaran` berhenti MEMANGGIL saat anggaran sudah habis;
    sisanya dilewati (bukan dipaksa jalan)
"""
import math

from .data import HARGA


def token_estimasi(teks):
    """Perkiraan token: ceil(len(teks) / 4)."""
    # TODO
    raise NotImplementedError


def hitung_biaya(tokens_in, tokens_out, model, harga=None):
    """Biaya USD dari tarif per 1 juta token. Model tak dikenal -> ValueError."""
    # TODO
    raise NotImplementedError


def biaya_hasil(hasil, harga=None):
    """Biaya sebuah hasil panggilan (dict dari provider)."""
    # TODO
    raise NotImplementedError


def jalankan_anggaran(provider, daftar_pesan, batas_usd, *, model="mini",
                      timeout_ms=5000, harga=None):
    """Jalankan pekerjaan sampai anggaran habis.

    Return: diproses, dilewati, total_biaya_usd, hasil (list of dict dengan
    index/teks/biaya_usd), berhenti_di (index pertama yang dilewati | None).
    """
    # TODO
    raise NotImplementedError
