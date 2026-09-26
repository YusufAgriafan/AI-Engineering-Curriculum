"""TODO Bagian 5a — Biaya API vs lokal + keputusan deployment.

Kontrak lengkap ada di tests/test_api_eval.py (kelas TestBiaya &
TestKeputusanDeployment).
"""
import math

from ftkit.data import HARGA


def biaya_api(token_in, token_out, model="api_mini"):
    """USD = in/1e6*harga_in + out/1e6*harga_out ('lokal' = 0.0)."""
    raise NotImplementedError


def biaya_bulanan(permintaan_per_hari, token_in, token_out,
                  model="api_mini", hari=30):
    """biaya_api × permintaan_per_hari × hari."""
    raise NotImplementedError


def token_estimasi(teks):
    """ceil(len(teks)/4) — cukup untuk proyeksi, bukan tagihan."""
    raise NotImplementedError


def keputusan_deployment(permintaan_per_hari, token_in, token_out,
                         biaya_gpu_per_bulan=60.0, hari=30):
    """biaya API bulanan vs GPU: api > gpu → 'lokal', selain itu 'api'.

    Return {'biaya_api_bulanan', 'biaya_lokal_bulanan', 'keputusan'}.
    """
    raise NotImplementedError
