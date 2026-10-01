"""GIVEN — Tokenizer mini + utilitas teks (JANGAN DIUBAH).

Sama dengan pola Bab 13 (ftkit.tokenizer): token = ceil(len/4). Cukup untuk
proyeksi biaya eval, bukan tagihan resmi tokenizer vendor.
"""
import math


def token_estimasi(teks):
    """Proyeksi token: ceil(len/4)."""
    return math.ceil(len(str(teks)) / 4.0)
