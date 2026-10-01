"""GIVEN — Tokenizer: kontrak estimasi token dari Bab 11/15 (JANGAN DIUBAH).

token_estimasi(teks) = ceil(len(teks)/4) — cukup untuk mengukur biaya tanpa
tokenizer asli. Di produksi ganti dengan tiktoken/tokenizer model.
"""
import math


def token_estimasi(teks):
    return math.ceil(len(str(teks)) / 4.0)
