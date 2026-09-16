"""Tokenisasi, vocabulary, encoding OOV-aman, dan padding.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_text.py.

Konvensi id:
    PAD_ID = 0   (padding)
    OOV_ID = 1   (kata tak dikenal)

Aturan emas: vocabulary dibangun dari data TRAIN saja, dan harus DETERMINISTIK —
vocab yang sama dari korpus yang sama, kapan pun dijalankan (karena diurut abjad).
"""

import numpy as np

__all__ = ["PAD_ID", "OOV_ID", "tokenize_kata", "bangun_vocab", "encode",
           "encode_aman", "pad_sequences"]

PAD_ID = 0
OOV_ID = 1


def tokenize_kata(teks):
    """Teks -> list kata (lowercase, split spasi)."""
    # TODO
    raise NotImplementedError


def bangun_vocab(kalimat_list):
    """Return (vocab, w2i, i2w): vocab terurut abjad, w2i = kata->id mulai 0.

    kalimat_list: list string ATAU list list-token — keduanya diterima.
    """
    # TODO: kumpulkan semua token (lowercase), sort, buat dua mapping
    raise NotImplementedError


def encode(kalimat, w2i):
    """Kalimat (string) -> list id. Kata tak dikenal -> OOV_ID (jangan crash)."""
    # TODO
    raise NotImplementedError


def encode_aman(kalimat, w2i):
    """Sama seperti encode() — dipertahankan untuk kejelasan kontrak OOV."""
    # TODO: delegasikan ke encode() atau tulis ulang
    raise NotImplementedError


def pad_sequences(seqs, maxlen):
    """Semua sequence jadi sama panjang (array int (n, maxlen)).

    - lebih pendek -> isi PAD_ID di BELAKANG (padding='post')
    - lebih panjang -> potong dari BELAKANG (truncating='post')
    - seqs kosong -> bentuk (0, maxlen)
    """
    # TODO
    raise NotImplementedError
