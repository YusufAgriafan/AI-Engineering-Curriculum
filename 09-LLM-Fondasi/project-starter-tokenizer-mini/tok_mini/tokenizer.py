"""Bagian 2 — Tokenizer end-to-end.  TODO: kamu yang mengisi.

Vocab memetakan simbol -> indeks (0..V-1). Semua karakter awal corpus DIJAMIN
ada di vocab — itulah fallback yang membuat teks apa pun bisa di-encode.
"""
from .bpe import apply_merges, ke_simbol, latih_bpe, merge_pasangan, normalisasi


def buat_vocab(corpus, n_merge):
    """Vocab = semua simbol awal + hasil merge, terurut leksikografis.

    Return dict {simbol: indeks} dengan indeks 0..V-1 tanpa celah.
    (Perlu untuk one-hot & embedding: index = baris matrix.)
    """
    # TODO: kumpulkan simbol awal semua kata corpus, tambah hasil tiap merge,
    # urutkan leksikografis (sorted()), map ke indeks 0..V-1.
    raise NotImplementedError


def encode(kata, merges, vocab):
    """Satu kata -> tuple token (hanya yang ada di vocab).

    Token tak dikenal di-SKIP — tapi semua karakter awal dijamin ada, jadi
    tidak ada informasi hilang untuk kata dari alfabet corpus.
    """
    # TODO
    raise NotImplementedError


def decode(token_tuple):
    """Tuple token -> string tanpa penanda </w>."""
    # TODO
    raise NotImplementedError


def encode_teks(teks, merges, vocab):
    """Teks multi-kata -> list of list token (normalisasi dulu)."""
    # TODO
    raise NotImplementedError


def decode_teks(tokens_per_kata):
    """List of list token -> teks ber-spasi (round-trip dari encode_teks)."""
    # TODO
    raise NotImplementedError
