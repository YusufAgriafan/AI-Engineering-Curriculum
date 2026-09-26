"""Tokenizer mini deterministik — GIVEN, JANGAN DIUBAH.

Vokabularium dibangun dari korpus train (urutan kemunculan pertama), OOV →
<unk>. Numerik: mean vektor one-hot per token (dimensi 16) — cukup untuk
memisahkan dua kelas kata yang arah konsistennya sengaja dibuat tegas.
"""
import re

from .data import DATA_SENTIMEN

_DIM = 16
_TOKEN_PAT = re.compile(r"[a-z0-9]+")
_PAD, _UNK = "<pad>", "<unk>"

KATA_NEGASI = ("tidak", "bukan")


def tokenisasi(teks):
    """Huruf kecil → [a-z0-9]+. Deterministik, tanpa dependensi."""
    return _TOKEN_PAT.findall(str(teks).lower())


def bangun_vocab(teks_list=None):
    """Vocab deterministik: <pad>, <unk>, lalu kata sesuai urutan kemunculan pertama."""
    vocab = [_PAD, _UNK]
    terlihat = {_PAD, _UNK}
    for teks in teks_list if teks_list is not None else [d["teks"] for d in DATA_SENTIMEN if d["split"] == "train"]:
        for tok in tokenisasi(teks):
            if tok not in terlihat:
                terlihat.add(tok)
                vocab.append(tok)
    return vocab


class TokenizerMini:
    """teks → vektor mean one-hot + flag negasi.

    - OOV → <unk> (bukan error — teks nyata selalu punya kata baru);
    - negasi: kalau ada 'tidak'/'bukan', flag = 1 → fitur ke-17 (dipakai
      model untuk membalik arah logit);
    - vektor mean one-hot (bukan jumlah) → panjang ulasan tidak menggeser skala.
    """

    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else bangun_vocab()
        self.indeks = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, teks):
        """teks → (vektor 16, flag_negasi)."""
        tokens = tokenisasi(teks)
        vec = [0.0] * _DIM
        for tok in tokens:
            idx = self.indeks.get(tok, self.indeks[_UNK])
            vec[idx % _DIM] += 1.0
        n = len(tokens)
        if n == 0:
            vec[self.indeks[_UNK] % _DIM] = 1.0
            n = 1
        vec = [v / n for v in vec]
        negasi = 1.0 if any(t in KATA_NEGASI for t in tokens) else 0.0
        return vec, negasi

    def ukuran_vocab(self):
        return len(self.vocab)


def fitur(teks, tok=None):
    """teks → vektor 17-dim (16 mean one-hot + flag negasi). Cukup untuk demo."""
    tok = TokenizerMini() if tok is None else tok
    vec, neg = tok.encode(teks)
    return vec + [neg]
