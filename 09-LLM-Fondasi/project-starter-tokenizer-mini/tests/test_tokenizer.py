"""Spesifikasi Bagian 2 — Tokenizer end-to-end (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/tokenizer.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.bpe import latih_bpe, normalisasi
from tok_mini.tokenizer import buat_vocab, decode, decode_teks, encode, encode_teks

CORPUS = ["low", "low", "lower", "lowest", "newer", "newer",
          "wider", "wider", "new", "new", "new"]


class TestVocab(unittest.TestCase):
    def setUp(self):
        self.vocab0 = buat_vocab(CORPUS, 0)
        self.vocab4 = buat_vocab(CORPUS, 4)

    def test_vocab_awal_isi(self):
        # vocab memuat SEMUA simbol AWAL: karakter alfabet corpus + </w>
        for kar in "delnostw":
            self.assertIn(kar, self.vocab0)
        self.assertIn("</w>", self.vocab0)
        self.assertEqual(len(self.vocab0), 11)       # 10 karakter unik + </w>

    def test_vocab_menambah_hasil_merge(self):
        self.assertIn("er</w>", self.vocab4)
        self.assertIn("new", self.vocab4)
        self.assertEqual(len(self.vocab4), 15)       # 11 awal + 4 merge

    def test_indeks_tanpa_celah(self):
        indeks = sorted(self.vocab4.values())
        self.assertEqual(indeks, list(range(len(self.vocab4))))


class TestEncodeDecode(unittest.TestCase):
    def setUp(self):
        self.merges, _ = latih_bpe(CORPUS, 4)
        self.vocab = buat_vocab(CORPUS, 4)

    def test_encode_kata_dipelajari(self):
        self.assertEqual(encode("lower", self.merges, self.vocab),
                         ("l", "o", "w", "er</w>"))

    def test_encode_hanya_token_vocab(self):
        toks = encode("new", self.merges, self.vocab)
        self.assertEqual(toks, ("new", "</w>"))

    def test_decode_tanpa_penanda(self):
        self.assertEqual(decode(("l", "o", "w", "er</w>")), "lower")
        self.assertEqual(decode(("new", "</w>")), "new")

    def test_roundtrip_satu_kata(self):
        kata = "Wider"
        self.assertEqual(decode(encode(kata.lower(), self.merges, self.vocab)), kata.lower())

    def test_roundtrip_teks(self):
        teks = "NEWER lowliest"      # hanya huruf dari alfabet corpus
        hasil = decode_teks(encode_teks(teks, self.merges, self.vocab))
        self.assertEqual(hasil, teks.lower())

    def test_encode_huruf_asing_diabaikan(self):
        # 'h' tak ada di alfabet corpus -> simbol (t, h, e, </w>) jadi tak ada di vocab
        toks = encode("the", self.merges, self.vocab)
        self.assertEqual(toks, ("t", "e", "</w>"))

    def test_encode_teks_normalisasi(self):
        toks = encode_teks("New low", self.merges, self.vocab)
        self.assertEqual(len(toks), 2)
        self.assertEqual(toks[0], ("new", "</w>"))


if __name__ == "__main__":
    unittest.main()
