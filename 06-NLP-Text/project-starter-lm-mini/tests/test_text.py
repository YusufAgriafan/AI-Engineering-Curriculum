"""Spesifikasi tokenisasi & padding dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di lm_mini/text.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_mini.text import OOV_ID, PAD_ID, bangun_vocab, encode, encode_aman, pad_sequences, tokenize_kata


class TestTokenize(unittest.TestCase):
    def test_lowercase_split(self):
        self.assertEqual(tokenize_kata("Saya Suka NASI"), ["saya", "suka", "nasi"])

    def test_spasi_berlebih(self):
        self.assertEqual(tokenize_kata("  saya   suka  "), ["saya", "suka"])

    def test_kosong(self):
        self.assertEqual(tokenize_kata(""), [])


class TestBangunVocab(unittest.TestCase):
    def test_terurut_dan_deterministik(self):
        v1, w2i1, i2w1 = bangun_vocab(["b a c", "a b"])
        v2, w2i2, i2w2 = bangun_vocab(["b a c", "a b"])
        self.assertEqual(v1, ["a", "b", "c"], "vocab harus terurut abjad")
        self.assertEqual(v1, v2)
        self.assertEqual(w2i1, w2i2)

    def test_roundtrip(self):
        vocab, w2i, i2w = bangun_vocab(["kucing itu lucu"])
        self.assertEqual(len(vocab), 3)
        self.assertEqual(len(w2i), 3)
        self.assertEqual(len(i2w), 3)
        self.assertTrue(all(i2w[w2i[w]] == w for w in vocab))

    def test_menerima_list_token(self):
        vocab, _, _ = bangun_vocab([["a", "b"], ["b", "c"]])
        self.assertEqual(vocab, ["a", "b", "c"])

    def test_id_mulai_nol(self):
        _, w2i, _ = bangun_vocab(["satu dua"])
        self.assertEqual(sorted(w2i.values()), [0, 1])


class TestEncode(unittest.TestCase):
    def test_encode_kalimat(self):
        vocab, w2i, _ = bangun_vocab(["saya suka nasi"])
        self.assertEqual(encode("saya suka nasi", w2i),
                         [w2i[t] for t in tokenize_kata("saya suka nasi")])

    def test_oov_tidak_crash(self):
        _, w2i, _ = bangun_vocab(["saya suka nasi"])
        ids = encode("saya makan bakso", w2i)
        self.assertIn(OOV_ID, ids, "'makan'/'bakso' tak dikenal -> OOV")
        self.assertEqual(ids[0], w2i["saya"])


class TestPadSequences(unittest.TestCase):
    def test_pad_post(self):
        out = pad_sequences([[5, 2], [3]], 3)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.tolist(), [[5, 2, 0], [3, 0, 0]])

    def test_truncate_post(self):
        out = pad_sequences([[3, 9, 7, 4]], 3)
        self.assertEqual(out.tolist(), [[3, 9, 7]], "potong dari belakang")

    def test_kosong(self):
        self.assertEqual(pad_sequences([], 2).shape, (0, 2))

    def test_maxlen_nol(self):
        self.assertEqual(pad_sequences([[1, 2]], 0).shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
