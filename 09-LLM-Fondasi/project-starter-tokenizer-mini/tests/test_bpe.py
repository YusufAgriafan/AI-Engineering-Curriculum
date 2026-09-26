"""Spesifikasi Bagian 1 — BPE (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/bpe.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.bpe import apply_merges, best_pair, hitung_pasangan, ke_simbol, latih_bpe, merge_pasangan, normalisasi


class TestNormalisasi(unittest.TestCase):
    def test_lowercase_dan_split(self):
        self.assertEqual(normalisasi("The NEWER"), ["the", "newer"])

    def test_spasi_berlebih(self):
        self.assertEqual(normalisasi("  low   new  "), ["low", "new"])


class TestSimbol(unittest.TestCase):
    def test_ke_simbol(self):
        self.assertEqual(ke_simbol("low"), ("l", "o", "w", "</w>"))

    def test_satu_karakter(self):
        self.assertEqual(ke_simbol("a"), ("a", "</w>"))


class TestHitungPasangan(unittest.TestCase):
    def setUp(self):
        self.sym = [ke_simbol(k) for k in ["low", "low", "new"]]

    def test_frekuensi_menghitung_semua_kemunculan(self):
        pc = hitung_pasangan(self.sym)
        self.assertEqual(pc[("l", "o")], 2)          # dua kali 'low'
        self.assertEqual(pc[("o", "w")], 2)
        self.assertEqual(pc[("w", "</w>")], 3)       # low x2 + new
        self.assertEqual(pc[("n", "e")], 1)

    def test_return_counter(self):
        import collections
        self.assertIsInstance(hitung_pasangan(self.sym), collections.Counter)


class TestBestPair(unittest.TestCase):
    def test_terfrekuensi(self):
        pc = hitung_pasangan([ke_simbol("low"), ke_simbol("new")])
        self.assertEqual(best_pair(pc), ("w", "</w>"))  # 3 vs 2

    def test_seri_leksikografis(self):
        pc = {("b", "z"): 3, ("a", "y"): 3, ("c", "x"): 1}
        self.assertEqual(best_pair(pc), ("a", "y"))


class TestMerge(unittest.TestCase):
    def test_merge_tengah(self):
        out = merge_pasangan([("l", "o", "w", "</w>")], ("l", "o"))
        self.assertEqual(out, [("lo", "w", "</w>")])

    def test_merge_berulang_dalam_satu_kata(self):
        out = merge_pasangan([("a", "a", "a", "</w>")], ("a", "a"))
        self.assertEqual(out, [("aa", "a", "</w>")])

    def test_input_tidak_dimodifikasi(self):
        asli = [("l", "o", "w", "</w>")]
        merge_pasangan(asli, ("l", "o"))
        self.assertEqual(asli, [("l", "o", "w", "</w>")])


class TestLatihBPE(unittest.TestCase):
    def setUp(self):
        self.corpus = ["low", "low", "lower", "lowest", "newer", "newer",
                       "wider", "wider", "new", "new", "new"]

    def test_urutan_merge_terkunci(self):
        merges, _ = latih_bpe(self.corpus, 4)
        self.assertEqual(merges[0], (("e", "r"), 5))
        self.assertEqual(merges[1], (("e", "w"), 5))
        self.assertEqual(merges[2], (("er", "</w>"), 5))
        self.assertEqual(merges[3], (("n", "ew"), 5))

    def test_states_panjang_dan_awal(self):
        merges, states = latih_bpe(self.corpus, 3)
        self.assertEqual(len(states), 4)             # awal + 3 merge
        self.assertEqual(states[0][0], ("l", "o", "w", "</w>"))

    def test_state_terakhir_mencerminkan_merge(self):
        merges, states = latih_bpe(self.corpus, 1)
        self.assertEqual(states[1][0], ("l", "o", "w", "</w>"))   # low tak tersentuh
        self.assertIn(("n", "e", "w", "er", "</w>"), [tuple(s) for s in states[1]])


class TestApplyMerges(unittest.TestCase):
    def setUp(self):
        self.corpus = ["low", "low", "lower", "lowest", "newer", "newer",
                       "wider", "wider", "new", "new", "new"]
        self.merges, _ = latih_bpe(self.corpus, 4)

    def test_kata_yang_dipelajari(self):
        self.assertEqual(apply_merges(ke_simbol("lower"), self.merges),
                         ("l", "o", "w", "er</w>"))

    def test_token_baru_dari_merge_keempat(self):
        self.assertEqual(apply_merges(ke_simbol("new"), self.merges),
                         ("new", "</w>"))

    def test_kata_tak_kenal_fallback_karakter(self):
        self.assertEqual(apply_merges(ke_simbol("lowest"), self.merges),
                         ("l", "o", "w", "e", "s", "t", "</w>"))

    def test_karakter_asing_tetap_ada(self):
        hasil = apply_merges(ke_simbol("xyz"), self.merges)
        self.assertEqual(hasil, ("x", "y", "z", "</w>"))


if __name__ == "__main__":
    unittest.main()
