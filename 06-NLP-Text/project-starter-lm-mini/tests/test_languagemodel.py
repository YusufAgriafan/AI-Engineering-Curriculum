"""Spesifikasi bigram LM dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di lm_mini/languagemodel.py lolos.
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_mini.corpus import KALIMAT_LM
from lm_mini.languagemodel import bigram_matrix, ee_bigram, prob_baris_dengan_temp, sample_bigram


class TestBigramMatrix(unittest.TestCase):
    def test_smoothing_dan_normalisasi(self):
        P = bigram_matrix([0, 1, 0, 2], 3, alpha=0.5)
        self.assertEqual(P.shape, (3, 3))
        self.assertTrue(np.allclose(P.sum(axis=1), 1.0), "tiap baris = distribusi")
        # baris 0: base 1.5 utk 3 kolom + transisi 0->1 dan 0->2 (masing +1)
        self.assertAlmostEqual(P[0, 0], 1.5 / 6.5)
        self.assertAlmostEqual(P[0, 1], 2.5 / 6.5)
        self.assertAlmostEqual(P[0, 2], 2.5 / 6.5)

    def test_seq_kosong_tetap_distribusi(self):
        P = bigram_matrix([], 3, alpha=0.0)
        self.assertTrue(np.allclose(P.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(P, 1.0 / 3.0))

    def test_alpha_nol_tanpa_transisi_nol(self):
        # alpha=0 + smoothing base 1.0 -> sel tak terlihat tetap punya probabilitas
        P = bigram_matrix([0, 1], 3, alpha=0.0)
        self.assertTrue((P > 0).all())


class TestEE(unittest.TestCase):
    def test_bigram_lebih_baik_dari_unigram(self):
        teks = "\n".join(KALIMAT_LM) + "\n"
        chars = sorted(set(teks))
        c2i = {c: i for i, c in enumerate(chars)}
        seq = [c2i[c] for c in teks]
        P = bigram_matrix(seq, len(chars), alpha=0.5)
        ee_b = ee_bigram(seq, P)
        unigram = np.bincount(seq, minlength=len(chars)) / len(seq)
        ee_u = -np.mean(np.log(unigram[seq[1:]] + 1e-12))
        self.assertLess(ee_b, ee_u - 0.2, "bigram harus jauh lebih baik dari unigram")

    def test_ee_rata_rata_log(self):
        P = np.array([[0.2, 0.8], [0.5, 0.5]])
        seq = [0, 1, 1]
        expected = -np.mean(np.log([P[0, 1], P[1, 1]]))
        self.assertAlmostEqual(ee_bigram(seq, P), expected, places=9)


class TestTemperature(unittest.TestCase):
    def test_temp_satu_identik(self):
        baris = np.array([0.1, 0.7, 0.2])
        self.assertTrue(np.allclose(prob_baris_dengan_temp(baris, 1.0), baris))

    def test_temp_rendah_mempertajam(self):
        baris = np.array([0.1, 0.7, 0.2])
        tajam = prob_baris_dengan_temp(baris, 0.5)
        self.assertGreater(tajam[1], baris[1], "pemenang harus makin dominan")
        self.assertAlmostEqual(tajam.sum(), 1.0)

    def test_temp_tinggi_meratakan(self):
        baris = np.array([0.7, 0.2, 0.1])
        datar = prob_baris_dengan_temp(baris, 2.0)
        self.assertLess(datar[0] - datar[1], baris[0] - baris[1], "selisih menyusut")
        self.assertAlmostEqual(datar.sum(), 1.0)

    def test_baris_nol_aman(self):
        out = prob_baris_dengan_temp(np.zeros(3), 0.7)
        self.assertEqual(out.shape, (3,))


class TestSampling(unittest.TestCase):
    def setUp(self):
        teks = "\n".join(KALIMAT_LM) + "\n"
        self.chars = sorted(set(teks))
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.seq = [self.c2i[c] for c in teks]
        self.P = bigram_matrix(self.seq, len(self.chars), alpha=0.5)

    def test_panjang_dan_domain(self):
        rng = np.random.default_rng(1)
        ids = sample_bigram(self.P, self.c2i["m"], 50, 1.0, rng)
        self.assertEqual(len(ids), 50)
        self.assertTrue(all(0 <= i < len(self.chars) for i in ids))

    def test_temp_rendah_repetitif(self):
        rng = np.random.default_rng(1)
        sampel = [sample_bigram(self.P, self.c2i["m"], 80, 0.3, rng) for _ in range(20)]

        def entropy(id_list):
            teks = "".join(self.chars[i] for i in id_list)
            vals = np.array(list(Counter(teks).values()), dtype=float)
            p = vals / vals.sum()
            return float(-(p * np.log(p)).sum())

        rng2 = np.random.default_rng(2)
        sampel_hi = [sample_bigram(self.P, self.c2i["m"], 80, 1.8, rng2) for _ in range(20)]
        self.assertLess(np.mean([entropy(s) for s in sampel]),
                        np.mean([entropy(s) for s in sampel_hi]),
                        "T rendah harus lebih repetitif (entropy lebih kecil)")

    def test_deterministik_pada_temp_sangat_rendah(self):
        rng = np.random.default_rng(3)
        ids = sample_bigram(self.P, self.c2i["m"], 30, 0.05, rng)
        # T sangat rendah ~ argmax: tiap langkah ambil argmax baris
        cur = self.c2i["m"]
        ekspektasi = []
        for _ in range(30):
            baris = prob_baris_dengan_temp(self.P[cur], 0.05)
            cur = int(np.argmax(baris))
            ekspektasi.append(cur)
        self.assertEqual(ids, ekspektasi)


if __name__ == "__main__":
    unittest.main()
