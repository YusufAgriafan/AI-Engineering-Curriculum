"""Spesifikasi Bagian 3 — Embedding (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/embedding.py lolos.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.embedding import analogi, cari_mirip, cos_sim, latih_embedding


class TestCosSim(unittest.TestCase):
    def test_searah(self):
        self.assertAlmostEqual(cos_sim(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 1.0)

    def test_ortogonal(self):
        self.assertAlmostEqual(cos_sim(np.array([1.0, 0.0]), np.array([0.0, 3.0])), 0.0)

    def test_berlawanan(self):
        self.assertAlmostEqual(cos_sim(np.array([1.0, 1.0]), np.array([-2.0, -2.0])), -1.0)

    def test_pembagi_nol_aman(self):
        self.assertEqual(cos_sim(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])), 0.0)


class TestCariMirip(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.W = rng.normal(0, 1, (8, 8))

    def test_mengembalikan_top_n(self):
        hasil = cari_mirip(self.W, self.W[3], exclude={3}, top_n=3)
        self.assertEqual(len(hasil), 3)
        self.assertNotIn(3, hasil)

    def test_terurut_menurun_dan_terbaik_pertama(self):
        skor = [cos_sim(self.W[i], self.W[3]) for i in range(8) if i != 3]
        hasil = cari_mirip(self.W, self.W[3], exclude={3}, top_n=3)
        self.assertAlmostEqual(cos_sim(self.W[hasil[0]], self.W[3]), max(skor), places=9)
        s1 = cos_sim(self.W[hasil[0]], self.W[3])
        s2 = cos_sim(self.W[hasil[1]], self.W[3])
        self.assertGreaterEqual(s1, s2)


class TestAnalogi(unittest.TestCase):
    def test_analogi_konsisten_dengan_cari_mirip(self):
        W = latih_embedding()
        target = W[4] - W[1] + W[2]
        expected = cari_mirip(W, target, exclude=(1, 4, 2), top_n=1)[0]
        self.assertEqual(analogi(1, 4, 2, W), expected)

    def test_exclude_dihormati(self):
        W = latih_embedding()
        hasil = analogi(1, 4, 2, W)
        self.assertNotIn(hasil, (1, 4, 2))


class TestLatihEmbedding(unittest.TestCase):
    def test_deterministik_seed(self):
        W1 = latih_embedding(seed=9)
        W2 = latih_embedding(seed=9)
        np.testing.assert_array_equal(W1, W2)

    def test_struktur_muncul_dari_data(self):
        W = latih_embedding(idx_vokal=(0, 4, 6), vocab=8, d=8, epochs=200, seed=9)
        dalem = np.mean([cos_sim(W[i], W[j]) for i in (0, 4, 6) for j in (0, 4, 6) if i < j])
        luar = np.mean([cos_sim(W[i], W[j]) for i in (0, 4, 6) for j in (1, 2, 5)])
        self.assertGreater(dalem, 0.9, f"cos dalam kelompok {dalem:.3f} harus tinggi")
        self.assertGreater(dalem, luar + 0.5, "struktur kelompok harus jelas")


if __name__ == "__main__":
    unittest.main()
