"""Spesifikasi Bagian 7 — Sampling (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/sampling.py lolos.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.sampling import sample_greedy, sample_topp, sample_topk

LOGITS = np.array([0.1, 2.2, 0.5, 1.3, 0.1, 0.9, 0.2, 0.4])


class TestGreedy(unittest.TestCase):
    def test_argmax(self):
        self.assertEqual(sample_greedy(LOGITS), 1)

    def test_ties_ambil_pertama(self):
        self.assertEqual(sample_greedy(np.array([1.0, 1.0, 0.0])), 0)


class TestTopK(unittest.TestCase):
    def test_hanya_dari_top_k(self):
        rng = np.random.default_rng(9)
        for _ in range(50):
            self.assertIn(sample_topk(LOGITS, 2, rng), (1, 3))

    def test_top1_greedy(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            self.assertEqual(sample_topk(LOGITS, 1, rng), 1)

    def test_distribusi_proporsional(self):
        rng = np.random.default_rng(0)
        counts = {1: 0, 3: 0}
        for _ in range(2000):
            counts[sample_topk(LOGITS, 2, rng)] += 1
        self.assertGreater(counts[1], counts[3], "token prob lebih besar harus lebih sering")

    def test_temperature_meratakan(self):
        rng = np.random.default_rng(0)
        counts = {1: 0, 3: 0}
        for _ in range(2000):
            counts[sample_topk(LOGITS, 2, rng, T=50.0)] += 1
        rasio = counts[1] / max(counts[3], 1)
        self.assertLess(rasio, 2.0, f"T besar harus mendekati merata, rasio {rasio:.2f}")


class TestTopP(unittest.TestCase):
    def test_nucleus_satu_token_deterministik(self):
        rng = np.random.default_rng(9)
        for _ in range(30):
            self.assertEqual(sample_topp(LOGITS, 0.3, rng), 1)

    def test_nucleus_p_sedang(self):
        rng = np.random.default_rng(9)
        for _ in range(50):
            self.assertIn(sample_topp(LOGITS, 0.6, rng), (1, 3, 5))

    def test_nucleus_p_satu_semua_kandidat(self):
        rng = np.random.default_rng(1)
        hasil = {sample_topp(LOGITS, 1.0, rng) for _ in range(1000)}
        self.assertEqual(hasil, set(range(8)))

    def test_kembalikan_int(self):
        rng = np.random.default_rng(3)
        self.assertIsInstance(sample_topp(LOGITS, 0.9, rng), int)


if __name__ == "__main__":
    unittest.main()
