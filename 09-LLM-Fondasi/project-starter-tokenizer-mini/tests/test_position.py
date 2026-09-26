"""Spesifikasi Bagian 4 — Positional encoding (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/position.py lolos.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.position import positional_encoding, rpe


class TestShape(unittest.TestCase):
    def test_bentuk(self):
        self.assertEqual(positional_encoding(12, 16).shape, (12, 16))

    def test_bounded(self):
        PE = positional_encoding(30, 8)
        self.assertTrue(np.all(PE >= -1) and np.all(PE <= 1))


class TestNilai(unittest.TestCase):
    def test_pos_nol(self):
        PE = positional_encoding(5, 16)
        np.testing.assert_allclose(PE[0, 0::2], 0.0, atol=1e-12)   # sin(0)
        np.testing.assert_allclose(PE[0, 1::2], 1.0, atol=1e-12)   # cos(0)

    def test_freq_tertinggi(self):
        PE = positional_encoding(5, 16)
        self.assertAlmostEqual(PE[1, 0], np.sin(1.0), places=9)
        self.assertAlmostEqual(PE[1, 1], np.cos(1.0), places=9)

    def test_baris_dua_pasang_sin_cos(self):
        PE = positional_encoding(10, 8)
        div = np.exp(-np.log(10000.0) * np.arange(0, 8, 2) / 8)
        np.testing.assert_allclose(PE[3, 0::2], np.sin(3 * div), atol=1e-9)
        np.testing.assert_allclose(PE[3, 1::2], np.cos(3 * div), atol=1e-9)


class TestProperti(unittest.TestCase):
    def test_dot_fungsi_selisih_posisi(self):
        PE = positional_encoding(20, 16)
        jarak3 = PE[0] @ PE[3]
        self.assertAlmostEqual(PE[5] @ PE[8], jarak3, places=6)
        self.assertAlmostEqual(PE[2] @ PE[5], jarak3, places=6)
        self.assertNotAlmostEqual(PE[0] @ PE[5], jarak3, places=3)

    def test_rpe_satu_baris(self):
        PE = positional_encoding(10, 8)
        np.testing.assert_array_equal(rpe(3, 8), PE[3])

    def test_rpe_independen_maxlen(self):
        np.testing.assert_array_equal(rpe(3, 8), positional_encoding(4, 8)[3])


if __name__ == "__main__":
    unittest.main()
