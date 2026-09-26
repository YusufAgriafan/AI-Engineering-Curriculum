"""Spesifikasi threshold & confusion dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di prod_mini/threshold.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prod_mini.threshold import confusion, pilih_threshold, prf


class TestConfusion(unittest.TestCase):
    def test_seimbang(self):
        cm = confusion(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.3, 0.1]), 0.5)
        self.assertEqual(cm, {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1})

    def test_tepat_di_threshold_adalah_positif(self):
        cm = confusion(np.array([0]), np.array([0.5]), 0.5)
        self.assertEqual(cm, {'tp': 0, 'fp': 1, 'fn': 0, 'tn': 0})

    def test_semua_benar(self):
        cm = confusion(np.array([1, 0, 0, 0]), np.array([1.0, 0.0, 0.4, 0.2]), 0.5)
        self.assertEqual(cm, {'tp': 1, 'fp': 0, 'fn': 0, 'tn': 3})

    def test_nilai_int(self):
        cm = confusion(np.array([1, 0]), np.array([0.7, 0.3]), 0.5)
        for v in cm.values():
            self.assertIsInstance(v, int)


class TestPrf(unittest.TestCase):
    def test_dasar(self):
        m = prf({'tp': 3, 'fp': 1, 'fn': 2, 'tn': 4})
        self.assertAlmostEqual(m['precision'], 0.75, places=12)
        self.assertAlmostEqual(m['recall'], 0.6, places=12)
        self.assertAlmostEqual(m['f1'], 2 * 0.75 * 0.6 / 1.35, places=12)

    def test_pembagi_nol_tidak_crash(self):
        m = prf({'tp': 0, 'fp': 0, 'fn': 5, 'tn': 5})
        self.assertEqual(m['precision'], 0.0)
        self.assertEqual(m['recall'], 0.0)
        self.assertEqual(m['f1'], 0.0)

    def test_sempurna(self):
        m = prf({'tp': 2, 'fp': 0, 'fn': 0, 'tn': 1})
        self.assertEqual(m['precision'], 1.0)
        self.assertEqual(m['recall'], 1.0)
        self.assertEqual(m['f1'], 1.0)


class TestPilihThreshold(unittest.TestCase):
    def test_pilih_f1_tertinggi(self):
        # dua kandidat: 0.5 jelas lebih baik daripada 3.0
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.3, 0.1])
        thr, cm, m = pilih_threshold(y, s, [0.5, 3.0])
        self.assertEqual(float(thr), 0.5)
        self.assertGreater(m['f1'], 0.0)

    def test_ties_ambil_pertama(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.3, 0.1])
        thr1, _, _ = pilih_threshold(y, s, [0.5, 0.5, 0.5])
        self.assertEqual(float(thr1), 0.5)


if __name__ == "__main__":
    unittest.main()
