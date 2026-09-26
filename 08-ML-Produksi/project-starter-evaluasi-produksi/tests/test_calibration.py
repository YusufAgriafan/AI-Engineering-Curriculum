"""Spesifikasi kalibrasi probability dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di prod_mini/calibration.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prod_mini.calibration import (apply_platt, binned_calibration, ece,
                                   fit_platt, sigmoid)


class TestSigmoid(unittest.TestCase):
    def test_nol(self):
        self.assertEqual(sigmoid(0.0), 0.5)

    def test_ekstrem_stabil(self):
        s = sigmoid(np.array([-1000.0, 1000.0]))
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertLess(s[0], 1e-20)
        self.assertGreater(s[1], 1.0 - 1e-15)

    def test_rentang(self):
        # float64: sigmoid(50) sudah tepat == 1.0; rentang 'dalam' yang diuji keketatannya
        p = sigmoid(np.linspace(-30, 30, 101))
        self.assertTrue(np.all(p > 0) and np.all(p < 1))
        p_ext = sigmoid(np.array([-60.0, 60.0]))   # batas clip
        self.assertTrue(np.all(np.isfinite(p_ext)) and np.all(p_ext >= 0) and np.all(p_ext <= 1))

    def test_nilai_dasar(self):
        np.testing.assert_allclose(
            sigmoid(np.array([-2.0, 3.0])),
            [0.1192029, 0.9525741], atol=1e-6)


class TestBinnedCalibration(unittest.TestCase):
    def test_konvensi_bin(self):
        yt = np.array([1, 0, 1, 0])
        pp = np.array([0.9, 0.1, 0.8, 0.2])
        bins = binned_calibration(yt, pp, 10)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[0]['n'], 0)          # tidak ada p di [0.0, 0.1)
        self.assertEqual(bins[1]['n'], 1)          # p=0.1 -> bin idx 1
        self.assertAlmostEqual(bins[1]['conf'], 0.1, places=12)
        self.assertAlmostEqual(bins[1]['acc'], 0.0, places=12)
        self.assertEqual(bins[9]['n'], 1)          # p=0.9 -> bin terakhir
        self.assertAlmostEqual(bins[9]['acc'], 1.0, places=12)

    def test_p_satu_masuk_bin_terakhir(self):
        bins = binned_calibration(np.array([1, 0]), np.array([1.0, 0.0]), 10)
        self.assertEqual(bins[9]['n'], 1)
        self.assertEqual(bins[0]['n'], 1)


class TestEce(unittest.TestCase):
    def test_contoh_kuis(self):
        e = ece(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]), 10)
        self.assertAlmostEqual(e, 0.15, places=12)

    def test_sempurna(self):
        e = ece(np.array([1, 0]), np.array([1.0, 0.0]), 10)
        self.assertAlmostEqual(e, 0.0, places=12)

    def test_overconfident(self):
        # yakin 100% pada 4 prediksi, 2 benar -> |0.5 - 1.0| = 0.5
        e = ece(np.array([1, 0, 1, 0]), np.array([1.0, 1.0, 1.0, 1.0]), 10)
        self.assertAlmostEqual(e, 0.5, places=12)


class TestPlatt(unittest.TestCase):
    def test_bentuk(self):
        s = np.linspace(-2, 2, 50)
        y = (s > 0).astype(int)
        a, b = fit_platt(s, y)
        self.assertIsInstance(a, float)
        self.assertIsInstance(b, float)

    def test_memperbaiki_kalibrasi(self):
        # skor mentah overconfident: sigmoid(4*s); Platt belajar memperkecil skala
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 400)
        y = (sigmoid(s) > rng.uniform(0, 1, 400)).astype(int)
        a, b = fit_platt(s, y)
        p_raw = sigmoid(4.0 * s)
        p_cal = apply_platt(a, b, 4.0 * s)
        # Platt setidaknya tidak memperburuk ECE pada data ini
        self.assertLessEqual(ece(y, p_cal, 10), ece(y, p_raw, 10) + 0.05)

    def test_rentang_output(self):
        a, b = fit_platt(np.linspace(-3, 3, 100), (np.linspace(-3, 3, 100) > 0).astype(int))
        p = apply_platt(a, b, np.array([-5.0, 0.0, 5.0]))
        self.assertTrue(np.all(p > 0) and np.all(p < 1))
        # nilai ekstrem: tetap finite dan di [0, 1]
        p_ext = apply_platt(a, b, np.array([-1000.0, 1000.0]))
        self.assertTrue(np.all(np.isfinite(p_ext)) and np.all(p_ext >= 0) and np.all(p_ext <= 1))


if __name__ == "__main__":
    unittest.main()
