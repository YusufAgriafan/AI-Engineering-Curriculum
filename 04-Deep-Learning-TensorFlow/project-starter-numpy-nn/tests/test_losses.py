"""Spesifikasi losses dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/losses.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.losses import bce, bce_with_logits_grad, mse, mse_grad  # noqa: E402


class TestBCE(unittest.TestCase):
    def test_nilai_dasar(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        p = np.array([0.2, 0.8, 0.6, 0.3])
        expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.assertAlmostEqual(bce(y, p), float(expected))

    def test_tebakan_50_50(self):
        y = np.array([0.0, 1.0])
        self.assertAlmostEqual(bce(y, np.array([0.5, 0.5])), np.log(2), places=12)

    def test_sempurna_dengan_clip(self):
        y = np.array([0.0, 1.0])
        p = np.array([0.0, 1.0])          # tanpa clip -> log(0) = -inf
        val = bce(y, p)
        self.assertTrue(np.isfinite(val), "clip eps harus mencegah inf")

    def test_return_python_float(self):
        val = bce([0.0, 1.0], [0.3, 0.7])
        self.assertIsInstance(val, float)

    def test_bentuk_tidak_cocok(self):
        with self.assertRaises(ValueError):
            bce([0.0, 1.0], [0.3, 0.7, 0.1])


class TestBCEGrad(unittest.TestCase):
    def test_gradien_numerik(self):
        """Cek gradien analitik vs selisih-hingga (numerik)."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=20).astype(float)
        p0 = rng.uniform(0.05, 0.95, size=20)
        eps = 1e-6
        num = (bce(y, p0 + eps) - bce(y, p0 - eps)) / (2 * eps)
        ana = float(np.sum(bce_with_logits_grad(p0, y)))
        self.assertAlmostEqual(ana, num, places=5)

    def test_mean_bukan_sum(self):
        y = np.array([0.0, 1.0])
        p = np.array([0.2, 0.8])
        g = bce_with_logits_grad(p, y)
        self.assertAlmostEqual(float(np.sum(g)), 0.0, places=9)
        # p=0.2 utk y=0: (0.2-0)/(0.2*0.8)/2 = 0.625 ; p=0.8 utk y=1: (0.8-1)/(0.8*0.2)/2 = -0.625
        self.assertAlmostEqual(float(g[0]), 0.625, places=9)
        self.assertAlmostEqual(float(g[1]), -0.625, places=9)


class TestMSE(unittest.TestCase):
    def test_dasar(self):
        self.assertAlmostEqual(mse([0.0, 0.0], [1.0, 3.0]), 5.0)   # (1 + 9)/2

    def test_nol(self):
        self.assertAlmostEqual(mse([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_bentuk_tidak_cocok(self):
        with self.assertRaises(ValueError):
            mse([0.0, 1.0], [0.0])


class TestMSEGrad(unittest.TestCase):
    def test_gradien_numerik(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, size=15)
        p0 = rng.normal(0, 1, size=15)
        eps = 1e-6
        num = (mse(y, p0 + eps) - mse(y, p0 - eps)) / (2 * eps)
        ana = float(np.sum(mse_grad(p0, y)))
        self.assertAlmostEqual(ana, num, places=5)


if __name__ == "__main__":
    unittest.main()
