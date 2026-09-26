"""Spesifikasi metrics dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di prod_mini/metrics.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prod_mini.metrics import mae, r2, rmse, skill_score


class TestRmse(unittest.TestCase):
    def test_dasar(self):
        self.assertAlmostEqual(rmse([1., 2.], [2., 4.]), np.sqrt(2.5), places=12)

    def test_sederhana(self):
        self.assertAlmostEqual(rmse([0., 0.], [3., 4.]), np.sqrt(12.5), places=12)

    def test_sempurna(self):
        self.assertEqual(rmse([1., 2., 3.], [1., 2., 3.]), 0.0)

    def test_return_float(self):
        self.assertIsInstance(rmse([1.], [2.]), float)


class TestMae(unittest.TestCase):
    def test_dasar(self):
        self.assertAlmostEqual(mae([1., 2.], [2., 4.]), 1.5, places=12)

    def test_rmse_ge_mae(self):
        # Cauchy-Schwarz: RMSE >= MAE selalu; error (1, 9) menonjolkan bedanya
        e, p = np.array([1., 9.]), np.zeros(2)
        self.assertGreaterEqual(rmse(e, p), mae(e, p) - 1e-12)

    def test_sempurna(self):
        self.assertEqual(mae([5.], [5.]), 0.0)


class TestR2(unittest.TestCase):
    def test_sempurna(self):
        self.assertAlmostEqual(r2([1., 2., 3.], [1., 2., 3.]), 1.0, places=12)

    def test_negatif_jika_lebih_buruk_dari_mean(self):
        self.assertLess(r2([1., 2., 3.], [3., 2., 1.]), 0.0)

    def test_mean_prediksi(self):
        # prediksi konstan = mean(y_true) -> R2 = 0
        self.assertAlmostEqual(r2([1., 2., 3.], [2., 2., 2.]), 0.0, places=12)


class TestSkillScore(unittest.TestCase):
    def test_setengah(self):
        self.assertAlmostEqual(skill_score(1.0, 2.0), 0.5, places=12)

    def test_setara(self):
        self.assertAlmostEqual(skill_score(2.0, 2.0), 0.0, places=12)

    def test_negatif(self):
        self.assertLess(skill_score(3.0, 2.0), 0.0)

    def test_sempurna(self):
        self.assertAlmostEqual(skill_score(0.0, 2.0), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
