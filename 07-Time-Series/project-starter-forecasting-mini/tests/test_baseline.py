"""Spesifikasi metrik & baseline dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ts_mini/baseline.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_mini.baseline import forecast_ham, forecast_naive, forecast_seasonal_naive, mae, rmse, smape


class TestMetrik(unittest.TestCase):
    def test_mae(self):
        self.assertAlmostEqual(mae(np.array([1.0, 2.0]), np.array([2.0, 4.0])), 1.5)

    def test_mae_sama(self):
        self.assertEqual(mae(np.array([5.0]), np.array([5.0])), 0.0)

    def test_rmse_hukum_error_besar(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, -1.0])
        c = np.array([2.0, 0.0])
        self.assertAlmostEqual(rmse(a, c), 2 ** 0.5)
        self.assertGreater(rmse(a, c), mae(a, c), "RMSE harus > MAE saat error tidak rata")

    def test_smape_simetris(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(smape(x, y), smape(y, x), places=9)

    def test_smape_aman_nol(self):
        v = smape(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        self.assertTrue(np.isfinite(v), "epsilon harus menjaga dari pembagian nol")

    def test_smape_sempurna(self):
        x = np.array([3.0, 4.0])
        self.assertEqual(smape(x, x), 0.0)


class TestNaive(unittest.TestCase):
    def test_konstan_dari_nilai_terakhir(self):
        out = forecast_naive(np.array([1.0, 2.0, 9.0]), 4)
        self.assertEqual(out.shape, (4,))
        np.testing.assert_array_equal(out, [9, 9, 9, 9])

    def test_train_tidak_berubah(self):
        tr = np.array([5.0, 6.0])
        forecast_naive(tr, 3)
        np.testing.assert_array_equal(tr, [5.0, 6.0])


class TestSeasonalNaive(unittest.TestCase):
    def test_satu_siklus_diulang(self):
        tr = np.arange(10, dtype=float)
        out = forecast_seasonal_naive(tr, 4, 5)
        np.testing.assert_array_equal(out, [5, 6, 7, 8])

    def test_h_lebih_besar_dari_period(self):
        tr = np.array([10.0, 20.0, 30.0])
        out = forecast_seasonal_naive(tr, 5, 3)
        np.testing.assert_array_equal(out, [10, 20, 30, 10, 20])

    def test_h_sama_period(self):
        tr = np.array([4.0, 1.0, 7.0])
        out = forecast_seasonal_naive(tr, 3, 3)
        np.testing.assert_array_equal(out, [4, 1, 7])


class TestHam(unittest.TestCase):
    def test_mean_train(self):
        out = forecast_ham(np.array([1.0, 2.0, 3.0]), 3)
        np.testing.assert_array_equal(out, [2, 2, 2])

    def test_bentuk(self):
        self.assertEqual(forecast_ham(np.arange(50.0), 7).shape, (7,))


if __name__ == "__main__":
    unittest.main()
