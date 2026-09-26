"""Spesifikasi AR(p) + forecast rekursif dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ts_mini/ar.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_mini.ar import ar_forecast, fit_ar


class TestFitAr(unittest.TestCase):
    def test_pulihkan_ar1(self):
        # AR(1) phi=0.7 -> estimasi dekat 0.7
        rng = np.random.default_rng(42)
        z = np.empty(2000)
        z[0] = 0.0
        for i in range(1, 2000):
            z[i] = 0.7 * z[i - 1] + rng.normal(0, 1.0)
        phi = fit_ar(z, 1)
        self.assertEqual(phi.shape, (1,))
        self.assertLess(abs(phi[0] - 0.7), 0.05)

    def test_bentuk_p(self):
        phi2 = fit_ar(np.arange(10, dtype=float), 2)
        self.assertEqual(phi2.shape, (2,))

    def test_ar2_exact_pada_data_deterministik(self):
        # y[t] = 0.5·y[t-1] - 0.2·y[t-2] (tanpa noise) -> koefisien harus persis
        n = 60
        y = np.empty(n)
        y[0], y[1] = 1.0, 0.4
        for i in range(2, n):
            y[i] = 0.5 * y[i - 1] - 0.2 * y[i - 2]
        phi = fit_ar(y, 2)
        # kontrak: coef[0] mengalikan nilai TERBARU (lag-1)
        self.assertAlmostEqual(phi[0], 0.5, places=6)
        self.assertAlmostEqual(phi[1], -0.2, places=6)

    def test_fit_dari_train_saja(self):
        # jumlah baris menyusut: n - p
        rng = np.random.default_rng(3)
        s = rng.normal(size=100)
        fit_ar(s, 7)  # tidak boleh error


class TestArForecast(unittest.TestCase):
    def test_satu_langkah_konvensi(self):
        # last_values=[10, 4] artinya y[t-2]=10, y[t-1]=4
        # coef=[0.5, 0.3] -> pred = 0.5·4 + 0.3·10 = 5.0
        out = ar_forecast(np.array([0.5, 0.3]), np.array([10.0, 4.0]), 1)
        self.assertEqual(out.shape, (1,))
        self.assertAlmostEqual(out[0], 5.0, places=9)

    def test_rekursif_memakai_prediksi(self):
        c = np.array([0.5, 0.3])
        out = ar_forecast(c, np.array([10.0, 4.0]), 2)
        # langkah-2: hist terbaru = pred langkah-1 (5.0), lalu 4.0
        self.assertAlmostEqual(out[1], 0.5 * 5.0 + 0.3 * 4.0, places=9)

    def test_bentuk_h(self):
        out = ar_forecast(np.array([1.0]), np.array([2.0]), 10)
        self.assertEqual(out.shape, (10,))

    def test_ar1_phi_satu_konstan(self):
        # phi=1, tanpa noise: forecast = nilai terakhir terus-menerus
        out = ar_forecast(np.array([1.0]), np.array([3.0, 7.0]), 5)
        np.testing.assert_array_equal(out, np.full(5, 7.0))


if __name__ == "__main__":
    unittest.main()
