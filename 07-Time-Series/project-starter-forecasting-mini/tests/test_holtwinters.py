"""Spesifikasi Holt-Winters dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ts_mini/holtwinters.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_mini.holtwinters import holt_winters_add, holt_winters_mult


class TestHWAdd(unittest.TestCase):
    def setUp(self):
        # seri deterministik: musiman murni tanpa noise
        self.m = 8
        t = np.arange(64, dtype=float)
        self.seri = 10.0 + 0.0 * t + 5.0 * np.sin(2 * np.pi * t / self.m)

    def test_bentuk_return(self):
        f, lv, tb, ss = holt_winters_add(self.seri, self.m, 0.3, 0.05, 0.3, 6)
        self.assertEqual(f.shape, (6,))
        self.assertEqual(ss.shape, (self.m,))
        self.assertIsInstance(lv, float)
        self.assertIsInstance(tb, float)

    def test_season_jumlah_nol(self):
        _, _, _, ss = holt_winters_add(self.seri, self.m, 0.3, 0.05, 0.3, 6)
        self.assertAlmostEqual(float(np.sum(ss)), 0.0, places=5)

    def test_forecast_mengikuti_pola(self):
        # pada seri musiman murni, forecast 1 langkah harus dekat nilai musiman berikutnya
        f, _, _, _ = holt_winters_add(self.seri, self.m, 0.5, 0.1, 0.5, self.m)
        lanjut = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(64, 64 + self.m) / self.m)
        self.assertLess(float(np.max(np.abs(f - lanjut))), 2.0,
                        "forecast harus mengikuti pola musiman deterministik")

    def test_deterministik(self):
        a = holt_winters_add(self.seri, self.m, 0.3, 0.05, 0.3, 4)[0]
        b = holt_winters_add(self.seri, self.m, 0.3, 0.05, 0.3, 4)[0]
        np.testing.assert_array_equal(a, b)

    def test_diterima_data_bab7(self):
        # pada seri sintetis Bab 7, angka harus masuk rentang kalibrasi
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
        from ts_mini.data import HORIZON, SPLIT_IDX, make_series
        seri, _ = make_series()
        train = seri[:SPLIT_IDX]
        f, lv, tb, ss = holt_winters_add(train, 365, 0.3, 0.05, 0.3, HORIZON)
        self.assertLess(abs(tb), 0.5)
        self.assertGreater(float(np.max(np.abs(ss))), 5.0)


class TestHWMult(unittest.TestCase):
    def test_bentuk(self):
        m = 8
        t = np.arange(64, dtype=float)
        seri = 10.0 * (1.0 + 0.2 * np.sin(2 * np.pi * t / m))
        f = holt_winters_mult(seri, m, 0.3, 0.05, 0.3, 6)
        self.assertEqual(f.shape, (6,))

    def test_pada_seri_bab7_mult_menang(self):
        # kalibrasi Bab 7: di seri multiplikatif, mult MAE < add MAE
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
        from ts_mini.data import HORIZON, SPLIT_IDX, make_mult_series
        from ts_mini.baseline import mae
        seri, _ = make_mult_series()
        train, val = seri[:SPLIT_IDX], seri[SPLIT_IDX:]
        f_add = holt_winters_add(train, 365, 0.3, 0.05, 0.3, HORIZON)[0]
        f_mult = holt_winters_mult(train, 365, 0.3, 0.05, 0.3, HORIZON)
        self.assertLess(mae(val, f_mult), mae(val, f_add))

    def test_deterministik(self):
        m = 8
        t = np.arange(64, dtype=float)
        seri = 10.0 * (1.0 + 0.2 * np.sin(2 * np.pi * t / m))
        a = holt_winters_mult(seri, m, 0.3, 0.05, 0.3, 4)
        b = holt_winters_mult(seri, m, 0.3, 0.05, 0.3, 4)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
