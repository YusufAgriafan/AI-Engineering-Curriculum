"""Spesifikasi windowing & split temporal dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ts_mini/windowing.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_mini.windowing import buat_window, buat_window_multi, split_waktu


class TestBuatWindow(unittest.TestCase):
    def test_bentuk_dan_isi(self):
        s = np.arange(10, dtype=float)
        X, y = buat_window(s, 3)
        self.assertEqual(X.shape, (7, 3))
        self.assertEqual(y.shape, (7,))
        np.testing.assert_array_equal(X[0], [0, 1, 2])
        self.assertEqual(y[0], 3.0)
        np.testing.assert_array_equal(X[-1], [6, 7, 8])
        self.assertEqual(y[-1], 9.0)

    def test_n_sampel(self):
        s = np.arange(100, dtype=float)
        X, y = buat_window(s, 35)
        self.assertEqual(X.shape, (65, 35))
        self.assertEqual(y.shape, (65,))

    def test_sliding_murni(self):
        # window ke-i HARUS mulai di indeks i (tanpa lompatan)
        rng = np.random.default_rng(0)
        s = rng.normal(size=50)
        X, y = buat_window(s, 7)
        for i in (0, 1, 25, 42):
            np.testing.assert_array_equal(X[i], s[i:i + 7])
            self.assertEqual(y[i], s[i + 7])

    def test_w_satu(self):
        s = np.arange(5, dtype=float)
        X, y = buat_window(s, 1)
        self.assertEqual(X.shape, (4, 1))
        np.testing.assert_array_equal(y, [1, 2, 3, 4])

    def test_w_terlalu_besar(self):
        with self.assertRaises(Exception):
            buat_window(np.arange(5, dtype=float), 5)


class TestSplitWaktu(unittest.TestCase):
    def test_urutan_terjaga(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        Xtr, ytr, Xva, yva = split_waktu(X, y, 0.8)
        self.assertEqual(len(Xtr), 40)
        self.assertEqual(len(Xva), 10)
        np.testing.assert_array_equal(Xtr, X[:40])
        np.testing.assert_array_equal(Xva, X[40:])
        np.testing.assert_array_equal(yva, y[40:])

    def test_bukan_acak(self):
        # split yang benar TIDAK mengacak: Xtr monoton naik
        X = np.arange(100).reshape(50, 2)
        Xtr, _, Xva, _ = split_waktu(X, np.zeros(50), 0.8)
        self.assertTrue(np.all(np.diff(Xtr[:, 0]) > 0))
        self.assertTrue(np.all(np.diff(Xva[:, 0]) > 0))

    def test_rasio(self):
        X = np.zeros((10, 1))
        Xtr, _, _, _ = split_waktu(X, np.zeros(10), 0.5)
        self.assertEqual(len(Xtr), 5)


class TestBuatWindowMulti(unittest.TestCase):
    def test_bentuk(self):
        s = np.arange(20, dtype=float)
        X, y = buat_window_multi(s, 5, 3)
        self.assertEqual(X.shape, (13, 5))
        self.assertEqual(y.shape, (13, 3))

    def test_isi(self):
        s = np.arange(20, dtype=float)
        X, y = buat_window_multi(s, 5, 3)
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(y[0], [5, 6, 7])
        np.testing.assert_array_equal(y[-1], [17, 18, 19])

    def test_n_sampel(self):
        s = np.arange(1000, dtype=float)
        X, y = buat_window_multi(s, 35, 1)
        # n_sampel = n - W - h + 1 = 965
        self.assertEqual(X.shape, (965, 35))
        self.assertEqual(y.shape, (965, 1))


if __name__ == "__main__":
    unittest.main()
