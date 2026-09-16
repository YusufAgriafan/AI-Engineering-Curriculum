"""Spesifikasi augmentasi dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/augment.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.augment import (augment_batch, flip_horizontal, flip_vertical,
                             random_noise, random_shift)


class TestFlip(unittest.TestCase):
    def test_horizontal_membalik_kolom(self):
        X = np.arange(12, dtype=float).reshape(1, 3, 4)
        out = flip_horizontal(X)
        self.assertTrue(np.allclose(out, X[:, :, ::-1]))
        self.assertTrue(np.allclose(out[0, 0], [3., 2., 1., 0.]))

    def test_vertical_membalik_baris(self):
        X = np.arange(12, dtype=float).reshape(1, 3, 4)
        out = flip_vertical(X)
        self.assertTrue(np.allclose(out, X[:, ::-1, :]))
        self.assertTrue(np.allclose(out[0, :, 0], [8., 4., 0.]))

    def test_flip_ganda_identitas(self):
        X = np.random.default_rng(0).normal(size=(2, 5, 5))
        self.assertTrue(np.allclose(flip_horizontal(flip_horizontal(X)), X))
        self.assertTrue(np.allclose(flip_vertical(flip_vertical(X)), X))

    def test_tidak_mengubah_asli(self):
        X = np.ones((2, 4, 4))
        asli = X.copy()
        flip_horizontal(X)
        flip_vertical(X)
        self.assertTrue(np.allclose(X, asli), "flip tidak boleh in-place")

    def test_label_tidak_pernah_disentuh(self):
        # garis horizontal di baris 1 — setelah flip horizontal TETAP horizontal
        X = np.zeros((1, 4, 4))
        X[0, 1, :] = 1.0
        out = flip_horizontal(X)
        self.assertTrue(np.allclose(out.sum(axis=2), [[0., 4., 0., 0.]]),
                        "garis tetap di baris 1 -> label 'horizontal' tetap valid")


class TestRandomNoise(unittest.TestCase):
    def test_deterministik_dengan_seed(self):
        X = np.ones((10, 8, 8))
        a = random_noise(X, np.random.default_rng(0))
        b = random_noise(X, np.random.default_rng(0))
        self.assertTrue(np.array_equal(a, b))

    def test_tidak_mengubah_asli(self):
        X = np.ones((10, 8, 8))
        asli = X.copy()
        random_noise(X, np.random.default_rng(1))
        self.assertTrue(np.allclose(X, asli))

    def test_bentuk_dan_proporsi(self):
        X = np.ones((20, 8, 8))
        out = random_noise(X, np.random.default_rng(2), p=0.25, strength=0.5)
        self.assertEqual(out.shape, X.shape)
        berubah = (~np.isclose(out, X)).sum(axis=(1, 2)) > 0
        self.assertEqual(int(berubah.sum()), 5, "p=0.25 dari 20 gambar = 5 gambar")
        delta = out[berubah][0] - X[berubah][0]
        self.assertGreaterEqual(delta.max(), 0.4)
        self.assertLessEqual(delta.max(), 0.9 + 1e-9)

    def test_p_nol_tanpa_perubahan(self):
        X = np.ones((10, 8, 8))
        out = random_noise(X, np.random.default_rng(3), p=0.0)
        self.assertTrue(np.allclose(out, X))


class TestRandomShift(unittest.TestCase):
    def test_shift_nol_identitas(self):
        X = np.ones((3, 8, 8))
        out = random_shift(X, np.random.default_rng(4), max_shift=0)
        self.assertTrue(np.allclose(out, X))

    def test_area_geser_dan_isi_nol(self):
        X = np.zeros((1, 4, 4))
        X[0, 1:3, 1:3] = 5.0                       # kotak 2x2 di tengah
        # deterministik: (dx, dy) = (1, 1) — pakai rng manual agar pasti
        rng = np.random.default_rng(11)
        rng.integers(-2, 3), rng.integers(-2, 3)   # konsumsi dua draw pertama?
        # Lebih aman: uji via properti, bukan nilai rng spesifik:
        out = random_shift(X, np.random.default_rng(0), max_shift=2)
        self.assertEqual(out.shape, X.shape)
        # area kosong tepat nol, total "massa" tidak pernah bertambah
        self.assertLessEqual(out.sum(), X.sum() + 1e-9)
        # dan tak ada nilai baru di luar {0, 5}
        self.assertTrue(np.isin(np.unique(np.round(out, 9)), [0.0, 5.0]).all())

    def test_konten_tetap_kotak(self):
        X = np.zeros((1, 8, 8))
        X[0, 3:5, 3:5] = 1.0
        rng = np.random.default_rng(7)
        for _ in range(8):
            out = random_shift(X, rng, max_shift=2)
            if out.sum() == 0:
                continue                            # tergeser penuh keluar frame
            ys, xs = np.nonzero(out[0])
            self.assertEqual(ys.max() - ys.min() + 1, 2, "kotak tetap utuh")
            self.assertEqual(xs.max() - xs.min() + 1, 2)

    def test_nilai_tidak_berubah(self):
        X = np.random.default_rng(5).uniform(0, 1, (6, 8, 8))
        out = random_shift(X, np.random.default_rng(9), max_shift=2)
        # translasi murni: nilai yang muncul harus berasal dari X sendiri
        # (atau nol), dan massa total tidak pernah bertambah
        nilai_X = np.unique(np.round(X.ravel(), 9))
        nilai_out = np.unique(np.round(out.ravel(), 9))
        self.assertTrue(np.isin(nilai_out, np.concatenate([[0.0], nilai_X])).all())
        self.assertLessEqual(out.sum(), X.sum() + 1e-9)


class TestAugmentBatch(unittest.TestCase):
    def test_bentuk_dan_label_utuh(self):
        X = np.random.default_rng(0).uniform(0, 1, (12, 8, 8))
        y = np.array([0, 1] * 6)
        ya = y.copy()
        Xa, ya2 = augment_batch(X, y, np.random.default_rng(1))
        self.assertEqual(Xa.shape, X.shape)
        self.assertTrue(np.array_equal(ya, ya2), "label TIDAK berubah")
        self.assertFalse(np.allclose(Xa, X), "minimal ada gambar yang berubah")

    def test_deterministik(self):
        X = np.random.default_rng(2).uniform(0, 1, (12, 8, 8))
        y = np.zeros(12)
        a = augment_batch(X, y, np.random.default_rng(3))
        b = augment_batch(X, y, np.random.default_rng(3))
        self.assertTrue(np.allclose(a[0], b[0]))


if __name__ == "__main__":
    unittest.main()
