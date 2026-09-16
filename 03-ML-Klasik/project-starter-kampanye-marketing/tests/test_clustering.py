"""Spesifikasi k-means from-scratch dalam bentuk test (mode TDD).

Jalankan dari folder project ini:
    python -m unittest discover -s tests -v

JANGAN DIUBAH — tugasmu membuat implementasi di ml_sederhana/clustering.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_sederhana.clustering import kmeans, standardize  # noqa: E402
from ml_sederhana.data import muat_semua  # noqa: E402

X_train, y_train, X_val, y_val, X_test, y_test, NAMA = muat_semua()
X_ALL = np.vstack([X_train, X_val, X_test])
Y_ALL = np.concatenate([y_train, y_val, y_test])


class TestStandardize(unittest.TestCase):
    def test_properti_zscore(self):
        Z = standardize(X_ALL)
        self.assertTrue(np.allclose(Z.mean(axis=0), 0, atol=1e-12))
        self.assertTrue(np.allclose(Z.std(axis=0), 1, atol=1e-12))

    def test_kolom_konstan_tidak_nan(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        Z = standardize(X)
        self.assertTrue(np.all(np.isfinite(Z)), "kolom konstan tidak boleh menghasilkan NaN")
        self.assertTrue(np.allclose(Z[:, 0], 0))


class TestKmeans(unittest.TestCase):
    def test_dua_grup_jelas(self):
        rng = np.random.default_rng(0)
        grupA = rng.normal(-10, 0.5, size=(50, 2))
        grupB = rng.normal(+10, 0.5, size=(50, 2))
        X = np.vstack([grupA, grupB])
        labels, centroids, inertia = kmeans(X, k=2, seed=0)
        self.assertEqual(len(labels), 100)
        self.assertEqual(len(np.unique(labels)), 2, "harus menemukan 2 grup")
        # grup yang sama harus label sama
        self.assertEqual(len(set(labels[:50])), 1)
        self.assertEqual(len(set(labels[50:])), 1)
        self.assertLess(inertia, 100.0, "klaster padat -> inertia kecil (≈ n·dim·σ²)")

    def test_inertia_menurun_dengan_k(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, size=(300, 3))
        _, _, iner2 = kmeans(X, k=2, seed=0)
        _, _, iner5 = kmeans(X, k=5, seed=0)
        self.assertLessEqual(iner5, iner2, "k lebih besar -> inertia tidak boleh naik")

    def test_angka_referensi_dataset(self):
        """Angka acuan: standardize + kmeans(k=2, seed=0) di seluruh 1.200 titik."""
        Z = standardize(X_ALL)
        labels, _, inertia = kmeans(Z, k=2, seed=0)
        self.assertAlmostEqual(inertia, 7585.360, places=2)

        sizes = np.bincount(labels, minlength=2)
        self.assertEqual(sizes.tolist(), [638, 562])

        pos_rate = [float(Y_ALL[labels == kk].mean()) for kk in range(2)]
        self.assertAlmostEqual(pos_rate[0], 0.1693, places=3)
        self.assertAlmostEqual(pos_rate[1], 0.0427, places=3)

    def test_k_satu(self):
        labels, centroids, inertia = kmeans(X_ALL[:50], k=1, seed=0)
        self.assertTrue((labels == 0).all())
        self.assertTrue(np.allclose(centroids[0], X_ALL[:50].mean(axis=0)))
        self.assertGreater(inertia, 0.0)

    def test_input_tidak_valid(self):
        with self.assertRaises(ValueError):
            kmeans(X_ALL[:10], k=0)
        with self.assertRaises(ValueError):
            kmeans(X_ALL[:3], k=5)


if __name__ == "__main__":
    unittest.main()
