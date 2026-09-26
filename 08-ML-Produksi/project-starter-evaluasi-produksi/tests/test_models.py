"""Spesifikasi models dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di prod_mini/models.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prod_mini.data import TRUE_B, TRUE_W, make_dataset, split_tiga_arah
from prod_mini.models import fit_ham, fit_neg, fit_ols, predict, predict_ham, predict_neg


class TestFitOls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_dataset()
        cls.Xtr, cls.ytr, cls.Xca, cls.yca, cls.Xte, cls.yte = split_tiga_arah(X, y)

    def test_pulihkan_resep(self):
        w, b = fit_ols(self.Xtr, self.ytr)
        np.testing.assert_allclose(w, TRUE_W, atol=0.15)
        self.assertAlmostEqual(b, TRUE_B, delta=0.2)

    def test_bentuk(self):
        w, b = fit_ols(self.Xtr, self.ytr)
        self.assertEqual(w.shape, (2,))
        self.assertIsInstance(b, float)

    def test_prediksi_shape(self):
        w, b = fit_ols(self.Xtr, self.ytr)
        p = predict(w, b, self.Xte)
        self.assertEqual(p.shape, (len(self.Xte),))

    def test_exact_2fitur(self):
        # data tanpa noise: OLS harus pulihkan resep persis
        X = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
        y = X @ TRUE_W + TRUE_B
        w, b = fit_ols(X, y)
        np.testing.assert_allclose(w, TRUE_W, atol=1e-8)
        self.assertAlmostEqual(b, TRUE_B, places=8)


class TestHam(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(fit_ham([1., 2., 3.]), 2.0, places=12)

    def test_prediksi_konstan(self):
        p = predict_ham(2.5, np.zeros((4, 2)))
        np.testing.assert_array_equal(p, [2.5, 2.5, 2.5, 2.5])

    def test_ham_kalah_ols(self):
        X, y = make_dataset()
        Xtr, ytr, _, _, Xte, yte = split_tiga_arah(X, y)
        w, b = fit_ols(Xtr, ytr)
        err_ols = float(np.mean((yte - predict(w, b, Xte)) ** 2))
        err_ham = float(np.mean((yte - predict_ham(fit_ham(ytr), Xte)) ** 2))
        self.assertLess(err_ols, err_ham)


class TestFitNeg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_dataset()
        cls.Xtr, cls.ytr, _, _, cls.Xte, cls.yte = split_tiga_arah(X, y)

    def test_bentuk(self):
        w, b = fit_neg(self.Xtr, self.ytr)
        self.assertEqual(w.shape, (1,))
        self.assertIsInstance(b, float)

    def test_fit_prediksi_konsisten(self):
        # fit & predict saling cocok (dua pembalikan): prediksi model == -x2·w + b
        w, b = fit_neg(self.Xtr, self.ytr)
        p = predict_neg(w, b, self.Xte[:5])
        manual = -self.Xte[:5, 1] * w[0] + b
        np.testing.assert_allclose(p, manual, atol=1e-12)

    def test_negatif_lebih_buruk_dari_ham(self):
        # model salah arah ini HARUS kalah dari HAM — itu inti pembelajarannya
        from prod_mini.metrics import rmse
        w, b = fit_neg(self.Xtr, self.ytr)
        rmse_neg = rmse(self.yte, predict_neg(w, b, self.Xte))
        rmse_ham = rmse(self.yte, predict_ham(fit_ham(self.ytr), self.Xte))
        self.assertGreater(rmse_neg, rmse_ham)


if __name__ == "__main__":
    unittest.main()
