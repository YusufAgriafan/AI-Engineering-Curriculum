"""Spesifikasi logistic regression from-scratch dalam bentuk test (mode TDD).

Jalankan dari folder project ini:
    python -m unittest discover -s tests -v

JANGAN DIUBAH — tugasmu membuat implementasi di ml_sederhana/model.py lolos.
Angka acuan dataset dihitung dari generator seed 42 yang dibekukan di
ml_sederhana/data.npz — tidak boleh berubah.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_sederhana.data import muat_semua  # noqa: E402
from ml_sederhana.model import (  # noqa: E402
    LogisticRegressionGD,
    binary_cross_entropy,
    sigmoid,
)

X_train, y_train, X_val, y_val, X_test, y_test, NAMA = muat_semua()


class TestSigmoid(unittest.TestCase):
    def test_nilai_kunci(self):
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5)
        self.assertAlmostEqual(float(sigmoid(2.0)), 1 / (1 + np.exp(-2.0)))

    def test_stabil_ekstrem(self):
        out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.all(np.isfinite(out)), "tidak boleh ada NaN/inf")
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[2]), 1.0, places=6)

    def test_bentuk(self):
        z = np.zeros((3, 4))
        self.assertEqual(sigmoid(z).shape, (3, 4))


class TestBCE(unittest.TestCase):
    def test_nilai_dasar(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        p = np.array([0.2, 0.8, 0.6, 0.3])
        expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.assertAlmostEqual(binary_cross_entropy(y, p), float(expected))

    def test_sempurna(self):
        y = np.array([0.0, 1.0])
        p = np.array([1e-15, 1 - 1e-15])
        self.assertAlmostEqual(binary_cross_entropy(y, p), 0.0, places=6)


class TestLogisticRegressionGD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LogisticRegressionGD(lr=0.1, n_iter=2000).fit(X_train, y_train)

    def test_loss_menurun(self):
        hist = self.model.loss_history
        self.assertGreaterEqual(len(hist), 2)
        self.assertLess(hist[-1], hist[0], "loss harus menurun setelah training")
        self.assertTrue(
            all(hist[i] >= hist[i + 1] - 1e-12 for i in range(len(hist) - 1)),
            "loss full-batch GD harus monoton turun",
        )

    def test_val_accuracy_threshold_default(self):
        pred = self.model.predict(X_val)  # threshold default 0.5
        acc = float((pred == y_val).mean())
        self.assertGreaterEqual(acc, 0.89, f"val accuracy terlalu rendah: {acc:.4f}")

    def test_angka_referensi_dataset(self):
        """Angka acuan dari generator seed 42 (membekukan dataset)."""
        p_val = self.model.predict_proba(X_val)
        self.assertAlmostEqual(float(p_val.mean()), 0.1151, places=3)

        pred = self.model.predict(X_val)
        TP = int(((y_val == 1) & (pred == 1)).sum())
        FP = int(((y_val == 0) & (pred == 1)).sum())
        FN = int(((y_val == 1) & (pred == 0)).sum())
        self.assertEqual(TP, 13)
        self.assertEqual(FP, 7)
        self.assertEqual(FN, 14)

    def test_custom_threshold_menaikkan_recall(self):
        """Threshold rendah harus menaikkan recall (mungkin turunkan precision)."""
        p_val = self.model.predict_proba(X_val)
        pred_rendah = (p_val >= 0.2).astype(int)
        pred_biasa = (p_val >= 0.5).astype(int)
        rec = lambda pred: int(((y_val == 1) & (pred == 1)).sum()) / int((y_val == 1).sum())  # noqa: E731
        self.assertGreaterEqual(rec(pred_rendah), rec(pred_biasa))

    def test_kontak_ke_fitur_terkuat(self):
        """Fitur 0 (jam_pakai) paling membedakan -> |w[0]| terbesar di antara fitur 0-5."""
        w = self.model.w
        idx_terkuat = int(np.argmax(np.abs(w[:6])))
        self.assertEqual(idx_terkuat, 0)

    def test_predict_sebelum_fit(self):
        model = LogisticRegressionGD()
        with self.assertRaises(RuntimeError):
            model.predict_proba(X_val[:5])


if __name__ == "__main__":
    unittest.main()
