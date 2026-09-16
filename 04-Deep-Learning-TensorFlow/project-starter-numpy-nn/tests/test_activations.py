"""Spesifikasi fungsi aktivasi dalam bentuk test (mode TDD).

Jalankan dari folder project ini:
    python -m unittest discover -s tests -v

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/activations.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.activations import (  # noqa: E402
    relu,
    relu_grad,
    sigmoid,
    sigmoid_from_logits_stable,
    softmax,
)


class TestReLU(unittest.TestCase):
    def test_dasar(self):
        self.assertTrue(np.allclose(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0]))

    def test_bentuk_2d(self):
        z = np.array([[-1.0, 2.0], [0.5, -0.5]])
        self.assertEqual(relu(z).shape, (2, 2))


class TestReLUGrad(unittest.TestCase):
    def test_dasar(self):
        g = relu_grad(np.array([-2.0, 0.0, 3.0]))
        self.assertTrue(np.allclose(g, [0.0, 0.0, 1.0]))

    def test_bentuk_dan_tipe(self):
        g = relu_grad(np.zeros((3, 4)))
        self.assertEqual(g.shape, (3, 4))
        self.assertTrue(np.all(g == 0))


class TestSigmoid(unittest.TestCase):
    def test_nilai_kunci(self):
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5)
        self.assertAlmostEqual(float(sigmoid(2.0)), 1 / (1 + np.exp(-2.0)))

    def test_stabil_ekstrem(self):
        out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.all(np.isfinite(out)), "tidak boleh NaN/inf (pakai dua cabang!)")
        self.assertAlmostEqual(float(out[0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[2]), 1.0, places=6)

    def test_monoton(self):
        z = np.linspace(-10, 10, 41)
        p = sigmoid(z)
        self.assertTrue(np.all(np.diff(p) >= 0), "sigmoid harus monoton naik")


class TestSigmoidFromLogits(unittest.TestCase):
    def test_pasangan_p_grad(self):
        logits = np.array([-3.0, 0.0, 2.0])
        p, grad = sigmoid_from_logits_stable(logits)
        self.assertTrue(np.allclose(p, 1 / (1 + np.exp(-logits))))
        self.assertTrue(np.allclose(grad, p * (1 - p)))

    def test_grad_maksimum_di_nol(self):
        _, grad = sigmoid_from_logits_stable(0.0)
        self.assertAlmostEqual(float(grad), 0.25, places=12)


class TestSoftmax(unittest.TestCase):
    def test_jumlah_satu(self):
        p = softmax(np.array([2.0, 1.0, 0.1]))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        self.assertTrue(np.all(p > 0))

    def test_invarian_shift(self):
        # menambah konstanta pada semua logits TIDAK mengubah softmax
        a = softmax(np.array([1.0, 2.0, 3.0]))
        b = softmax(np.array([1001.0, 1002.0, 1003.0]))   # tanpa shift -> overflow
        self.assertTrue(np.allclose(a, b))

    def test_per_baris_2d(self):
        p = softmax(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
        self.assertEqual(p.shape, (2, 3))
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0))

    def test_stabil_besar(self):
        p = softmax(np.array([1000.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p[0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
