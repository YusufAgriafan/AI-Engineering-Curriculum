"""Spesifikasi konvolusi, pooling & flatten dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/conv.py lolos.
Termasuk GRADIENT CHECK numerik: bukti backward benar, bukan cuma "kodenya jalan".
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.conv import (Conv2D, Flatten, conv2d_backward, conv2d_forward,
                          maxpool2x2_backward, maxpool2x2_forward, pad_zero)


def conv_ref(X, K, stride=1):
    """Referensi konvolusi dengan loop eksplisit (untuk membandingkan hasil)."""
    n, h, w = X.shape
    kh, kw = K.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((n, oh, ow))
    for i in range(n):
        for r in range(oh):
            for c in range(ow):
                out[i, r, c] = (X[i, r * stride:r * stride + kh,
                                  c * stride:c * stride + kw] * K).sum()
    return out


class TestPadZero(unittest.TestCase):
    def test_bentuk_dan_nol(self):
        X = np.ones((2, 3, 4))
        Xp = pad_zero(X, 1)
        self.assertEqual(Xp.shape, (2, 5, 6))
        self.assertTrue(np.allclose(Xp[:, 0, :], 0.0))
        self.assertTrue(np.allclose(Xp[:, -1, :], 0.0))
        self.assertTrue(np.allclose(Xp[:, :, 0], 0.0))
        self.assertTrue(np.allclose(Xp[:, :, -1], 0.0))
        self.assertTrue(np.allclose(Xp[:, 1:-1, 1:-1], 1.0))

    def test_pad_nol_tidak_alias(self):
        X = np.ones((2, 3, 3))
        Xp = pad_zero(X, 0)
        Xp[0, 0, 0] = 99.0
        self.assertEqual(X[0, 0, 0], 1.0, "pad=0 harus return salinan")

    def test_pad_negatif_error(self):
        with self.assertRaises(ValueError):
            pad_zero(np.ones((2, 3, 3)), -1)


class TestConvForward(unittest.TestCase):
    def test_kernel_satu_titik_copy(self):
        X = np.arange(12, dtype=float).reshape(1, 3, 4)
        K = np.array([[1.0]])
        self.assertTrue(np.allclose(conv2d_forward(X, K, stride=1), X))

    def test_detektor_tepi_vertikal(self):
        X = np.ones((1, 4, 4))
        K = np.array([[-1.0, 1.0]])          # detektor tepi vertikal
        out = conv2d_forward(X, K, stride=1)
        self.assertEqual(out.shape, (1, 4, 3))
        self.assertTrue(np.allclose(out, 0.0), "area konstan -> tepi = 0")

    def test_cocok_referensi_acak(self):
        rng = np.random.default_rng(0)
        for stride in (1, 2):
            X = rng.normal(size=(3, 7, 8))
            K = rng.normal(size=(3, 2))
            self.assertTrue(np.allclose(conv2d_forward(X, K, stride=stride),
                                        conv_ref(X, K, stride=stride)),
                            f"stride={stride}")

    def test_validasi_bentuk(self):
        with self.assertRaises(ValueError):
            conv2d_forward(np.zeros((3, 3)), np.ones((1, 1)))       # X bukan 3D
        with self.assertRaises(ValueError):
            conv2d_forward(np.zeros((1, 3, 3)), np.ones(2))          # K bukan 2D
        with self.assertRaises(ValueError):
            conv2d_forward(np.zeros((1, 3, 3)), np.ones((4, 1)))     # K > X
        with self.assertRaises(ValueError):
            conv2d_forward(np.zeros((1, 3, 3)), np.ones((1, 1)), stride=0)


class TestConvBackward(unittest.TestCase):
    def test_gradien_numerik_kernel(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(2, 5, 5))
        K = rng.normal(size=(3, 3))
        R = rng.normal(size=conv2d_forward(X, K).shape)   # L = sum(out * R)
        dXp, dK = conv2d_backward(X, K, R, stride=1)
        eps = 1e-6
        for u in range(3):
            for v in range(3):
                Kp, Km = K.copy(), K.copy()
                Kp[u, v] += eps
                Km[u, v] -= eps
                num = (conv2d_forward(X, Kp) * R).sum() - (conv2d_forward(X, Km) * R).sum()
                self.assertAlmostEqual(dK[u, v], num / (2 * eps), places=4)

    def test_gradien_numerik_input(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(2, 6, 6))
        K = rng.normal(size=(2, 2))
        stride = 2
        R = rng.normal(size=conv2d_forward(X, K, stride).shape)
        dXp, _ = conv2d_backward(X, K, R, stride=stride)
        eps = 1e-6
        idx = [(0, 0), (1, 3), (4, 5), (5, 0), (3, 3)]
        for (i, r, c) in [(0, a, b) for (a, b) in idx]:
            Xp, Xm = X.copy(), X.copy()
            Xp[i, r, c] += eps
            Xm[i, r, c] -= eps
            num = (conv2d_forward(Xp, K, stride) * R).sum() - (conv2d_forward(Xm, K, stride) * R).sum()
            self.assertAlmostEqual(dXp[i, r, c], num / (2 * eps), places=4)


class TestMaxPool(unittest.TestCase):
    def test_forward_nilai_maks(self):
        X = np.array([[[1., 2., 5., 6.],
                       [3., 4., 7., 8.],
                       [9., 10., 13., 14.],
                       [11., 12., 15., 16.]]])
        out = maxpool2x2_forward(X)
        self.assertEqual(out.shape, (1, 2, 2))
        self.assertTrue(np.allclose(out, [[4., 8.], [12., 16.]]))

    def test_backward_rute_argmax(self):
        X = np.array([[[0.1, 0.9, 0.3, 0.7],
                       [0.3, 0.2, 0.6, 0.4],
                       [0.7, 0.05, 0.2, 0.3],
                       [0.1, 0.4, 0.9, 0.05]]])
        grad = np.array([[[1., 2.], [3., 4.]]])
        dX = maxpool2x2_backward(X, grad)
        self.assertEqual(dX.shape, X.shape)
        self.assertAlmostEqual(dX[0, 0, 1], 1.0)   # max jendela 1: posisi (0,1)
        self.assertAlmostEqual(dX[0, 0, 3], 2.0)   # max jendela 2: posisi (0,3)
        self.assertAlmostEqual(dX[0, 2, 0], 3.0)   # max jendela 3: posisi (2,0)
        self.assertAlmostEqual(dX[0, 3, 2], 4.0)   # max jendela 4: posisi (3,2)
        self.assertAlmostEqual(dX.sum(), 10.0, "gradien hanya lewat 4 posisi max")

    def test_gradien_numerik_tanpa_seri(self):
        rng = np.random.default_rng(3)
        X = rng.permutation(np.arange(16, dtype=float)).reshape(1, 4, 4)  # semua unik
        R = rng.normal(size=(1, 2, 2))
        dX = maxpool2x2_backward(X, R)
        eps = 1e-6
        for (r, c) in [(0, 0), (1, 2), (3, 3), (2, 1)]:
            Xp, Xm = X.copy(), X.copy()
            Xp[0, r, c] += eps
            Xm[0, r, c] -= eps
            num = (maxpool2x2_forward(Xp) * R).sum() - (maxpool2x2_forward(Xm) * R).sum()
            self.assertAlmostEqual(dX[0, r, c], num / (2 * eps), places=4)

    def test_dimensi_ganjil_error(self):
        with self.assertRaises(ValueError):
            maxpool2x2_forward(np.zeros((1, 3, 4)))


class TestFlatten(unittest.TestCase):
    def test_maju_mundur_konsisten(self):
        F = Flatten()
        X = np.arange(24, dtype=float).reshape(2, 3, 4)
        out = F.forward(X)
        self.assertEqual(out.shape, (2, 12))
        self.assertTrue(np.allclose(out[0], X[0].ravel()))
        g = np.arange(24, dtype=float).reshape(2, 12)
        self.assertTrue(np.allclose(F.backward(g), g.reshape(2, 3, 4)))


class TestConv2DLayer(unittest.TestCase):
    def test_lazy_init_deterministik(self):
        L1 = Conv2D(3, 3, pad=1, seed=7)
        L2 = Conv2D(3, 3, pad=1, seed=7)
        X = np.random.default_rng(0).normal(size=(2, 8, 8))
        L1.forward(X)
        L2.forward(X)
        self.assertEqual(L1.K.shape, (3, 3))
        self.assertTrue(np.array_equal(L1.K, L2.K), "seed sama -> filter sama")

    def test_padding_dibuang_di_backward(self):
        L = Conv2D(3, 3, pad=1, seed=0)
        X = np.random.default_rng(1).normal(size=(2, 8, 8))
        out = L.forward(X)
        self.assertEqual(out.shape, (2, 8, 8), "pad=1 + kernel 3x3 menjaga ukuran")
        dX = L.backward(np.ones_like(out))
        self.assertEqual(dX.shape, (2, 8, 8), "padding harus dibuang dari dXp")
        self.assertEqual(L.dK.shape, (3, 3))

    def test_gradient_check_end_to_end(self):
        rng = np.random.default_rng(4)
        L = Conv2D(3, 3, pad=1, seed=5)
        X = rng.normal(size=(2, 6, 6))
        R = rng.normal(size=(2, 6, 6))
        out = L.forward(X)
        dX = L.backward(R)
        eps = 1e-6
        # cek dK
        for (u, v) in [(0, 0), (1, 2), (2, 1)]:
            Kp, Km = L.K.copy(), L.K.copy()
            Kp[u, v] += eps
            Km[u, v] -= eps
            num = (conv2d_forward(pad_zero(X, 1), Kp) * R).sum() - \
                  (conv2d_forward(pad_zero(X, 1), Km) * R).sum()
            self.assertAlmostEqual(L.dK[u, v], num / (2 * eps), places=4)
        # cek dX (dalam area bukan padding)
        for (r, c) in [(0, 0), (2, 3), (5, 5)]:
            Xp, Xm = X.copy(), X.copy()
            Xp[:, r, c] += eps
            Xm[:, r, c] -= eps
            num = (conv2d_forward(pad_zero(Xp, 1), L.K) * R).sum() - \
                  (conv2d_forward(pad_zero(Xm, 1), L.K) * R).sum()
            self.assertAlmostEqual(dX[:, r, c].sum(), num / (2 * eps), places=4)


if __name__ == "__main__":
    unittest.main()
