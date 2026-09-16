"""Spesifikasi Dense & Sequential dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/layers.py lolos.
Termasuk gradien check numerik: bukti backward benar, bukan cuma "kodenya jalan".
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.layers import Dense, Sequential  # noqa: E402
from nn_mini.losses import bce  # noqa: E402


class TestDenseForward(unittest.TestCase):
    def test_bentuk_output(self):
        L = Dense(5, activation="relu", seed=0)
        a = L.forward(np.zeros((7, 3)))
        self.assertEqual(a.shape, (7, 5))
        self.assertEqual(L.W.shape, (3, 5))
        self.assertEqual(L.b.shape, (5,))

    def test_bias_nol_output_nol(self):
        L = Dense(4, activation="linear", seed=1)
        a = L.forward(np.zeros((6, 3)))
        self.assertTrue(np.allclose(a, 0.0), "W acak x X nol + b nol = nol")

    def test_init_he_vs_xavier(self):
        # He (relu): std = sqrt(2/fan_in) -> lebih lebar dari Xavier sqrt(1/fan_in)
        W_relu = Dense(256, activation="relu", seed=0)
        W_relu.forward(np.zeros((10, 64)))
        W_lin = Dense(256, activation="linear", seed=0)
        W_lin.forward(np.zeros((10, 64)))
        self.assertGreater(W_relu.W.std(), W_lin.W.std(),
                           "init He (relu) harus lebih lebar dari Xavier")

    def test_aktivasi_dipakai(self):
        L = Dense(3, activation="sigmoid", seed=2)
        L.W = np.full((2, 3), 100.0)     # z besar positif -> sigmoid ~ 1
        L.b = np.zeros(3)
        a = L.forward(np.ones((1, 2)))
        self.assertTrue(np.allclose(a, 1.0, atol=1e-6))

    def test_input_bukan_2d(self):
        L = Dense(3, seed=0)
        with self.assertRaises(ValueError):
            L.forward(np.zeros(5))

    def test_aktivasi_tidak_dikenal(self):
        with self.assertRaises(ValueError):
            Dense(3, activation="gelu_klaim_sendiri")


class TestDenseBackward(unittest.TestCase):
    def test_gradien_numerik_sebuah_layer(self):
        """dL/dX dari backward harus cocok dengan selisih-hingga numerik."""
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, size=(8, 4))
        L = Dense(3, activation="relu", seed=5)
        a = L.forward(X)
        G = rng.normal(0, 1, size=a.shape)          # dL/da acak

        dX = L.backward(G)

        # numerik: ganggu satu elemen X, ukur perubahan L = sum(a * G)
        loss = lambda Xv: np.sum(L.forward(Xv) * G)  # noqa: E731
        eps = 1e-6
        for i in range(8):
            for j in range(4):
                Xp, Xm = X.copy(), X.copy()
                Xp[i, j] += eps
                Xm[i, j] -= eps
                num = (loss(Xp) - loss(Xm)) / (2 * eps)
                self.assertAlmostEqual(dX[i, j], num, places=4,
                                       msg=f"dL/dX[{i},{j}] analitik vs numerik")

    def test_dW_db_bentuk(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, size=(12, 5))
        L = Dense(4, activation="linear", seed=9)
        a = L.forward(X)
        dX = L.backward(np.ones_like(a))
        self.assertEqual(L.dW.shape, (5, 4))
        self.assertEqual(L.db.shape, (4,))
        self.assertTrue(np.allclose(L.db, 12.0), "db = sum dz; linear + grad 1 + n=12 -> db = 12")


class TestSequential(unittest.TestCase):
    def _model(self):
        return Sequential([Dense(8, activation="relu", seed=1),
                           Dense(1, activation="sigmoid", seed=2)])

    def test_forward_bentuk(self):
        model = self._model()
        out = model.forward(np.zeros((10, 3)))
        self.assertEqual(out.shape, (10, 1))
        self.assertTrue(np.allclose(out, 0.5), "sigmoid(0)=0.5 (b nol, X nol)")

    def test_backward_bentuk(self):
        model = self._model()
        X = np.random.default_rng(0).normal(size=(10, 3))
        out = model.forward(X)
        dX = model.backward(np.ones_like(out))
        self.assertEqual(dX.shape, (10, 3))

    def test_parameters_urutan(self):
        model = self._model()
        model.forward(np.zeros((4, 3)))
        params = model.parameters()
        self.assertEqual(len(params), 4)             # W1, b1, W2, b2
        self.assertEqual(params[0].shape, (3, 8))
        self.assertEqual(params[1].shape, (8,))
        self.assertEqual(params[2].shape, (8, 1))
        self.assertEqual(params[3].shape, (1,))

    def test_full_network_gradient_check(self):
        """Gradien check END-TO-END: loss MLP vs selisih-hingga per bobot."""
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, size=(15, 2))
        y = rng.integers(0, 2, size=15).astype(float)
        model = self._model()
        model.forward(X)
        params = model.parameters()

        def loss_fn():
            p = model.forward(X).ravel()
            return bce(y, p)

        # analitik: backward dari gradien BCE terhadap output
        eps = 1e-12
        p = model.forward(X).ravel()
        pc = np.clip(p, eps, 1 - eps)
        dL_da = ((pc - y) / (pc * (1 - pc)) / len(y)).reshape(-1, 1)
        model.backward(dL_da)

        # numerik pada subset bobot (sampling biar cepat tapi tetap tegas)
        rng2 = np.random.default_rng(42)
        for k in range(len(params)):
            P = params[k]
            idxs = [tuple(int(rng2.integers(0, s)) for s in P.shape)] if P.ndim else [()]
            for idx in idxs:
                idx = idx if isinstance(idx, tuple) else ()
                orig = P[idx]
                P[idx] = orig + 1e-6
                lp = loss_fn()
                P[idx] = orig - 1e-6
                lm = loss_fn()
                P[idx] = orig
                num = (lp - lm) / 2e-6
                grad_analitik = (model.layers[k // 2].dW if k % 2 == 0
                                 else model.layers[k // 2].db)
                # ambil elemen sesuai idx
                ga = grad_analitik[idx] if idx else grad_analitik
                self.assertAlmostEqual(float(ga), num, places=4,
                                       msg=f"param#{k} idx={idx}")

    def test_sequential_kosong(self):
        with self.assertRaises(ValueError):
            Sequential([])


if __name__ == "__main__":
    unittest.main()
