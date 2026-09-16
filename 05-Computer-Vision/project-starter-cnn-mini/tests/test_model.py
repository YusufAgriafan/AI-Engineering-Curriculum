"""Spesifikasi CNNMini (rakit blok) dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/model.py lolos.
Termasuk GRADIENT CHECK end-to-end dan test dataset bekukan.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.conv import Conv2D
from nn_mini.data import muat_semua
from nn_mini.model import CNNMini, MaxPool2x2, ReLU
from nn_mini.train import train_model

X_train, y_train, X_val, y_val, X_test, y_test = muat_semua()


class TestBlokTanpaParameter(unittest.TestCase):
    def test_relu_block(self):
        B = ReLU()
        X = np.array([[[-1., 2.], [0., -3.]]])
        self.assertTrue(np.allclose(B.forward(X), [[0., 2.], [0., 0.]]))
        g = np.ones_like(X)
        self.assertTrue(np.allclose(B.backward(g), [[0., 1.], [0., 0.]]),
                        "gradien lewat hanya di posisi z > 0")

    def test_pool_block_backward_butuh_forward(self):
        B = MaxPool2x2()
        X = np.random.default_rng(0).random((2, 6, 6))
        out = B.forward(X)
        dX = B.backward(np.ones_like(out))
        self.assertEqual(dX.shape, X.shape)


class TestCNNMiniForward(unittest.TestCase):
    def test_bentuk_output(self):
        m = CNNMini(seed=0)
        p = m.forward(X_train[:5])
        self.assertEqual(p.shape, (5, 1), "forward: (n,1); predict_proba: (n,)")

    def test_probabilitas_valid(self):
        m = CNNMini(seed=1)
        p = m.predict_proba(X_val[:20])
        self.assertTrue(((p >= 0.0) & (p <= 1.0)).all())

    def test_input_bukan_3d(self):
        m = CNNMini(seed=2)
        with self.assertRaises(ValueError):
            m.forward(X_train[:5].reshape(5, 64))


class TestCNNMiniBackward(unittest.TestCase):
    def test_gradient_check_end_to_end(self):
        """L = sum(p * R): dL/dK dan dL/dW harus cocok selisih-hingga numerik.

        Catatan penting (poin belajar!): max-pool punya SUBGRADIEN di jendela
        yang seri — termasuk jendela "mati" (semua z ≤ 0 setelah ReLU = 0.0).
        Supaya cek numerik adil, kita paksa z > 0 di mana-mana (X > 0 dan K > 0)
        sehingga ReLU identitas dan tiap jendela pool punya argmax unik.
        Di training nyata, ketidak-pastian kecil ini normal dan tidak masalah.
        """
        rng = np.random.default_rng(3)
        m = CNNMini(seed=9)
        X = np.abs(X_train[:8]) + 0.1
        R = rng.normal(size=8)
        m.forward(X)
        m.conv.K = np.abs(m.conv.K) + 0.5          # z > 0 di mana-mana
        p = m.forward(X)
        g = R.reshape(-1, 1)
        m.backward(g)
        eps = 1e-6

        dK_ref = m.conv.dK.copy()
        dW_ref = m.dense.dW.copy()
        db_ref = m.dense.db.copy()

        for (u, v) in [(0, 0), (1, 1), (2, 2)]:
            Kp, Km = m.conv.K.copy(), m.conv.K.copy()
            Kp[u, v] += eps
            Km[u, v] -= eps
            K_ory = m.conv.K.copy()
            m.conv.K = Kp
            lp = (m.forward(X).ravel() * R).sum()
            m.conv.K = Km
            lm = (m.forward(X).ravel() * R).sum()
            m.conv.K = K_ory
            self.assertAlmostEqual(dK_ref[u, v], (lp - lm) / (2 * eps), places=4)

        for j in range(dW_ref.shape[0]):
            Wp, Wm = m.dense.W.copy(), m.dense.W.copy()
            Wp[j, 0] += eps
            Wm[j, 0] -= eps
            W_ory = m.dense.W.copy()
            m.dense.W = Wp
            lp = (m.forward(X).ravel() * R).sum()
            m.dense.W = Wm
            lm = (m.forward(X).ravel() * R).sum()
            m.dense.W = W_ory
            self.assertAlmostEqual(dW_ref[j, 0], (lp - lm) / (2 * eps), places=4)

        # bias
        bp, bm = m.dense.b.copy(), m.dense.b.copy()
        bp[0] += eps
        bm[0] -= eps
        b_ory = m.dense.b.copy()
        m.dense.b = bp
        lp = (m.forward(X).ravel() * R).sum()
        m.dense.b = bm
        lm = (m.forward(X).ravel() * R).sum()
        m.dense.b = b_ory
        self.assertAlmostEqual(db_ref[0], (lp - lm) / (2 * eps), places=4)

    def test_grads_urut_sama_dengan_parameters(self):
        m = CNNMini(seed=4)
        m.forward(X_train[:4])
        m.backward(np.ones((4, 1)))
        self.assertEqual(len(m.parameters()), len(m.grads()))
        self.assertEqual(m.conv.K.shape, m.grads()[0].shape)
        self.assertEqual(m.dense.W.shape, m.grads()[1].shape)


class TestTraining(unittest.TestCase):
    def test_dataset_bekukan(self):
        self.assertEqual(len(y_train), 420)
        self.assertEqual(len(y_val), 90)
        self.assertEqual(len(y_test), 90)
        self.assertEqual(int(y_train.sum()), 189, "seed 42: 189 gambar vertikal di train")
        self.assertEqual(int(y_val.sum()), 46, "seed 42: 46 gambar vertikal di val")
        self.assertEqual(int(y_test.sum()), 49, "seed 42: 49 gambar vertikal di test")

    def test_latihan_cepat_konvergen(self):
        m = CNNMini(seed=0)
        h = train_model(m, X_train, y_train, X_val, y_val,
                        epochs=30, batch_size=32, lr=0.5, seed=0)
        self.assertGreater(h["val_acc"][-1], 0.80,
                           "CNN 30 epoch harus sudah tembus 0.80 di val")
        self.assertLess(h["val_loss"][-1], 0.5)

    def test_early_stopping(self):
        m = CNNMini(seed=1)
        h = train_model(m, X_train, y_train, X_val, y_val,
                        epochs=500, batch_size=32, lr=0.5, seed=0, patience=15)
        self.assertLess(len(h["val_loss"]), 500,
                        "patience=15 harus menghentikan sebelum 500 epoch")


if __name__ == "__main__":
    unittest.main()
