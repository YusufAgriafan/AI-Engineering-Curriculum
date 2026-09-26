"""Spesifikasi Bagian 6 — Mini language model (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/lm.py lolos.
Loss & training butuh beberapa detik (finite-difference) — itu wajar.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.lm import cross_entropy, init_params, loss_fn, one_hot, prediksi_berikut, train_lm

SEQ = [0, 1, 0, 1, 0, 1, 0, 1]


class TestOneHot(unittest.TestCase):
    def test_bentuk_dan_isi(self):
        oh = one_hot([1, 2], 4)
        self.assertEqual(oh.shape, (2, 4))
        self.assertEqual(oh.sum(axis=1).tolist(), [1.0, 1.0])
        self.assertEqual(oh[0, 1], 1.0)
        self.assertEqual(oh[1, 2], 1.0)


class TestCrossEntropy(unittest.TestCase):
    def test_uniform(self):
        ce = cross_entropy(np.zeros((2, 4)), np.array([1, 2]))
        self.assertAlmostEqual(ce, np.log(4), places=9)

    def test_prediksi_hampir_sempurna(self):
        logits = np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
        ce = cross_entropy(logits, np.array([0, 1]))
        self.assertLess(ce, 0.001)   # ~1e-4: e^-10 per posisi

    def test_stabil_logit_ekstrem(self):
        ce = cross_entropy(np.array([[1000.0, -1000.0]]), np.array([0]))
        self.assertTrue(np.isfinite(ce))
        self.assertLess(ce, 1e-6)


class TestLossFn(unittest.TestCase):
    def test_loss_awal_dekat_ln_vocab(self):
        params = init_params(9, vocab=4, d=16)
        ce = loss_fn(params, SEQ)
        self.assertTrue(1.0 < ce < 1.9, f"loss awal {ce:.3f} harus dekat ln(4)=1.386")

    def test_shift_target(self):
        # loss = CE(rata-rata) antara logits[:-1] dan ids[1:] — bandingkan manual
        params = init_params(9, vocab=4, d=16)
        from tok_mini.lm import forward
        logits = forward(params, SEQ)
        manual = cross_entropy(logits[:-1], np.asarray(SEQ[1:]))
        self.assertAlmostEqual(loss_fn(params, SEQ), manual, places=12)


class TestTrainLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.losses = train_lm(init_params(9, vocab=4, d=16), SEQ,
                                          epochs=120, lr=0.05)

    def test_jumlah_epoch(self):
        self.assertEqual(len(self.losses), 120)

    def test_loss_turun(self):
        self.assertLess(self.losses[-1], 0.5, f"loss akhir {self.losses[-1]:.3f}")
        self.assertLess(self.losses[-1], self.losses[0] * 0.4)

    def test_prediksi_pola_bergantian(self):
        self.assertEqual(prediksi_berikut(self.params, [0]), 1)
        self.assertEqual(prediksi_berikut(self.params, [0, 1]), 0)
        self.assertEqual(prediksi_berikut(self.params, [0, 1, 0]), 1)


if __name__ == "__main__":
    unittest.main()
