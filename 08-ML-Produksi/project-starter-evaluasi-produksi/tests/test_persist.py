"""Spesifikasi persistensi model dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di prod_mini/persist.py lolos.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prod_mini.data import make_dataset, split_tiga_arah
from prod_mini.models import fit_ols, predict
from prod_mini.persist import load_model, roundtrip_ok, save_model, sha256_file


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_roundtrip_nilai(self):
        w = np.array([1.5, -2.0])
        p = os.path.join(self.tmpdir, "m.npz")
        save_model(p, w, 0.25)
        w2, b2 = load_model(p)
        np.testing.assert_array_equal(w2, w)
        self.assertEqual(b2, 0.25)

    def test_deterministik(self):
        # dua kali save dari objek yang sama -> byte identik
        w = np.array([1.5, -2.0])
        p1 = os.path.join(self.tmpdir, "a.npz")
        p2 = os.path.join(self.tmpdir, "b.npz")
        save_model(p1, w, 0.25)
        save_model(p2, w, 0.25)
        self.assertEqual(sha256_file(p1), sha256_file(p2))

    def test_hash_berbeda_untuk_model_berbeda(self):
        p1 = os.path.join(self.tmpdir, "a.npz")
        p2 = os.path.join(self.tmpdir, "b.npz")
        save_model(p1, np.array([1.0]), 0.0)
        save_model(p2, np.array([1.1]), 0.0)
        self.assertNotEqual(sha256_file(p1), sha256_file(p2))

    def test_hash_format(self):
        p = os.path.join(self.tmpdir, "m.npz")
        save_model(p, np.array([1.0]), 0.0)
        h = sha256_file(p)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)


class TestRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_dataset()
        cls.Xtr, cls.ytr, _, _, cls.Xte, cls.yte = split_tiga_arah(X, y)
        cls.w, cls.b = fit_ols(cls.Xtr, cls.ytr)

    def test_roundtrip_ok_true(self):
        pred_sebelum = predict(self.w, self.b, self.Xte)
        self.assertTrue(roundtrip_ok(self.w, self.b, self.Xte, pred_sebelum))

    def test_roundtrip_mendeteksi_perubahan(self):
        # prediksi "sebelum" yang dipalsukan HARUS terdeteksi -> False
        pred_palsu = predict(self.w, self.b, self.Xte) + 1.0
        self.assertFalse(roundtrip_ok(self.w, self.b, self.Xte, pred_palsu))

    def test_prediksi_identik_bit_per_bit(self):
        # jalur manual: save -> load -> prediksi == prediksi asli (array_equal)
        p = os.path.join(tempfile.mkdtemp(), "m.npz")
        save_model(p, self.w, self.b)
        w2, b2 = load_model(p)
        self.assertTrue(np.array_equal(predict(w2, b2, self.Xte),
                                       predict(self.w, self.b, self.Xte)))


if __name__ == "__main__":
    unittest.main()
