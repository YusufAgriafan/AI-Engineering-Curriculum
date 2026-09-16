"""Spesifikasi optimizer, batching & training loop (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di nn_mini/train.py lolos.
Angka acuan dataset dihitung dari generator seed 42 yang dibekukan di
nn_mini/data.npz — tidak boleh berubah.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn_mini.data import muat_semua  # noqa: E402
from nn_mini.layers import Dense, Sequential  # noqa: E402
from nn_mini.train import SGD, iter_batch, train_model  # noqa: E402

X_train, y_train, X_val, y_val, X_test, y_test = muat_semua()


class TestSGD(unittest.TestCase):
    def test_step_inplace(self):
        opt = SGD(lr=0.5)
        W = np.array([[1.0, 2.0]])
        b = np.array([3.0])
        gW = np.array([[0.2, -0.4]])
        gb = np.array([1.0])
        opt.step([W, b], [gW, gb])
        self.assertTrue(np.allclose(W, [[0.9, 2.2]]))
        self.assertTrue(np.allclose(b, [2.5]))

    def test_panjang_tidak_cocok(self):
        opt = SGD(lr=0.1)
        with self.assertRaises(ValueError):
            opt.step([np.zeros(2)], [np.zeros(2), np.zeros(2)])


class TestIterBatch(unittest.TestCase):
    def test_mengcover_semua_data_tepat_sekali(self):
        X = np.arange(20).reshape(10, 2).astype(float)
        y = np.arange(10)
        terlihat = []
        for Xb, yb in iter_batch(X, y, batch_size=3, seed=0):
            self.assertEqual(len(Xb), len(yb))
            terlihat.extend(yb.tolist())
        self.assertEqual(sorted(terlihat), list(range(10)), "semua sampel, tanpa duplikat")

    def test_batch_terakhir_boleh_kecil(self):
        X = np.zeros((7, 1))
        y = np.zeros(7)
        sizes = [len(yb) for _, yb in iter_batch(X, y, batch_size=3, seed=1)]
        self.assertEqual(sizes, [3, 3, 1])

    def test_shuffle_beda_seed_beda_urutan(self):
        X = np.arange(100).reshape(50, 2).astype(float)
        y = np.arange(50)
        urutan1 = [int(yb[0]) for _, yb in iter_batch(X, y, 10, seed=1)]
        urutan2 = [int(yb[0]) for _, yb in iter_batch(X, y, 10, seed=2)]
        self.assertNotEqual(urutan1, urutan2, "seed beda -> urutan epoch beda")


class TestTrainModel(unittest.TestCase):
    def _model(self):
        return Sequential([Dense(16, activation="relu", seed=7),
                           Dense(8, activation="relu", seed=8),
                           Dense(1, activation="sigmoid", seed=9)])

    def test_dataset_bekukan(self):
        """Angka acuan split (generator seed 42)."""
        self.assertEqual((len(y_train), len(y_val), len(y_test)), (420, 90, 90))
        self.assertEqual((int(y_train.sum()), int(y_val.sum()), int(y_test.sum())),
                         (206, 49, 45))

    def test_loss_menurun_dan_akurasi_target(self):
        model = self._model()
        hist = train_model(model, X_train, y_train, X_val, y_val,
                           epochs=300, batch_size=32, lr=0.1, seed=0)
        self.assertLess(hist["train_loss"][-1], hist["train_loss"][0],
                        "train loss harus turun")
        self.assertLess(hist["val_loss"][-1], 0.15, "val loss harus jauh di bawah ln 2")
        self.assertGreaterEqual(hist["val_acc"][-1], 0.94,
                                "MLP harus mengalahkan baseline linear 0.889 dengan jelas")

    def test_early_stopping_berhenti_lebih_awal(self):
        model = self._model()
        hist = train_model(model, X_train, y_train, X_val, y_val,
                           epochs=1000, batch_size=32, lr=0.1, patience=25, seed=0)
        self.assertLess(len(hist["val_loss"]), 1000,
                        "patience harus menghentikan training sebelum 1000 epoch")
        self.assertGreaterEqual(hist["val_acc"][-1], 0.90)

    def test_tanpa_early_stopping_jalankan_penuh(self):
        model = self._model()
        hist = train_model(model, X_train, y_train, X_val, y_val,
                           epochs=60, batch_size=32, lr=0.1, patience=None, seed=0)
        self.assertEqual(len(hist["val_loss"]), 60)

    def test_lr_terlalu_besar_mengganggu(self):
        """Eksperimen Bab 4: lr = 50 seharusnya jauh lebih buruk dari lr = 0.1."""
        model_ok = self._model()
        hist_ok = train_model(model_ok, X_train, y_train, X_val, y_val,
                              epochs=120, batch_size=32, lr=0.1, seed=0)
        model_bad = self._model()
        hist_bad = train_model(model_bad, X_train, y_train, X_val, y_val,
                               epochs=120, batch_size=32, lr=50.0, seed=0)
        self.assertLess(hist_ok["val_loss"][-1], hist_bad["val_loss"][-1])

    def test_mlp_mengalahkan_baseline_di_test(self):
        """Test set disentuh sekali: MLP >= 0.94; baseline linear ±0.87 (dibuktikan di notebook)."""
        model = self._model()
        train_model(model, X_train, y_train, X_val, y_val,
                    epochs=300, batch_size=32, lr=0.1, seed=0)
        p_test = model.predict_proba(X_test)
        acc = float(((p_test >= 0.5) == y_test).mean())
        self.assertGreaterEqual(acc, 0.94, f"test accuracy: {acc:.4f}")


if __name__ == "__main__":
    unittest.main()
