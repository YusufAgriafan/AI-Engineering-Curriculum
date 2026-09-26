"""Spesifikasi Bagian 4 — Loop SFT + early stopping (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ftkit/trainer.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftkit.model import inisialisasi, loss_batch
from ftkit.trainer import dataset_ftkit, epoch_satu, latih


class TestDataset(unittest.TestCase):
    def test_split_terkunci(self):
        data = dataset_ftkit()
        self.assertEqual((len(data["train"]), len(data["val"])), (16, 8))

    def test_dimensi_fitur(self):
        data = dataset_ftkit()
        x, y = data["train"][0]
        self.assertEqual(len(x), 17)
        self.assertIn(y, (0.0, 1.0))


class TestEpochSatu(unittest.TestCase):
    def test_loss_turun_satu_epoch(self):
        data = dataset_ftkit()
        w0, b0 = inisialisasi()
        loss_awal = loss_batch(data["train"], w0, b0)
        w1, b1, tr = epoch_satu(data["train"], w0, b0, lr=0.5)
        self.assertLess(loss_batch(data["train"], w1, b1), loss_awal)

    def test_angka_terkunci_epoch_pertama(self):
        data = dataset_ftkit()
        w0, b0 = inisialisasi()
        w1, b1, tr = epoch_satu(data["train"], w0, b0, lr=0.5)
        self.assertAlmostEqual(tr, 0.739636, places=4)   # loss SEBELUM update per contoh
        self.assertAlmostEqual(b1, -0.114574, places=4)


class TestLatih(unittest.TestCase):
    def test_angka_terkunci_normal(self):
        """Data bersih: konvergen, val acc 1.0, tidak berhenti dini dalam 60 epoch."""
        data = dataset_ftkit()
        h = latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
        self.assertFalse(h["berhenti_dini"])
        self.assertEqual(h["epochs_dijalankan"], 60)
        self.assertEqual(h["riwayat"]["val_acc"][-1], 1.0)
        self.assertAlmostEqual(h["riwayat"]["train_loss"][0], 0.739636, places=4)

    def test_angka_terkunci_early_stopping(self):
        """2 label train sengaja dibalik (data kontradiktif) + lr tinggi:
        val loss terbaik di epoch 61, berhenti dini di 66 (sabar=5),
        bobot TERBAIK yang dipulihkan — bukan bobot terakhir."""
        data = dataset_ftkit()
        train = [(x, (1.0 - y) if i in (6, 14) else y)
                 for i, (x, y) in enumerate(data["train"])]
        h = latih(train, data["val"], lr=3.0, epochs=100, sabar=5)
        self.assertTrue(h["berhenti_dini"])
        self.assertEqual(h["epochs_dijalankan"], 66)
        self.assertEqual(h["best_epoch"], 61)
        r = h["riwayat"]
        self.assertAlmostEqual(min(r["val_loss"]), r["val_loss"][60], places=4)
        self.assertAlmostEqual(r["val_loss"][-1], 0.4188, places=3)
        # restore best: val acc pada bobot akhir = val acc di best_epoch
        self.assertAlmostEqual(r["val_acc"][60], 0.75, places=9)


if __name__ == "__main__":
    unittest.main()
