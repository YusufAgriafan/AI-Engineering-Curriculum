"""Spesifikasi metrik klasifikasi dalam bentuk test (mode TDD).

Jalankan dari folder project ini:
    python -m unittest discover -s tests -v

JANGAN DIUBAH — tugasmu membuat implementasi di ml_sederhana/metrik.py lolos.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_sederhana.metrik import (  # noqa: E402
    accuracy,
    confusion_matrix,
    f1_score,
    precision_recall_f1,
    roc_auc,
)


class TestConfusionMatrix(unittest.TestCase):
    def test_kasus_dasar(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm["TP"], 4)
        self.assertEqual(cm["TN"], 4)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["FN"], 1)

    def test_tanpa_prediksi_positif(self):
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 0, 0])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm["TP"], 0)
        self.assertEqual(cm["FN"], 2)

    def test_label_tidak_valid(self):
        with self.assertRaises(ValueError):
            confusion_matrix([0, 2], [0, 1])


class TestAccuracy(unittest.TestCase):
    def test_kasus_dasar(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.8)

    def test_sempurna_dan_terburuk(self):
        y = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(accuracy(y, y), 1.0)
        self.assertAlmostEqual(accuracy(y, 1 - y), 0.0)


class TestPrecisionRecallF1(unittest.TestCase):
    def test_kasus_dasar(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        self.assertAlmostEqual(p, 0.8)
        self.assertAlmostEqual(r, 0.8)
        self.assertAlmostEqual(f1, 0.8)

    def test_edge_case_nol(self):
        # tanpa prediksi positif: precision & recall = 0 (konvensi sklearn)
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 0, 0])
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))

    def test_f1_rendah_karena_recall(self):
        # precision 1.0 tapi recall 0.25 -> F1 = 0.4
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 0, 0])
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 0.25)
        self.assertAlmostEqual(f1, 0.4)

    def test_f1_score_shortcut(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
        self.assertAlmostEqual(f1_score(y_true, y_pred), 0.8)


class TestRocAuc(unittest.TestCase):
    def test_sempurna(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(y, s), 1.0)

    def test_terbalik_dan_acak(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.8, 0.9, 0.1, 0.2])
        self.assertAlmostEqual(roc_auc(y, s), 0.0)
        self.assertAlmostEqual(roc_auc(y, np.array([0.3, 0.4, 0.3, 0.4])), 0.5)

    def test_nilai_tengah(self):
        # pos (0.25) mengalahkan neg 0.1 & 0.2, kalah dari neg 0.3 -> 2/3
        y = np.array([0, 0, 0, 1])
        s = np.array([0.1, 0.2, 0.3, 0.25])
        self.assertAlmostEqual(roc_auc(y, s), 2 / 3)

    def test_tie_dirata_rata(self):
        # skor sama -> pasangan tie bernilai 0.5, bukan 1 atau 0
        y = np.array([0, 1, 0])
        s = np.array([0.5, 0.5, 0.9])
        # pos(0.5) vs neg1(0.5) = tie (0.5), pos(0.5) vs neg2(0.9) = kalah (0)
        # AUC = (0.5 + 0) / 2 = 0.25
        self.assertAlmostEqual(roc_auc(y, s), 0.25)

    def test_satu_kelas_saja(self):
        with self.assertRaises(ValueError):
            roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
