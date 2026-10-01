"""Spesifikasi Bagian 5 — Monitor: biaya & latensi (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di evalkit/monitor.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.data import LATENSI_MS
from evalkit.monitor import biaya_run, estimasi_biaya, latensi_persentil
from evalkit.runner import jalankan_eval
from evalkit.tokenizer import token_estimasi


class TestEstimasiBiaya(unittest.TestCase):
    def test_angka_terkunci_api_mini(self):
        # 'abc' = 1 token, 'abcd' = 1 token
        # 1/1e6*0.15 + 1/1e6*0.60 = 7.5e-07
        self.assertAlmostEqual(estimasi_biaya("abc", "abcd"), 7.5e-07, places=14)

    def test_angka_terkunci_api_besar(self):
        biaya = estimasi_biaya("abc", "abcd", "api_besar")
        self.assertAlmostEqual(biaya, 1 / 1e6 * 2.50 + 1 / 1e6 * 10.0, places=14)

    def test_lokal_nol(self):
        self.assertEqual(estimasi_biaya("abc", "abcd", "lokal"), 0.0)

    def test_model_tak_dikenal_raise(self):
        with self.assertRaises(KeyError):
            estimasi_biaya("abc", "abcd", "model_hantu")

    def test_token_dari_token_estimasi(self):
        tanya, jawab = "a" * 8, "b" * 7   # 2 + 2 token
        biaya = estimasi_biaya(tanya, jawab, "api_mini")
        self.assertAlmostEqual(biaya, (2 * 0.15 + 2 * 0.60) / 1e6, places=14)
        self.assertEqual(token_estimasi(tanya), 2)


class TestBiayaRun(unittest.TestCase):
    def test_angka_terkunci_run_v3_api_mini(self):
        run = jalankan_eval("v3")
        self.assertAlmostEqual(biaya_run(run["hasil"]), 0.00013305, places=10)

    def test_run_kosong_nol(self):
        self.assertEqual(biaya_run([]), 0.0)

    def test_api_besar_lebih_mahal(self):
        run = jalankan_eval("v3")
        self.assertGreater(biaya_run(run["hasil"], "api_besar"),
                           biaya_run(run["hasil"], "api_mini"))


class TestLatensiPersentil(unittest.TestCase):
    def test_angka_terkunci_api_mini(self):
        self.assertEqual(latensi_persentil(list(LATENSI_MS["api_mini"])),
                         {"p50": 480.0, "p90": 900.0, "p99": 900.0})

    def test_angka_terkunci_lokal(self):
        self.assertEqual(latensi_persentil(list(LATENSI_MS["lokal"])),
                         {"p50": 210.0, "p90": 400.0, "p99": 400.0})

    def test_nearest_rank_acak(self):
        # n=4, p50 → rank ceil(0.5*4)=2 → elemen ke-2 urut naik
        self.assertEqual(latensi_persentil([30.0, 10.0, 40.0, 20.0])["p50"], 20.0)
        # p90 → rank ceil(0.9*4)=4 → elemen ke-4 = max
        self.assertEqual(latensi_persentil([30.0, 10.0, 40.0, 20.0])["p90"], 40.0)

    def test_tunggal_dan_kosong(self):
        self.assertEqual(latensi_persentil([5.0]), {"p50": 5.0, "p90": 5.0, "p99": 5.0})
        self.assertEqual(latensi_persentil([]), {"p50": 0.0, "p90": 0.0, "p99": 0.0})

    def test_custom_persen(self):
        self.assertEqual(latensi_persentil([1.0, 2.0, 3.0, 4.0], persen=(25,))["p25"], 1.0)


if __name__ == "__main__":
    unittest.main()
