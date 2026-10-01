"""Spesifikasi Bagian 4 — Regression: diff dua run + gate rilis (TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di evalkit/regression.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.regression import bandingkan, gate_rilis
from evalkit.runner import jalankan_eval


class TestBandingkan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_v1 = jalankan_eval("v1")
        cls.run_v2 = jalankan_eval("v2")
        cls.run_v3 = jalankan_eval("v3")

    def test_angka_terkunci_v1_ke_v3(self):
        lap = bandingkan(self.run_v1, self.run_v3)
        self.assertEqual(lap["skor_rata"],
                         {"lama": 0.3333, "baru": 0.8333, "delta": 0.5})
        self.assertEqual(lap["regresi"], [])
        self.assertEqual(lap["perbaikan"],
                         ["abstain", "injection", "hallucination", "edge"])
        self.assertEqual(lap["per_kategori"]["injection"],
                         {"lama": 0.0, "baru": 1.0, "delta": 1.0})
        self.assertEqual(lap["per_kategori"]["kebijakan"],
                         {"lama": 1.0, "baru": 1.0, "delta": 0.0})

    def test_semua_kategori_muncul(self):
        lap = bandingkan(self.run_v1, self.run_v2)
        self.assertEqual(len(lap["per_kategori"]), 6)

    def test_deteksi_regresi_terurut(self):
        # v2 → v1: abstain, hallucination, edge turun 1.0
        lap = bandingkan(self.run_v2, self.run_v1)
        self.assertEqual(lap["regresi"], ["abstain", "hallucination", "edge"])
        self.assertEqual(lap["perbaikan"], [])

    def test_delta_bulat_4(self):
        lap = bandingkan(self.run_v1, self.run_v2)
        self.assertEqual(lap["skor_rata"]["delta"], 0.3334)
        self.assertEqual(lap["per_kategori"]["edge"]["delta"], 1.0)


class TestGateRilis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_v1 = jalankan_eval("v1")
        cls.run_v2 = jalankan_eval("v2")
        cls.run_v3 = jalankan_eval("v3")

    def test_angka_terkunci_v3_lolos(self):
        keputusan = gate_rilis(bandingkan(self.run_v1, self.run_v3))
        self.assertTrue(keputusan["lolos"])
        self.assertIn("0.8333", keputusan["alasan"])

    def test_angka_terkunci_regresi_ditolak(self):
        keputusan = gate_rilis(bandingkan(self.run_v2, self.run_v1))
        self.assertFalse(keputusan["lolos"])
        self.assertIn("regresi", keputusan["alasan"])
        self.assertIn("abstain", keputusan["alasan"])

    def test_skor_di_bawah_ambang_ditolak(self):
        lap = bandingkan(self.run_v1, self.run_v1)  # tanpa regresi, skor 0.3333
        keputusan = gate_rilis(lap, skor_min=0.75)
        self.assertFalse(keputusan["lolos"])
        self.assertIn("di bawah ambang", keputusan["alasan"])

    def test_ambang_skor_min_dipatuhan(self):
        lap = bandingkan(self.run_v1, self.run_v3)  # skor baru 0.8333
        self.assertTrue(gate_rilis(lap, skor_min=0.8)["lolos"])
        self.assertFalse(gate_rilis(lap, skor_min=0.9)["lolos"])


if __name__ == "__main__":
    unittest.main()
