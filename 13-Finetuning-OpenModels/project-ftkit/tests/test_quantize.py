"""Spesifikasi Bagian 3 — Simulasi kuantisasi (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ftkit/quantize.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftkit.quantize import (bandingkan_varian, kuantisasi_simulasi,
                            laju_kompresi, ukuran_model_gb)


class TestKuantisasiSimulasi(unittest.TestCase):
    def test_angka_terkunci_q4(self):
        hasil = kuantisasi_simulasi([0.9, -0.4, 0.15], 4)
        self.assertAlmostEqual(hasil[0], 0.9, places=3)      # max → tepat
        self.assertAlmostEqual(hasil[1], -0.3857, places=3)
        self.assertAlmostEqual(hasil[2], 0.1286, places=3)

    def test_mendekati_asli_bila_bit_tinggi(self):
        bobot = [0.9, -0.4, 0.15, 0.33]
        q8 = kuantisasi_simulasi(bobot, 8)
        for a, b in zip(bobot, q8):
            self.assertLess(abs(a - b), 0.01)

    def test_output_bentuk_sama(self):
        hasil = kuantisasi_simulasi([0.9, -0.4, 0.15], 4)
        self.assertEqual(len(hasil), 3)

    def test_bit_32_identity(self):
        bobot = [0.9, -0.4, 0.15]
        self.assertEqual(kuantisasi_simulasi(bobot, 32), bobot)

    def test_skala_nol_tidak_crash(self):
        self.assertEqual(kuantisasi_simulasi([0.0, 0.0], 4), [0.0, 0.0])


class TestKompresi(unittest.TestCase):
    def test_angka_terkunci(self):
        self.assertAlmostEqual(laju_kompresi(4), 8.0, places=9)
        self.assertAlmostEqual(laju_kompresi(8), 4.0, places=9)
        self.assertAlmostEqual(laju_kompresi(16), 2.0, places=9)


class TestUkuran(unittest.TestCase):
    def test_angka_terkunci_7b(self):
        self.assertAlmostEqual(ukuran_model_gb(7.0, 16), 14.0, places=9)
        self.assertAlmostEqual(ukuran_model_gb(7.0, 8), 7.0, places=9)
        self.assertAlmostEqual(ukuran_model_gb(7.0, 4), 3.5, places=9)


class TestBandingkanVarian(unittest.TestCase):
    def test_tabel_varian_terkunci(self):
        tabel = bandingkan_varian()
        by_name = {t["nama"]: t for t in tabel}
        self.assertEqual(by_name["fp16"]["ukuran_gb"], 14.0)
        self.assertEqual(by_name["int8"]["ukuran_gb"], 7.0)
        self.assertEqual(by_name["q4"]["ukuran_gb"], 3.5)
        self.assertEqual(by_name["fp16"]["skor"], 0.82)
        self.assertEqual(by_name["q4"]["skor"], 0.78)


if __name__ == "__main__":
    unittest.main()
