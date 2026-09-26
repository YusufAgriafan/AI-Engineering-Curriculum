"""Spesifikasi Bagian 1 — Token, biaya, penjaga anggaran (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/cost.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.cost import (biaya_hasil, hitung_biaya, jalankan_anggaran,
                         token_estimasi)
from llmkit.transport import provider_utama

PESAN = [{"role": "user", "content": "Jelaskan apa itu machine learning dalam 2 kalimat."}]


class TestTokenEstimasi(unittest.TestCase):
    def test_kosong(self):
        self.assertEqual(token_estimasi(""), 0)

    def test_bulat_ke_atas(self):
        self.assertEqual([token_estimasi("x" * n) for n in (1, 4, 5, 9)], [1, 1, 2, 3])

    def test_bukan_string(self):
        self.assertEqual(token_estimasi(1234), 1)


class TestHitungBiaya(unittest.TestCase):
    def test_mini(self):
        self.assertAlmostEqual(hitung_biaya(1000, 300, "mini"), 0.00033, places=9)

    def test_besar_jauh_lebih_mahal(self):
        self.assertAlmostEqual(hitung_biaya(1000, 300, "besar"), 0.0095, places=9)

    def test_nol_token(self):
        self.assertEqual(hitung_biaya(0, 0, "mini"), 0.0)

    def test_output_lebih_mahal_dari_input(self):
        self.assertGreater(hitung_biaya(0, 1000, "mini"), hitung_biaya(1000, 0, "mini"))

    def test_model_tak_dikenal(self):
        with self.assertRaises(ValueError):
            hitung_biaya(10, 10, "model_ajaib")

    def test_biaya_hasil_sesuai_hasil_provider(self):
        h = provider_utama().panggil("mini", PESAN)
        self.assertAlmostEqual(biaya_hasil(h),
                               hitung_biaya(h["tokens_in"], h["tokens_out"], h["model"]), places=12)


class TestAnggaran(unittest.TestCase):
    def test_berhenti_saat_anggaran_habis(self):
        h = jalankan_anggaran(provider_utama(), [PESAN] * 4, 0.00003, model="mini")
        self.assertEqual(h["diproses"], 3)
        self.assertEqual(h["dilewati"], 1)
        self.assertEqual(h["berhenti_di"], 3)
        self.assertAlmostEqual(h["total_biaya_usd"], 4.005e-05, places=12)

    def test_anggaran_longgar_semua_diproses(self):
        h = jalankan_anggaran(provider_utama(), [PESAN] * 3, 1.0, model="mini")
        self.assertEqual((h["diproses"], h["dilewati"], h["berhenti_di"]), (3, 0, None))

    def test_anggaran_nol_tidak_memanggil_sama_sekali(self):
        p = provider_utama()
        h = jalankan_anggaran(p, [PESAN] * 2, 0.0, model="mini")
        self.assertEqual((h["diproses"], h["dilewati"], h["berhenti_di"]), (0, 2, 0))
        self.assertEqual(p.log, [], "tidak boleh ada panggilan API saat anggaran nol")

    def test_hasil_mencatat_index_dan_biaya(self):
        h = jalankan_anggaran(provider_utama(), [PESAN] * 2, 1.0, model="mini")
        self.assertEqual([r["index"] for r in h["hasil"]], [0, 1])
        self.assertAlmostEqual(h["hasil"][0]["biaya_usd"], 1.335e-05, places=12)

    def test_total_sama_dengan_jumlah_biaya(self):
        h = jalankan_anggaran(provider_utama(), [PESAN] * 3, 1.0, model="mini")
        self.assertAlmostEqual(h["total_biaya_usd"],
                               sum(r["biaya_usd"] for r in h["hasil"]), places=12)


if __name__ == "__main__":
    unittest.main()
