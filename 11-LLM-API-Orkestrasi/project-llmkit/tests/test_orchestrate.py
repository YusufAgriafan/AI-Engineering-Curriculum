"""Spesifikasi Bagian 7 — Orkestrasi: prompt chaining (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/orchestrate.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.data import HARGA, LANGKAH_PIPELINE, TIKET_CONTOH
from llmkit.orchestrate import jalankan_pipeline
from llmkit.transport import ProviderPalsu, provider_utama


class TestPipelineNormal(unittest.TestCase):
    def setUp(self):
        self.h = jalankan_pipeline(provider_utama(), LANGKAH_PIPELINE, TIKET_CONTOH)

    def test_semua_langkah_berhasil(self):
        self.assertEqual((self.h["berhasil"], self.h["gagal"]), (3, 0))
        self.assertEqual(len(self.h["langkah"]), 3)

    def test_urutan_dan_model_per_langkah(self):
        self.assertEqual([l["nama"] for l in self.h["langkah"]],
                         ["klasifikasi", "ekstraksi", "balasan"])
        self.assertEqual([l["model"] for l in self.h["langkah"]], ["mini", "mini", "sedang"])

    def test_total_biaya_dan_latensi_terkunci(self):
        self.assertAlmostEqual(self.h["total_biaya_usd"], 0.00016075, places=12)
        # mini x2 (184 + 187) + sedang sekali retry (1000 backoff + 249) = 1620 ms
        self.assertEqual(self.h["total_ms"], 1620)
        self.assertEqual([l["attempts"] for l in self.h["langkah"]], [1, 1, 2])

    def test_total_sama_dengan_jumlah_per_langkah(self):
        self.assertAlmostEqual(self.h["total_biaya_usd"],
                               sum(l["biaya_usd"] for l in self.h["langkah"]), places=12)
        self.assertEqual(self.h["total_ms"], sum(l["total_ms"] for l in self.h["langkah"]))

    def test_teks_akhir_dari_langkah_terakhir(self):
        self.assertEqual(self.h["teks_akhir"], self.h["langkah"][-1]["teks"])
        self.assertIn("balasan pelanggan", self.h["teks_akhir"])

    def test_teks_berawalan_nama_model(self):
        self.assertTrue(all(l["teks"].startswith(f'[{l["model"]}]')
                            for l in self.h["langkah"]))


class TestPipelineGagal(unittest.TestCase):
    def test_langkah_model_sedang_gagal_saat_maks_satu(self):
        h = jalankan_pipeline(provider_utama(), LANGKAH_PIPELINE, TIKET_CONTOH,
                              maks_percobaan=1)
        self.assertEqual((h["berhasil"], h["gagal"]), (2, 1))
        gagal = [l for l in h["langkah"] if not l["sukses"]][0]
        self.assertEqual(gagal["nama"], "balasan")
        self.assertEqual(gagal["galat"], "semua_gagal")
        self.assertIsNone(gagal["teks"])
        self.assertEqual(gagal["biaya_usd"], 0.0)

    def test_berhenti_saat_gagal(self):
        rusak = ProviderPalsu("utama", HARGA, {"mini": ["rate_limit"], "sedang": ["ok"]})
        h = jalankan_pipeline(rusak, LANGKAH_PIPELINE, TIKET_CONTOH, maks_percobaan=1,
                              berhenti_saat_gagal=True)
        self.assertEqual(len(h["langkah"]), 1)
        self.assertEqual((h["berhasil"], h["gagal"]), (0, 1))
        self.assertIsNone(h["teks_akhir"])

    def test_lanjut_saat_gagal(self):
        rusak = ProviderPalsu("utama", HARGA, {"mini": ["rate_limit"], "sedang": ["ok"]})
        h = jalankan_pipeline(rusak, LANGKAH_PIPELINE, TIKET_CONTOH, maks_percobaan=1)
        self.assertEqual(len(h["langkah"]), 3)
        self.assertEqual((h["berhasil"], h["gagal"]), (2, 1))
        # langkah 1 gagal (mini panggilan pertama), langkah 2 sukses (mini pulih)
        self.assertEqual([l["sukses"] for l in h["langkah"]], [False, True, True])

    def test_retry_lolos_langkah_kedua_tautan(self):
        # sedang gagal sekali lalu berhasil: dengan maks_percobaan=2 langkah 3 sukses
        h = jalankan_pipeline(provider_utama(), LANGKAH_PIPELINE, TIKET_CONTOH,
                              maks_percobaan=2)
        self.assertEqual(h["berhasil"], 3)
        self.assertEqual(h["langkah"][2]["attempts"], 2)


class TestSetelanPipeline(unittest.TestCase):
    def test_settings_diteruskan(self):
        tunggu = []
        h = jalankan_pipeline(provider_utama(), LANGKAH_PIPELINE, TIKET_CONTOH,
                              maks_percobaan=3, settings={"basis_ms": 10,
                                                          "sleep_fn": tunggu.append})
        self.assertEqual(h["berhasil"], 3)
        self.assertEqual(tunggu, [10], "sedang gagal sekali -> satu backoff 10ms")

    def test_pipeline_kosong(self):
        h = jalankan_pipeline(provider_utama(), [], TIKET_CONTOH)
        self.assertEqual((h["berhasil"], h["gagal"], h["total_ms"]), (0, 0, 0))
        self.assertIsNone(h["teks_akhir"])

    def test_langkah_tunggal(self):
        h = jalankan_pipeline(provider_utama(), [LANGKAH_PIPELINE[0]], TIKET_CONTOH)
        self.assertEqual((h["berhasil"], len(h["langkah"])), (1, 1))


if __name__ == "__main__":
    unittest.main()
