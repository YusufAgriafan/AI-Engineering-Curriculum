"""Spesifikasi Bagian 5 — Biaya API vs lokal + evaluasi & OOD (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ftkit/api.py dan
ftkit/evaluasi.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftkit.api import biaya_api, biaya_bulanan, keputusan_deployment, token_estimasi
from ftkit.evaluasi import (evaluasi_sebelum_sesudah, kebijakan_ood, laporan_eval,
                            prediksi)
from ftkit.model import inisialisasi
from ftkit.trainer import dataset_ftkit, latih


class TestBiaya(unittest.TestCase):
    def test_angka_terkunci_mini(self):
        self.assertAlmostEqual(biaya_api(1000, 300, "api_mini"), 0.00033, places=8)

    def test_angka_terkunci_besar(self):
        self.assertAlmostEqual(biaya_api(1000, 300, "api_besar"), 0.0055, places=8)

    def test_lokal_nol(self):
        self.assertEqual(biaya_api(1000, 300, "lokal"), 0.0)

    def test_bulanan_terkunci(self):
        self.assertAlmostEqual(biaya_bulanan(2000, 1000, 300, "api_mini"), 19.8, places=6)

    def test_token_estimasi(self):
        self.assertEqual(token_estimasi(""), 0)
        self.assertEqual(token_estimasi("abcd"), 1)
        self.assertEqual(token_estimasi("abcde"), 2)


class TestKeputusanDeployment(unittest.TestCase):
    def test_volume_kecil_pakai_api(self):
        d = keputusan_deployment(200, 1000, 300)
        self.assertEqual(d["keputusan"], "api")
        self.assertAlmostEqual(d["biaya_api_bulanan"], 1.98, places=6)

    def test_volume_besar_terkunci(self):
        d = keputusan_deployment(10000, 1000, 300)
        self.assertEqual(d["keputusan"], "lokal")
        self.assertAlmostEqual(d["biaya_api_bulanan"], 99.0, places=6)


class TestPrediksi(unittest.TestCase):
    def test_yakin_jarak_ke_ambang(self):
        data = dataset_ftkit()
        h = latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
        p = prediksi("Barang bagus, pengiriman cepat", h["w"], h["b"])
        self.assertEqual(p["label"], "positif")
        self.assertGreater(p["yakin"], 0.8)


class TestEvalSebelumSesudah(unittest.TestCase):
    def test_angka_terkunci(self):
        data = dataset_ftkit()
        h = latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
        w0, b0 = inisialisasi()
        ev = evaluasi_sebelum_sesudah(data["val"], w0, b0, h["w"], h["b"])
        self.assertEqual(ev["sebelum"], 0.75)
        self.assertEqual(ev["sesudah"], 1.0)
        self.assertAlmostEqual(ev["delta"], 0.25, places=9)


class TestKebijakanOod(unittest.TestCase):
    def test_ood_terkunci_rendah_yakin(self):
        """'Lumayan standar saja sih' → skor ~0.4988, yakin ~0.0024 → JANGAN tampilkan."""
        data = dataset_ftkit()
        h = latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
        p = prediksi("Lumayan standar saja sih", h["w"], h["b"])
        self.assertAlmostEqual(p["skor"], 0.4988, places=3)
        self.assertLess(p["yakin"], 0.1)
        keputusan = kebijakan_ood(p)
        self.assertFalse(keputusan["tampilkan"])

    def test_yakin_tampil(self):
        data = dataset_ftkit()
        h = latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
        p = prediksi("Barang bagus, pengiriman cepat", h["w"], h["b"])
        keputusan = kebijakan_ood(p)
        self.assertTrue(keputusan["tampilkan"])


class TestLaporanEval(unittest.TestCase):
    def test_laporan_model_terlatih(self):
        data = dataset_ftkit()
        h = latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
        p_ood = prediksi("Lumayan standar saja sih", h["w"], h["b"])
        lap = laporan_eval(data["val"], h["w"], h["b"], ood_hasil=p_ood)
        self.assertEqual(lap["akurasi"], 1.0)
        self.assertEqual(lap["salah"], [])
        self.assertFalse(lap["ood"]["tampilkan"])


if __name__ == "__main__":
    unittest.main()
