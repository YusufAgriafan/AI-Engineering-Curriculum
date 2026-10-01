"""Spesifikasi Bagian 4 — Canary rollout (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di deploykit/canary.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploykit.canary import evaluasi_canary, rencana_rollout
from deploykit.data import CANARY_KASUS, ENV_PRODUKSI
from deploykit.konfig import config_layanan


class TestEvaluasiCanary(unittest.TestCase):
    def test_angka_terkunci_lima_kasus(self):
        hasil = {}
        for nama, (err, skor, p50, biaya_b, biaya_l) in CANARY_KASUS.items():
            hasil[nama] = evaluasi_canary(nama, err, skor, p50, biaya_b, biaya_l,
                                          p50_lama_ms=480.0, skor_min=0.75)
        self.assertEqual(hasil["sehat"]["putuskan"], "lanjut")
        self.assertEqual(hasil["sehat"]["alasan"], "semua SLO terpenuhi")
        self.assertEqual(hasil["error"]["putuskan"], "tahan")
        self.assertIn("error rate", hasil["error"]["alasan"])
        self.assertEqual(hasil["skor_turun"]["putuskan"], "tahan")
        self.assertIn("skor eval", hasil["skor_turun"]["alasan"])
        self.assertEqual(hasil["lambat"]["putuskan"], "tahan")
        self.assertIn("latensi", hasil["lambat"]["alasan"])
        self.assertEqual(hasil["mahal"]["putuskan"], "tahan")
        self.assertIn("biaya", hasil["mahal"]["alasan"])

    def test_batas_tepat_slo(self):
        # error tepat 0.02 (tidak >) → lanjut; skor tepat 0.75 (tidak <) → lanjut
        h = evaluasi_canary("x", 0.02, 0.75, 480.0, 0.00033, 0.00033)
        self.assertEqual(h["putuskan"], "lanjut")

    def test_latensi_slo_rasio_dari_data(self):
        # 900 ms vs p50 lama 480 → rasio 1.875 > 1.5 → tahan
        h = evaluasi_canary("x", 0.0, 0.9, 900.0, 0.00033, 0.00033)
        self.assertEqual(h["putuskan"], "tahan")
        # 720 ms = 1.5 × 480 → tepat di SLO → lanjut
        h2 = evaluasi_canary("x", 0.0, 0.9, 720.0, 0.00033, 0.00033)
        self.assertEqual(h2["putuskan"], "lanjut")

    def test_biaya_dua_kali_lama(self):
        # 0.00070 > 2 × 0.00033 = 0.00066 → tahan
        h = evaluasi_canary("x", 0.0, 0.9, 480.0, 0.00070, 0.00033)
        self.assertEqual(h["putuskan"], "tahan")
        self.assertIn("biaya", h["alasan"])

    def test_skor_min_dari_eval_gate(self):
        # skor 0.8 lolos default, tapi gagal bila gate dinaikkan ke 0.85
        h = evaluasi_canary("x", 0.0, 0.8, 480.0, 0.00033, 0.00033, skor_min=0.85)
        self.assertEqual(h["putuskan"], "tahan")
        self.assertIn("skor eval", h["alasan"])

    def test_return_memuat_nama(self):
        h = evaluasi_canary("canary-a", 0.0, 0.9, 480.0, 0.00033, 0.00033)
        self.assertEqual(h["nama"], "canary-a")


class TestRencanaRollout(unittest.TestCase):
    def test_angka_terkunci_config_produksi(self):
        cfg = config_layanan(ENV_PRODUKSI)          # CANARY_PERSEN = 20
        rencana = rencana_rollout(cfg)
        persen = [r["persen"] for r in rencana]
        self.assertEqual(persen, [20, 50, 100])
        self.assertFalse(rencana[0]["penuh"])
        self.assertFalse(rencana[1]["penuh"])
        self.assertTrue(rencana[2]["penuh"])

    def test_config_asli_tidak_berubah(self):
        cfg = config_layanan(ENV_PRODUKSI)
        rencana_rollout(cfg)
        self.assertEqual(cfg["CANARY_PERSEN"], 20)

    def test_dengan_canary_menyalin_config(self):
        cfg = config_layanan(ENV_PRODUKSI)
        rencana = rencana_rollout(cfg)
        langkah_50 = [r for r in rencana if r["persen"] == 50][0]
        self.assertEqual(langkah_50["dengan_canary"]["CANARY_PERSEN"], 50)
        self.assertEqual(langkah_50["dengan_canary"]["PORT"], 9000)
        self.assertEqual(langkah_50["dengan_canary"]["SKOR_MIN"], 0.8)

    def test_langkah_kustom(self):
        cfg = {"CANARY_PERSEN": 5, "PORT": 8000}
        rencana = rencana_rollout(cfg, langkah=(5, 25, 75, 100))
        self.assertEqual([r["persen"] for r in rencana], [5, 25, 75, 100])


if __name__ == "__main__":
    unittest.main()
