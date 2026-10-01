"""Spesifikasi Bagian 5 — Monitoring & drift (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di deploykit/monitor.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploykit.data import DRIFT, DRIFT_JAM_PENUH, HEALTH_METRIK
from deploykit.monitor import deteksi_drift, health_score, kueri_ke_kata, ringkas_permintaan
from deploykit.sistem import model_mini


class TestHealthScore(unittest.TestCase):
    def test_angka_terkunci_sehat_penuh(self):
        h = health_score()      # semua 4 metrik memenuhi target
        self.assertEqual(h["skor"], 1.0)
        self.assertTrue(h["ok"])
        self.assertEqual(len(h["rincian"]), 4)

    def test_rincian_memuat_semua_metrik(self):
        h = health_score()
        metrik = [r["metrik"] for r in h["rincian"]]
        self.assertEqual(metrik, list(HEALTH_METRIK.keys()))

    def test_satu_metrik_gagal(self):
        metrik = dict(HEALTH_METRIK)
        metrik["error_rate"] = (0.05, 0.01, "maks")     # 0.05 > 0.01 → gagal
        h = health_score(metrik)
        self.assertEqual(h["skor"], 0.75)
        self.assertFalse(h["ok"])
        gagal = [r for r in h["rincian"] if not r["ok"]]
        self.assertEqual(gagal[0]["metrik"], "error_rate")

    def test_arah_min(self):
        metrik = dict(HEALTH_METRIK)
        metrik["skor_eval"] = (0.5, 0.75, "min")        # 0.5 < 0.75 → gagal
        h = health_score(metrik)
        self.assertEqual(h["skor"], 0.75)

    def test_semua_gagal(self):
        metrik = {k: ((1e9, v[1], v[2]) if v[2] == "maks" else (-1, v[1], v[2]))
                  for k, v in HEALTH_METRIK.items()}
        h = health_score(metrik)
        self.assertEqual(h["skor"], 0.0)
        self.assertFalse(h["ok"])


class TestKueriKeKata(unittest.TestCase):
    def test_buang_kata_pendek(self):
        kata = kueri_ke_kata("Apakah toko punya cabang di Surabaya?")
        self.assertIn("surabaya", kata)
        self.assertIn("cabang", kata)
        self.assertNotIn("di", kata)
        self.assertNotIn("apakah", kata)

    def test_kosong(self):
        self.assertEqual(kueri_ke_kata("   "), set())


class TestDeteksiDrift(unittest.TestCase):
    def test_angka_terkunci_data_produksi(self):
        d = deteksi_drift()
        self.assertIn("qris", d["kata_baru"])
        self.assertNotIn("pengiriman", d["kata_baru"])
        self.assertEqual(d["kosakata_lama"], 15)
        # periode baru = 4 kueri, semuanya menyebut qris → 1.0
        self.assertEqual(d["proporsi_baru"], 1.0)
        self.assertTrue(d["drift"])

    def test_ambang_dinilai_dari_data(self):
        # ambang 1.01 → 1.0 < ambang → tidak drift
        d = deteksi_drift(ambang=1.01)
        self.assertFalse(d["drift"])

    def test_periode_baru_kosong(self):
        d = deteksi_drift([(0, "Berapa lama pengiriman reguler?")], jam_penuh=1)
        self.assertEqual(d["proporsi_baru"], 0.0)
        self.assertFalse(d["drift"])

    def test_kata_baru_urut_abjad_unik(self):
        d = deteksi_drift()
        self.assertEqual(d["kata_baru"], sorted(set(d["kata_baru"])))


class TestRingkasPermintaan(unittest.TestCase):
    def test_baris_log_terkunci(self):
        import math
        h = model_mini("Berapa lama pengiriman reguler?")
        baris = ringkas_permintaan(h)
        self.assertEqual(baris["model"], "api_mini")
        self.assertFalse(baris["abstain"])
        self.assertEqual(baris["token_out"], math.ceil(len(h["jawaban"]) / 4))
        self.assertEqual(baris["singkat"], h["jawaban"][:40])

    def test_abstain_tercatat(self):
        h = model_mini("QRIS kadaluarsa berapa lama?")   # tak ada di KB → abstain
        baris = ringkas_permintaan(h)
        self.assertTrue(baris["abstain"])


if __name__ == "__main__":
    unittest.main()
