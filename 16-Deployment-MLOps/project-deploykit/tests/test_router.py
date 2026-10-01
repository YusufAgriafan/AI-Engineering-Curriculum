"""Spesifikasi Bagian 3 — Router 2-tier + fallback + biaya (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di deploykit/router.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploykit.data import HARGA, PROFIL_TOKEN
from deploykit.router import (biaya_permintaan, breakeven_req_per_hari,
                              jawab_fallback, jawab_router, pilih_tier)
from deploykit.sistem import model_besar, model_mini


class TestPilihTier(unittest.TestCase):
    def test_angka_terkunci_mudah(self):
        self.assertEqual(pilih_tier("Berapa lama pengiriman reguler?"), "api_mini")

    def test_angka_terkunci_sulit(self):
        self.assertEqual(pilih_tier("Kenapa pesanan saya dikembalikan?"), "api_besar")
        self.assertEqual(pilih_tier("BANDINGKAN pengiriman kilat dan reguler"), "api_besar")

    def test_case_insensitive(self):
        self.assertEqual(pilih_tier("KENAPA barang ditolak?"), "api_besar")

    def test_kata_di_dalam_kalimat(self):
        self.assertEqual(pilih_tier("tolong jelaskan kebijakan"), "api_besar")


class TestJawabRouter(unittest.TestCase):
    def test_angka_terkunci_mudah_ke_mini(self):
        h = jawab_router("Berapa lama pengiriman reguler?")
        self.assertEqual(h["meta"]["model"], "api_mini")
        self.assertEqual(h["meta"]["tier"], "api_mini")
        self.assertEqual(h["meta"]["router"], "mudah")
        self.assertIn("reguler", h["jawaban"].lower())

    def test_sulit_ke_besar(self):
        h = jawab_router("Jelaskan kebijakan pengembalian barang elektronik.")
        self.assertEqual(h["meta"]["model"], "api_besar")
        self.assertEqual(h["meta"]["router"], "hard")
        # jawaban model besar = 2 kalimat
        self.assertGreaterEqual(h["jawaban"].count(". "), 1)

    def test_abstain_ikuti_ambang_konfig(self):
        # skor retrieval 0.5 (cashback+wallet di d3)
        h = jawab_router("Bagaimana cara cashback e-wallet?", ambang=0.9)
        self.assertTrue(h["meta"]["abstain"])
        h2 = jawab_router("Bagaimana cara cashback e-wallet?", ambang=0.05)
        self.assertFalse(h2["meta"]["abstain"])


class TestJawabFallback(unittest.TestCase):
    def test_utama_sehat(self):
        h = jawab_fallback("Berapa lama pengiriman reguler?")
        self.assertEqual(h["meta"]["tier"], "utama")
        self.assertEqual(h["meta"]["model"], "api_mini")
        self.assertNotIn("error_utama", h["meta"])

    def test_angka_terkunci_fallback_terpakai(self):
        def gagal(teks, ambang=None):
            return {"jawaban": "", "skor_retrieval": 0.0,
                    "meta": {"model": "api_mini", "error": "timeout"}}

        h = jawab_fallback("Berapa lama pengiriman reguler?", gagal_fn=gagal)
        self.assertEqual(h["meta"]["tier"], "cadangan")
        self.assertEqual(h["meta"]["model"], "api_besar")
        self.assertEqual(h["meta"]["error_utama"], "timeout")
        self.assertIn("reguler", h["jawaban"].lower())

    def test_fallback_dengan_model_injeksi(self):
        def utama_gagal(teks, ambang=None):
            return {"jawaban": "", "skor_retrieval": 0.0,
                    "meta": {"error": "rate limit"}}

        def cadangan_ok(teks, ambang=None):
            return {"jawaban": "Jawaban cadangan.", "skor_retrieval": 0.9,
                    "meta": {"model": "lokal"}}

        h = jawab_fallback("q", gagal_fn=utama_gagal, cadangan_fn=cadangan_ok)
        # model tiruan diganti: meta dari cadangan_ok + tier/error_utama
        self.assertEqual(h["meta"]["tier"], "cadangan")
        self.assertEqual(h["meta"]["error_utama"], "rate limit")
        self.assertEqual(h["jawaban"], "Jawaban cadangan.")


class TestBiaya(unittest.TestCase):
    def test_biaya_permintaan_terkunci(self):
        meta = {"token_in": 1000, "token_out": 300}
        biaya = biaya_permintaan(meta, "api_mini")
        self.assertAlmostEqual(biaya, (1000 * 0.15 + 300 * 0.60) / 1e6, places=14)
        self.assertAlmostEqual(biaya, 0.00033, places=10)

    def test_biaya_api_besar_lebih_mahal(self):
        meta = {"token_in": 1000, "token_out": 300}
        self.assertGreater(biaya_permintaan(meta, "api_besar"),
                           biaya_permintaan(meta, "api_mini"))

    def test_lokal_nol(self):
        self.assertEqual(
            biaya_permintaan({"token_in": 100, "token_out": 10}, "lokal"), 0.0)

    def test_angka_terkunci_breakeven(self):
        # (60/30) / ((1000*0.15 + 300*0.60)/1e6) = 2.0 / 0.00033 = 6060.6… → 6061
        self.assertEqual(breakeven_req_per_hari(), 6061)
        self.assertEqual(breakeven_req_per_hari(30.0), 3030)


if __name__ == "__main__":
    unittest.main()
