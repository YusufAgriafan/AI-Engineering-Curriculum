"""Spesifikasi Bagian 0 — Perilaku provider PALSU (mode TDD).

Provider-nya GIVEN, tapi test ini penting: dokumen ini adalah "kontrak" yang
membuat kegagalan bisa dijadwalkan. Kalau kamu paham transport, kamu paham
kenapa retry/fallback/cache bisa diuji tanpa jaringan.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

from llmkit.transport import (GalatPermintaanBuruk, GalatRateLimit, GalatServer,
                              GalatTimeout, ProviderPalsu, provider_cadangan,
                              provider_utama)
from llmkit.data import HARGA, RENCANA  # noqa: F401

PESAN = [{"role": "user", "content": "Halo dunia"}]


class TestRencanaKegagalan(unittest.TestCase):
    def test_rencana_habis_lalu_ok(self):
        p = provider_utama()
        urutan = []
        for _ in range(3):
            urutan.append(p.status_berikut("besar"))
            try:
                p.panggil("besar", PESAN)
            except Exception:
                pass
        self.assertEqual(urutan, ["rate_limit", "server_error", "ok"])

    def test_status_berubah_setelah_dipanggil(self):
        p = provider_utama()
        self.assertEqual(p.status_berikut("besar"), "rate_limit")
        with self.assertRaises(GalatRateLimit):
            p.panggil("besar", PESAN)
        self.assertEqual(p.status_berikut("besar"), "server_error")
        with self.assertRaises(GalatServer):
            p.panggil("besar", PESAN)
        self.assertEqual(p.status_berikut("besar"), "ok")
        self.assertIsNotNone(p.panggil("besar", PESAN)["teks"])

    def test_model_tak_dikenal(self):
        p = provider_utama()
        self.assertEqual(p.status_berikut("ajaib"), "model_tidak_dikenal")
        with self.assertRaises(GalatPermintaanBuruk):
            p.panggil("ajaib", PESAN)

    def test_provider_terpisah_punya_keadaan_sendiri(self):
        p1, p2 = provider_utama(), provider_utama()
        with self.assertRaises(GalatRateLimit):
            p1.panggil("sedang", PESAN)
        self.assertEqual(p2.jumlah_panggilan.get("sedang", 0), 0)
        self.assertEqual(p2.status_berikut("sedang"), "rate_limit", "p2 belum dipakai sama sekali")
        self.assertEqual(p1.status_berikut("sedang"), "ok", "p1 sudah lewat rencana pertama")

    def test_cadangan_selalu_ok(self):
        p = provider_cadangan()
        self.assertEqual(p.panggil("murah", PESAN)["provider"], "cadangan")


class TestHasilPanggilan(unittest.TestCase):
    def test_token_dihitung_dari_prompt(self):
        p = provider_utama()
        h = p.panggil("mini", PESAN)
        prompt = "Halo dunia"
        self.assertEqual(h["tokens_in"], math.ceil(len(prompt) / 4))
        self.assertEqual(h["tokens_out"], math.ceil(len(h["teks"]) / 4))

    def test_latensi_deterministik(self):
        a = provider_utama().panggil("mini", PESAN)["latency_ms"]
        b = provider_utama().panggil("mini", PESAN)["latency_ms"]
        self.assertEqual(a, b)

    def test_model_besar_lebih_lambat(self):
        p = provider_utama()
        kecil = p.latensi_untuk("mini", "x")
        besar = p.latensi_untuk("besar", "x")
        self.assertGreater(besar, kecil)

    def test_timeout_bila_latensi_melebihi_batas(self):
        bersih = ProviderPalsu("utama", HARGA, {"besar": ["ok"]})
        with self.assertRaises(GalatTimeout):
            bersih.panggil("besar", PESAN, timeout_ms=50)

    def test_log_mencatat_setiap_panggilan(self):
        p = provider_utama()
        p.panggil("mini", PESAN)
        with self.assertRaises(GalatRateLimit):
            p.panggil("sedang", PESAN)
        self.assertEqual(len(p.log), 2)
        self.assertEqual(p.log[1]["status"], "rate_limit")
        self.assertEqual(p.log[1]["percobaan"], 1)

    def test_attempts_mencerminkan_panggilan_ke_n(self):
        p = provider_utama()
        with self.assertRaises(GalatRateLimit):
            p.panggil("sedang", PESAN)
        self.assertEqual(p.panggil("sedang", PESAN)["attempts"], 2)


class TestJadwalKegagalanTerkunci(unittest.TestCase):
    def test_rencana_data(self):
        self.assertEqual(RENCANA["mini"], ["ok"])
        self.assertEqual(RENCANA["sedang"], ["rate_limit", "ok"])
        self.assertEqual(RENCANA["besar"], ["rate_limit", "server_error", "ok"])
        self.assertEqual(list(HARGA), ["mini", "sedang", "besar"])


if __name__ == "__main__":
    unittest.main()
