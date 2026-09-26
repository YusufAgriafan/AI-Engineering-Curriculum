"""Spesifikasi Bagian 2 — Retry + backoff (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/retry.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.retry import (GalatSemuaPercobaan, backoff_ms, panggil_dengan_retry)
from llmkit.transport import (GalatPermintaanBuruk, ProviderPalsu, provider_utama)

PESAN = [{"role": "user", "content": "Halo, tolong jawab singkat."}]


class TestBackoff(unittest.TestCase):
    def test_eksponensial(self):
        self.assertEqual([backoff_ms(n) for n in (1, 2, 3, 4)], [1000, 2000, 4000, 8000])

    def test_dibatasi_maks(self):
        self.assertEqual(backoff_ms(9), 8000)

    def test_basis_dan_faktor_custom(self):
        self.assertEqual([backoff_ms(n, 100, 3, 500) for n in (1, 2, 3, 4)], [100, 300, 500, 500])


class TestRetrySukses(unittest.TestCase):
    def test_langsung_berhasil(self):
        h = panggil_dengan_retry(provider_utama(), "mini", PESAN)
        self.assertEqual(h["attempts"], 1)
        self.assertEqual(h["timpenungguan_ms"], [])
        self.assertEqual(h["riwayat"], [])
        self.assertEqual(h["total_ms"], h["latency_ms"])

    def test_gagal_sekali_lalu_berhasil(self):
        tunggu = []
        h = panggil_dengan_retry(provider_utama(), "sedang", PESAN, sleep_fn=tunggu.append)
        self.assertEqual(h["attempts"], 2)
        self.assertEqual(h["riwayat"], [(1, "rate_limit")])
        self.assertEqual(h["timpenungguan_ms"], [1000])
        self.assertEqual(tunggu, [1000], "sleep_fn harus dipanggil dengan jeda backoff")
        self.assertEqual(h["total_ms"], h["latency_ms"] + 1000)

    def test_dua_kegagalan_lalu_berhasil(self):
        h = panggil_dengan_retry(provider_utama(), "besar", PESAN, maks_percobaan=3)
        self.assertEqual(h["attempts"], 3)
        self.assertEqual(h["riwayat"], [(1, "rate_limit"), (2, "server_error")])
        self.assertEqual(h["timpenungguan_ms"], [1000, 2000])
        self.assertEqual(h["total_ms"], h["latency_ms"] + 3000)

    def test_hasil_lengkap(self):
        h = panggil_dengan_retry(provider_utama(), "mini", PESAN)
        for k in ("teks", "tokens_in", "tokens_out", "latency_ms", "model", "attempts"):
            self.assertIn(k, h)


class TestRetryGagalTotal(unittest.TestCase):
    def test_percobaan_habis(self):
        p = provider_utama()
        with self.assertRaises(GalatSemuaPercobaan) as ctx:
            panggil_dengan_retry(p, "besar", PESAN, maks_percobaan=2)
        self.assertEqual(ctx.exception.riwayat, [(1, "rate_limit"), (2, "server_error")])
        self.assertEqual(p.jumlah_panggilan["besar"], 2, "harus tepat 2 panggilan, tidak lebih")

    def test_tidak_menunggu_setelah_percobaan_terakhir(self):
        tunggu = []
        with self.assertRaises(GalatSemuaPercobaan):
            panggil_dengan_retry(provider_utama(), "besar", PESAN, maks_percobaan=2,
                                 sleep_fn=tunggu.append)
        self.assertEqual(tunggu, [1000], "backoff hanya untuk percobaan yang masih tersisa")

    def test_galat_fatal_tidak_diulang(self):
        p = provider_utama()
        with self.assertRaises(GalatPermintaanBuruk):
            panggil_dengan_retry(p, "model_ajaib", PESAN)
        self.assertEqual(p.jumlah_panggilan["model_ajaib"], 1, "bad_request = fatal, jangan retry")

    def test_galat_fatal_tidak_dibungkus(self):
        with self.assertRaises(GalatPermintaanBuruk) as ctx:
            panggil_dengan_retry(provider_utama(), "model_ajaib", PESAN)
        self.assertEqual(ctx.exception.jenis, "bad_request")


class TestSetelanCustom(unittest.TestCase):
    def test_maks_percobaan_satu_tidak_retry(self):
        p = provider_utama()
        with self.assertRaises(GalatSemuaPercobaan):
            panggil_dengan_retry(p, "sedang", PESAN, maks_percobaan=1)
        self.assertEqual(p.jumlah_panggilan["sedang"], 1)

    def test_basis_backoff_custom(self):
        tunggu = []
        panggil_dengan_retry(provider_utama(), "sedang", PESAN, basis_ms=50,
                             sleep_fn=tunggu.append)
        self.assertEqual(tunggu, [50])

    def test_sleep_fn_opsional(self):
        # tanpa sleep_fn tidak boleh menunggu sungguhan (harus tetap cepat)
        h = panggil_dengan_retry(provider_utama(), "sedang", PESAN)
        self.assertEqual(h["timpenungguan_ms"], [1000])

    def test_temperature_diteruskan_ke_provider(self):
        seen = {}
        p = ProviderPalsu("x", {"mini": {"input": 0.15, "output": 0.6}}, {"mini": ["ok"]})

        asli = p.panggil

        def dibungkus(model, messages, **kw):
            seen.update(kw)
            return asli(model, messages, **kw)

        p.panggil = dibungkus
        panggil_dengan_retry(p, "mini", PESAN, temperature=0.7, maks_token=99)
        self.assertEqual(seen["temperature"], 0.7)
        self.assertEqual(seen["maks_token"], 99)


if __name__ == "__main__":
    unittest.main()
