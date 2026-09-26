"""Spesifikasi Bagian 3 — Rantai fallback multi-provider (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/fallback.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.data import HARGA, HARGA_CADANGAN, RANTAI_FALLBACK
from llmkit.fallback import panggil_dengan_fallback
from llmkit.retry import GalatSemuaPercobaan
from llmkit.transport import ProviderPalsu, provider_cadangan, provider_utama

PESAN = [{"role": "user", "content": "Halo, jawab singkat."}]


def provs(utama=None, cadangan=None):
    return {"utama": utama or provider_utama(), "cadangan": cadangan or provider_cadangan()}


class TestFallbackDasar(unittest.TestCase):
    def test_tautan_pertama_berhasil(self):
        h = panggil_dengan_fallback(provs(), RANTAI_FALLBACK, PESAN,
                                    settings={"maks_percobaan": 3})
        self.assertEqual(h["model"], "besar")
        self.assertEqual(h["lompatan"], 0)
        self.assertEqual(len(h["dicoba"]), 1)
        self.assertTrue(h["dicoba"][0]["sukses"])

    def test_lompat_ke_tautan_kedua(self):
        h = panggil_dengan_fallback(provs(), RANTAI_FALLBACK, PESAN,
                                    settings={"maks_percobaan": 2})
        # besar gagal (rate_limit + server_error), mini berhasil
        self.assertEqual(h["model"], "mini")
        self.assertEqual(h["lompatan"], 1)
        self.assertEqual([d["sukses"] for d in h["dicoba"]], [False, True])
        self.assertEqual(h["dicoba"][0]["jenis"], "semua_gagal")

    def test_lompat_ke_provider_cadangan(self):
        rusak = ProviderPalsu("utama", HARGA, {"besar": ["rate_limit"], "mini": ["rate_limit"]})
        h = panggil_dengan_fallback(provs(utama=rusak), RANTAI_FALLBACK, PESAN,
                                    settings={"maks_percobaan": 1})
        self.assertEqual(h["model"], "murah")
        self.assertEqual(h["provider"], "cadangan")
        self.assertEqual(h["lompatan"], 2)

    def test_hasil_membawa_metadata_fallback(self):
        h = panggil_dengan_fallback(provs(), RANTAI_FALLBACK, PESAN,
                                    settings={"maks_percobaan": 2})
        for k in ("dicoba", "lompatan", "tautan", "teks", "tokens_in", "tokens_out"):
            self.assertIn(k, h)


class TestFallbackGagalTotal(unittest.TestCase):
    def test_seluruh_rantai_gagal(self):
        rusak = ProviderPalsu("utama", HARGA,
                              {"besar": ["rate_limit"], "mini": ["server_error"]})
        rusak_cad = ProviderPalsu("cadangan", HARGA_CADANGAN, {"murah": ["rate_limit"]})
        with self.assertRaises(GalatSemuaPercobaan) as ctx:
            panggil_dengan_fallback(provs(utama=rusak, cadangan=rusak_cad),
                                    RANTAI_FALLBACK, PESAN, settings={"maks_percobaan": 1})
        self.assertEqual(len(ctx.exception.riwayat), 3)

    def test_provider_tidak_ada_dilewati(self):
        rantai = [{"provider": "hilang", "model": "besar"},
                  {"provider": "utama", "model": "mini"}]
        h = panggil_dengan_fallback(provs(), rantai, PESAN, settings={"maks_percobaan": 1})
        self.assertEqual(h["model"], "mini")
        self.assertEqual(h["dicoba"][0]["jenis"], "provider_tidak_ada")
        self.assertEqual(h["lompatan"], 1)

    def test_rantai_kosong_gagal(self):
        with self.assertRaises(GalatSemuaPercobaan):
            panggil_dengan_fallback(provs(), [], PESAN)


class TestSetelanPerTautan(unittest.TestCase):
    def test_tautan_menimpa_setelan_dasar(self):
        PESAN2 = [{"role": "user", "content": "Halo."}]
        h = panggil_dengan_fallback(provs(), RANTAI_FALLBACK, PESAN2,
                                    settings={"maks_percobaan": 2})
        self.assertEqual(h["model"], "mini")
        # percobaan pada tautan kedua tercatat di `dicoba`
        self.assertEqual(h["dicoba"][1]["attempts"], 1)

    def test_setelan_tautan_override(self):
        rantai = [
            {"provider": "utama", "model": "besar", "settings": {"maks_percobaan": 1}},
            {"provider": "utama", "model": "mini", "settings": {"maks_percobaan": 1}},
        ]
        h = panggil_dengan_fallback(provs(), rantai, PESAN, settings={"maks_percobaan": 5})
        self.assertEqual(h["model"], "mini")


if __name__ == "__main__":
    unittest.main()
