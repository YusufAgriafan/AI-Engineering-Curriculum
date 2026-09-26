"""Spesifikasi Bagian 6 — Function calling: skema, validasi, loop (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/tools.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.tools import (jalankan_tool, loop_tool, skema_parameter, skema_tools,
                          validasi_argumen)
from llmkit.transport import provider_utama


class TestSkema(unittest.TestCase):
    def test_bentuk_untuk_api(self):
        skema = skema_tools()
        self.assertEqual(len(skema), 3)
        for s in skema:
            self.assertEqual(s["type"], "function")
            self.assertIn("name", s["function"])
            self.assertIn("description", s["function"])
            self.assertIn("parameters", s["function"])

    def test_tipe_parameter_benar(self):
        p = skema_parameter("get_cuaca")
        self.assertEqual(p["type"], "object")
        self.assertEqual(p["properties"]["kota"]["type"], "string")
        self.assertEqual(sorted(p["required"]), ["kota", "satuan"])

    def test_integer_dikenali(self):
        self.assertEqual(skema_parameter("cari_catatan")["properties"]["batas"]["type"],
                         "integer")


class TestValidasiArgumen(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validasi_argumen("get_cuaca", {"kota": "Bandung", "satuan": "celsius"}), [])

    def test_field_hilang(self):
        self.assertEqual(validasi_argumen("get_cuaca", {}), ["missing:kota", "missing:satuan"])

    def test_tipe_salah(self):
        self.assertEqual(validasi_argumen("get_cuaca", {"kota": 5, "satuan": "celsius"}),
                         ["type:kota"])

    def test_boolean_bukan_integer(self):
        self.assertEqual(validasi_argumen("cari_catatan",
                                          {"kata_kunci": "x", "batas": True}), ["type:batas"])

    def test_field_tak_dikenal(self):
        self.assertEqual(validasi_argumen("get_cuaca",
                                          {"kota": "a", "satuan": "c", "x": 1}),
                         ["tak_dikenal:x"])

    def test_tool_tak_dikenal(self):
        self.assertEqual(validasi_argumen("nuklir", {}), ["tak_dikenal:nuklir"])

    def test_argumen_bukan_dict(self):
        self.assertEqual(validasi_argumen("get_cuaca", "bukan dict"), ["type:<argumen>"])


class TestJalankanTool(unittest.TestCase):
    def test_sukses(self):
        r = jalankan_tool("get_cuaca", {"kota": "Jakarta", "satuan": "celsius"})
        self.assertIsNone(r["galat"])
        self.assertEqual(r["hasil"]["suhu"], 28)
        self.assertEqual(r["errors"], [])

    def test_gagal_tidak_memanggil_handler(self):
        r = jalankan_tool("get_cuaca", {"kota": 5})
        self.assertIsNone(r["hasil"])
        self.assertEqual(r["errors"], ["missing:satuan", "type:kota"])
        self.assertEqual(r["galat"], "missing:satuan; type:kota")

    def test_tool_hitung(self):
        r = jalankan_tool("hitung", {"ekspresi": "15 * 8 + 20"})
        self.assertEqual(r["hasil"]["hasil"], 140)


class TestLoopTool(unittest.TestCase):
    def test_tanpa_tool_satu_iterasi(self):
        h = loop_tool(provider_utama(), "mini", "Jelaskan singkat apa itu embedding.")
        self.assertEqual(h["iterasi"], 1)
        self.assertEqual(h["tool_dipakai"], [])
        self.assertIn("embedding", h["jawaban"])

    def test_satu_tool(self):
        h = loop_tool(provider_utama(), "mini", "Berapa suhu Jakarta hari ini?")
        self.assertEqual(h["tool_dipakai"], ["get_cuaca"])
        self.assertEqual(h["iterasi"], 2)
        self.assertEqual(h["riwayat"][0]["hasil"]["suhu"], 28)
        self.assertIn("suhu Jakarta", h["jawaban"])

    def test_dua_tool_berurutan(self):
        h = loop_tool(provider_utama(), "mini",
                      "Cari catatan tentang machine learning, lalu hitung 15 * 8 + 20.")
        self.assertEqual(h["tool_dipakai"], ["cari_catatan", "hitung"])
        self.assertEqual(h["iterasi"], 3)
        self.assertEqual(h["riwayat"][1]["hasil"]["hasil"], 140)

    def test_berhenti_saat_maks_iterasi(self):
        h = loop_tool(provider_utama(), "mini",
                      "Cari catatan tentang machine learning, lalu hitung 15 * 8 + 20.",
                      maks_iterasi=2)
        self.assertEqual(h["berhenti"], "maks_iterasi")
        self.assertIsNone(h["jawaban"])

    def test_tool_yang_sudah_dipakai_tidak_diulang(self):
        h = loop_tool(provider_utama(), "mini", "Berapa suhu Jakarta? Suhu ya, bukan cuaca kota lain.")
        self.assertEqual(h["tool_dipakai"], ["get_cuaca"], "tool sama tidak boleh dipanggil dua kali")

    def test_token_dan_latensi_dijumlahkan(self):
        h = loop_tool(provider_utama(), "mini", "Berapa suhu Jakarta hari ini?")
        self.assertEqual(h["riwayat"][0]["iterasi"], 1)
        self.assertGreater(h["tokens_in"], 0)
        self.assertGreater(h["total_ms"], 0)

    def test_galat_argumen_dikirim_balik_ke_model(self):
        from llmkit.transport import ProviderPalsu

        class ProviderNakal(ProviderPalsu):
            """Simulasi model yang memanggil tool dengan argumen cacat."""

            def panggil(self, model, messages, **kw):
                h = super().panggil(model, messages, **kw)
                for tc in h.get("tool_calls", []):
                    tc["argumen"] = {}
                return h

        h = loop_tool(ProviderNakal("utama"), "mini", "Berapa suhu Jakarta hari ini?")
        self.assertEqual(h["riwayat"][0]["galat"], "missing:kota; missing:satuan")
        self.assertIsNone(h["riwayat"][0]["hasil"])
        self.assertEqual(h["tool_dipakai"], ["get_cuaca"], "loop tetap lanjut, bukan crash")


if __name__ == "__main__":
    unittest.main()
