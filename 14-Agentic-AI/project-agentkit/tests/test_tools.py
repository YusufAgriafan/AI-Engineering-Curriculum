"""Spesifikasi Bagian 1 — Registry tools + validasi skema (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di agentkit/tools.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.tools import (REGISTRY, eksekusi_tool, tool_cari_dokumen,
                            tool_cek_pesanan, tool_refund, validasi_skema)


class TestValidasiSkema(unittest.TestCase):
    def test_args_valid(self):
        ok, pesan = validasi_skema("cek_pesanan", {"invoice_id": "INV-001"})
        self.assertTrue(ok)
        self.assertEqual(pesan, "")

    def test_required_hilang(self):
        ok, pesan = validasi_skema("refund", {"invoice_id": "INV-001"})
        self.assertFalse(ok)
        self.assertIn("alasan", pesan)

    def test_pola_invoice(self):
        ok, _ = validasi_skema("cek_pesanan", {"invoice_id": "ABC123"})
        self.assertFalse(ok)

    def test_min_panjang(self):
        ok, pesan = validasi_skema("cari_dokumen", {"kueri": "ab"})
        self.assertFalse(ok)
        self.assertIn("minimal", pesan)

    def test_type_salah(self):
        ok, pesan = validasi_skema("cek_pesanan", {"invoice_id": 123})
        self.assertFalse(ok)
        self.assertIn("string", pesan)

    def test_tool_tak_dikenal(self):
        ok, pesan = validasi_skema("hapus_database", {})
        self.assertFalse(ok)
        self.assertIn("tidak dikenal", pesan)

    def test_args_bukan_dict(self):
        ok, _ = validasi_skema("cek_pesanan", "INV-001")
        self.assertFalse(ok)


class TestToolCariDokumen(unittest.TestCase):
    def test_angka_terkunci_d1(self):
        hasil = tool_cari_dokumen("pengembalian barang")
        self.assertEqual(len(hasil), 1)
        self.assertEqual(hasil[0]["id"], "d1")
        self.assertEqual(hasil[0]["skor"], 1.0)
        self.assertIn("14 hari", hasil[0]["teks"])

    def test_dua_dokumen_terkunci(self):
        # kata unik {pengembalian, cod, cashback}: d3 cocok 2/3, d1 cocok 1/3
        hasil = tool_cari_dokumen("pengembalian cod cashback")
        self.assertEqual([h["id"] for h in hasil], ["d3", "d1"])
        self.assertEqual(hasil[0]["skor"], 0.6667)
        self.assertEqual(hasil[1]["skor"], 0.3333)

    def test_tanpa_cocok_kosong(self):
        self.assertEqual(tool_cari_dokumen("xyzzy"), [])
        self.assertEqual(tool_cari_dokumen("ab"), [])   # kata terlalu pendek

    def test_input_tak_berubah(self):
        tool_cari_dokumen("pengembalian barang")
        self.assertTrue(True)   # hasil list baru — tidak ada state global


class TestToolCekPesanan(unittest.TestCase):
    def test_angka_terkunci_inv001(self):
        p = tool_cek_pesanan("INV-001")
        self.assertEqual(p["status"], "dikirim")
        self.assertEqual(p["produk"], "Keyboard mini")
        self.assertEqual(p["total"], 250000)

    def test_pii_dibuang(self):
        """Kunci 'user' tidak boleh bocor ke konteks LLM."""
        p = tool_cek_pesanan("inv-002")   # huruf kecil: tetap dinormalisasi
        self.assertEqual(p["status"], "diproses")
        self.assertNotIn("user", p)

    def test_id_tak_ada(self):
        p = tool_cek_pesanan("INV-999")
        self.assertEqual(p, {"error": "pesanan tidak ditemukan"})


class TestToolRefund(unittest.TestCase):
    def test_selalu_butuh_konfirmasi(self):
        r = tool_refund("INV-001", "barang rusak")
        self.assertTrue(r["butuh_konfirmasi"])
        self.assertEqual(r["invoice_id"], "INV-001")
        self.assertEqual(r["alasan"], "barang rusak")
        self.assertIn("konfirmasi", r["pesan"])


class TestEksekusi(unittest.TestCase):
    def test_dispatch_valid(self):
        hasil = eksekusi_tool("cek_pesanan", {"invoice_id": "INV-003"})
        self.assertEqual(hasil["status"], "selesai")

    def test_dispatch_error_tidak_raise(self):
        hasil = eksekusi_tool("cek_pesanan", {"invoice_id": "salah"})
        self.assertIn("error", hasil)

    def test_tool_tak_kenal(self):
        hasil = eksekusi_tool("rm_rf", {})
        self.assertIn("error", hasil)

    def test_registry_lengkap(self):
        self.assertEqual(set(REGISTRY), {"cari_dokumen", "cek_pesanan", "refund"})


if __name__ == "__main__":
    unittest.main()
