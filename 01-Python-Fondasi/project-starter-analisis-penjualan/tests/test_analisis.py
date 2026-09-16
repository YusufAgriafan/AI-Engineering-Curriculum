"""Spesifikasi fungsi analisis penjualan dalam bentuk test (mode TDD).

Jalankan dari folder project ini:
    python -m unittest discover -s tests -v

JANGAN DIUBAH — tugasmu membuat implementasi di penjualan/analisis.py lolos.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penjualan.analisis import (  # noqa: E402
    bersihkan_harga,
    pendapatan_per_produk,
    produk_terlaris,
    rata_rata_harga_per_kota,
    total_pendapatan,
    transaksi_di_atas,
)
from penjualan.data import TRANSAKSI, TRANSAKSI_KOSONG, TRANSAKSI_SATU  # noqa: E402


class TestTotalPendapatan(unittest.TestCase):
    def test_normal(self):
        # Kopi: 10*25000 + 8*27000 + 12*26000 = 778000; Teh: 273500; Susu: 124000
        self.assertEqual(total_pendapatan(TRANSAKSI), 1_175_500)

    def test_kosong(self):
        self.assertEqual(total_pendapatan(TRANSAKSI_KOSONG), 0.0)

    def test_satu(self):
        self.assertEqual(total_pendapatan(TRANSAKSI_SATU), 24000)


class TestPendapatanPerProduk(unittest.TestCase):
    def test_normal(self):
        hasil = pendapatan_per_produk(TRANSAKSI)
        self.assertEqual(hasil["Kopi"], 778_000)
        self.assertEqual(hasil["Teh"], 273_500)
        self.assertEqual(hasil["Susu"], 124_000)

    def test_semua_produk_muncul(self):
        self.assertEqual(set(pendapatan_per_produk(TRANSAKSI)), {"Kopi", "Teh", "Susu"})

    def test_kosong(self):
        self.assertEqual(pendapatan_per_produk(TRANSAKSI_KOSONG), {})


class TestProdukTerlaris(unittest.TestCase):
    def test_berdasarkan_unit(self):
        # Kopi 30 unit, Teh 18, Susu 7
        self.assertEqual(produk_terlaris(TRANSAKSI), "Kopi")

    def test_kosong(self):
        self.assertIsNone(produk_terlaris(TRANSAKSI_KOSONG))

    def test_satu(self):
        self.assertEqual(produk_terlaris(TRANSAKSI_SATU), "Gula")


class TestTransaksiDiAtas(unittest.TestCase):
    def test_filter_dan_urutan(self):
        hasil = transaksi_di_atas(TRANSAKSI, 100_000)
        pendapatan = [t["unit"] * t["harga"] for t in hasil]
        self.assertEqual(pendapatan, sorted(pendapatan, reverse=True))
        self.assertTrue(all(p > 100_000 for p in pendapatan))

    def test_semua(self):
        self.assertEqual(len(transaksi_di_atas(TRANSAKSI, 0)), 8)

    def test_tidak_ada(self):
        self.assertEqual(transaksi_di_atas(TRANSAKSI, 10_000_000), [])


class TestRataRataHargaPerKota(unittest.TestCase):
    def test_normal(self):
        hasil = rata_rata_harga_per_kota(TRANSAKSI)
        # Bandung: (25000+15000+15500+26000)/4 = 20375.0 ; Jakarta: (27000+18000+17500+15000)/4 = 19375.0
        self.assertEqual(hasil["Bandung"], 20375.0)
        self.assertEqual(hasil["Jakarta"], 19375.0)

    def test_pembulatan(self):
        data = [{"produk": "X", "kota": "K", "unit": 1, "harga": 1000},
                {"produk": "Y", "kota": "K", "unit": 1, "harga": 1001},
                {"produk": "Z", "kota": "K", "unit": 1, "harga": 1000}]
        hasil = rata_rata_harga_per_kota(data)
        self.assertEqual(hasil["K"], 1000.33)  # 1000.333... -> 2 desimal


class TestBersihkanHarga(unittest.TestCase):
    def test_format_rp_dengan_titik(self):
        self.assertEqual(bersihkan_harga("Rp 2.000"), 2000.0)

    def test_format_rp_desimal_koma(self):
        self.assertEqual(bersihkan_harga("Rp1.250,75"), 1250.75)

    def test_spasi_dan_tanpa_rp(self):
        self.assertEqual(bersihkan_harga("  15000 "), 15000.0)
        self.assertEqual(bersihkan_harga("3,14"), 3.14)

    def test_tidak_valid(self):
        self.assertEqual(bersihkan_harga("abc"), 0.0)
        self.assertEqual(bersihkan_harga(""), 0.0)


if __name__ == "__main__":
    unittest.main()
