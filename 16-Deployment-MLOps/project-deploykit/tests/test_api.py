"""Spesifikasi Bagian 2 — Validasi permintaan + rate limit (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di deploykit/api.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploykit.api import RateLimiter, validasi_chat
from deploykit.data import MAKS_KARAKTER, MAKS_PESAN


def _body(n_pesan=1, teks="halo"):
    return {"pesan": [{"peran": "user", "teks": teks}] * n_pesan}


class TestValidasiChat(unittest.TestCase):
    def test_body_valid(self):
        status, balasan = validasi_chat(_body())
        self.assertEqual(status, 200)
        self.assertEqual(balasan, _body())

    def test_bukan_dict(self):
        self.assertEqual(validasi_chat("halo")[0], 422)
        self.assertEqual(validasi_chat(None)[0], 422)
        self.assertEqual(validasi_chat("halo")[1], {"error": "body harus dict"})

    def test_pesan_hilang_bukan_list_kosong(self):
        self.assertEqual(validasi_chat({})[0], 422)
        self.assertEqual(validasi_chat({"pesan": "halo"})[0], 422)
        self.assertEqual(validasi_chat({"pesan": []})[0], 422)
        self.assertEqual(validasi_chat({})[1], {"error": "pesan harus list non-kosong"})

    def test_format_pesan_salah(self):
        self.assertEqual(validasi_chat({"pesan": ["halo"]})[0], 422)
        self.assertEqual(validasi_chat({"pesan": [{"peran": "user"}]})[0], 422)
        self.assertEqual(validasi_chat({"pesan": [{"peran": "user", "teks": "   "}]})[0], 422)
        self.assertEqual(validasi_chat({"pesan": ["halo"]})[1],
                         {"error": "format pesan salah"})

    def test_terlalu_banyak_pesan(self):
        status, _ = validasi_chat(_body(n_pesan=MAKS_PESAN + 1))
        self.assertEqual(status, 422)

    def test_tepat_batas_masih_200(self):
        self.assertEqual(validasi_chat(_body(n_pesan=MAKS_PESAN))[0], 200)

    def test_pesan_terlalu_panjang(self):
        status, balasan = validasi_chat(_body(teks="a" * (MAKS_KARAKTER + 1)))
        self.assertEqual(status, 422)
        self.assertEqual(balasan, {"error": "pesan terlalu panjang"})

    def test_tepat_batas_karakter_masih_200(self):
        self.assertEqual(validasi_chat(_body(teks="a" * MAKS_KARAKTER))[0], 200)


class TestRateLimiter(unittest.TestCase):
    def test_batas_nol_ditolak(self):
        with self.assertRaises(ValueError):
            RateLimiter(0)

    def test_angka_terkunci_60_per_menit(self):
        rl = RateLimiter(60)
        hasil = [rl.izinkan("u1", t) for t in range(60)]   # detik 0..59
        self.assertTrue(all(ok for ok, _ in hasil))
        self.assertEqual(hasil[-1], (True, 0))      # kuota terakhir terpakai
        ok, sisa = rl.izinkan("u1", 59.5)           # masih jendela [0, 60)
        self.assertEqual((ok, sisa), (False, 0))
        self.assertEqual(rl.izinkan("u1", 60), (True, 59))   # jendela baru [60,120)

    def test_jendela_baru_mengembalikan_kuota(self):
        rl = RateLimiter(60)
        for t in range(60):
            rl.izinkan("u1", t)
        self.assertEqual(rl.izinkan("u1", 61), (True, 59))  # jendela 60-120

    def test_klien_terpisah(self):
        rl = RateLimiter(1)
        self.assertEqual(rl.izinkan("a", 0), (True, 0))
        self.assertEqual(rl.izinkan("b", 0), (True, 0))
        self.assertEqual(rl.izinkan("a", 0.5), (False, 0))

    def test_reset_detik(self):
        rl = RateLimiter(60)
        rl.izinkan("u1", 0.0)                       # jendela [0, 60)
        self.assertEqual(rl.reset_detik("u1", 59.2), 1)   # ceil(0.8) = 1
        self.assertEqual(rl.reset_detik("u1", 30.0), 30)  # ceil(30) = 30
        self.assertEqual(rl.reset_detik("u1", 60.0), 0)   # jendela sudah berakhir

    def test_reset_detik_klien_baru(self):
        rl = RateLimiter(60)
        self.assertEqual(rl.reset_detik("baru", 100.0), 0)


if __name__ == "__main__":
    unittest.main()
