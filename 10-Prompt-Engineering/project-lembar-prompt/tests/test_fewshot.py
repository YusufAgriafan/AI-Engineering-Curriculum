"""Spesifikasi Bagian 3 — Few-shot: format, pemilihan, pengurutan (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di plib/fewshot.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plib.data import GOLDEN
from plib.fewshot import (bangun_fewshot, format_contoh, pilih_contoh,
                          urutkan_contoh)


def _non_null(kasus):
    return sum(1 for v in kasus["expected"].values() if v is not None)


class TestFormatContoh(unittest.TestCase):
    def test_format_tepat(self):
        out = format_contoh("hai", {"product_name": "X", "price": 100})
        self.assertEqual(out, 'Input: "hai"\nOutput: {"product_name": "X", "price": 100}')

    def test_unicode_tidak_di_escape(self):
        out = format_contoh("a", {"product_name": "Kursi Tamu"})
        self.assertIn("Kursi Tamu", out)
        self.assertNotIn("\\u", out)

    def test_nilai_null_tetap_json(self):
        out = format_contoh("a", {"price": None})
        self.assertIn('"price": null', out)


class TestPilihContoh(unittest.TestCase):
    def test_k_lebih_besar_dari_n(self):
        self.assertEqual(len(pilih_contoh(GOLDEN, 99)), len(GOLDEN))

    def test_k_satu_ambil_pertama(self):
        terpilih = pilih_contoh(GOLDEN, 1)
        self.assertEqual(terpilih, [GOLDEN[0]])

    def test_sebar_merata_angka_terkunci(self):
        terpilih = pilih_contoh(GOLDEN, 3)
        self.assertEqual([GOLDEN.index(c) for c in terpilih], [0, 2, 5])

    def test_deterministik(self):
        a = [c["teks"] for c in pilih_contoh(GOLDEN, 4)]
        b = [c["teks"] for c in pilih_contoh(GOLDEN, 4)]
        self.assertEqual(a, b)

    def test_tidak_memodifikasi_input(self):
        sebelum = list(GOLDEN)
        pilih_contoh(GOLDEN, 3)
        self.assertEqual(GOLDEN, sebelum)


class TestUrutkanContoh(unittest.TestCase):
    def test_paling_kosong_dulu_paling_kaya_terakhir(self):
        urut = urutkan_contoh(GOLDEN)
        skor = [_non_null(c) for c in urut]
        self.assertEqual(skor, sorted(skor))
        self.assertEqual(urut[0]["teks"].startswith("Selamat pagi"), True)
        self.assertEqual(urut[-1]["teks"].startswith("Smart TV"), True)

    def test_stabil_untuk_skor_sama(self):
        urut = urutkan_contoh(GOLDEN)
        # kasus 2 dan 5 sama-sama 4 field terisi -> urutan asli dipertahankan
        pos2 = urut.index(GOLDEN[1])
        pos5 = urut.index(GOLDEN[4])
        self.assertLess(pos2, pos5)

    def test_bukan_sortir_in_place(self):
        sebelum = [c["teks"] for c in GOLDEN]
        urutkan_contoh(GOLDEN)
        self.assertEqual([c["teks"] for c in GOLDEN], sebelum)


class TestBangunFewshot(unittest.TestCase):
    def test_jumlah_pasangan(self):
        blok = bangun_fewshot(GOLDEN, 2)
        self.assertEqual(blok.count("Input:"), 2)
        self.assertEqual(blok.count("Output:"), 2)

    def test_contoh_tersulit_terakhir(self):
        blok = bangun_fewshot(GOLDEN, 3)
        # k=3 -> {0,2,5}; diurutkan -> terakhir = Oven listrik (5 field terisi)
        self.assertTrue(blok.rstrip().endswith('}'))
        self.assertIn("Oven listrik", blok.split("Input:")[-1])

    def test_contoh_polos_ikut_terpilih(self):
        # k=3 -> indeks {0,2,5} lalu diurutkan -> "Selamat pagi" tetap ada
        blok = bangun_fewshot(GOLDEN, 3)
        self.assertIn("Selamat pagi", blok)

    def test_deterministik(self):
        self.assertEqual(bangun_fewshot(GOLDEN, 3), bangun_fewshot(GOLDEN, 3))


if __name__ == "__main__":
    unittest.main()
