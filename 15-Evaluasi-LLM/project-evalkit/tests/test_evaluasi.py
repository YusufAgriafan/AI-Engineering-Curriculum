"""Spesifikasi Bagian 7 — Evaluasi end-to-end + error analysis + feedback (TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di evalkit/evaluasi.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.data import VERSI
from evalkit.evaluasi import (analisis_error, antrean_feedback,
                              bandingkan_versi, evaluasi_versi, laporan_lengkap)


class TestEvaluasiVersi(unittest.TestCase):
    def test_angka_terkunci_total(self):
        for versi, total in (("v1", 0.2667), ("v2", 0.5667), ("v3", 0.8667)):
            lap = evaluasi_versi(versi)
            self.assertEqual(lap["versi"], versi)
            self.assertEqual(lap["total"], total, versi)
            self.assertEqual(lap["run"]["agregat"]["n"], 12)

    def test_evalset_kustom(self):
        from evalkit.data import EVALSET
        lap = evaluasi_versi("v3", evalset=[EVALSET[0]])
        self.assertEqual(lap["run"]["agregat"]["n"], 1)


class TestBandingkanVersi(unittest.TestCase):
    def test_angka_terkunci_urut_versi(self):
        laporan = bandingkan_versi()
        self.assertEqual([l["versi"] for l in laporan], list(VERSI))
        self.assertEqual([l["total"] for l in laporan], [0.2667, 0.5667, 0.8667])


class TestAnalisisError(unittest.TestCase):
    def test_angka_terkunci_v3(self):
        run = evaluasi_versi("v3")["run"]
        gagal = analisis_error(run)
        self.assertEqual([g["id"] for g in gagal], ["e05", "e06"])
        self.assertTrue(all(g["skor"] < 1.0 for g in gagal))

    def test_v1_delapan_gagal(self):
        gagal = analisis_error(evaluasi_versi("v1")["run"])
        self.assertEqual(len(gagal), 8)

    def test_urut_terburuk_dulu_dan_kunci_lengkap(self):
        gagal = analisis_error(evaluasi_versi("v1")["run"])
        for g in gagal:
            self.assertEqual(set(g), {"id", "kategori", "tanya", "jawaban", "skor"})
        skor = [g["skor"] for g in gagal]
        self.assertEqual(skor, sorted(skor))

    def test_semua_bersih_kosong(self):
        gagal = analisis_error(evaluasi_versi("v3",
                                              evalset=[{"id": "x", "kategori": "kebijakan",
                                                        "tanya": "Berapa lama barang dikembalikan?",
                                                        "kunci": "14 hari"}])["run"])
        self.assertEqual(gagal, [])


class TestAntreanFeedback(unittest.TestCase):
    def test_angka_terkunci(self):
        q1 = antrean_feedback(evaluasi_versi("v1")["run"])
        q3 = antrean_feedback(evaluasi_versi("v3")["run"])
        self.assertEqual(q1, {"antrean": ["e05", "e06", "e07", "e08", "e09", "e10", "e11", "e12"],
                              "n": 8})
        self.assertEqual(q3, {"antrean": ["e05", "e06"], "n": 2})

    def test_ambang_ketat(self):
        q = antrean_feedback(evaluasi_versi("v3")["run"], ambang=1.0)
        self.assertEqual(q["n"], 2)   # hanya e05 & e06 yang 0.0


class TestLaporanLengkap(unittest.TestCase):
    def test_angka_terkunci(self):
        L = laporan_lengkap()
        self.assertEqual({v: d["total"] for v, d in L["per_versi"].items()},
                         {"v1": 0.2667, "v2": 0.5667, "v3": 0.8667})
        self.assertEqual({v: d["n_gagal"] for v, d in L["per_versi"].items()},
                         {"v1": 8, "v2": 4, "v3": 2})
        self.assertEqual(L["terbaik"], "v3")
        self.assertEqual(L["regresi_v1_v3"], 0.6)

    def test_sinyal_tersimpan(self):
        L = laporan_lengkap()
        self.assertEqual(L["per_versi"]["v3"]["sinyal"],
                         {"cakupan": 0.6667, "jujur": 1.0, "aman": 1.0})


if __name__ == "__main__":
    unittest.main()
