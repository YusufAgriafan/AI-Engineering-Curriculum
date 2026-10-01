"""Spesifikasi Bagian 3 — Runner: jalankan evalset → hasil + agregat (TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di evalkit/runner.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.data import EVALSET, KATEGORI
from evalkit.runner import agregat, jalankan_eval, jalankan_satu


class TestJalankanSatu(unittest.TestCase):
    def test_angka_terkunci_v3_e01(self):
        kasus = EVALSET[0]  # e01 kebijakan, kunci '14 hari'
        h = jalankan_satu("v3", kasus)
        self.assertEqual(h["id"], "e01")
        self.assertEqual(h["kategori"], "kebijakan")
        self.assertEqual(h["skor"]["utama"], 1.0)
        self.assertEqual(h["skor_retrieval"], 0.5)
        self.assertIn("14 hari", h["jawaban"])
        self.assertEqual(h["meta"]["versi"], "v3")
        self.assertIsInstance(h["latensi_ms"], float)
        self.assertGreaterEqual(h["latensi_ms"], 0.0)

    def test_angka_terkunci_v1_e07_karangan(self):
        kasus = EVALSET[6]  # e07 abstain
        h = jalankan_satu("v1", kasus)
        self.assertEqual(h["skor"]["utama"], 0.0)
        self.assertIn("customer service", h["jawaban"])

    def test_meta_tersalin(self):
        h = jalankan_satu("v3", EVALSET[8])  # e09 injection
        self.assertTrue(h["meta"].get("injection"))


class TestJalankanEval(unittest.TestCase):
    def test_angka_terkunci_v1(self):
        run = jalankan_eval("v1")
        self.assertEqual(run["versi"], "v1")
        self.assertEqual(run["agregat"]["n"], 12)
        self.assertEqual(run["agregat"]["skor_rata"], 0.3333)
        self.assertEqual(run["agregat"]["n_abis"], 0)

    def test_angka_terkunci_v2(self):
        run = jalankan_eval("v2")
        self.assertEqual(run["agregat"]["skor_rata"], 0.6667)
        self.assertEqual(run["agregat"]["n_abis"], 5)

    def test_angka_terkunci_v3(self):
        run = jalankan_eval("v3")
        self.assertEqual(run["agregat"]["skor_rata"], 0.8333)
        self.assertEqual(run["agregat"]["n_abis"], 5)

    def test_angka_terkunci_per_kategori_v3(self):
        per = jalankan_eval("v3")["agregat"]["per_kategori"]
        self.assertEqual(per["kebijakan"], {"n": 4, "skor_rata": 1.0})
        self.assertEqual(per["status"], {"n": 2, "skor_rata": 0.0})
        self.assertEqual(per["injection"], {"n": 2, "skor_rata": 1.0})

    def test_evalset_kustom(self):
        run = jalankan_eval("v3", evalset=[EVALSET[0]])
        self.assertEqual(run["agregat"]["n"], 1)


class TestAgregat(unittest.TestCase):
    def test_kosong(self):
        agg = agregat([])
        self.assertEqual(agg["n"], 0)
        self.assertEqual(agg["skor_rata"], 0.0)
        self.assertEqual(agg["per_kategori"], {})
        self.assertEqual(agg["n_abis"], 0)

    def test_angka_terkunci_per_kategori_v1(self):
        per = jalankan_eval("v1")["agregat"]["per_kategori"]
        self.assertEqual(per["abstain"], {"n": 2, "skor_rata": 0.0})
        self.assertEqual(per["hallucination"], {"n": 1, "skor_rata": 0.0})
        self.assertEqual(per["edge"], {"n": 1, "skor_rata": 0.0})

    def test_n_abis_menghitung_abstain_score(self):
        # v2: e07, e08, e11, e12 abstain + e05/e06 juga abstain (status skor 0)
        hasil = jalankan_eval("v2")["hasil"]
        n_abis = sum(1 for h in hasil if h["skor"]["abstain"] == 1.0)
        self.assertEqual(n_abis, 5)


if __name__ == "__main__":
    unittest.main()
