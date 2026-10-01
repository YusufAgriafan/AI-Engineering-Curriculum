"""Spesifikasi Bagian 1 — Evalset: muat, validasi, split (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di evalkit/evalset.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.data import EVALSET
from evalkit.evalset import hitung_kategori, split_dev_test, validasi_evalset


class TestValidasiEvalset(unittest.TestCase):
    def test_evalset_resmi_valid(self):
        ok, pesan = validasi_evalset(EVALSET)
        self.assertTrue(ok, pesan)
        self.assertEqual(pesan, "")

    def test_bukan_list_dan_kosong(self):
        self.assertFalse(validasi_evalset([])[0])
        self.assertFalse(validasi_evalset("bukan")[0])

    def test_id_duplikat_ditolak(self):
        duplikat = [dict(EVALSET[0]), dict(EVALSET[0])]
        ok, pesan = validasi_evalset(duplikat)
        self.assertFalse(ok)
        self.assertIn("duplikat", pesan)

    def test_kategori_tak_dikenal(self):
        kasus = [{"id": "x1", "kategori": "apsing", "tanya": "q", "kunci": None}]
        ok, pesan = validasi_evalset(kasus)
        self.assertFalse(ok)
        self.assertIn("kategori", pesan)

    def test_tanya_kosong_non_edge_ditolak(self):
        kasus = [{"id": "x1", "kategori": "kebijakan", "tanya": "   ", "kunci": "k"}]
        ok, pesan = validasi_evalset(kasus)
        self.assertFalse(ok)
        self.assertIn("tanya", pesan)

    def test_edge_boleh_kosong(self):
        kasus = [{"id": "x1", "kategori": "edge", "tanya": "  ", "kunci": None}]
        self.assertTrue(validasi_evalset(kasus)[0])

    def test_kunci_terisi_pada_abstain_ditolak(self):
        kasus = [{"id": "x1", "kategori": "abstain", "tanya": "q", "kunci": "14 hari"}]
        ok, pesan = validasi_evalset(kasus)
        self.assertFalse(ok)
        self.assertIn("None", pesan)

    def test_kunci_terisi_pada_injection_ditolak(self):
        kasus = [{"id": "x1", "kategori": "injection", "tanya": "q", "kunci": "x"}]
        self.assertFalse(validasi_evalset(kasus)[0])

    def test_kunci_wajib_ada_pada_kebijakan_dengan_kunci(self):
        kasus = [{"id": "x1", "kategori": "kebijakan", "tanya": "q", "kunci": "k"}]
        self.assertTrue(validasi_evalset(kasus)[0])


class TestHitungKategori(unittest.TestCase):
    def test_angka_terkunci_evalset_resmi(self):
        self.assertEqual(hitung_kategori(EVALSET),
                         {"kebijakan": 4, "status": 2, "abstain": 2,
                          "injection": 2, "hallucination": 1, "edge": 1})

    def test_semua_kategori_muncul(self):
        hitung = hitung_kategori([EVALSET[0]])
        self.assertEqual(set(hitung), set(hitung_kategori(EVALSET)))
        self.assertEqual(hitung["status"], 0)


class TestSplitDevTest(unittest.TestCase):
    def test_angka_terkunci_rasio_default(self):
        dev, test = split_dev_test(EVALSET)
        self.assertEqual([k["id"] for k in test], ["e01", "e02"])
        self.assertEqual(len(dev), 10)

    def test_angka_terkunci_rasio_05(self):
        dev, test = split_dev_test(EVALSET, rasio_test=0.5)
        self.assertEqual((len(dev), len(test)), (7, 5))
        self.assertEqual([k["id"] for k in test],
                         ["e01", "e02", "e03", "e04", "e05"])

    def test_split_melengkapi_tanpa_overlap(self):
        dev, test = split_dev_test(EVALSET)
        id_dev = {k["id"] for k in dev}
        id_test = {k["id"] for k in test}
        self.assertEqual(id_dev | id_test, {k["id"] for k in EVALSET})
        self.assertEqual(id_dev & id_test, set())

    def test_urutan_asli_terjaga(self):
        dev, test = split_dev_test(EVALSET)
        for kasus in dev + test:
            self.assertIn(kasus, EVALSET)

    def test_tanpa_random_deterministik(self):
        d1, t1 = split_dev_test(EVALSET)
        d2, t2 = split_dev_test(EVALSET)
        self.assertEqual(d1, d2)
        self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()
