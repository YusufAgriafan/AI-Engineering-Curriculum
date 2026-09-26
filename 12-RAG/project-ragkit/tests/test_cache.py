"""Spesifikasi Bagian 4 — Cache konteks retrieval (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/cache.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.cache import CacheRetrieval, retrieve_bercache
from ragkit.retriever import buat_retriever


class TestCacheRetrieval(unittest.TestCase):
    def setUp(self):
        self.c = CacheRetrieval(ttl=10)

    def test_miss_lalu_hit(self):
        self.assertIsNone(self.c.ambil("k", 0))
        self.c.simpan("k", ["chunk"], 0)
        self.assertEqual(self.c.ambil("k", 1), ["chunk"])

    def test_kunci_dinormalisasi(self):
        self.assertEqual(self.c.kunci("  Halo Dunia "), "halo dunia")
        self.assertEqual(self.c.kunci("HALO"), self.c.kunci("halo"))

    def test_kedaluwarsa_pada_batas_ttl(self):
        self.c.simpan("k", ["chunk"], 0)
        self.assertEqual(self.c.ambil("k", 9), ["chunk"])    # 9 < 10
        self.assertIsNone(self.c.ambil("k", 10))             # 10 >= 10

    def test_expired_bukan_miss(self):
        self.c.simpan("k", ["chunk"], 0)
        self.c.ambil("k", 10)
        s = self.c.statistik()
        self.assertEqual((s["hit"], s["miss"], s["expired"]), (0, 0, 1))

    def test_statistik_hit_rate(self):
        self.c.ambil("x", 0)              # miss
        self.c.simpan("x", [1], 0)
        self.c.ambil("x", 1)              # hit
        s = self.c.statistik()
        self.assertEqual((s["hit"], s["miss"]), (1, 1))
        self.assertAlmostEqual(s["hit_rate"], 0.5, places=9)

    def test_hit_rate_kosong(self):
        self.assertEqual(CacheRetrieval().statistik()["hit_rate"], 0.0)


class TestRetrieveBercache(unittest.TestCase):
    def test_pola_miss_hit_hit_miss(self):
        retrieve = buat_retriever()
        c = CacheRetrieval(ttl=10)
        pola, panggilan = [], {"n": 0}

        def retrieve_hitung(kueri, top_k=3):
            panggilan["n"] += 1
            return retrieve(kueri, top_k)

        aturan = ["Berapa hari cuti tahunan untuk karyawan tetap?",
                  "berapa hari cuti tahunan untuk karyawan tetap?",   # sama (kunci lower)
                  "Berapa hari cuti tahunan untuk karyawan tetap?",
                  "Berapa lama maksimal cuti sakit tanpa surat dokter?"]
        for i, q in enumerate(aturan):
            r = retrieve_bercache(retrieve_hitung, q, c, sekarang=i, top_k=3)
            pola.append(r["cache"])
        self.assertEqual(pola, ["miss", "hit", "hit", "miss"])
        self.assertEqual(panggilan["n"], 2)

    def test_hit_mengembalikan_chunks_sama(self):
        retrieve = buat_retriever()
        c = CacheRetrieval(ttl=10)
        q = "Berapa hari cuti tahunan untuk karyawan tetap?"
        r1 = retrieve_bercache(retrieve, q, c, sekarang=0)
        r2 = retrieve_bercache(retrieve, q, c, sekarang=1)
        self.assertEqual(r1["hasil"], r2["hasil"])

    def test_latensi_hit_lebih_cepat(self):
        retrieve = buat_retriever()
        c = CacheRetrieval(ttl=10)
        q = "cuti sakit"
        r1 = retrieve_bercache(retrieve, q, c, sekarang=0)
        r2 = retrieve_bercache(retrieve, q, c, sekarang=1)
        self.assertEqual(r1["latensi_ms"], 16)     # 2 * 8 (miss)
        self.assertEqual(r2["latensi_ms"], 2)      # 2 (hit)

    def test_hit_tidak_melakukan_pencarian(self):
        retrieve = buat_retriever()
        c = CacheRetrieval(ttl=10)
        q = "cuti sakit"
        retrieve_bercache(retrieve, q, c, sekarang=0)
        dipanggil = {"n": 0}

        def retrieve_hitung(kueri, top_k=3):
            dipanggil["n"] += 1
            return retrieve(kueri, top_k)

        retrieve_bercache(retrieve_hitung, q, c, sekarang=1)
        self.assertEqual(dipanggil["n"], 0)


if __name__ == "__main__":
    unittest.main()
