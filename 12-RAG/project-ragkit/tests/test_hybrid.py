"""Spesifikasi Bagian 5 — Hybrid search: BM25 mini + RRF (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/hybrid.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.chunking import buat_semua_chunks
from ragkit.data import KUERI_UJI
from ragkit.hybrid import cari_bm25, cari_hybrid, gabung_rrf
from ragkit.retriever import buat_retriever


class TestBM25(unittest.TestCase):
    def test_exact_match_menang(self):
        chunks = buat_semua_chunks()
        hasil = cari_bm25("iPhone 15", chunks, top_k=3)
        # korpus tidak punya 'iphone' → BM25 tetap jalan, skor semua 0 → index kecil
        self.assertEqual(len(hasil), 3)

    def test_bm25_kena_kata_kunci(self):
        chunks = buat_semua_chunks()
        hasil = cari_bm25("surat dokter cuti sakit", chunks, top_k=3)
        self.assertEqual(hasil[0]["chunk"]["source"], "kebijakan-cuti-kesehatan.pdf")

    def test_bm25_korpus_kosong(self):
        self.assertEqual(cari_bm25("apa saja", [], top_k=3), [])

    def test_bm25_skor_tertinggi_pertama(self):
        chunks = buat_semua_chunks()
        hasil = cari_bm25("cuti melahirkan", chunks, top_k=5)
        skor = [h["score"] for h in hasil]
        self.assertEqual(skor, sorted(skor, reverse=True))


class TestRRF(unittest.TestCase):
    def test_rrf_dasar(self):
        a = [{"chunk": {"chunk_index": 0}}, {"chunk": {"chunk_index": 1}}]
        b = [{"chunk": {"chunk_index": 1}}, {"chunk": {"chunk_index": 2}}]
        skor = gabung_rrf([a, b], k=60)
        # index 1 menang rank-0 di b dan rank-1 di a → tertinggi
        self.assertEqual(max(skor, key=skor.get), 1)
        self.assertAlmostEqual(skor[1], 1 / 62 + 1 / 61, places=9)
        self.assertAlmostEqual(skor[0], 1 / 61, places=9)
        self.assertAlmostEqual(skor[2], 1 / 62, places=9)

    def test_rrf_dengan_k_kustom(self):
        a = [{"chunk": {"chunk_index": 3}}]
        skor = gabung_rrf([a], k=10)
        self.assertAlmostEqual(skor[3], 1 / 11, places=9)


class TestHybrid(unittest.TestCase):
    def test_hybrid_top_k(self):
        retrieve = buat_retriever()
        chunks = buat_semua_chunks()
        hasil = cari_hybrid(retrieve, chunks,
                            "Berapa hari cuti tahunan untuk karyawan tetap?",
                            top_k=3)
        self.assertEqual(len(hasil), 3)
        self.assertEqual(len({h["chunk"]["chunk_index"] for h in hasil}), 3)

    def test_angka_terkunci_hybrid_q1(self):
        retrieve = buat_retriever()
        chunks = buat_semua_chunks()
        hasil = cari_hybrid(retrieve, chunks,
                            "Berapa hari cuti tahunan untuk karyawan tetap?",
                            top_k=3)
        self.assertEqual([h["chunk"]["chunk_index"] for h in hasil], [5, 0, 4])

    def test_hybrid_hit_rate_100(self):
        retrieve = buat_retriever()
        chunks = buat_semua_chunks()
        hit = 0
        for q in KUERI_UJI:
            sumber = [h["chunk"]["source"]
                      for h in cari_hybrid(retrieve, chunks, q["teks"], top_k=3)]
            hit += q["source_emas"] in sumber
        self.assertEqual(hit, 5)


if __name__ == "__main__":
    unittest.main()
