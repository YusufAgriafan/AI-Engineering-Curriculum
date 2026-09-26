"""Spesifikasi Bagian 2 — Retriever + evaluasi hit@k (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/retriever.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.data import KUERI_UJI
from ragkit.embed import EncoderEmbedding
from ragkit.retriever import buat_retriever, evaluasi_retrieval


class TestBuatRetriever(unittest.TestCase):
    def test_retrieve_bentuk_hasil(self):
        retrieve = buat_retriever()
        hasil = retrieve("Berapa hari cuti tahunan untuk karyawan tetap?", 3)
        self.assertEqual(len(hasil), 3)
        self.assertEqual({len(h["chunk"]["text"]) > 0 for h in hasil}, {True})
        self.assertTrue(all(0.0 <= h["score"] <= 1.0 for h in hasil))

    def test_angka_terkunci_q1(self):
        retrieve = buat_retriever()
        hasil = retrieve("Berapa hari cuti tahunan untuk karyawan tetap?", 3)
        self.assertEqual([h["chunk"]["chunk_index"] for h in hasil], [5, 0, 4])
        self.assertAlmostEqual(hasil[0]["score"], 0.5868, places=3)

    def test_encoder_kustom(self):
        enc = EncoderEmbedding(dim=128)
        retrieve = buat_retriever(encoder=enc)
        hasil = retrieve("cuti sakit", 2)
        self.assertEqual(len(hasil), 2)


class TestEvaluasiRetrieval(unittest.TestCase):
    def test_angka_terkunci_hit3(self):
        retrieve = buat_retriever()
        ev = evaluasi_retrieval(retrieve, top_k=3)
        self.assertAlmostEqual(ev["hit_rate"], 1.0, places=9)
        self.assertEqual(ev["kueri_gagal"], [])

    def test_angka_terkunci_hit1(self):
        retrieve = buat_retriever()
        ev = evaluasi_retrieval(retrieve, top_k=1)
        self.assertAlmostEqual(ev["hit_rate"], 0.8, places=9)
        self.assertEqual(ev["kueri_gagal"], ["q1"])

    def test_posisi_1based(self):
        retrieve = buat_retriever()
        ev = evaluasi_retrieval(retrieve, top_k=3)
        by_id = {p["id"]: p for p in ev["per_kueri"]}
        # q1: chunk FAQ (index 5) menang top-1 → dokumen kebijakan di posisi 2
        self.assertEqual(by_id["q1"]["posisi"], 2)
        self.assertEqual(by_id["q2"]["posisi"], 1)
        self.assertEqual(by_id["q3"]["posisi"], 1)

    def test_hit_rate_kosong(self):
        retrieve = buat_retriever()
        ev = evaluasi_retrieval(retrieve, kueri_uji=[], top_k=3)
        self.assertEqual(ev["hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
