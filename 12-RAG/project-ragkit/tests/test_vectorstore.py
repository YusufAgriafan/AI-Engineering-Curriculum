"""Spesifikasi Bagian 1 — VectorStore cosine similarity (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/vectorstore.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.chunking import buat_semua_chunks
from ragkit.embed import EncoderEmbedding
from ragkit.vectorstore import VectorStore


def _store_contoh():
    chunks = buat_semua_chunks()
    enc = EncoderEmbedding()
    return chunks, enc, enc.embed([c["text"] for c in chunks])


class TestTambahDanCari(unittest.TestCase):
    def test_add_lalu_ukuran(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        self.assertEqual(len(s.chunks), 7)
        self.assertEqual(len(s.matriks), 7)

    def test_cari_mengembalikan_top_k(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        qv = enc.embed_query("Berapa hari cuti tahunan untuk karyawan tetap?")
        hasil = s.cari(qv, top_k=3)
        self.assertEqual(len(hasil), 3)
        for r in hasil:
            self.assertIn("chunk", r)
            self.assertIn("score", r)
            self.assertIn("index", r)

    def test_skor_tertinggi_pertama(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        qv = enc.embed_query("Berapa hari cuti tahunan untuk karyawan tetap?")
        skor = [r["score"] for r in s.cari(qv, top_k=7)]
        self.assertEqual(skor, sorted(skor, reverse=True))

    def test_skor_terbaik_relevan_tinggi(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        qv = enc.embed_query("Berapa hari cuti tahunan untuk karyawan tetap?")
        terbaik = s.cari(qv, top_k=1)[0]
        self.assertGreater(terbaik["score"], 0.5)

    def test_top_k_lebih_besar_dari_jumlah(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        qv = enc.embed_query("cuti")
        hasil = s.cari(qv, top_k=99)
        self.assertEqual(len(hasil), 7)

    def test_tie_dipecah_index_kecil(self):
        s = VectorStore()
        v = [1.0, 0.0, 0.0]
        chunks = [{"chunk_index": 0, "text": "a", "source": "s", "page": 1,
                   "doc_id": "d"},
                  {"chunk_index": 1, "text": "b", "source": "s", "page": 1,
                   "doc_id": "d"}]
        s.add(chunks, [v, v])
        hasil = s.cari([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual([r["index"] for r in hasil], [0, 1])

    def test_store_kosong_tidak_crash(self):
        s = VectorStore()
        self.assertEqual(s.cari([1.0, 2.0], top_k=3), [])

    def test_cosine_eksplisit_tak_ternormalisasi(self):
        s = VectorStore()
        chunks = [{"chunk_index": 0, "text": "a", "source": "s", "page": 1,
                   "doc_id": "d"}]
        s.add(chunks, [[2.0, 0.0]])
        r = s.cari([5.0, 0.0], top_k=1)[0]
        self.assertAlmostEqual(r["score"], 1.0, places=9)


class TestKonsistensiKorpus(unittest.TestCase):
    def test_setiap_kueri_emas_kena_top1_atau_top2(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        from ragkit.data import KUERI_UJI
        for q in KUERI_UJI:
            sumber = [r["chunk"]["source"]
                      for r in s.cari(enc.embed_query(q["teks"]), top_k=2)]
            self.assertIn(q["source_emas"], sumber, msg=q["id"])

    def test_chunk_index_konsisten_dengan_posisi(self):
        chunks, enc, M = _store_contoh()
        s = VectorStore()
        s.add(chunks, M)
        qv = enc.embed_query("cuti sakit dokter")
        for r in s.cari(qv, top_k=5):
            self.assertEqual(r["chunk"]["chunk_index"], r["index"])


if __name__ == "__main__":
    unittest.main()
