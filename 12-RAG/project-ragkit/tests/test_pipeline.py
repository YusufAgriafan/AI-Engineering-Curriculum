"""Spesifikasi Bagian 7 — Pipeline RAG end-to-end (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/pipeline.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.data import KUERI_LUAR_CORPUS, SEP_DETEKSI_TIDAK_ADA
from ragkit.pipeline import jalankan_rag
from ragkit.retriever import buat_retriever


class TestJalankanRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # staticmethod: fungsi biasa di class attribute akan jadi bound method
        cls.retrieve = staticmethod(buat_retriever())

    def test_bentuk_hasil_lengkap(self):
        h = jalankan_rag(self.retrieve,
                         "Berapa hari cuti tahunan untuk karyawan tetap?")
        for kunci in ("kueri", "chunks", "jawaban", "sitasi", "sumber",
                      "abstain", "skor_terbaik", "skema", "biaya_usd", "tahap"):
            self.assertIn(kunci, h)

    def test_jawaban_q1_dengan_sitasi(self):
        h = jalankan_rag(self.retrieve,
                         "Berapa hari cuti tahunan untuk karyawan tetap?")
        self.assertIn("12", h["jawaban"])
        self.assertFalse(h["abstain"])
        self.assertEqual(h["sumber"][0]["source"], "kebijakan-ketenagakerjaan.pdf")
        self.assertEqual(h["sumber"][0]["page"], 1)

    def test_skema_memuat_konteks_dan_aturan(self):
        h = jalankan_rag(self.retrieve, "Berapa hari cuti melahirkan?")
        self.assertIn("KONTEKS", h["skema"])
        self.assertIn("tidak ada di dokumen", h["skema"])
        self.assertIn("[1]", h["skema"])

    def test_angka_terkunci_biaya_q1(self):
        h = jalankan_rag(self.retrieve,
                         "Berapa hari cuti tahunan untuk karyawan tetap?")
        self.assertAlmostEqual(h["biaya_usd"], 5.91e-05, places=9)

    def test_angka_terkunci_tahap(self):
        h = jalankan_rag(self.retrieve, "cuti")
        # index 7 chunk → 8 + 2*7 = 22 (+25) ; generate = 50
        self.assertEqual(h["tahap"]["retrieve_ms"], 22 + 25)
        self.assertEqual(h["tahap"]["generate_ms"], 50)

    def test_luar_korpus_diabstainkan(self):
        h = jalankan_rag(self.retrieve, KUERI_LUAR_CORPUS["teks"])
        self.assertTrue(h["abstain"])
        self.assertEqual(h["jawaban"], SEP_DETEKSI_TIDAK_ADA)
        self.assertEqual(h["sitasi"], [])
        # skor retrieval terbaik jauh di bawah ambang 0.30
        self.assertLess(h["skor_terbaik"], 0.30)

    def test_gate_menimpa_jawaban_ketika_skor_rendah(self):
        # skor DI BAWAH ambang tetapi generator BISA menjawab (tidak abstain)
        # → gate wajib menimpa. Konteks dibuat relevan agar generator aktif.
        dapat = self.retrieve("Berapa hari cuti tahunan untuk karyawan tetap?", 1)

        def retrieve_skor_rendah(kueri, top_k=3):
            return [{"chunk": dapat[0]["chunk"], "score": 0.2, "index": 0}]

        h = jalankan_rag(retrieve_skor_rendah,
                         "Berapa hari cuti tahunan untuk karyawan tetap?")
        self.assertTrue(h["abstain"])
        self.assertEqual(h["jawaban"], SEP_DETEKSI_TIDAK_ADA)
        self.assertAlmostEqual(h["abstaikan_oleh"], 0.2, places=4)

    def test_ambang_nol_menonaktifkan_gate_bukan_generator(self):
        # ambang -1 mematikan GATE; kueri luar korpus tetap abstain karena
        # GENERATOR sendiri tidak menemukan kalimat yang menjawab.
        h = jalankan_rag(self.retrieve, KUERI_LUAR_CORPUS["teks"], ambang=-1.0)
        self.assertTrue(h["abstain"])
        self.assertNotIn("abstaikan_oleh", h)   # bukan gate yang menimpa

    def test_skor_terbaik_q1(self):
        h = jalankan_rag(self.retrieve,
                         "Berapa hari cuti tahunan untuk karyawan tetap?")
        self.assertAlmostEqual(h["skor_terbaik"], 0.5868, places=3)

    def test_biaya_nol_bila_teks_kosong(self):
        # kueri kosong: retrieval tetap jalan (semua skor rendah) → abstain;
        # biaya dihitung dari prompt+ jawaban → > 0. Di sini hanya cek >= 0.
        h = jalankan_rag(self.retrieve, "")
        self.assertGreaterEqual(h["biaya_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
