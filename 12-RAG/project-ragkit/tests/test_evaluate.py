"""Spesifikasi Bagian 8 — Evaluasi ujung-ke-ujung (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/evaluate.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.evaluate import evaluasi_ujung_ke_ujung
from ragkit.pipeline import jalankan_rag
from ragkit.retriever import buat_retriever


class TestEvaluasiUjungKeUjung(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # staticmethod: fungsi biasa di class attribute akan jadi bound method
        cls.retrieve = staticmethod(buat_retriever())
        cls.ev = evaluasi_ujung_ke_ujung(
            lambda q: jalankan_rag(cls.retrieve, q))

    def test_angka_terkunci_akurasi(self):
        self.assertAlmostEqual(self.ev["akurasi"], 1.0, places=9)

    def test_angka_terkunci_abstain(self):
        self.assertTrue(self.ev["abstain_benar"])
        self.assertEqual(self.ev["jawaban_luar"], "tidak ada di dokumen")

    def test_per_kueri_lengkap(self):
        self.assertEqual(len(self.ev["per_kueri"]), 5)
        self.assertTrue(all(p["benar"] for p in self.ev["per_kueri"]))
        self.assertEqual([p["id"] for p in self.ev["per_kueri"]],
                         ["q1", "q2", "q3", "q4", "q5"])

    def test_gate_bukan_satu2nya_pertahanan(self):
        # ambang -1 mematikan gate; generator ekstraktif tetap abstain untuk
        # kueri luar korpus — dua lapis pertahanan yang saling mengisi.
        ev = evaluasi_ujung_ke_ujung(
            lambda q: jalankan_rag(self.retrieve, q, ambang=-1.0))
        self.assertTrue(ev["abstain_benar"])

    def test_kueri_uji_kustom(self):
        ev = evaluasi_ujung_ke_ujung(
            lambda q: jalankan_rag(self.retrieve, q),
            kueri_uji=[{"id": "x1", "teks": "Berapa hari cuti melahirkan?",
                        "jawaban_emas": "90"}])
        self.assertAlmostEqual(ev["akurasi"], 1.0, places=9)
        self.assertEqual([p["id"] for p in ev["per_kueri"]], ["x1"])


if __name__ == "__main__":
    unittest.main()
