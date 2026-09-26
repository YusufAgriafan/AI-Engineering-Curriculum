"""Spesifikasi Bagian 3 — Abstain gate (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/abstain.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.abstain import abstaikan
from ragkit.data import SEP_DETEKSI_TIDAK_ADA


def _hasil(skor, abstain=False, jawaban="12 hari"):
    return {"jawaban": jawaban, "sitasi": [0], "sumber": [{"n": 1}],
            "abstain": abstain, "skor_terbaik": skor}


class TestAbstaikan(unittest.TestCase):
    def test_skor_di_bawah_ambang_mengabstain(self):
        out = abstaikan(_hasil(0.10), ambang=0.30)
        self.assertEqual(out["jawaban"], SEP_DETEKSI_TIDAK_ADA)
        self.assertEqual(out["sitasi"], [])
        self.assertEqual(out["sumber"], [])
        self.assertTrue(out["abstain"])
        self.assertAlmostEqual(out["abstaikan_oleh"], 0.10, places=4)

    def test_skor_di_ambang_tidak_mengabstain(self):
        h = _hasil(0.30)
        out = abstaikan(h, ambang=0.30)
        self.assertEqual(out["jawaban"], "12 hari")
        self.assertNotIn("abstaikan_oleh", out)
        self.assertFalse(out["abstain"])

    def test_skor_tinggi_utuh(self):
        h = _hasil(0.59)
        out = abstaikan(h, ambang=0.30)
        self.assertEqual(out["jawaban"], "12 hari")
        self.assertEqual(out["sitasi"], [0])

    def test_sudah_abstain_dibiarkan(self):
        h = _hasil(0.29, abstain=True, jawaban=SEP_DETEKSI_TIDAK_ADA)
        out = abstaikan(h, ambang=0.30)
        self.assertTrue(out["abstain"])
        self.assertNotIn("abstaikan_oleh", out)

    def test_tanpa_skor_dianggap_nol(self):
        h = {"jawaban": "x", "sitasi": [0], "sumber": [], "abstain": False}
        out = abstaikan(h, ambang=0.30)
        self.assertTrue(out["abstain"])

    def test_input_tidak_diubah(self):
        h = _hasil(0.10)
        abstaikan(h, ambang=0.30)
        self.assertEqual(h["jawaban"], "12 hari")
        self.assertFalse(h["abstain"])

    def test_skor_luar_korpus_terkunci(self):
        # skor terbaik q6 = 0.065 → di bawah ambang default
        out = abstaikan(_hasil(0.0649), ambang=0.30)
        self.assertTrue(out["abstain"])


if __name__ == "__main__":
    unittest.main()
