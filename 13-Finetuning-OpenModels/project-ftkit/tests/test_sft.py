"""Spesifikasi Bagian 1 — Dataset SFT ala Alpaca (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ftkit/sft.py lolos.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftkit.data import FORMAT_SFT
from ftkit.sft import ke_contoh_sft, muat_jsonl, oversample, split_sft, tulis_jsonl


class TestKeContohSft(unittest.TestCase):
    def test_format_prompt(self):
        prompt, respons = ke_contoh_sft(FORMAT_SFT[0])
        self.assertIn(FORMAT_SFT[0]["instruksi"], prompt)
        self.assertIn("Input: " + FORMAT_SFT[0]["input"], prompt)
        self.assertTrue(prompt.rstrip().endswith("Jawaban:"))
        self.assertEqual(respons, "positif")

    def test_semua_baris_konsisten(self):
        for b in FORMAT_SFT:
            prompt, respons = ke_contoh_sft(b)
            self.assertIn(b["input"], prompt)
            self.assertEqual(respons, b["respons"])


class TestJsonl(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.jsonl"
            tulis_jsonl(p, FORMAT_SFT)
            hasil = muat_jsonl(p)
        self.assertEqual(len(hasil), len(FORMAT_SFT))
        self.assertEqual(hasil[0]["input"], FORMAT_SFT[0]["input"])
        self.assertEqual(hasil[-1]["respons"], FORMAT_SFT[-1]["respons"])

    def test_baris_kosong_dilewati(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.jsonl"
            p.write_text(
                '{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
            hasil = muat_jsonl(p)
        self.assertEqual(len(hasil), 2)


class TestOversample(unittest.TestCase):
    def test_angka_terkunci_per_label_6(self):
        hasil = oversample(FORMAT_SFT, per_label=6)
        n_pos = sum(1 for b in hasil if b["respons"] == "positif")
        n_neg = sum(1 for b in hasil if b["respons"] == "negatif")
        self.assertEqual((n_pos, n_neg), (6, 6))
        self.assertEqual(len(hasil), 12)

    def test_sudah_seimbang_tidak_diubah(self):
        seimbang = [b for b in FORMAT_SFT if b["respons"] == "positif"][:2] + \
                   [b for b in FORMAT_SFT if b["respons"] == "negatif"][:2]
        hasil = oversample(seimbang, per_label=2)
        self.assertEqual(len(hasil), 4)

    def test_original_pertahannya(self):
        hasil = oversample(FORMAT_SFT, per_label=6)
        self.assertEqual(hasil[0], FORMAT_SFT[0])   # urutan asli dipertahankan


class TestSplitSft(unittest.TestCase):
    def test_angka_terkunci_8_contoh(self):
        train, val = split_sft(FORMAT_SFT, rasio_val=0.25)
        self.assertEqual((len(train), len(val)), (5, 3))

    def test_total_terkirim(self):
        train, val = split_sft(FORMAT_SFT, rasio_val=0.25)
        self.assertEqual(len(train) + len(val), len(FORMAT_SFT))

    def test_rasio_nol(self):
        train, val = split_sft(FORMAT_SFT, rasio_val=0.0)
        self.assertEqual((len(train), len(val)), (8, 0))


if __name__ == "__main__":
    unittest.main()
