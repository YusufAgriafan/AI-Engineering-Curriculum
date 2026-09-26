"""Spesifikasi Bagian 1 — Anatomi prompt, bagian wajib, anggaran token (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di plib/sections.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plib.data import BAGIAN, PROMPT_NAIF, PROMPT_PRODUKSI
from plib.sections import (bagian_ditemukan, bagian_hilang, build_prompt,
                           estimasi_token, potong_budget, prompt_lengkap)

PROMPT_SEPI = "[ROLE]\nKamu asisten.\n\n[TUGAS]\nJawab singkat."


def _build_contoh_lengkap(contoh=""):
    return build_prompt(
        role="Kamu asisten ekstraksi data.",
        konteks="<dokumen>\n{{dokumen}}\n</dokumen>",
        tugas="Ekstrak pesanan.",
        format_rules="Balas JSON saja.",
        constraints="Jangan mengarang.",
        pertanyaan="Jawab sekarang.",
        contoh=contoh,
    )


class TestBagian(unittest.TestCase):
    def test_urutan_kanonik(self):
        p = _build_contoh_lengkap(contoh="Input: a\nOutput: {}")
        self.assertEqual(bagian_ditemukan(p), BAGIAN)

    def test_bagian_hilang_untuk_prompt_sepi(self):
        self.assertEqual(bagian_hilang(PROMPT_SEPI), ["KONTEKS", "FORMAT", "CONSTRAINT", "INPUT"])

    def test_bagian_ditemukan_prompt_sepi(self):
        self.assertEqual(bagian_ditemukan(PROMPT_SEPI), ["ROLE", "TUGAS"])

    def test_prompt_lengkap(self):
        self.assertFalse(prompt_lengkap(PROMPT_SEPI))
        self.assertTrue(prompt_lengkap(_build_contoh_lengkap()))

    def test_contoh_tidak_dihitung_wajib(self):
        # CONTOH opsional: absennya tidak membuat prompt "tidak lengkap"
        p = _build_contoh_lengkap(contoh="")
        self.assertTrue(prompt_lengkap(p))
        self.assertEqual(bagian_ditemukan(p), ["ROLE", "KONTEKS", "TUGAS", "FORMAT",
                                               "CONSTRAINT", "INPUT"])

    def test_kata_kunci_dua_arah(self):
        # kaya dan miskin harus dibedakan, bukan sekadar "ada [ROLE]"
        self.assertLess(len(bagian_ditemukan(PROMPT_SEPI)), len(bagian_ditemukan(_build_contoh_lengkap())))


class TestBuildPrompt(unittest.TestCase):
    def test_urutan_bagian(self):
        p = _build_contoh_lengkap(contoh="Input: a\nOutput: {}")
        posisi = [p.index(f"[{b}]") for b in BAGIAN]
        self.assertEqual(posisi, sorted(posisi))

    def test_bagian_kosong_dilewati(self):
        p = build_prompt(role="r", konteks="", tugas="t", format_rules="f",
                         constraints="c", pertanyaan="i")
        self.assertNotIn("[KONTEKS]", p)
        self.assertNotIn("[CONTOH]", p)

    def test_isi_dirapikan(self):
        p = build_prompt(role="  r  ", konteks="k", tugas="t", format_rules="f",
                         constraints="c", pertanyaan="i")
        self.assertTrue(p.startswith("[ROLE]\nr"))

    def test_bukan_str_format(self):
        # prompt dengan kurung kurawal JSON harus lolos apa adanya
        p = _build_contoh_lengkap(contoh='Input: "a"\nOutput: {"x": null}')
        self.assertIn('{"x": null}', p)


class TestTokenBudget(unittest.TestCase):
    def test_estimasi_naif(self):
        self.assertEqual(estimasi_token(""), 0)
        self.assertEqual(estimasi_token("abcd"), 1)
        self.assertEqual(estimasi_token("abcde"), 2)

    def test_angka_terkunci(self):
        self.assertEqual(estimasi_token(PROMPT_NAIF), 24)
        self.assertEqual(estimasi_token(PROMPT_PRODUKSI), 277)

    def test_potong_tidak_perlu(self):
        self.assertEqual(potong_budget("aaa bbb", 10), "aaa bbb")

    def test_potong_di_batas_kata(self):
        self.assertEqual(potong_budget("aaa bbb ccc ddd eee", 3), "aaa bbb ccc")

    def test_potong_lebih_pendek_dari_anggaran(self):
        h = potong_budget("aaa bbb ccc ddd eee", 3)
        self.assertLessEqual(len(h), 3 * 4)


if __name__ == "__main__":
    unittest.main()
