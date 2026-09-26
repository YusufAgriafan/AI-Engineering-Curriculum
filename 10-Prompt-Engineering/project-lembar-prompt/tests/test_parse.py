"""Spesifikasi Bagian 5 — Parse output berisik & perbaikan JSON (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di plib/parse.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plib.data import GOLDEN, PROMPT_PRODUKSI, PROMPT_TERSTRUKTUR
from plib.parse import ekstrak_json, naive_json_ok, parse_output, perbaiki_json
from plib.template import render
from plib.mock_llm import panggil

TEKS1 = GOLDEN[0]["teks"]
BENAR1 = GOLDEN[0]["expected"]


class TestNaiveJson(unittest.TestCase):
    def test_json_murni(self):
        self.assertTrue(naive_json_ok('{"a": 1}'))

    def test_code_fence_gagal_mentah(self):
        self.assertFalse(naive_json_ok('```json\n{"a": 1}\n```'))

    def test_prosa_gagal_mentah(self):
        self.assertFalse(naive_json_ok('Tentu! {"a": 1} semoga membantu'))

    def test_bukan_json(self):
        self.assertFalse(naive_json_ok("halo dunia"))


class TestPerbaikiJson(unittest.TestCase):
    def test_kutip_tunggal(self):
        self.assertEqual(perbaiki_json("{'a': 1}"), {"a": 1})

    def test_python_literal(self):
        self.assertEqual(perbaiki_json('{"a": True, "b": None}'), {"a": True, "b": None})

    def test_koma_menggantung(self):
        self.assertEqual(perbaiki_json('{"a": 1, "b": 2,}'), {"a": 1, "b": 2})

    def test_komentar(self):
        self.assertEqual(perbaiki_json('{"a": 1} // catatan'), {"a": 1})

    def test_gabungan_semua(self):
        self.assertEqual(perbaiki_json("{'a': True, // x\n 'b': 2,}"), {"a": True, "b": 2})

    def test_json_biasa_tetap_jalan(self):
        self.assertEqual(perbaiki_json('{"a": [1, 2]}'), {"a": [1, 2]})


class TestEkstrakJson(unittest.TestCase):
    def test_murni(self):
        self.assertEqual(ekstrak_json('{"a": 1}'), {"a": 1})

    def test_code_fence(self):
        self.assertEqual(ekstrak_json('bla\n```json\n{"a": 1}\n```\nbla'), {"a": 1})

    def test_prosa_sekitar(self):
        self.assertEqual(ekstrak_json('Tentu! Berikut hasilnya {"a": 1}. Semoga membantu!'), {"a": 1})

    def test_nested(self):
        self.assertEqual(ekstrak_json('x {"a": {"b": [1, 2]}} y'), {"a": {"b": [1, 2]}})

    def test_kurung_di_dalam_string(self):
        # brace counter naif akan pecah di sini
        self.assertEqual(ekstrak_json('x {"a": "}{"} y'), {"a": "}{"})

    def test_get_pertama_saja(self):
        self.assertEqual(ekstrak_json('{"a": 1} {"b": 2}'), {"a": 1})

    def test_tanpa_json_error(self):
        with self.assertRaises(ValueError):
            ekstrak_json("Sepertinya pelanggan membahas iPhone 15.")

    def test_kurung_timpang_error(self):
        with self.assertRaises(ValueError):
            ekstrak_json("{ ini bukan json }")

    def test_perbaiki_dipakai_saat_dibutuhkan(self):
        self.assertEqual(ekstrak_json("Hasil: {'a': 1}"), {"a": 1})


class TestParseOutput(unittest.TestCase):
    def test_json_sesuai_skema(self):
        self.assertEqual(parse_output('{"a": 1}', {"type": "object"}), {"a": 1})

    def test_output_mock_terstruktur_bisa_diparse(self):
        prompt = render(PROMPT_TERSTRUKTUR, teks=TEKS1)
        out = panggil(prompt, TEKS1)
        self.assertFalse(naive_json_ok(out))          # butuh pembersihan...
        data = parse_output(out, None)                # ...tapi tetap bisa dipakai
        self.assertEqual(data["product_name"], "iPhone 15")
        self.assertEqual(data["price"], 25000000)
        # isinya belum benar: prompt ini belum mengatur nilai kosong -> halusinasi
        self.assertEqual(data["shipping_estimate"], "1-3 hari")

    def test_output_mock_produksi_langsung_bersih(self):
        prompt = render(PROMPT_PRODUKSI, dokumen=TEKS1)
        out = panggil(prompt, TEKS1)
        self.assertTrue(naive_json_ok(out))
        self.assertEqual(parse_output(out, None), BENAR1)

    def test_skema_dilanggar(self):
        with self.assertRaises(ValueError):
            parse_output('{"a": 1}', {"type": "object", "required": ["b"]})

    def test_tanpa_json(self):
        with self.assertRaises(ValueError):
            parse_output("tidak ada apa-apa di sini", None)


if __name__ == "__main__":
    unittest.main()
