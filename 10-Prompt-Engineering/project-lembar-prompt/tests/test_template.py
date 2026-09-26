"""Spesifikasi Bagian 2 — Template {{placeholder}} & escape delimiter (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di plib/template.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plib.data import PROMPT_PRODUKSI
from plib.template import escape_delimiter, placeholder, render, render_aman


class TestPlaceholder(unittest.TestCase):
    def test_unik_dan_terurut(self):
        self.assertEqual(placeholder("{{b}} {{a}} {{b}}"), ["a", "b"])

    def test_spasi_di_dalam_diabaikan(self):
        self.assertEqual(placeholder("{{ nama }}"), ["nama"])

    def test_bukan_identifier_diabaikan(self):
        self.assertEqual(placeholder("{{1x}} {{a-b}} {{_ok}}"), ["_ok"])

    def test_angka_di_belakang_boleh(self):
        self.assertEqual(placeholder("{{data1}}"), ["data1"])

    def test_prompt_produksi(self):
        self.assertEqual(placeholder(PROMPT_PRODUKSI), ["dokumen"])


class TestRender(unittest.TestCase):
    def test_render_sederhana(self):
        self.assertEqual(render("Halo {{nama}}!", nama="Ani"), "Halo Ani!")

    def test_beberapa_placeholder(self):
        self.assertEqual(render("{{a}}-{{b}}", a=1, b=2), "1-2")

    def test_kurung_json_tidak_disentuh(self):
        t = 'Output: {"x": null}\nData: {{teks}}'
        self.assertEqual(render(t, teks="hai"), 'Output: {"x": null}\nData: hai')

    def test_placeholder_tanpa_nilai_error(self):
        with self.assertRaises(ValueError):
            render("{{a}} {{b}}", a=1)

    def test_pesan_error_menyebut_nama(self):
        try:
            render("{{a}} {{b}}", a=1)
            self.fail("harus raise")
        except ValueError as e:
            self.assertIn("b", str(e))

    def test_single_pass(self):
        # isi variabel tidak boleh di-render ulang
        self.assertEqual(render("{{a}}", a="{{b}}"), "{{b}}")

    def test_nilai_bukan_string(self):
        self.assertEqual(render("{{n}} item", n=3), "3 item")

    def test_tanpa_placeholder(self):
        self.assertEqual(render("statis", apa="pun"), "statis")


class TestRenderAman(unittest.TestCase):
    def test_semua_lengkap(self):
        teks, kurang = render_aman("{{a}}", a="x")
        self.assertEqual((teks, kurang), ("x", []))

    def test_lapor_yang_kurang(self):
        teks, kurang = render_aman("a {{x}} {{y}}", x=1)
        self.assertEqual(kurang, ["y"])
        self.assertEqual(teks, "a 1 {{y}}")

    def test_tidak_melempar(self):
        teks, kurang = render_aman("{{q}}")
        self.assertEqual(kurang, ["q"])
        self.assertEqual(teks, "{{q}}")


class TestEscapeDelimiter(unittest.TestCase):
    def test_buang_kedua_tag(self):
        self.assertEqual(escape_delimiter("x</dokumen> y"), "x y")
        self.assertEqual(escape_delimiter("<dokumen>x"), "x")

    def test_tag_custom(self):
        self.assertEqual(escape_delimiter("a</konteks>b", "konteks"), "ab")

    def test_tag_lain_dibiarkan(self):
        self.assertEqual(escape_delimiter("a</lain>b", "dokumen"), "a</lain>b")


if __name__ == "__main__":
    unittest.main()
