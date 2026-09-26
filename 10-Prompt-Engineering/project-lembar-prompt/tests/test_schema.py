"""Spesifikasi Bagian 4 — Validasi JSON-schema mini (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di plib/schema.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plib.data import GOLDEN, SKEMA_PENUH
from plib.schema import valid, validate

BENAR = GOLDEN[0]["expected"]


class TestStrukturDasar(unittest.TestCase):
    def test_data_benar(self):
        self.assertEqual(validate(BENAR), [])

    def test_semua_kasus_golden_valid(self):
        for kasus in GOLDEN:
            self.assertEqual(validate(kasus["expected"]), [], kasus["teks"])

    def test_bukan_object(self):
        self.assertEqual(validate([1, 2]), ["type:<root>"])
        self.assertEqual(validate("halo"), ["type:<root>"])

    def test_helper_valid(self):
        self.assertTrue(valid(BENAR))
        self.assertFalse(valid({}))


class TestRequired(unittest.TestCase):
    def test_field_hilang_berurutan(self):
        self.assertEqual(
            validate({"product_name": "a"}),
            ["missing:price", "missing:quantity", "missing:shipping_estimate",
             "missing:extracted", "missing:confidence"],
        )

    def test_field_null_bukan_hilang(self):
        data = {**BENAR, "price": None}
        self.assertEqual(validate(data), [])


class TestTipe(unittest.TestCase):
    def test_tipe_salah(self):
        self.assertEqual(validate({**BENAR, "price": "mahal"}), ["type:price"])

    def test_boolean_bukan_integer(self):
        self.assertEqual(validate({**BENAR, "price": True}), ["type:price"])

    def test_integer_bukan_boolean(self):
        self.assertEqual(validate({**BENAR, "extracted": 1}), ["type:extracted"])

    def test_number_menerima_integer_dan_float(self):
        self.assertEqual(validate({**BENAR, "confidence": 1}), [])
        self.assertEqual(validate({**BENAR, "confidence": 0.25}), [])

    def test_null_diizinkan_karena_daftar_tipe(self):
        self.assertEqual(validate({**BENAR, "shipping_estimate": None}), [])

    def test_tipe_tidak_dikenal(self):
        with self.assertRaises(ValueError):
            validate({"x": 1}, {"type": "object", "properties": {"x": {"type": "tanggal"}}})


class TestBatas(unittest.TestCase):
    def test_minimum(self):
        self.assertEqual(validate({**BENAR, "quantity": 0}), ["minimum:quantity"])

    def test_maximum(self):
        self.assertEqual(validate({**BENAR, "confidence": 1.5}), ["maximum:confidence"])

    def test_batas_dilewati_saat_null(self):
        self.assertEqual(validate({**BENAR, "quantity": None}), [])

    def test_enum(self):
        skema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok", "gagal"]}}}
        self.assertEqual(validate({"status": "ok"}, skema), [])
        self.assertEqual(validate({"status": "lain"}, skema), ["enum:status"])

    def test_panjang_string(self):
        skema = {"type": "object", "properties": {"x": {"type": "string", "minLength": 3, "maxLength": 5}}}
        self.assertEqual(validate({"x": "abcd"}, skema), [])
        self.assertEqual(validate({"x": "ab"}, skema), ["length:x"])
        self.assertEqual(validate({"x": "abcdef"}, skema), ["length:x"])


class TestExtra(unittest.TestCase):
    def test_additional_properties_false(self):
        skema = {"type": "object", "properties": {"price": {"type": "integer"}},
                 "additionalProperties": False}
        self.assertEqual(validate({"price": 1}, skema), [])
        self.assertEqual(validate({"price": 1, "bonus": "x"}, skema), ["extra:bonus"])

    def test_default_mengizinkan_extra(self):
        self.assertEqual(validate({**BENAR, "catatan": "bebas"}, SKEMA_PENUH), [])


if __name__ == "__main__":
    unittest.main()
