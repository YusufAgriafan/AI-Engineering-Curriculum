"""Spesifikasi Bagian 1 — Konfigurasi 12-factor (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di deploykit/konfig.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploykit.data import ENV_PRODUKSI, SCHEMA_ENV
from deploykit.konfig import baca_env, config_layanan


class TestBacaEnv(unittest.TestCase):
    def test_tipe_int_dan_float(self):
        cfg = baca_env({"PORT": "9000", "AMBAT_ABSTAIN": "0.25"})
        self.assertEqual(cfg["PORT"], 9000)
        self.assertEqual(cfg["AMBAT_ABSTAIN"], 0.25)

    def test_default_dipakai_bila_tidak_ada(self):
        cfg = baca_env({})
        self.assertEqual(cfg["PORT"], 8000)
        self.assertEqual(cfg["AMBAT_ABSTAIN"], 0.20)
        self.assertEqual(cfg["MODEL_API"], "api_mini")

    def test_angka_terkunci_env_produksi(self):
        cfg = baca_env(ENV_PRODUKSI)
        self.assertEqual(cfg, {"PORT": 9000, "AMBAT_ABSTAIN": 0.25,
                               "SKOR_MIN": 0.8, "RATE_LIMIT_PER_MENIT": 120,
                               "MODEL_API": "api_besar", "CANARY_PERSEN": 20,
                               "BIAYA_HARIAN_MAKS": 3.5})

    def test_kunci_asing_diabaikan(self):
        cfg = baca_env({"PORT": "7000", "SECRET_KEY": "h4x"})
        self.assertEqual(cfg["PORT"], 7000)
        self.assertNotIn("SECRET_KEY", cfg)

    def test_env_kosong_string_tetap_diterima(self):
        cfg = baca_env({"MODEL_API": ""})
        self.assertEqual(cfg["MODEL_API"], "")

    def test_konversi_gagal_nyaring(self):
        with self.assertRaises(ValueError) as ctx:
            baca_env({"PORT": "delapan ribu"})
        self.assertIn("PORT", str(ctx.exception))

    def test_float_gagal_nyaring(self):
        with self.assertRaises(ValueError) as ctx:
            baca_env({"SKOR_MIN": "tiga per empat"})
        self.assertIn("SKOR_MIN", str(ctx.exception))


class TestConfigLayanan(unittest.TestCase):
    def test_env_produksi_valid(self):
        cfg = config_layanan(ENV_PRODUKSI)
        self.assertEqual(cfg["PORT"], 9000)
        self.assertEqual(cfg["MODEL_API"], "api_besar")

    def test_default_valid(self):
        cfg = config_layanan({})
        self.assertEqual(cfg["PORT"], 8000)

    def test_port_nol_ditolak(self):
        with self.assertRaises(ValueError) as ctx:
            config_layanan({"PORT": "0"})
        self.assertIn("PORT", str(ctx.exception))

    def test_ambang_luar_jangkauan_ditolak(self):
        with self.assertRaises(ValueError) as ctx:
            config_layanan({"AMBAT_ABSTAIN": "1.5"})
        self.assertIn("AMBAT_ABSTAIN", str(ctx.exception))

    def test_skor_min_nol_ditolak(self):
        with self.assertRaises(ValueError) as ctx:
            config_layanan({"SKOR_MIN": "0"})
        self.assertIn("SKOR_MIN", str(ctx.exception))

    def test_model_tak_dikenal_ditolak(self):
        with self.assertRaises(ValueError) as ctx:
            config_layanan({"MODEL_API": "model_hantu"})
        self.assertIn("MODEL_API", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
