"""Spesifikasi Bagian 2 — Matematika LoRA + gradient check (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ftkit/lora.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftkit.lora import (buat_lora, delta_w, gradient_check, hitung_param,
                        kontribusi_adapter, rasio_param)


class TestBuatLora(unittest.TestCase):
    def test_bentuk_adapter(self):
        ad = buat_lora(4, 3, 2)
        self.assertEqual(len(ad["A"]), 2)                 # rank × dim_in
        self.assertEqual(len(ad["A"][0]), 4)
        self.assertEqual(len(ad["B"]), 3)                 # dim_out × rank
        self.assertEqual(len(ad["B"][0]), 2)

    def test_delta_awal_nol(self):
        """B diinisialisasi NOL → ΔW awal 0 → model mulai dari base utuh."""
        ad = buat_lora(4, 3, 2)
        self.assertTrue(all(v == 0.0 for row in delta_w(ad) for v in row))


class TestDeltaW(unittest.TestCase):
    def test_matematika_b_dot_a(self):
        ad = buat_lora(2, 2, 1)
        ad["A"] = [[1.0, 2.0]]
        ad["B"] = [[3.0], [4.0]]
        d = delta_w(ad)
        self.assertEqual(d[0][0], 3.0)
        self.assertEqual(d[0][1], 6.0)
        self.assertEqual(d[1][0], 4.0)
        self.assertEqual(d[1][1], 8.0)


class TestParam(unittest.TestCase):
    def test_angka_terkunci_kecil(self):
        # rank*(dim_in+dim_out) = 2*(4+3) = 14 ; full = 12 → rasio > 1 (dim kecil)
        ad = buat_lora(4, 3, 2)
        self.assertEqual(hitung_param(ad), 14)

    def test_angka_terkunci_besar(self):
        """4096×4096 rank 16: 131072 param vs 16.7jt → rasio 0.0078 (0.78%)."""
        self.assertAlmostEqual(rasio_param(4096, 4096, 16), 0.0078125, places=9)


class TestKontribusi(unittest.TestCase):
    def test_kontribusi_adapter(self):
        ad = buat_lora(2, 1, 1)
        ad["A"] = [[1.0, 2.0]]
        ad["B"] = [[3.0]]
        # (B @ A @ x)[0] = 3*(1*1 + 2*2) = 15
        self.assertAlmostEqual(kontribusi_adapter([1.0, 2.0], ad), 15.0, places=9)

    def test_adapter_nol_tak_mengubah(self):
        ad = buat_lora(2, 1, 1)
        self.assertAlmostEqual(kontribusi_adapter([1.0, 2.0], ad), 0.0, places=9)


class TestGradientCheck(unittest.TestCase):
    def test_gradien_analitik_sesuai_numerik(self):
        maks = gradient_check()
        self.assertLess(maks, 1e-6, f"gradien analitik meleset: {maks}")


if __name__ == "__main__":
    unittest.main()
