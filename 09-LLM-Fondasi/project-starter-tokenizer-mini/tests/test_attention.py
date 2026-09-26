"""Spesifikasi Bagian 5 — Self-attention (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di tok_mini/attention.py lolos.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tok_mini.attention import self_attention, softmax_stabil


class TestSoftmax(unittest.TestCase):
    def test_distribusi_sederhana(self):
        p = softmax_stabil(np.array([0.0, np.log(3.0)]))
        np.testing.assert_allclose(p, [0.25, 0.75], atol=1e-12)

    def test_baris_jumlah_satu(self):
        x = np.random.default_rng(0).normal(0, 30, (5, 7))
        np.testing.assert_allclose(softmax_stabil(x).sum(axis=-1), 1.0, atol=1e-9)

    def test_stabil_skala_besar(self):
        x = np.array([[1000.0, 1001.0, 999.0]])
        p = softmax_stabil(x)
        self.assertTrue(np.all(np.isfinite(p)), "harus finite tanpa overflow")
        self.assertAlmostEqual(float(p[0, 1]), np.exp(1.0) / (np.exp(1.0) + 1 + np.exp(-1.0)), places=9)

    def test_axis_argumen(self):
        x = np.random.default_rng(1).normal(0, 3, (2, 4, 6))
        p = softmax_stabil(x, axis=1)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


class TestAttention(unittest.TestCase):
    def setUp(self):
        np.random.seed(9)
        self.Q = np.random.randn(4, 8)
        self.K = np.random.randn(4, 8)
        self.V = np.random.randn(4, 8)

    def test_bentuk_output(self):
        out, attn = self_attention(self.Q, self.K, self.V)
        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(attn.shape, (4, 4))

    def test_baris_softmax(self):
        _, attn = self_attention(self.Q, self.K, self.V)
        np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-9)
        self.assertTrue(np.all(attn > 0))

    def test_output_campuran_berbobot(self):
        out, attn = self_attention(self.Q, self.K, self.V)
        np.testing.assert_allclose(out, attn @ self.V, atol=1e-12)

    def test_scaling_sqrt_dk(self):
        _, attn = self_attention(self.Q, self.K, self.V)
        e = np.exp(self.Q @ self.K.T / np.sqrt(8))
        manual = e / e.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(attn, manual, atol=1e-12)

    def test_tanpa_scaling_berbeda(self):
        _, attn = self_attention(self.Q, self.K, self.V)
        e = np.exp(self.Q @ self.K.T)          # tanpa sqrt(d_k)
        noscale = e / e.sum(axis=-1, keepdims=True)
        self.assertFalse(np.allclose(attn, noscale))

    def test_causal_mask(self):
        _, attn = self_attention(self.Q, self.K, self.V, causal=True)
        np.testing.assert_allclose(np.triu(attn, k=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(attn[0], [1, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-9)

    def test_causal_baris_manual(self):
        _, attn = self_attention(self.Q, self.K, self.V, causal=True)
        S = self.Q @ self.K.T / np.sqrt(8)
        z = np.array([S[2, 0], S[2, 1], S[2, 2], -1e9])
        ez = np.exp(z - z.max())
        np.testing.assert_allclose(attn[2], ez / ez.sum(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
