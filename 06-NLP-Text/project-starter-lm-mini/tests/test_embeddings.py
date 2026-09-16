"""Spesifikasi skip-gram dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di lm_mini/embeddings.py lolos.
Termasuk GRADIENT CHECK numerik: bukti gradien benar, bukan cuma "loss turun".
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_mini.corpus import KAT_OF, muat_korpus_topik
from lm_mini.embeddings import frekuensi_unigram, pasangan_skipgram, sigmoid, train_skipgram
from lm_mini.text import bangun_vocab


class TestSigmoid(unittest.TestCase):
    def test_nilai_dan_stabil(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertAlmostEqual(sigmoid(100.0), 1.0, places=6, msg="tidak boleh overflow")
        self.assertAlmostEqual(sigmoid(-100.0), 0.0, places=6)


class TestPasangan(unittest.TestCase):
    def test_dua_arah_tanpa_diri(self):
        w2i = {"a": 0, "b": 1, "c": 2}
        got = sorted(pasangan_skipgram([["a", "b", "c"]], w2i, window=1))
        self.assertEqual(got, [(0, 1), (1, 0), (1, 2), (2, 1)])

    def test_window_dua(self):
        w2i = {"a": 0, "b": 1, "c": 2}
        got = sorted(pasangan_skipgram([["a", "b", "c"]], w2i, window=2))
        self.assertEqual(got, [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])

    def test_korpus_riil_ribuan(self):
        korpus = muat_korpus_topik()
        _, w2i, _ = bangun_vocab([" ".join(s) for s in korpus])
        pairs = pasangan_skipgram(korpus, w2i, window=2)
        self.assertGreater(len(pairs), 4000)


class TestFrekuensi(unittest.TestCase):
    def test_jumlah_tepat_satu(self):
        korpus = muat_korpus_topik()
        vocab, w2i, _ = bangun_vocab([" ".join(s) for s in korpus])
        freq = frekuensi_unigram(korpus, w2i)
        self.assertEqual(freq.shape, (len(vocab),))
        self.assertAlmostEqual(float(freq.sum()), 1.0, places=12)
        self.assertTrue((freq > 0).all())


def loss_pasangan(W_in, W_out, t, c, negs):
    """Loss skip-gram satu pasangan (untuk gradient check)."""
    s = sigmoid(float(W_in[t] @ W_out[c]))
    L = -np.log(max(s, 1e-12))
    for n_id in negs:
        if n_id == c:
            continue
        sn = sigmoid(float(W_in[t] @ W_out[n_id]))
        L += -np.log(max(1 - sn, 1e-12))
    return L


class TestTrainSkipGram(unittest.TestCase):
    def test_loss_turun(self):
        korpus = muat_korpus_topik()
        vocab, w2i, _ = bangun_vocab([" ".join(s) for s in korpus])
        pairs = pasangan_skipgram(korpus, w2i, window=2)[:1500]
        freq = frekuensi_unigram(korpus, w2i)
        _, hist = train_skipgram(pairs, len(vocab), freq, dim=16, epochs=3, seed=0)
        self.assertEqual(len(hist), 3)
        self.assertLess(hist[-1], hist[0], "loss harus turun")

    def test_gradient_check_numerik(self):
        """Gradien analitik (σ(z) − y)·konteks vs selisih-hingga numerik."""
        rng = np.random.default_rng(9)
        V, D = 6, 8
        Win0 = rng.normal(0, 0.5, (V, D))
        Wout0 = rng.normal(0, 0.5, (V, D))
        t0, c0, negs0 = 2, 4, [0, 3, 5]
        # gradien analitik pada v_t
        v_ = Win0[t0]
        g_an = (sigmoid(float(v_ @ Wout0[c0])) - 1.0) * Wout0[c0]
        for n_id in negs0:
            if n_id == c0:
                continue
            g_an += sigmoid(float(v_ @ Wout0[n_id])) * Wout0[n_id]
        # numerik
        eps = 1e-6
        Win0[t0, 0] += eps
        Lp = loss_pasangan(Win0, Wout0, t0, c0, negs0)
        Win0[t0, 0] -= 2 * eps
        Lm = loss_pasangan(Win0, Wout0, t0, c0, negs0)
        Win0[t0, 0] += eps
        g_num = (Lp - Lm) / (2 * eps)
        self.assertAlmostEqual(g_an[0], g_num, places=5)

    def test_purity_geografis(self):
        """Tetangga terdekat (cosine) kata konten harus satu kategori."""
        korpus = muat_korpus_topik()
        vocab, w2i, i2w = bangun_vocab([" ".join(s) for s in korpus])
        pairs = pasangan_skipgram(korpus, w2i, window=2)
        freq = frekuensi_unigram(korpus, w2i)
        E, _ = train_skipgram(pairs, len(vocab), freq, dim=16, epochs=5, seed=0)

        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        cats = {"hewan", "kendaraan", "makanan"}
        benar = total = 0
        for i, w in enumerate(vocab):
            if KAT_OF[w] not in cats:
                continue
            total += 1
            best_j = max((cos(E[i], E[j]), j) for j in range(len(vocab)) if j != i)[1]
            benar += KAT_OF[i2w[best_j]] == KAT_OF[w]
        purity = benar / total
        self.assertGreaterEqual(purity, 0.9,
                                f"embedding harus mengelompok per kategori: {purity:.2f}")


if __name__ == "__main__":
    unittest.main()
