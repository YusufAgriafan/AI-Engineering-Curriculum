"""Spesifikasi RNN char-level dalam bentuk test (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di lm_mini/rnn.py lolos.
Termasuk GRADIENT CHECK BPTT: bukti gradien sepanjang waktu benar.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_mini.corpus import KALIMAT_LM
from lm_mini.languagemodel import bigram_matrix
from lm_mini.rnn import onehot_batch, rnn_backward, rnn_forward, rnn_step_backward, rnn_step_forward, softmax


def buat_data_lm(ctx=4):
    teks = "\n".join(KALIMAT_LM) + "\n"
    chars = sorted(set(teks))
    c2i = {c: i for i, c in enumerate(chars)}
    seq = [c2i[c] for c in teks]
    X, y = [], []
    for i in range(len(seq) - ctx):
        X.append(seq[i:i + ctx])
        y.append(seq[i + ctx])
    X, y = np.array(X), np.array(y)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    n_tr = int(0.75 * len(X))
    return chars, X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


class TestSoftmax(unittest.TestCase):
    def test_baris_distribusi_dan_stabil(self):
        out = softmax(np.array([[1000.0, 1001.0], [-1000.0, -1001.0]]))
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.allclose(out.sum(axis=1), 1.0))
        self.assertFalse(np.isnan(out).any(), "tidak boleh overflow/nan")


class TestStep(unittest.TestCase):
    def test_forward_bentuk_dan_cache(self):
        r = np.random.default_rng(0)
        x_t = r.normal(size=(2, 5))
        h_prev = r.normal(size=(2, 3))
        Wx, Wh, bh = r.normal(size=(5, 3)), r.normal(size=(3, 3)), np.zeros(3)
        h, cache = rnn_step_forward(x_t, h_prev, Wx, Wh, bh)
        self.assertEqual(h.shape, (2, 3))
        self.assertTrue(np.allclose(h, np.tanh(x_t @ Wx + h_prev @ Wh.T + bh)))
        h_prev_c, h_pre = cache
        self.assertTrue(np.allclose(h_prev_c, h_prev))
        self.assertTrue(np.allclose(np.tanh(h_pre), h))

    def test_backward_turunan_tanh(self):
        r = np.random.default_rng(0)
        x_t = r.normal(size=(2, 5))
        h_prev = r.normal(size=(2, 3))
        Wx, Wh, bh = r.normal(size=(5, 3)), r.normal(size=(3, 3)), np.zeros(3)
        h, cache = rnn_step_forward(x_t, h_prev, Wx, Wh, bh)
        dh = np.ones_like(h)
        *_, dbh = rnn_step_backward(dh, cache, x_t, Wx, Wh)
        # dbh = da.sum(0) dengan dh=1 -> da = 1 - tanh(h_pre)^2 = 1 - h^2
        self.assertTrue(np.allclose(dbh, (1 - h ** 2).sum(axis=0)))
        _, dh_prev, *_ = rnn_step_backward(dh, cache, x_t, Wx, Wh)
        self.assertTrue(np.allclose(dh_prev, (1 - h ** 2) @ Wh))

    def test_step_gradient_check(self):
        rng = np.random.default_rng(1)
        x_t = rng.normal(size=(2, 4))
        h_prev = rng.normal(size=(2, 3))
        Wx, Wh, bh = rng.normal(size=(4, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3,))
        dh = rng.normal(size=(2, 3))

        def loss_fn(Wx_, Wh_, bh_):
            h, _ = rnn_step_forward(x_t, h_prev, Wx_, Wh_, bh_)
            return float((h * dh).sum())

        dx, dh_prev, dWx, dWh, dbh = rnn_step_backward(dh, rnn_step_forward(
            x_t, h_prev, Wx, Wh, bh)[1], x_t, Wx, Wh)
        eps = 1e-6
        for nama, P, G in [("Wx", Wx, dWx), ("Wh", Wh, dWh), ("bh", bh, dbh)]:
            it = np.nditer(P, flags=["multi_index"])
            cek = 0
            while not it.finished:
                idx = it.multi_index
                orig = P[idx]
                P[idx] = orig + eps
                Lp = loss_fn(Wx, Wh, bh)
                P[idx] = orig - eps
                Lm = loss_fn(Wx, Wh, bh)
                P[idx] = orig
                num = (Lp - Lm) / (2 * eps)
                self.assertAlmostEqual(G[idx], num, places=4, msg=f"{nama}{idx}")
                cek += 1
                it.iternext()
                if cek >= 3:
                    break


class TestForwardBackward(unittest.TestCase):
    def setUp(self):
        self.chars, self.Xtr, self.ytr, self.Xva, self.yva = buat_data_lm(ctx=4)
        self.V = len(self.chars)
        self.CTX = 4

    def _params(self, hidden=8, seed=11):
        rng = np.random.default_rng(seed)
        return [rng.normal(0, 0.4, (self.V, hidden)), rng.normal(0, 0.4, (hidden, hidden)),
                rng.normal(0, 0.1, (hidden, self.V)), np.zeros(hidden), np.zeros(self.V)]

    def test_forward_probs(self):
        params = self._params()
        probs, cache = rnn_forward(self.Xtr[:4], params, self.CTX)
        self.assertEqual(probs.shape, (4, self.V))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        hs, Hs = cache
        self.assertEqual(len(hs), self.CTX + 1)
        self.assertEqual(len(Hs), self.CTX)
        self.assertTrue(np.allclose(hs[0], 0.0))

    def test_bptt_gradient_check(self):
        params = self._params()
        Xb, yb = self.Xtr[:4], self.ytr[:4]
        probs, cache = rnn_forward(Xb, params, self.CTX)
        grads = rnn_backward(probs, yb, Xb, params, cache, self.CTX)

        def loss_fn(p):
            pr, _ = rnn_forward(Xb, p, self.CTX)
            return float(-np.mean(np.log(pr[np.arange(4), yb] + 1e-12)))

        eps = 1e-6
        nama = ["dWx", "dWh", "dWy", "dbh", "dby"]
        for pi in range(5):
            P_ = params[pi]
            G_ = grads[pi]
            it = np.nditer(P_, flags=["multi_index"])
            cek = 0
            while not it.finished:
                idx = it.multi_index
                orig = P_[idx]
                P_[idx] = orig + eps
                Lp = loss_fn(params)
                P_[idx] = orig - eps
                Lm = loss_fn(params)
                P_[idx] = orig
                num = (Lp - Lm) / (2 * eps)
                denom = abs(num) + abs(G_[idx]) + 1e-8
                if denom > 1e-8:
                    rel = abs(G_[idx] - num) / denom
                    self.assertLess(rel, 1e-4, f"{nama[pi]}{idx}: {G_[idx]} vs {num}")
                    cek += 1
                it.iternext()
                if cek >= 3:
                    break

    def test_rnn_mengalahkan_bigram(self):
        """Latih RNN kecil beberapa epoch; harus menang atas baseline bigram."""
        fwd = rnn_forward
        hidden, epochs, lr = 32, 150, 0.2
        rng = np.random.default_rng(0)
        params = [rng.normal(0, 0.5, (self.V, hidden)), rng.normal(0, 0.5, (hidden, hidden)),
                  rng.normal(0, 0.1, (hidden, self.V)), np.zeros(hidden), np.zeros(self.V)]
        for ep in range(epochs):
            order = rng.permutation(len(self.Xtr))
            for s in range(0, len(order), 32):
                bidx = order[s:s + 32]
                Xb, yb = self.Xtr[bidx], self.ytr[bidx]
                probs, cache = fwd(Xb, params, self.CTX)
                grads = rnn_backward(probs, yb, Xb, params, cache, self.CTX)
                for g in grads:
                    np.clip(g, -5, 5, out=g)
                for P, G in zip(params, grads):
                    P -= lr * G
        probs_va, _ = fwd(self.Xva, params, self.CTX)
        acc_rnn = float((probs_va.argmax(1) == self.yva).mean())

        counts = np.ones((self.V, self.V))
        teks = "\n".join(KALIMAT_LM) + "\n"
        c2i = {c: i for i, c in enumerate(self.chars)}
        seq = [c2i[c] for c in teks]
        n_tr = len(self.Xtr)
        for a, b in zip(seq[:n_tr + self.CTX], seq[1:n_tr + self.CTX]):
            counts[a, b] += 1
        P_bi = counts / counts.sum(1, keepdims=True)
        acc_bi = float((np.array([P_bi[x[-1]].argmax() for x in self.Xva]) == self.yva).mean())

        self.assertGreater(acc_rnn, acc_bi,
                           f"RNN {acc_rnn:.3f} harus > bigram {acc_bi:.3f}")


if __name__ == "__main__":
    unittest.main()
