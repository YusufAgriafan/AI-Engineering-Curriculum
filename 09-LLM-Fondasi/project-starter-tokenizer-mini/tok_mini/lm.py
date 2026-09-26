"""Bagian 6 — Mini language model.

TODO kamu: `loss_fn` dan `train_lm`. Sisanya GIVEN — fokuslah pada loss
next-token (SHIFT target!) dan loop gradient descent, bukan pada forward pass.
"""
import numpy as np


def one_hot(ids, vocab):
    """(n,) int -> (n, vocab) one-hot float.  (GIVEN)"""
    out = np.zeros((len(ids), vocab))
    out[np.arange(len(ids)), ids] = 1.0
    return out


def cross_entropy(logits, targets):
    """Mean cross-entropy; log-softmax stabil. logits (n, V), targets (n,) int.  (GIVEN)"""
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    z = logits - np.max(logits, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    n = len(targets)
    return float(-np.sum(logp[np.arange(n), targets]) / n)


def init_params(seed, vocab, d):
    """Parameter awal: We, PE (fixed), 2 blok dense, output projection.  (GIVEN)"""
    from .position import positional_encoding

    r = np.random.default_rng(seed)
    return {
        "We": r.normal(0, 0.5, (vocab, d)),
        "PE": positional_encoding(32, d) * 0.1,
        "W1": r.normal(0, 0.5, (d, d)), "b1": np.zeros(d),
        "W2": r.normal(0, 0.5, (d, d)), "b2": np.zeros(d),
        "Wo": r.normal(0, 0.1, (d, vocab)), "bo": np.zeros(vocab),
    }


def loss_fn(params, ids):
    """Next-token CE: logits di posisi t diprediksi dari ids[t+1] (SHIFT!).

    Return float. Jangan lupa: prediksi = logits[:-1], target = ids[1:].
    """
    # TODO
    raise NotImplementedError


def train_lm(params, seq, epochs, lr):
    """Gradient descent dengan gradien finite-difference (GIVEN helper numeric_grads).

    PE TIDAK dilatih (fixed). Return (params, losses) — losses sepanjang epochs.
    """
    # TODO
    raise NotImplementedError


def numeric_grads(f, params, seq, eps=1e-5):
    """Finite-difference gradient semua parameter float.  (GIVEN)"""
    grads = {}
    for k, arr in params.items():
        if arr is None:
            continue
        g = np.zeros_like(arr)
        flat, gflat = arr.ravel(), g.ravel()
        for i in range(len(flat)):
            orig = flat[i]
            flat[i] = orig + eps
            fp = f(params, seq)
            flat[i] = orig - eps
            fm = f(params, seq)
            flat[i] = orig
            gflat[i] = (fp - fm) / (2 * eps)
        grads[k] = g
    return grads


def forward(params, ids):
    """Embedding + PE, 2 blok dense ReLU, output projection.  (GIVEN)"""
    n = len(ids)
    X = params["We"][np.asarray(ids)] + params["PE"][:n]
    A1 = np.maximum(X @ params["W1"] + params["b1"], 0.0)
    A2 = np.maximum(A1 @ params["W2"] + params["b2"], 0.0)
    return A2 @ params["Wo"] + params["bo"]


def prediksi_berikut(params, ids):
    """Token dengan logits tertinggi di posisi terakhir.  (GIVEN)"""
    return int(np.argmax(forward(params, ids)[-1]))
