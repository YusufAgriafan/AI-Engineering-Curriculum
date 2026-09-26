"""Bagian 7 — Sampling: dari logits ke token.  TODO: kamu yang mengisi."""
import numpy as np

from .attention import softmax_stabil


def sample_greedy(logits):
    """Return int argmax — deterministik."""
    # TODO
    raise NotImplementedError


def sample_topk(logits, k, rng, T=1.0):
    """Sampling dari k token probabilitas tertinggi (setelah temperature).

    idx = argsort(logits)[::-1][:k]; p = softmax(logits[idx] / T);
    return rng.choice(idx, p=p) sebagai int.
    """
    # TODO
    raise NotImplementedError


def sample_topp(logits, p_min, rng, T=1.0):
    """Nucleus: kandidat terkecil yang kumulatif prob >= p_min (minimal 1 token).

    order = argsort menurun; cum = cumsum(softmax(order / T));
    kc = 1 + searchsorted(cum, p_min, side='left')  (clamp ke len(order));
    sampling dari order[:kc].
    """
    # TODO
    raise NotImplementedError
