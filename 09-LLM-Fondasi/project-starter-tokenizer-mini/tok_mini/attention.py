"""Bagian 5 — Self-attention (scaled dot-product).  TODO: kamu yang mengisi."""
import numpy as np


def softmax_stabil(x, axis=-1):
    """Softmax numerically stable: kurangi max sebelum exp. Return array sama bentuk."""
    # TODO
    raise NotImplementedError


def self_attention(Q, K, V, causal=False):
    """Scaled dot-product attention. Q, K, V: (n, d_k). Return (output, attn).

    scores = Q @ K.T / sqrt(d_k)
    causal=True  -> scores += triu(full((n, n), -1e9), k=1) sebelum softmax.
    attn: baris ke-i = distribusi probabilitas token yang dilihat token i.
    """
    # TODO
    raise NotImplementedError
