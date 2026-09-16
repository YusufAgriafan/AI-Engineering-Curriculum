"""Fungsi aktivasi + turunannya — numpy murni.

Tugasmu: isi fungsi bertanda TODO. Kontrak ada di docstring dan di
tests/test_activations.py — baca test-nya dulu (TDD).

Ingat tabel Bab 4:
- ReLU    -> hidden layer (paling umum)
- Sigmoid -> output klasifikasi biner
- Softmax -> output klasifikasi multikelas
"""

import numpy as np


def relu(z):
    """ReLU: max(0, z). Element-wise, bentuk output = bentuk input."""
    z = np.asarray(z, dtype=float)
    # TODO
    raise NotImplementedError


def relu_grad(z):
    """Turunan ReLU terhadap z: 1 jika z > 0, selain itu 0.

    Return array float sebesar z. (Di z = 0 gradien dianggap 0.)
    """
    z = np.asarray(z, dtype=float)
    # TODO
    raise NotImplementedError


def sigmoid(z):
    """Sigmoid versi stabil-numerik (wajib!):

    - z >= 0 : 1 / (1 + exp(-z))
    - z < 0  : exp(z) / (1 + exp(z))   -> exp tidak pernah overflow

    Return array float dengan bentuk sama seperti z.
    """
    z = np.asarray(z, dtype=float)
    # TODO
    raise NotImplementedError


def sigmoid_from_logits_stable(logits):
    """Sigmoid stabil yang SEKALIGUS mengembalikan d(sigmoid)/d(logits).

    Return (p, grad) di mana:
      p    = sigmoid(logits)
      grad = p * (1 - p)
    """
    logits = np.asarray(logits, dtype=float)
    # TODO: p = sigmoid(logits); grad = p * (1 - p)
    raise NotImplementedError


def softmax(logits):
    """Softmax stabil: kurangi dengan max per baris dulu (nilai sama, aman overflow).

    - logits 1D -> hasil 1D (jumlah = 1)
    - logits 2D (n, k) -> softmax per BARIS (tiap baris jumlah = 1)
    """
    logits = np.asarray(logits, dtype=float)
    # TODO: shifted = logits - max(...); e = exp(shifted); return e / e.sum(...)
    raise NotImplementedError
