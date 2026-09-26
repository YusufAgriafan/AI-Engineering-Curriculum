"""Model klasifier tiny + loss BCE — GIVEN, JANGAN DIUBAH.

Model: mean embedding (+flag negasi) → 1 logit → sigmoid → P(positif). Kecil
tapi SUNGGUHAN: SGD di atas loss BCE dengan gradien analitik, gradient check
numerik lolos 1e-6. Semua operasi list murni (stdlib) supaya mesinnya terlihat.

Analogi LoRA untuk Bab 13: bobot embedding dibekukan (base model), dan yang
dilatih hanya w & b (kepala kecil / "adapter"). Update-param-kecil di atas
fungsi loss sungguhan — skala naik, bentuknya sama.
"""
import math

from .data import LABEL_NEG, LABEL_POS

DIM = 17  # 16 mean one-hot + 1 flag negasi


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def label_ke_y(label):
    """positif → 1.0, negatif → 0.0."""
    return 1.0 if label == LABEL_POS else 0.0


def forward(x, w, b):
    """logit = w·x + b → p = sigmoid(logit)."""
    logit = sum(xi * wi for xi, wi in zip(x, w)) + b
    return sigmoid(logit), logit


def loss_bce(p, y):
    """Binary cross-entropy numerik-stabil."""
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def gradien(x, p, y):
    """Gradien analitik dL/dw = (p - y) * x ; dL/db = (p - y)."""
    err = p - y
    return [err * xi for xi in x], err


def loss_batch(contoh, w, b):
    """Rata-rata BCE atas list (x, y)."""
    if not contoh:
        return 0.0
    return sum(loss_bce(forward(x, w, b)[0], y) for x, y in contoh) / len(contoh)


def akurasi(contoh, w, b):
    """P(positif) >= 0.5 → prediksi positif."""
    if not contoh:
        return 0.0
    benar = sum(1 for x, y in contoh if (1.0 if forward(x, w, b)[0] >= 0.5 else 0.0) == y)
    return benar / len(contoh)


def inisialisasi(seed=13, dim=DIM):
    """Bobot awal deterministik: w = 0.01*i (kecil, teratur), b = 0.0."""
    return [0.01 * ((i % 7) - 3) for i in range(dim)], 0.0
