"""TODO Bagian 3 — Simulasi kuantisasi & kompresi.

Kontrak lengkap ada di tests/test_quantize.py.
"""
def kuantisasi_simulasi(bobot, bit):
    """Kuantisasi simetris: skala = max|w| / (2^(bit-1) - 1);
    q = clamp(round(w/skala)); dekuanti = q*skala. Return list baru.
    bit >= 32 → salinan identik; skala 0 → salinan identik."""
    raise NotImplementedError


def laju_kompresi(bit):
    """FP32 → bit: rasio ukuran = 32/bit."""
    raise NotImplementedError


def ukuran_model_gb(param_miliar, bit):
    """Ukuran bobot (GB) = param × bit / 8 → GB (1e9)."""
    raise NotImplementedError


def bandingkan_varian(varian=None):
    """VARIAN_KUANTISASI → tabel dict {nama, bit, ukuran_gb (7B, round 2),
    kompresi (round 1), skor}."""
    raise NotImplementedError
