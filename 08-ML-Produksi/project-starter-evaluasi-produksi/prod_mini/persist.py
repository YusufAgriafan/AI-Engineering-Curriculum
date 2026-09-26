"""TODO: Persistensi model — save/load deterministik, hash, round-trip test.

Spesifikasi lengkap: tests/test_persist.py
"""

import hashlib
from pathlib import Path

import numpy as np


def save_model(path, w, b):
    """Simpan (w, b) ke file .npz secara DETERMINISTIK.

    w disimpan sebagai float64, b sebagai np.array skalar float.
    Deterministik = file yang sama dibuat dua kali -> byte identik.

    Petunjuk: np.savez(path, w=np.asarray(w, dtype=float), b=np.array(float(b)))
    """
    # TODO
    raise NotImplementedError


def load_model(path):
    """Baca kembali (w, b) dari file .npz. Return (w array, b float)."""
    # TODO
    raise NotImplementedError


def sha256_file(path):
    """SHA-256 hex dari ISI file (bukan nama/atributnya). Return string hex."""
    # TODO
    raise NotImplementedError


def roundtrip_ok(w, b, X_uji, prediksi_sebelum):
    """Round-trip test: save -> load -> prediksi ulang -> bandingkan.

    Return True jika prediksi sesudah round-trip IDENTIK bit-per-bit dengan
    prediksi_sebelum (np.array_equal, bukan allclose!).
    """
    # TODO
    # 1. save ke path sementara (tempfile.mkdtemp() / NamedTemporaryFile)
    # 2. load kembali
    # 3. prediksi dengan (w2, b2) pada X_uji
    # 4. return np.array_equal(prediksi_baru, prediksi_sebelum)
    raise NotImplementedError
