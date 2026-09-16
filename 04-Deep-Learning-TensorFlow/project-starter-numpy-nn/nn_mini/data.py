"""Loader dataset "dua bulan" — JANGAN DIUBAH (dipakai tests/).

Dataset non-linear klasik: dua setengah-lingkaran yang saling mengaitkan +
noise (seed 42). Sengaja DIPILIH non-linear supaya perbedaan model linear vs
MLP terlihat jelas:

- Logistic regression (linear) mentok di ±0.89 val accuracy.
- MLP kecil bisa ±0.97+ — backpropagation menemukan batas melengkung.

Split: 420 train / 90 val / 90 test. Fitur: (x, y) 2 dimensi.
"""

from pathlib import Path

import numpy as np

_DATA_PATH = Path(__file__).resolve().parent / "data.npz"


def muat_data():
    """Return (dict X per split, dict y per split)."""
    d = np.load(_DATA_PATH)
    X = {k: d[f"X_{k}"].astype(np.float64) for k in ("train", "val", "test")}
    y = {k: d[f"y_{k}"].astype(np.int64) for k in ("train", "val", "test")}
    return X, y


def muat_semua():
    """Return (X_train, y_train, X_val, y_val, X_test, y_test)."""
    X, y = muat_data()
    return X["train"], y["train"], X["val"], y["val"], X["test"], y["test"]
