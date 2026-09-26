"""TODO: Model — OLS, baseline HAM, dan model 'negatif' (simulasi eksperimen salah arah).

Spesifikasi lengkap: tests/test_models.py
"""

import numpy as np


def fit_ols(Xa, ya):
    """OLS dengan kolom bias: matriks [x1, x2, ..., 1], lstsq. Return (w, b).

    w: array (n_fitur,), b: float.
    """
    # TODO
    raise NotImplementedError


def predict(w, b, Xa):
    """Prediksi linear: X @ w + b. Return array (n,)."""
    # TODO
    raise NotImplementedError


def fit_ham(ya):
    """Baseline HAM: mean(ya). Return float."""
    # TODO
    raise NotImplementedError


def predict_ham(mean_y, Xa):
    """Prediksi konstan mean_y untuk semua baris. Return array (n,)."""
    # TODO
    raise NotImplementedError


def fit_neg(Xa, ya):
    """Model 'negatif': OLS pada fitur kedua DENGAN tanda dibalik, lalu kembalikan
    koefisien yang juga dibalik. Return (w berbentuk (1,), b).

    Langkah:
      1. X1 = kolom kedua Xa, bentuk (n, 1)
      2. lstsq pada matriks [-X1, 1] -> coef
      3. return (np.array([-coef[0]]), float(coef[1]))

    Prediksi (predict_neg) = -x2 * w + b. Dua pembalikan = model salah arah yang
    KONSISTEN: fit dan prediksi saling cocok, hasilnya tetap buruk — persis
    seperti eksperimen nyata yang salah arah tapi dilaporkan dengan percaya diri.
    """
    # TODO
    raise NotImplementedError


def predict_neg(w, b, Xa):
    """Prediksi model negatif: -x2 * w + b. Return array (n,)."""
    # TODO
    raise NotImplementedError
