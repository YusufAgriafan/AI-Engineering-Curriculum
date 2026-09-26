"""AR(p) — regresi linear pada lag, plus forecast rekursif multi-step.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_ar.py.

Kontrak penting: coef[0] selalu mengalikan nilai TERBARU (lag-1), coef[1]
lag-2, dst. Baris ke-i matriks fitur = [y[i+p-1], y[i+p-2], ..., y[i]]
(lag-1 dulu), target = y[i+p]. Forecast rekursif memakai prediksi sendiri
sebagai input — di situlah error menumpuk.
"""

import numpy as np

__all__ = ["fit_ar", "ar_forecast"]


def fit_ar(series, p):
    """Fit AR(p) dengan OLS. Return koefisien φ berbentuk (p,).

    Baris ke-i: fitur = [series[i+p-1], series[i+p-2], ..., series[i]]
    (lag-1 dulu!), target = series[i+p].
    Petunjuk: kolom-kolom X bisa dibangun dengan slicing
    series[p-k-1 : len(series)-k-1] untuk k = 0..p-1.
    Data menyusut: n -> n-p baris.
    """
    # TODO
    raise NotImplementedError


def ar_forecast(coef, last_values, h):
    """Forecast h langkah secara rekursif.

    last_values: p nilai terakhir URUT WAKTU (tertua -> terbaru), seperti
    train[-p:]. Di dalam: balik dulu sehingga hist[0] = nilai TERBARU.
    Prediksi baru dimasukkan ke histori (menjadi input langkah berikutnya).
    Return array (h,).
    """
    # TODO
    raise NotImplementedError
