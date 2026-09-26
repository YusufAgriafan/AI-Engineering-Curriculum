"""Data sintetis terkunci untuk project forecasting mini Bab 7.

JANGAN DIUBAH — semua angka di tests dan starter.ipynb mengacu pada seri yang
dihasilkan file ini (seed terkunci 42, resep identik dengan lab).

Resep seri aditif (make_series):
    y[t] = (100 + 0.05·t) + 20·sin(2π·t/365) + N(0, 5²)
Resep seri multiplikatif (make_mult_series):
    y[t] = (100 + 0.05·t) · (1 + 0.15·sin(2π·t/365)) + N(0, 4²)

Panjang: 1095 hari (3 tahun). Split standar: 80% train (876), 20% val (219).
"""

import numpy as np

__all__ = ["N_HARI", "SPLIT_IDX", "HORIZON", "make_series", "make_mult_series"]

N_HARI = 365 * 3
SPLIT_IDX = int(N_HARI * 0.8)   # 876
HORIZON = N_HARI - SPLIT_IDX    # 219


def make_series(seed=42, n_days=N_HARI):
    """Seri aditif: trend linear + musiman tahunan + noise. Return (seri, t)."""
    t = np.arange(n_days, dtype=float)
    rng = np.random.default_rng(seed)
    trend = 100.0 + 0.05 * t
    seasonality = 20.0 * np.sin(2 * np.pi * t / 365.0)
    noise = rng.normal(0.0, 5.0, n_days)
    return trend + seasonality + noise, t


def make_mult_series(seed=42, n_days=N_HARI):
    """Seri multiplikatif: amplitudo musiman membesar bersama level. Return (seri, t)."""
    t = np.arange(n_days, dtype=float)
    rng = np.random.default_rng(seed)
    level = 100.0 + 0.05 * t
    seri = level * (1.0 + 0.15 * np.sin(2 * np.pi * t / 365.0))
    seri = seri + rng.normal(0.0, 4.0, n_days)
    return seri, t


if __name__ == "__main__":
    s, t = make_series()
    sm, _ = make_mult_series()
    print(f"aditif : n={len(s)} mean={s.mean():.2f} std={s.std():.2f}")
    print(f"mult   : n={len(sm)} mean={sm.mean():.2f} std={sm.std():.2f}")
    print(f"split={SPLIT_IDX}, horizon={HORIZON}")
