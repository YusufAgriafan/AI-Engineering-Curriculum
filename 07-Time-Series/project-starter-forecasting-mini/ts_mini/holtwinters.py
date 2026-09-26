"""Holt-Winters (triple exponential smoothing) aditif & multiplikatif.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_holtwinters.py.

Urutan update per langkah (versi aditif):
    level  = α·(y[t] − s[t−m]) + (1−α)·(level + trend)
    trend  = β·(level_baru − level_lama) + (1−β)·trend
    s[t]   = γ·(y[t] − level_baru) + (1−γ)·s[t−m]
Forecast: ŷ[t+h] = level + h·trend + s[t+h−m]   (aditif)
          ŷ[t+h] = (level + h·trend) · s[t+h−m] (multiplikatif)

Inisialisasi (aditif): level0 = mean(y[:m]), season0 = y[:m] − level0.
Inisialisasi (multiplikatif): season0 = y[:m] / (level0 + eps), eps = 1e-8.
Loop mulai dari i = m (setelah satu siklus pertama penuh).
"""

import numpy as np

__all__ = ["holt_winters_add", "holt_winters_mult"]


def holt_winters_add(tr, period, alpha, beta, gamma, h):
    """Holt-Winters aditif. Return (forecast (h,), level, trend, season (m,)).

    season yang dikembalikan = m nilai musiman TERAKHIR (s[n-m:n]).
    """
    # TODO
    raise NotImplementedError


def holt_winters_mult(tr, period, alpha, beta, gamma, h):
    """Holt-Winters multiplikatif. Return forecast (h,).

    Gunakan eps = 1e-8 pada SEMUA pembagi yang bisa nol (season, level).
    """
    # TODO
    raise NotImplementedError
