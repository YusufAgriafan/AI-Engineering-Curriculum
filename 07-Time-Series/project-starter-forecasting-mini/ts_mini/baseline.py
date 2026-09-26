"""Metrik forecasting + tiga baseline yang wajib dikalahkan.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_baseline.py.

Kaidah lab: semua baseline HANYA boleh melihat train — evaluasi selalu di
data yang belum pernah disentuh model.
"""

import numpy as np

__all__ = ["mae", "rmse", "smape", "forecast_naive", "forecast_seasonal_naive",
           "forecast_ham"]


def mae(y_true, y_pred):
    """Mean Absolute Error: mean(|y_true - y_pred|). Return float."""
    # TODO
    raise NotImplementedError


def rmse(y_true, y_pred):
    """Root Mean Squared Error: sqrt(mean((y_true - y_pred)^2)). Return float."""
    # TODO
    raise NotImplementedError


def smape(y_true, y_pred):
    """Symmetric MAPE: mean(2·|e| / (|y_true| + |y_pred| + 1e-8)). Return float.

    Epsilon 1e-8 menjaga pembagi saat kedua nilai nol.
    """
    # TODO
    raise NotImplementedError


def forecast_naive(train, h):
    """Baseline naive: h prediksi bernilai train[-1] (konstan). Return array (h,)."""
    # TODO
    raise NotImplementedError


def forecast_seasonal_naive(train, h, period):
    """Baseline seasonal naive: pred[j] = train[len(train) - period + j % period].

    Untuk j < period itu berarti satu siklus penuh terakhir diulang.
    Return array (h,).
    """
    # TODO
    raise NotImplementedError


def forecast_ham(train, h):
    """Baseline HAM: h prediksi bernilai mean(train). Return array (h,)."""
    # TODO
    raise NotImplementedError
