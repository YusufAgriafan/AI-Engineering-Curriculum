"""TODO: Metrik regresi — RMSE, MAE, R², dan skill score.

Semua fungsi menerima array-like dan return float. Hitung dari nol dengan numpy.
Spesifikasi lengkap: tests/test_metrics.py
"""


def rmse(y_true, y_pred):
    """Root Mean Squared Error: sqrt(mean((y_true - y_pred)^2)). Return float."""
    # TODO
    raise NotImplementedError


def mae(y_true, y_pred):
    """Mean Absolute Error: mean(|y_true - y_pred|). Return float."""
    # TODO
    raise NotImplementedError


def r2(y_true, y_pred):
    """Koefisien determinasi: 1 - SS_res/SS_tot. Boleh NEGATIF. Return float."""
    # TODO
    raise NotImplementedError


def skill_score(rmse_model, rmse_ref):
    """Skill score relatif baseline: 1 - rmse_model/rmse_ref. Return float.

    > 0 = lebih baik dari baseline; 0 = setara; < 0 = TOLAK model.
    """
    # TODO
    raise NotImplementedError
