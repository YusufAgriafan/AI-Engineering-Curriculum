"""prod_mini — pipeline evaluasi model produksi, dibangun dari nol (Bab 8).

Modul yang kamu isi:
    metrics.py      — RMSE, MAE, R², skill score
    models.py       — OLS, HAM baseline, model 'negatif' (anti-model)
    persist.py      — save/load deterministik + hash + round-trip test
    threshold.py    — confusion matrix, precision/recall/F1, threshold tuning
    calibration.py  — sigmoid, binning, ECE, Platt scaling

Spesifikasi ada di tests/ — kerjakan dengan mode TDD sampai 40 test hijau.
"""
