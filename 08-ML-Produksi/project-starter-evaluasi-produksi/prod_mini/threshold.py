"""TODO: Keputusan & threshold — confusion matrix, P/R/F1, threshold tuning.

Konvensi: prediksi positif jika y_score >= thr (perhatikan >=, bukan >).
Pembagi nol -> metrik 0.0 (konvensi produksi: jangan crash).
Spesifikasi lengkap: tests/test_threshold.py
"""


def confusion(y_true01, y_score, thr):
    """Confusion matrix dari skor kontinu + threshold.

    y_true01: array 0/1. y_score: array kontinu.
    Prediksi positif jika y_score >= thr.
    Return dict {'tp', 'fp', 'fn', 'tn'} berisi int.
    """
    # TODO
    raise NotImplementedError


def prf(cm):
    """Precision, recall, F1 dari confusion matrix. Return dict.

    pembagi nol -> 0.0. F1 = 2PR/(P+R) — harmonic mean.
    """
    # TODO
    raise NotImplementedError


def pilih_threshold(y_true01, y_score, kandidat):
    """Pilih threshold dengan F1 tertinggi pada data yang diberikan (SEHARUSNYA calib).

    kandidat: iterable threshold. Ties: ambil yang PERTAMA ditemui.
    Return (thr float, cm dict, metrik dict) pada pemenang.
    """
    # TODO
    raise NotImplementedError
