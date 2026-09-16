"""Metrik klasifikasi — diimplementasikan dari nol (numpy murni).

Tugasmu: isi semua fungsi bertanda TODO. Kontrak tiap fungsi ada di
docstring dan di tests/test_metrik.py — baca test-nya dulu (TDD).
"""


def _cek_kelas(y):
    """Validasi: y hanya boleh berisi 0 dan 1."""
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y hanya boleh berisi 0 dan 1")
    return y


def confusion_matrix(y_true, y_pred):
    """Return dict {'TP':…, 'TN':…, 'FP':…, 'FN':…} dari dua array 0/1."""
    y_true = _cek_kelas(y_true)
    y_pred = _cek_kelas(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true dan y_pred harus sama bentuk")
    # TODO: hitung TP, TN, FP, FN
    raise NotImplementedError


def accuracy(y_true, y_pred):
    """(TP + TN) / total. Return float."""
    cm = confusion_matrix(y_true, y_pred)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    # TODO: guard pembagian nol lalu return akurasi
    raise NotImplementedError


def precision_recall_f1(y_true, y_pred):
    """Return tuple (precision, recall, f1) sebagai float.

    Konvensi edge case (SAMA dengan sklearn):
    - precision: TP+FP == 0  -> 0.0
    - recall:    TP+FN == 0  -> 0.0
    - f1:        P+R == 0    -> 0.0
    """
    cm = confusion_matrix(y_true, y_pred)
    # TODO: hitung ketiganya dengan guard seperti di atas
    raise NotImplementedError


def f1_score(y_true, y_pred):
    """Shortcut: return komponen F1 dari precision_recall_f1."""
    return precision_recall_f1(y_true, y_pred)[2]


def roc_auc(y_true, y_score):
    """ROC-AUC dari nol memakai rank statistic (Mann-Whitney U).

    y_score: skor kontinu (mis. probabilitas dari sigmoid), bukan 0/1.
    Handle tie: semua titik dengan skor sama dapat RATA-RATA rank-nya.
    Return float di [0, 1].
    """
    y_true = _cek_kelas(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true dan y_score harus sama bentuk")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC butuh kedua kelas hadir di y_true")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    # TODO: isi `ranks` — tiap grup tie dapat rata-rata rank 1-based-nya,
    #       lalu hitung AUC:
    #       AUC = (Σ rank(pos) − n_pos(n_pos+1)/2) / (n_pos·n_neg)
    raise NotImplementedError
