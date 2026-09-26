"""TODO Bagian 5b — Evaluasi sebelum/sesudah + kebijakan OOD (abstain).

Kontrak lengkap ada di tests/test_api_eval.py (kelas TestPrediksi,
TestEvalSebelumSesudah, TestKebijakanOod, TestLaporanEval).
Sambungan Bab 12: model yang tahu dia tidak yakin = dasar abstain.
"""
from ftkit.data import LABEL_POS
from ftkit.model import akurasi, forward
from ftkit.tokenizer import TokenizerMini, fitur


def prediksi(teks, w, b, tok=None, ambang=0.5):
    """teks → {'label' (positif bila p>=ambang), 'skor' (p), 
    'yakin' = |p - ambang| * 2}."""
    raise NotImplementedError


def evaluasi_sebelum_sesudah(val, w_sebelum, b_sebelum, w_sesudah, b_sesudah):
    """{'sebelum': acc, 'sesudah': acc, 'delta': sesudah - sebelum}."""
    raise NotImplementedError


def kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman="kurang yakin, mohon periksa"):
    """yakin < skor_min → {'tampilkan': False, 'aksi': jawaban_aman};
    selain itu {'tampilkan': True, 'aksi': 'tampilkan label'}.
    Jangan mengubah hasil_pred — kembalikan dict baru (merge)."""
    raise NotImplementedError


def laporan_eval(val, w, b, ood_hasil=None, skor_min=0.6):
    """{'akurasi': acc(val), 'salah': [{'p','y'} untuk contoh salah],
    'ood': kebijakan_ood(ood_hasil) bila diberikan, selain itu None}."""
    raise NotImplementedError
