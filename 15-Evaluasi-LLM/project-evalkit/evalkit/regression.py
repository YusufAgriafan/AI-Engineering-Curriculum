"""TODO Bagian 4 — Regression: bandingkan dua run + gate rilis.

Kontrak lengkap ada di tests/test_regression.py.

Prinsip inti Bab 15: yang dibandingkan adalah VERSI vs VERSI (regression
testing), bukan angka absolut. Gate rilis = keputusan otomatis yang bisa
dipasang di CI: skor turun > ambang → FAIL.
"""
from .data import KATEGORI


def bandingkan(run_lama, run_baru, ambang=0.05):
    """Dua hasil jalankan_eval → laporan diff.

    Return {'per_kategori': {kategori: {'lama', 'baru', 'delta'}}
            (delta = baru - lama, round 4; semua kategori KATEGORI muncul),
            'skor_rata': {'lama', 'baru', 'delta'},
            'regresi': list kategori dengan delta < -ambang (urut delta naik),
            'perbaikan': list kategori dengan delta > ambang (urut delta turun)}.
    """
    # TODO
    raise NotImplementedError


def gate_rilis(laporan, skor_min=0.75):
    """Laporan bandingkan() → keputusan rilis {'lolos': bool, 'alasan': str}.

    GAGAL bila:
    - ada kategori 'regresi' (skor turun melewati ambang), ATAU
    - skor_rata run_baru < skor_min.
    Alasan berisi ringkasan angka (dipakai sebagai pesan CI).
    """
    # TODO
    raise NotImplementedError
