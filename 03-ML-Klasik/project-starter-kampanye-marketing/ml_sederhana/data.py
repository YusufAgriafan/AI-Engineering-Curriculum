"""Loader dataset sintetis "kampanye marketing" — JANGAN DIUBAH (dipakai tests/).

Dataset dibuat dengan generator terkontrol (seed 42) sehingga angka
referensi di notebook starter & rubrik selalu reproducible:

- 1.200 pelanggan, 8 fitur, 11% kelas positif (membeli setelah kampanye).
- Fitur 0 ("jam_pakai_per_minggu") paling membedakan kelas (bimodal).
- Fitur 1-5 informatif tapi saling tumpang tindih (overlap moderat).
- Fitur 6-7 redundant (kombinasi linear fitur informatif).
- Sedikit label noise (flip_y = 0.01) — seperti data nyata.

Split: 720 train / 240 val / 240 test (stratified).
"""

from pathlib import Path

import numpy as np

_NAMA_FITUR = [
    "jam_pakai_per_minggu", "transaksi_bulanan", "umur_akun_bulan",
    "skor_engagement", "diskon_diterima", "kunjungan_30hari",
    "interaksi_lintas_produk", "referrer_score",
]

_DATA_PATH = Path(__file__).resolve().parent / "data.npz"


def muat_data():
    """Return (dict X per split, dict y per split, list nama fitur)."""
    d = np.load(_DATA_PATH)
    X = {k: d[f"X_{k}"].astype(np.float64) for k in ("train", "val", "test")}
    y = {k: d[f"y_{k}"].astype(np.int64) for k in ("train", "val", "test")}
    return X, y, list(_NAMA_FITUR)


def muat_semua():
    """Return (X_train, y_train, X_val, y_val, X_test, y_test, nama_fitur)."""
    X, y, nama = muat_data()
    return X["train"], y["train"], X["val"], y["val"], X["test"], y["test"], nama
