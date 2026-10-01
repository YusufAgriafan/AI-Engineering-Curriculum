"""TODO Bagian 3 — Runner: jalankan evalset → hasil run + agregat.

Kontrak lengkap ada di tests/test_runner.py.

Runner MEMANGGIL sistem (`sistem.jalankan_sistem`), scorer menilai hasilnya.
Struktur hasil run dirancang seperti laporan eval nyata: skor per kasus,
agregat per kategori, dan agregat keseluruhan.
"""
import time

from .data import EVALSET
from .scorers import skor_kasus
from .sistem import jalankan_sistem


def jalankan_satu(versi, kasus):
    """Jalankan SATU kasus: panggil sistem + skor + catatan.

    Return dict {'id', 'kategori', 'tanya', 'jawaban', 'skor_retrieval',
    'skor': skor_kasus(...), 'meta': hasil['meta'], 'latensi_ms': float}.
    latensi_ms diukur dengan time.perf_counter di sekitar panggilan sistem
    (nilai asli tidak dikunci; yang diuji hanya ada & >= 0).
    """
    # TODO
    raise NotImplementedError


def jalankan_eval(versi, evalset=None):
    """Jalankan SEMUA kasus untuk satu versi.

    Return {'versi': str, 'hasil': [jalankan_satu...], 'agregat': agregat(...)}.
    evalset None → EVALSET dari data.
    """
    # TODO
    raise NotImplementedError


def agregat(hasil_list):
    """List hasil kasus → agregat.

    Return {'n': int, 'skor_rata': rata-rata skor 'utama' (0.0 bila kosong),
            'per_kategori': {kategori: {'n', 'skor_rata'}} untuk SEMUA kategori
            yang muncul di hasil_list (tampilkan apa adanya, tanpa dipaksa
            lengkap), 'n_abis': jumlah kasus dengan 'abstain' == 1.0}.
    Pembulatan skor_rata: round(x, 4) di level kategori & keseluruhan.
    """
    # TODO
    raise NotImplementedError
