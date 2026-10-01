"""TODO Bagian 7 — Evaluasi end-to-end + error analysis + feedback loop.

Kontrak lengkap ada di tests/test_evaluasi.py.

Ini lapisan yang menyatukan semuanya (pola agent.py di agentkit):
jalankan → agregat → skor sinyal → error analysis → queue feedback.
"""
from .data import BOBOT_SINYAL, EVALSET
from .judge import skor_sinyal, skor_total
from .runner import jalankan_eval
from .scorers import skor_kasus


def evaluasi_versi(versi, evalset=None):
    """Versi → laporan lengkap {'versi', 'run', 'sinyal', 'total'}.

    - run = jalankan_eval(versi, evalset);
    - sinyal = skor_sinyal(run['hasil']);
    - total = skor_total(sinyal).
    """
    # TODO
    raise NotImplementedError


def bandingkan_versi(evalset=None):
    """Semua versi di data.VERSI → list laporan evaluasi_versi urut VERSI."""
    # TODO
    raise NotImplementedError


def analisis_error(run):
    """Run → daftar kasus gagal untuk dianalisis.

    Gagal = skor 'utama' < 1.0. Return list dict {'id', 'kategori', 'tanya',
    'jawaban', 'skor'} urut skor naik (terburuk dulu). Bila semua bersih →
    list kosong."""
    # TODO
    raise NotImplementedError


def antrean_feedback(run, ambang=0.99):
    """Simulasi feedback loop produksi (thumbs down → antrean review).

    Semua kasus dengan skor 'utama' < ambang masuk antrean.
    Return {'antrean': [id kasus urut skor naik], 'n': len} — kasus ini
    nantinya menjadi KASUS EVAL BARU (lingkaran eval selesai)."""
    # TODO
    raise NotImplementedError


def laporan_lengkap(evalset=None):
    """Semua versi → ringkasan siap laporan.

    Return {'per_versi': {versi: {'total', 'sinyal', 'n_gagal'}},
            'terbaik': nama versi dengan total tertinggi (tie → versi awal
            urut VERSI), 'regresi_v1_v3': delta total v3 - v1 (round 4)}.
    """
    # TODO
    raise NotImplementedError
