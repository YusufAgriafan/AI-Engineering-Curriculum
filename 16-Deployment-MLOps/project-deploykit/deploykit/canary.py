"""TODO Bagian 4 — Canary rollout: keputusan rilis berbasis metrik, bukan selera.

Canary = arahkan persen kecil traffic ke versi baru, bandingkan metriknya dengan
versi lama, putuskan: lanjut (promote), tahan, atau rollback. Kriteria GAGAL
(cek berurutan, pertama cocok jadi alasan) ada di data.CANARY_KASUS.
"""
from .data import SLO_LATENSI_RASIO


def evaluasi_canary(nama, error_rate, skor_eval, p50_baru_ms, biaya_baru,
                    biaya_lama, p50_lama_ms=480.0, skor_min=0.75):
    """Metrik canary → keputusan.

    Return {'nama', 'putuskan': 'lanjut'|'tahan', 'alasan': str}:
    - error_rate > 0.02                              → tahan, 'error rate ...'
    - skor_eval < skor_min                           → tahan, 'skor eval ...'
    - p50_baru > SLO_LATENSI_RASIO × p50_lama        → tahan, 'latensi ...'
    - biaya_baru > 2 × biaya_lama                    → tahan, 'biaya ...'
    - else                                           → lanjut, 'semua SLO terpenuhi'
    Angka desimal di alasan dibulatkan round 4."""
    # TODO
    raise NotImplementedError


def rencana_rollout(config, langkah=(5, 20, 50, 100)):
    """Config (config_layanan) + daftar langkah persen →
    [{'persen': p, 'dengan_canary': {kunci: nilai}, 'penuh': bool}].

    - persen diambil dari config['CANARY_PERSEN'] sebagai langkah pertama,
      lalu langkah yang LEBIH BESAR dari itu (urut naik) sampai 100.
    - 'dengan_canary': SALINAN config dengan CANARY_PERSEN diganti nilai
      langkah (config asli TIDAK berubah — fungsi murni).
    - langkah 100 → 'penuh': True."""
    # TODO
    raise NotImplementedError
