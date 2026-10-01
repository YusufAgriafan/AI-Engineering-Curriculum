"""TODO Bagian 5 — Monitoring & drift: health score + deteksi drift vocab.

Monitor = angka, bukan kesan: health score dari metrik terhadap target, dan
deteksi drift dari perubahan DISTRIBUSI permintaan (kata-kata baru yang tak
pernah muncul di periode lama) — bukan dari error.
"""
from .data import AMBAT_DRIFT, DRIFT, DRIFT_JAM_PENUH, HEALTH_METRIK, STOPWORD
from .tokenizer import token_estimasi


def health_score(metrik=None):
    """HEALTH_METRIK (bila None) → {'skor': 0..1, 'ok': bool, 'rincian': [...]}.

    Tiap metrik dinilai: nilai memenuhi target (arah 'maks': nilai ≤ target;
    arah 'min': nilai ≥ target) → 1.0, else 0.0.
    skor = rata penilaian round 4; ok = semua terpenuhi.
    rincian = [{'metrik', 'nilai', 'target', 'ok'}] urutan metrik asli."""
    # TODO
    raise NotImplementedError


def kueri_ke_kata(kueri):
    """Kueri → set kata bermakna: token [a-z0-9]+, lowercase, len ≥ 3, dan
    BUKAN STOPWORD (kata tanya/fungsi dibuang supaya topik yang terlihat)."""
    # TODO
    raise NotImplementedError


def deteksi_drift(data_drift=None, ambang=None, jam_penuh=None):
    """Deteksi drift KOSAKATA antara periode lama (jam < jam_penuh) dan baru
    (jam ≥ jam_penuh). Kosakata lama = gabungan kata dari semua kueri lama.

    Return {'kosakata_lama': int, 'kata_baru': [kata unik periode baru yang
    TIDAK ada di kosakata lama, urut abjad], 'proporsi_baru': n kueri periode
    baru yang mengandung ≥ 1 kata baru ÷ total kueri baru (round 4; kosong →
    0.0), 'drift': bool (proporsi_baru ≥ ambang)}."""
    # TODO
    raise NotImplementedError


def ringkas_permintaan(hasil_model):
    """Baris log terstruktur untuk trace produksi:
    {'model': ..., 'abstain': bool, 'token_out': n, 'singkat': 40 char pertama
    jawaban} — token_out dihitung dari jawaban (token_estimasi)."""
    # TODO
    raise NotImplementedError
