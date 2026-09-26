"""Bagian 3 — Abstain gate dari skor retrieval.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `abstaikan(hasil, ambang=0.30)`: bila skor retrieval terbaik < ambang,
    timpa hasil jadi abstain:
        jawaban  = "tidak ada di dokumen" (SEP dari data.py)
        sitasi   = []
        sumber   = []
        abstain  = True
        + kunci "abstaikan_oleh" = skor pembuktian (round 4 desimal)
  - hasil yang SUDAH abstain → dikembalikan apa adanya (tanpa kunci baru);
  - hasil yang aman (skor >= ambang) → kembali UTUH, tidak ada kunci baru;
  - jangan mengubah hasil yang diberikan (dict BARU).
"""
from ragkit.data import SEP_DETEKSI_TIDAK_ADA


def abstaikan(hasil, ambang=0.30):
    """Timpa jawaban jadi abstain bila skor terbaik < ambang."""
    # TODO
    raise NotImplementedError
