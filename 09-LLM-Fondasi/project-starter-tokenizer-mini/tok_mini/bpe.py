"""Bagian 1 — BPE dari nol.  TODO: kamu yang mengisi.

Konvensi wajib (dipakai tests/):
  - simbol kata = tuple karakter + penanda akhir kata "</w>"
  - merge terpilih: frekuensi tertinggi; SERI -> leksikografis TERKECIL
  - encode: apply merges BERURUTAN
"""
from collections import Counter


def ke_simbol(kata):
    """'low' -> ('l', 'o', 'w', '</w>').  (GIVEN)"""
    return tuple(kata) + ("</w>",)


def normalisasi(teks):
    """Lowercase + pisah spasi. Return list kata.  (GIVEN)"""
    return teks.lower().split()


def hitung_pasangan(simbol_per_kata):
    """Counter semua pasangan bersebelahan dari list of tuple simbol."""
    # TODO
    raise NotImplementedError


def best_pair(counter):
    """Pasangan terfrekuensi tertinggi; seri -> leksikografis terkecil.

    Petunjuk: min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    """
    # TODO
    raise NotImplementedError


def merge_pasangan(simbol_per_kata, pasangan):
    """Gabungkan setiap kemunculan bersebelahan `pasangan` jadi satu simbol.

    ('l','o','w','</w>') dengan merge ('l','o') -> ('lo','w','</w>').
    Return list of tuple baru (jangan memodifikasi input).
    """
    # TODO
    raise NotImplementedError


def latih_bpe(corpus, n_merge):
    """Jalankan n_merge iterasi BPE. Return (merges, states).

    merges: list of (pasangan, frekuensi) berurutan.
    states: snapshot list-of-tuples SETELAH tiap merge — states[0] = kondisi awal
    (panjang states = jumlah merge + 1). Berhenti lebih awal jika tak ada pasangan.
    """
    # TODO
    raise NotImplementedError


def apply_merges(simbol, merges):
    """Apply merges BERURUTAN ke satu tuple simbol. Return tuple hasil."""
    # TODO
    raise NotImplementedError
