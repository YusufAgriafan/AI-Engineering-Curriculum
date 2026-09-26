"""Bagian 1 — Anatomi prompt, bagian wajib, anggaran token.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Header bagian = "[NAMA]" dalam huruf besar, urut sesuai `data.BAGIAN`
  - Bagian wajib: ROLE, KONTEKS, TUGAS, FORMAT, CONSTRAINT, INPUT (CONTOH opsional)
  - Estimasi token = ceil(len(teks) / rasio), rasio default 4.0 karakter/token
"""
import math

from .data import BAGIAN, BAGIAN_WAJIB


def bagian_ditemukan(prompt):
    """Bagian mana saja yang benar-benar ada di prompt (urutan kanonik)."""
    # TODO
    raise NotImplementedError


def bagian_hilang(prompt):
    """Bagian WAJIB yang belum ada (urutan kanonik)."""
    # TODO
    raise NotImplementedError


def prompt_lengkap(prompt):
    """True bila tidak ada bagian wajib yang hilang."""
    # TODO
    raise NotImplementedError


def build_prompt(*, role, konteks, tugas, format_rules, constraints,
                 pertanyaan, contoh=""):
    """Rakit prompt produksi; bagian dengan isi kosong DILEWATI.

    Bentuk satu bagian: "[NAMA]\\n<isi tanpa spasi berlebih>",
    antar bagian dipisah satu baris kosong.
    """
    # TODO
    raise NotImplementedError


def estimasi_token(teks, rasio=4.0):
    """Perkiraan jumlah token: ceil(len(teks) / rasio)."""
    # TODO
    raise NotImplementedError


def potong_budget(teks, maks_token, rasio=4.0):
    """Potong ke anggaran token TANPA memotong kata di tengah."""
    # TODO
    raise NotImplementedError
