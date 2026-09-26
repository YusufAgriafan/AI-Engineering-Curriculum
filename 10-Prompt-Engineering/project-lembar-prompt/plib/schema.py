"""Bagian 4 — Validasi JSON-schema mini.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Return DAFTAR error (kosong = valid), format stabil:
      "missing:<field>" | "type:<field>" | "minimum:<field>" | "maximum:<field>"
      "enum:<field>"   | "length:<field>"| "extra:<field>"  | "type:<root>"
  - "type" boleh string atau DAFTAR tipe (mis. ["string", "null"])
  - `True`/`False` BUKAN integer (hati-hati: bool adalah subclass int di Python)
  - Batas minimum/maximum dilewati bila nilainya null
"""
from .data import SKEMA_PENUH


def validate(data, skema=None):
    """Daftar error; kosong berarti data valid. `skema=None` -> SKEMA_PENUH."""
    # TODO
    raise NotImplementedError


def valid(data, skema=None):
    """True bila tidak ada error."""
    # TODO
    raise NotImplementedError
