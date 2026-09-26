"""Bagian 2 — Retry + backoff eksponensial.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Hanya galat TRANSIEN (`transport.JENIS_TRANSIEN`) yang dicoba ulang
  - Galat fatal (bad_request / model tidak dikenal) LANGSUNG dilempar
  - Backoff: basis * faktor^(n-1), dibatasi maks_ms; TIDAK menunggu setelah
    percobaan terakhir
  - Semua waktu disuntik: `sleep_fn(ms)` — jangan panggil time.sleep
  - Gagal total -> raise `GalatSemuaPercobaan` (punya atribut `riwayat`:
    daftar (percobaan, jenis))
"""
from .transport import JENIS_TRANSIEN, GalatAPI


class GalatSemuaPercobaan(GalatAPI):
    """Semua percobaan gagal. `riwayat` = daftar (percobaan, jenis)."""

    def __init__(self, model, riwayat, pesan="semua percobaan gagal"):
        super().__init__("semua_gagal", model, pesan)
        self.riwayat = list(riwayat)


def backoff_ms(percobaan, basis_ms=1000, faktor=2, maks_ms=8000):
    """Exponential backoff: basis * faktor^(n-1), dibatasi maks_ms."""
    # TODO
    raise NotImplementedError


def panggil_dengan_retry(provider, model, messages, *, maks_percobaan=4, basis_ms=1000,
                         faktor=2, maks_ms=8000, timeout_ms=5000, sleep_fn=None,
                         **params):
    """Panggil provider dengan retry + backoff.

    Hasil sukses ditambah kunci: attempts, timpenungguan_ms, total_ms, riwayat.
    `total_ms` = latensi panggilan yang berhasil + total jeda backoff.
    """
    # TODO
    raise NotImplementedError
