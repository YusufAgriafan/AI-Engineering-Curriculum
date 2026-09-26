"""Bagian 3 — Rantai fallback multi-provider.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `providers`: {"utama": ProviderPalsu, "cadangan": ProviderPalsu}
  - `rantai`: [{"provider", "model", "settings" (opsional)}, ...]
  - Setelan tautan MENIMPA setelan dasar (`settings` argumen)
  - Provider tidak ada -> dicatat `jenis="provider_tidak_ada"` lalu lanjut
  - Semua tautan gagal -> raise `retry.GalatSemuaPercobaan` dengan riwayat lengkap
  - Hasil sukses ditambah: dicoba, lompatan (index tautan), tautan
"""
from .retry import GalatSemuaPercobaan, panggil_dengan_retry
from .transport import GalatAPI


def panggil_dengan_fallback(providers, rantai, messages, *, settings=None, **params):
    """Coba tiap tautan rantai sampai satu berhasil."""
    # TODO
    raise NotImplementedError
