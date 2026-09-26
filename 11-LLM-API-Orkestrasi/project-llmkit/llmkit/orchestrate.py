"""Bagian 7 — Orkestrasi: prompt chaining + laporan per langkah.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Tiap langkah: prompt = `template.format(teks=teks)`, model & suhu per langkah
  - Tiap langkah memakai retry (`retry.panggil_dengan_retry`) dengan `maks_percobaan`
  - Langkah gagal DICATAT (`sukses=False`, `galat=<jenis>`, biaya 0.0)
  - `berhenti_saat_gagal=True` -> stop setelah langkah pertama yang gagal
  - Return: langkah (rincian), berhasil, gagal, total_biaya_usd, total_ms, teks_akhir
"""
from .cost import biaya_hasil
from .retry import GalatSemuaPercobaan, panggil_dengan_retry
from .transport import GalatAPI


def jalankan_pipeline(provider, langkah, teks, *, maks_percobaan=3, timeout_ms=5000,
                      berhenti_saat_gagal=False, settings=None):
    """Jalankan langkah berurutan, laporkan tiap langkah."""
    # TODO
    raise NotImplementedError
