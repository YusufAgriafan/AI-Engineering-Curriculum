"""TODO Bagian 1 — Konfigurasi 12-factor: env → config, validasi skema.

Prinsip: konfigurasi berbeda antar environment (dev/staging/prod), kode SAMA.
Konfigurasi masuk lewat environment variable; paket tidak boleh hardcode.
"""
from .data import SCHEMA_ENV


def baca_env(env, schema=None):
    """Env dict (string) → config dict bertipe sesuai SCHEMA_ENV.

    Aturan (12-factor + fail fast):
    - kunci yang ADA di env  → dikonversi ke tipe skema ('int'/'float'/'str');
      konversi gagal → raise ValueError dengan pesan memuat NAMA KUNCI
      (gagal nyaring saat deploy, bukan diam-diam pakai default).
    - kunci yang TIDAK ada   → pakai default dari skema.
    - kunci env di luar skema → diabaikan (tidak bocor ke config).
    - skema None → pakai SCHEMA_ENV.
    """
    # TODO
    raise NotImplementedError


def config_layanan(env=None):
    """Bungkus baca_env → config siap pakai; validasi nilai kritis:
    - PORT, RATE_LIMIT_PER_MENIT, CANARY_PERSEN harus > 0
    - AMBAT_ABSTAIN, SKOR_MIN harus 0 < nilai < 1
    - MODEL_API harus kunci HARGA (biaya bisa dihitung)
    Melanggar → ValueError (pesan memuat nama kunci)."""
    # TODO
    raise NotImplementedError
