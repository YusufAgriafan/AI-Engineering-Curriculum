"""TODO Bagian 3 — Guardrail: aksi berisiko + sanitasi injection.

Kontrak lengkap ada di tests/test_guardrail.py. Isi fungsi-fungsi di bawah
sampai semua test hijau.
"""
from .data import AKSI_BERISIKO

POLA_INJECTION = ("ignore previous", "abaikan instruksi", "system prompt",
                  "lupakan aturan", "reveal", "kirim kode", "api key")


def butuh_konfirmasi(nama_aksi, aksi_berisiko=None):
    """Nama aksi → True bila termasuk AKSI_BERISIKO (perbandingan persis)."""
    # TODO
    raise NotImplementedError


def sanitasi(teks):
    """Netralkan perintah tersembunyi di teks (tool output / input user).

    - Return SALINAN; teks asli tidak berubah;
    - Setiap frasa POLA_INJECTION yang muncul (case-insensitive) diganti
      '[dihapus]' — sekali per kemunculan;
    - Baris baru diganti spasi (anti 'instruksi di baris baru');
    - Hasil dipangkas strip().
    """
    # TODO
    raise NotImplementedError


def amankan_observasi(observasi):
    """Observasi tool → versi aman untuk konteks LLM.

    - dict → salinan dengan SEMUA nilai string di-sanitasi (rekursif ke
      dalam list/dict); kunci tidak diubah;
    - list → salinan berisi elemen yang diamankan;
    - string → sanitasi langsung; tipe lain → dilewatkan apa adanya.
    Input TIDAK boleh berubah (fungsi murni).
    """
    # TODO
    raise NotImplementedError
