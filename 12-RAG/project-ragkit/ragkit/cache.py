"""Bagian 4 — Cache konteks retrieval.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `CacheRetrieval` meniru CachePrompt Bab 11 tapi untuk KONTEKS (bukan
    jawaban LLM): kunci = kueri.strip().lower(), TTL tick logis, dan
    hit/miss/expired DIHITUNG TERPISAH (expired ≠ miss!);
      - `ambil(kunci, sekarang)` → None (miss/expired) atau hasil tersimpan;
      - `simpan(kunci, hasil, sekarang)`;
      - `statistik()` → {"hit","miss","expired","ukuran","hit_rate"};
        hit_rate = 0.0 bila belum ada permintaan.
  - `retrieve_bercache(retrieve, kueri, cache, sekarang=0, top_k=3,
    latensi_ms=LATENSI_CACHE_MS)`:
      - hit  → {"hasil": chunks, "cache": "hit",  "latensi_ms": latensi_ms}
      - miss → embed + cari + simpan, latensi = latensi_ms * 8, "cache": "miss";
      - hasil yang disimpan = LIST CHUNK (bukan dict retrieval penuh).
"""
from ragkit.data import LATENSI_CACHE_MS


class CacheRetrieval:
    """Cache konteks retrieval: kunci = kueri; TTL tick logis."""

    def __init__(self, ttl=10):
        # TODO
        raise NotImplementedError

    def kunci(self, kueri):
        """Kunci cache: kueri tanpa spasi tepi, huruf kecil."""
        # TODO
        raise NotImplementedError

    def ambil(self, kunci, sekarang):
        """Hasil bila masih segar; None bila miss ATAU kedaluwarsa."""
        # TODO
        raise NotImplementedError

    def simpan(self, kunci, hasil, sekarang):
        # TODO
        raise NotImplementedError

    def statistik(self):
        """hit, miss, expired, ukuran, hit_rate (0.0 bila kosong)."""
        # TODO
        raise NotImplementedError


def retrieve_bercache(retrieve, kueri, cache, sekarang=0, top_k=3,
                      latensi_ms=LATENSI_CACHE_MS):
    """Retrieve lewat cache: hit → tanpa embedding/pencarian ulang."""
    # TODO
    raise NotImplementedError


def tolok_ukur(retrieve, n_permintaan=None, aturan=None, ambang=0.30):
    """Ukur manfaat cache retrieval di atas pipeline (untuk notebook).

    aturan: daftar teks kueri per permintaan (data.TOLOK_UKUR bila None).
    Return: {"latensi_ms": [...], "cache": [...], "hit_rate": float,
             "panggilan_pencarian": int, "total_biaya_usd": float,
             "jawaban": [...]}
    Lihat solusi/ragkit_ref.py sebagai referensi penuh (opsional).
    """
    # TODO (bagian notebook — tidak diuji tests/)
    raise NotImplementedError
