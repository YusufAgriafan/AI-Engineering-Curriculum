"""Bagian 4 — Cache konten-address + TTL.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `kunci` = sha256 dari JSON kanonik {"model", "messages", "params"} (sort_keys)
    dipotong 16 karakter hex
  - TTL dihitung dalam "tick" logis (integer), bukan jam dinding:
    entri kedaluwarsa bila `sekarang - tick_simpan >= ttl`
  - `ambil` menaikkan hit / miss / expired; entri kedaluwarsa DIHAPUS
  - `hit_rate` = hit / (hit + miss + expired); 0.0 bila belum ada apa-apa
"""
import hashlib
import json


class CachePrompt:
    """Cache konten-address dengan TTL dalam satuan tick logis."""

    def __init__(self, ttl=3):
        self.ttl = ttl
        self._isi = {}
        self.hit = 0
        self.miss = 0
        self.expired = 0

    def kunci(self, model, messages, params=None):
        """Kunci cache deterministik (16 hex)."""
        # TODO
        raise NotImplementedError

    def ambil(self, kunci, sekarang):
        """Hasil bila masih segar; None bila miss/kedaluwarsa."""
        # TODO
        raise NotImplementedError

    def simpan(self, kunci, hasil, sekarang):
        """Simpan hasil pada tick `sekarang`."""
        # TODO
        raise NotImplementedError

    def bersihkan(self, sekarang):
        """Buang semua entri kedaluwarsa; return jumlah yang dibuang."""
        # TODO
        raise NotImplementedError

    def statistik(self):
        """{'hit', 'miss', 'expired', 'ukuran', 'hit_rate'}."""
        # TODO
        raise NotImplementedError


def panggil_bercache(provider, model, messages, cache, sekarang, **params):
    """Panggil provider, lewati kalau cache masih segar. Hasil punya kunci 'cache'."""
    # TODO
    raise NotImplementedError
