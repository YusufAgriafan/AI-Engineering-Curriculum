"""TODO Bagian 4 — Memory: riwayat giliran + ringkasan + cache retrieval.

Kontrak lengkap ada di tests/test_memory.py. Isi class/fungsi di bawah
sampai semua test hijau.
"""
from .nlu import token_estimasi


class MemoriSesi:
    """Riwayat percakapan + cache retrieval per sesi.

    - tambah(peran, teks): catat {'peran', 'teks', 'token'} ke riwayat;
    - konteks(): max_giliran TERAKHIR dalam urutan ASLI (bukan dibalik!);
    - ringkas(): semua giliran → 'peran: teks' per baris;
    - cache: kunci = kueri strip().lower() → hasil; hit/miss tercatat.
    """

    def __init__(self, maks_giliran=4):
        # TODO
        raise NotImplementedError

    def tambah(self, peran, teks):
        """Catat satu giliran; return dict giliran yang disimpan."""
        # TODO
        raise NotImplementedError

    def konteks(self):
        """max_giliran terakhir, urutan asli — konteks yang dikirim ke LLM."""
        # TODO
        raise NotImplementedError

    def ringkas(self):
        """Seluruh riwayat → satu string 'peran: teks' per baris."""
        # TODO
        raise NotImplementedError

    def cache_get(self, kueri):
        """Ambil dari cache; None bila miss. Kunci dinormalisasi strip().lower()."""
        # TODO
        raise NotImplementedError

    def cache_put(self, kueri, nilai):
        """Simpan ke cache dengan kunci dinormalisasi."""
        # TODO
        raise NotImplementedError


def hitung_biaya(pesan, model="api_mini"):
    """Total token riwayat (token_estimasi per giliran) × tarif HARGA → USD."""
    # TODO
    raise NotImplementedError
