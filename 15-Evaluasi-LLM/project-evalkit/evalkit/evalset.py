"""TODO Bagian 1 — Evalset: muat, validasi, split deterministik.

Kontrak lengkap ada di tests/test_evalset.py. Isi fungsi di bawah sampai
semua test hijau.

Sambungan: split dev/test persis filosofi Bab 3-13 — dev untuk iterasi,
test hanya disentuh saat rilis besar. Split WAJIB deterministik (tanpa
random): evaluasi yang bisa berubah tiap run tidak bisa dibandingkan.
"""
from .data import KATEGORI


def validasi_evalset(evalset):
    """Evalset (list dict) → (ok, pesan_error).

    Aturan: list non-kosong; tiap item dict dengan kunci 'id', 'kategori',
    'tanya'; 'kategori' ada di KATEGORI; 'id' unik; 'tanya' string non-kosong
    SETELAH strip — kecuali kategori 'edge' (memang menguji input kosong).
    Kasus 'abstain'/'injection'/'hallucination'/'edge' harus punya kunci None
    (tidak ada jawaban benar) — kalau diisi, itu evalset yang menyesatkan.
    """
    # TODO
    raise NotImplementedError


def hitung_kategori(evalset):
    """Evalset → dict kategori → jumlah kasus (semua kategori KATEGORI muncul,
    kategori tanpa kasus bernilai 0)."""
    # TODO
    raise NotImplementedError


def split_dev_test(evalset, rasio_test=0.25):
    """Split deterministik interleave TANPA random — pola Bab 13.

    Index yang masuk test: (i*7 + SEED) % 100 < rasio_test*100 dengan
    SEED = 15 (dari .data). Return (dev, test) — dua list baru, urutan asli
    dipertahankan di masing-masing list.
    """
    # TODO
    raise NotImplementedError
