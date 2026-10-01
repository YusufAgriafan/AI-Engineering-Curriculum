"""GIVEN — Sistem RAG toko (Bab 15) + dua "model" API mini untuk lapisan ops.

Semua deterministik. Dua model tiruan:
- model_besar: generator ekstraktif 2 kalimat (jawaban "lebih lengkap", mahal)
- model_mini : generator ekstraktif 1 kalimat (murah, cukup untuk kasus mudah)

Keduanya kontrak sama dengan sistem Bab 15:
    jawab(teks, ambang=None) -> {'jawaban': str, 'skor_retrieval': float,
                                 'meta': {'model': ..., 'token_in': int,
                                          'token_out': int}}
Ambang abstain datang dari KONFIGURASI (12-factor) — bukan hardcode di model.
"""
import re

from .data import SCHEMA_ENV

AMBAT_DEFAULT = SCHEMA_ENV["AMBAT_ABSTAIN"][1]   # 0.20 (default 12-factor)

FRASA_TIDAK_TAHU = "Maaf, saya tidak memiliki informasi tentang hal tersebut di dokumen."

KB = [
    {"id": "d1", "judul": "Kebijakan Pengembalian",
     "teks": "Barang bisa dikembalikan dalam 14 hari sejak diterima dengan tagihan asli. "
             "Barang elektronik butuh kelengkapan dus dan aksesori."},
    {"id": "d2", "judul": "Pengiriman",
     "teks": "Pengiriman reguler datang 3 sampai 5 hari kerja. Pengiriman kilat datang 1 hari kerja "
             "untuk kota besar. Biaya kirim reguler gratis untuk belanja di atas 100 ribu."},
    {"id": "d3", "judul": "Pembayaran",
     "teks": "Pembayaran bisa transfer bank, e-wallet, dan COD. COD maksimal 2 juta rupiah. "
             "Pembayaran e-wallet dapat cashback 5 persen setiap hari Jumat."},
]


def _kata(teks):
    return {k for k in re.findall(r"[a-z0-9]+", str(teks).lower()) if len(k) > 2}


def cari(kueri):
    """Retrieval kata kunci → [(doc, skor)] skor > 0 urut menurun (Bab 15)."""
    kata = _kata(kueri)
    if not kata:
        return []
    hasil = []
    for doc in KB:
        gabung = (doc["judul"] + " " + doc["teks"]).lower()
        cocok = sum(1 for k in kata if k in gabung)
        skor = cocok / len(kata)
        if skor > 0:
            hasil.append((doc, round(skor, 4)))
    hasil.sort(key=lambda h: (-h[1], h[0]["id"]))
    return hasil


def _kalimat_terbaik(konteks, kueri):
    """Kalimat konteks paling banyak memuat kata kueri (generator ekstraktif)."""
    kata = _kata(kueri)
    terbaik, skor_b = "", -1
    for kalimat in re.split(r"(?<=[.!?])\s+", konteks):
        if not kalimat.strip():
            continue
        k = kalimat.lower()
        skor = sum(1 for w in kata if w in k)
        if skor > skor_b:
            terbaik, skor_b = kalimat.strip(), skor
    return terbaik


def _jawab_model(nama, jumlah_kalimat, teks, ambang=None):
    """Mesin generator bersama: abstain bila skor retrieval < ambang konfigurasi."""
    ambang = AMBAT_DEFAULT if ambang is None else ambang
    asli = str(teks)
    hasil = cari(asli)
    skor = hasil[0][1] if hasil else 0.0
    if not asli.strip() or skor < ambang:
        return {"jawaban": FRASA_TIDAK_TAHU, "skor_retrieval": skor,
                "meta": {"model": nama, "abstain": True}}
    konteks = hasil[0][0]["teks"]
    kalimat = [s for s in re.split(r"(?<=[.!?])\s+", konteks) if s.strip()]
    # Kalimat diurutkan 'paling relevan dulu', lalu kembalikan N kalimat pertama
    # dalam URUTAN ASLI teks (bacaan natural, bukan urutan skor).
    berperingkat = sorted(kalimat,
                          key=lambda s: -sum(1 for w in _kata(asli) if w in s.lower()))
    pilihan = berperingkat[:jumlah_kalimat]
    terpilih = [s for s in kalimat if s in pilihan]
    jawaban = " ".join(terpilih)
    return {"jawaban": jawaban, "skor_retrieval": skor,
            "meta": {"model": nama, "abstain": False}}


def model_besar(teks, ambang=None):
    """"api_besar": 2 kalimat — lebih lengkap, token & biaya lebih besar."""
    return _jawab_model("api_besar", 2, teks, ambang)


def model_mini(teks, ambang=None):
    """"api_mini": 1 kalimat — murah, cukup untuk kasus mudah."""
    return _jawab_model("api_mini", 1, teks, ambang)
