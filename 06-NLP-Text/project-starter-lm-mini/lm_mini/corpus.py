"""Korpus mini terkunci untuk project LM-mini Bab 6.

JANGAN DIUBAH — semua angka di tests dan starter.ipynb mengacu pada korpus yang
dihasilkan file ini (seed terkunci).

Struktur:
    KORPUS_TOPIK : list kalimat (token) — topik (hewan/kendaraan/makanan) +
                   sifat positif/negatif + kata fungsional. Untuk skip-gram.
    KAT_OF       : kamus kategori kata (hanya untuk EVALUASI purity).
    POS, NEG     : daftar kata sifat positif/negatif (untuk label sentimen).
    KALIMAT_LM   : list kalimat (string) pola sederhana — untuk bigram & RNN
                   char-level.
"""

import numpy as np

__all__ = ["TOPIK", "POS", "NEG", "FUNGSIO", "KAT_OF", "buat_korpus_topik",
           "muat_korpus_topik", "KALIMAT_LM"]

TOPIK = {
    "hewan": ["kucing", "anjing", "burung", "kelinci"],
    "kendaraan": ["mobil", "motor", "sepeda", "kereta"],
    "makanan": ["nasi", "roti", "bakso", "soto"],
}
POS = ["bagus", "enak", "senang", "mantap"]
NEG = ["jelek", "payah", "kecewa", "buruk"]
FUNGSIO = ["dan", "itu", "di"]

KAT_OF = {}
for _k, _ws in TOPIK.items():
    for _w in _ws:
        KAT_OF[_w] = _k
for _w in POS:
    KAT_OF[_w] = "sifat_pos"
for _w in NEG:
    KAT_OF[_w] = "sifat_neg"
for _w in FUNGSIO:
    KAT_OF[_w] = "fungsional"


def buat_korpus_topik(rng, n=400):
    """Kalimat sintetis: klausa topik + (70%) klausa sentimen 'itu <sifat>'."""
    kalimat = []
    kat3 = list(TOPIK)
    for _ in range(n):
        kat = rng.choice(kat3)
        toks = [str(x) for x in rng.choice(TOPIK[kat], size=int(rng.integers(2, 5)),
                                           replace=True)]
        for _f in range(int(rng.integers(0, 3))):
            toks.insert(int(rng.integers(0, len(toks) + 1)), str(rng.choice(FUNGSIO)))
        if rng.random() < 0.70:
            pool = POS if rng.random() < 0.5 else NEG
            toks.append("itu")
            for _s in range(int(rng.integers(1, 3))):
                toks.append(str(rng.choice(pool)))
            if rng.random() < 0.3:
                toks.append(str(rng.choice(pool)))
        kalimat.append(toks)
    return kalimat


def muat_korpus_topik(seed=42, n=400):
    """Return (korpus_token, label_sentimen per kalimat: 1 pos / -1 neg / 0 netral)."""
    rng = np.random.default_rng(seed)
    return buat_korpus_topik(rng, n=n)


KALIMAT_LM = [
    "adi makan nasi", "budi minum air", "citra membaca buku",
    "dewi menulis surat", "eka bermain bola", "fajar melihat kaca",
    "gita menyanyi lagu", "hadi membuka kelas", "indah menanam bunga",
    "joko menghitung uang", "kartika menjahit baju", "lina menjual buah",
    "maya menyapu lantai", "nanda mendengar kabar", "okta membeli roti",
    "putri menyiram tanaman", "rizky mengejar bus", "sari mencuci piring",
    "tono memperbaiki motor", "utami mengaduk adonan", "vina mengambil paket",
    "wawan mengangkat meja", "yuni menjahit kerudung", "zahra mengeringkan baju",
    "bagus membawa kasur", "candra mengisi gelas", "dian menyusun kartu",
    "fitri menata piring", "gilang menyalakan lampu", "husni menutup pintu",
    "irma menggulai sayur", "krisna meniup terompet",
    "makan nasi itu enak", "minum air itu menyegarkan",
    "membaca buku itu bagus", "menulis surat itu mudah",
    "bermain bola itu seru", "melihat kaca itu aneh",
    "menyanyi lagu itu indah", "membuka kelas itu pagi",
    "adi minum teh", "budi makan roti", "citra menjahit baju",
    "dewi menyiram bunga", "eka membaca koran", "fajar mengejar kucing",
    "gita membeli buku", "hadi menyapu halaman", "indah menjahit gaun",
    "joko menanam padi", "kartika menyusun bunga", "lina menghitung uang",
    "maya membuka pintu", "nanda memperbaiki sepeda", "okta mengambil air",
    "putri menulis surat", "rizky menyiram tanaman", "sari mengupas kentang",
    "tono membawa tas", "utami meniup peluit", "vina menjahit sarung",
    "wawan mengaduk kopi", "yuni menata meja", "zahra mengeringkan rambut",
    "makan roti itu enak", "minum teh itu nikmat", "membaca koran itu bagus",
    "menulis surat itu mudah", "bermain bola itu seru", "melihat kaca itu aneh",
]

if __name__ == "__main__":
    korpus = muat_korpus_topik()
    print("contoh kalimat topik:", [" ".join(s) for s in korpus[:3]])
    print("jumlah kalimat topik:", len(korpus))
    print("jumlah kalimat LM  :", len(KALIMAT_LM))
    print("contoh LM          :", KALIMAT_LM[:2])
