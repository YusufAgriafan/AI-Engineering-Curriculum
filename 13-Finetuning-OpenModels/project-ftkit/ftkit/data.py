"""Dataset, tarif, & konfigurasi terkunci project Bab 13 — JANGAN DIUBAH.

TANPA GPU dan TANPA jaringan — pola yang sama dengan Bab 9-12:

- "Model yang di-fine-tune" di sini adalah `ftkit/model.py`: klasifier
  bag-of-words TINY (mean embedding → 1 logit → sigmoid) yang dilatih dengan
  SGD sungguhan di CPU (< 0,1 detik). Semua matematika training asli:
  forward, loss BCE, gradien analitik, gradient check numerik, early stopping.

- "LLM besarnya" tidak ada — yang dilatih adalah KEPALA keputusan kecil. Ide
  pedagogisnya: LoRA mengganti HANYA adapter kecil di atas bobot beku; di sini
  kamu melihat matematika update-param-kecil di atas fungsi loss sungguhan,
  tanpa perlu VRAM. Skala naik, bentuknya sama.

Angka-angka (loss, akurasi, epoch, rasio LoRA, VRAM) TERKUNCI SEED 13 dan
diverifikasi `tests/` + `_calib.py` + `_verify_project.py`.
"""
SEED = 13

# Label kanonik
LABEL_POS = "positif"
LABEL_NEG = "negatif"
LABELS = (LABEL_POS, LABEL_NEG)

# --------------------------------------------------------------------------
# 1. Dataset sentimen ulasan e-commerce (24 contoh: 16 train + 8 val)
#    Kata sengaja KONSISTEN arahnya (bagus/puas/cepat selalu positif;
#    jelek/kecewa/lambat selalu negatif) supaya model tiny bisa konvergen
#    dan angka training terkunci. 1 contoh SENGAJA ambigu (DATA_OOD).
# --------------------------------------------------------------------------
DATA_SENTIMEN = [
    # ---- train (16) ----
    {"id": "t01", "split": "train", "teks": "Barang bagus, pengiriman cepat", "label": "positif"},
    {"id": "t02", "split": "train", "teks": "Kualitas jelek, saya menyesal", "label": "negatif"},
    {"id": "t03", "split": "train", "teks": "Puas dengan pembelian ini, mantap", "label": "positif"},
    {"id": "t04", "split": "train", "teks": "Aplikasinya ribet dan lambat", "label": "negatif"},
    {"id": "t05", "split": "train", "teks": "Mudah dipasang, hasil rapi", "label": "positif"},
    {"id": "t06", "split": "train", "teks": "Paket datang rusak, kecewa", "label": "negatif"},
    {"id": "t07", "split": "train", "teks": "Harga murah tapi kualitas bagus", "label": "positif"},
    {"id": "t08", "split": "train", "teks": "Baterai boros, payah", "label": "negatif"},
    {"id": "t09", "split": "train", "teks": "Pelayanan ramah, sangat puas", "label": "positif"},
    {"id": "t10", "split": "train", "teks": "Barang tidak sesuai deskripsi, kecewa", "label": "negatif"},
    {"id": "t11", "split": "train", "teks": "Pengiriman cepat, packing rapi", "label": "positif"},
    {"id": "t12", "split": "train", "teks": "Sudah dua kali rusak, payah", "label": "negatif"},
    {"id": "t13", "split": "train", "teks": "Suka sekali, recommended banget", "label": "positif"},
    {"id": "t14", "split": "train", "teks": "Suara bocor, minta ganti, jelek", "label": "negatif"},
    {"id": "t15", "split": "train", "teks": "Mantap, worth it, puas", "label": "positif"},
    {"id": "t16", "split": "train", "teks": "Lambat banget, menyesal beli", "label": "negatif"},
    # ---- val (8) ----
    {"id": "v01", "split": "val", "teks": "Kualitas oke, harga terjangkau, puas", "label": "positif"},
    {"id": "v02", "split": "val", "teks": "Kecewa, barang rusak saat tiba", "label": "negatif"},
    {"id": "v03", "split": "val", "teks": "Cepat sampai, barang bagus", "label": "positif"},
    {"id": "v04", "split": "val", "teks": "Ribet cara pakainya, menyesal", "label": "negatif"},
    {"id": "v05", "split": "val", "teks": "Suka, mudah dipakai, mantap", "label": "positif"},
    {"id": "v06", "split": "val", "teks": "Baterai boros, suara jelek", "label": "negatif"},
    {"id": "v07", "split": "val", "teks": "Pengiriman cepat, sangat puas", "label": "positif"},
    {"id": "v08", "split": "val", "teks": "Payah, lambat, tidak puas", "label": "negatif"},
]

# Contoh di luar distribusi: kata-kata tidak ada di train → keyakinan rendah.
# Sambungan Bab 12: model yang tahu dia tidak yakin = dasar abstain.
DATA_OOD = {
    "id": "o1",
    "teks": "Lumayan standar saja sih",
    "label": None,
    "catatan": "ambigu/netral — tidak pernah muncul di train",
}

# --------------------------------------------------------------------------
# 2. Format SFT (JSONL ala Alpaca): instruksi → respons.
#    SENGAJA tidak seimbang (6 positif, 2 negatif) untuk latihan oversample.
# --------------------------------------------------------------------------
TEMPLATE_INSTRUKSI = ("Klasifikasikan sentimen ulasan berikut. "
                      "Jawab HANYA satu kata: positif atau negatif.")

FORMAT_SFT = [
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Barang bagus, pengiriman cepat", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Kualitas jelek, saya menyesal", "respons": "negatif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Puas dengan pembelian ini, mantap", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Aplikasinya ribet dan lambat", "respons": "negatif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Mudah dipasang, hasil rapi", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Paket datang rusak, kecewa", "respons": "negatif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Pelayanan ramah, sangat puas", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Suara bocor, minta ganti, jelek", "respons": "positif"},
]

# ---------------------------------------------------------------------------
# 3. Tarif API (USD per 1 JUTA token) — konsisten dengan Bab 11.
#    "lokal" = open-weight di GPU sendiri: biaya marginal token = 0.
# ---------------------------------------------------------------------------
HARGA = {
    "api_besar": {"input": 2.50, "output": 10.0},
    "api_mini": {"input": 0.15, "output": 0.60},
    "lokal": {"input": 0.0, "output": 0.0},
}

# --------------------------------------------------------------------------
# 4. Varian kuantisasi (skor = akurasi eval pada task yang sama).
#    Pola nyata: makin agresif kuantisasi, makin kecil & cepat, skor turun.
# --------------------------------------------------------------------------
VARIAN_KUANTISASI = [
    {"nama": "fp16", "bit": 16, "skor": 0.82},
    {"nama": "int8", "bit": 8, "skor": 0.81},
    {"nama": "q4", "bit": 4, "skor": 0.78},
]
