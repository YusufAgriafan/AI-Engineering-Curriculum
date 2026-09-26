"""Bagian 8 — Evaluasi ujung-ke-ujung.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `evaluasi_ujung_ke_ujung(jalankan, kueri_uji=None, ambang=0.30)`:
      untuk tiap kueri uji → `jalankan(kueri)` → dict hasil pipeline.
      "benar" bila `jawaban_emas` (lowercase) ada di dalam jawaban (lowercase).
      Ditambah KUERI_LUAR_CORPUS: "abstain_benar" bila hasil abstain DAN
      jawabannya memuat SEP_DETEKSI_TIDAK_ADA.
      Return {"per_kueri": [{"id", "benar", "abstain", "jawaban"(60 char)}...],
              "akurasi": benar/jumlah, "abstain_benar": bool,
              "jawaban_luar": str}
"""
from ragkit.data import KUERI_LUAR_CORPUS, KUERI_UJI, SEP_DETEKSI_TIDAK_ADA


def evaluasi_ujung_ke_ujung(jalankan, kueri_uji=None, ambang=0.30):
    """Jawab semua kueri uji (+1 di luar korpus) → akurasi & abstain."""
    # TODO
    raise NotImplementedError
