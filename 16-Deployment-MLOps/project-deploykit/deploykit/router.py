"""TODO Bagian 3 — Router 2-tier + fallback: murah dulu, mahal kalau perlu.

Router = classifier mini (skill Bab 3) di depan dua model: pertanyaan dengan
kata sulit (KATA_SULIT) → api_besar; else → api_mini. Fallback: model utama
error → model cadangan. Keduanya mengecat meta agar keputusan bisa diaudit.
"""
from .data import HARGA, KATA_SULIT, PROFIL_TOKEN
from .sistem import model_besar, model_mini
from .tokenizer import token_estimasi


def pilih_tier(pertanyaan):
    """Pertanyaan → 'api_besar' bila memuat salah satu KATA_SULIT
    (case-insensitive, cek frasa di dalam teks), else 'api_mini'."""
    # TODO
    raise NotImplementedError


def jawab_router(pertanyaan, ambang=None, fn_besar=None, fn_mini=None):
    """Jawab dengan tier hasil pilih_tier; meta diextend:
    meta['tier'] = tier terpilih; meta['router'] = 'hard' | 'mudah'.
    fn_besar/fn_mini untuk injeksi model tiruan (default model_* dari sistem).
    Kontrak jawaban SAMA dengan model: {'jawaban', 'skor_retrieval', 'meta'}."""
    # TODO
    raise NotImplementedError


def jawab_fallback(pertanyaan, utama="api_mini", cadangan="api_besar",
                   ambang=None, gagal_fn=None):
    """Model utama "error" (gagal_fn mengembalikan dict meta['error'])
    → coba cadangan. meta: tier='cadangan' bila fallback terpakai, plus
    meta['error_utama'] = pesan error model utama. Utama sukses → tier='utama'.
    (Di produksi: exception/timeout kena pembungkus ini.)"""
    # TODO
    raise NotImplementedError


def biaya_permintaan(meta, model):
    """Meta hasil model (token_in/token_out) + nama model → USD:
    tok_in/1e6 × harga_in + tok_out/1e6 × harga_out (HARGA per 1 jt token)."""
    # TODO
    raise NotImplementedError


def breakeven_req_per_hari(biaya_gpu_per_bulan=60.0, model="api_mini"):
    """Breakeven API vs GPU lokal (Bab 13): biaya GPU per hari ÷ biaya per
    permintaan (PROFIL_TOKEN, harga `model`). Round ke bilangan bulat terdekat."""
    # TODO
    raise NotImplementedError
