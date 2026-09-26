"""TODO Bagian 1 — Dataset SFT ala Alpaca + oversampling.

Kontrak lengkap ada di tests/test_sft.py. Isi fungsi-fungsi di bawah sampai
semua test hijau.
"""
import json


def ke_contoh_sft(baris):
    """Dict {instruksi, input, respons} → (teks_prompt, respons).

    Format prompt (diuji): instruksi + baris kosong + 'Input: <input>' +
    baris kosong + 'Jawaban:'.
    """
    raise NotImplementedError


def muat_jsonl(path):
    """Baca JSONL → list of dict; baris kosong dilewati."""
    raise NotImplementedError


def tulis_jsonl(path, baris_list):
    """List of dict → JSONL (satu objek per baris, ensure_ascii=False)."""
    raise NotImplementedError


def oversample(baris_list, per_label=4):
    """Samakan jumlah contoh per label respons dengan duplikasi deterministik.

    Urutan asli dipertahankan; minoritas diduplikasi berulang sampai kuota.
    """
    raise NotImplementedError


def split_sft(baris_list, rasio_val=0.25, seed=13):
    """Bagi deterministik interleaved: contoh ke-i → val bila
    (i*7 + seed) % 10 < rasio_val*10. Return (train, val)."""
    raise NotImplementedError
