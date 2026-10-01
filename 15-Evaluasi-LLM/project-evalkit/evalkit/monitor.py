"""TODO Bagian 5 — Monitor: biaya run & latensi (observability mini).

Kontrak lengkap ada di tests/test_monitor.py.

Sambungan Bab 11 (biaya token) & Bab 13 (tarif): eval punya biaya —
monitoring biaya mencegah "eval mahal dibiarkan tidak jalan". Percentile
latensi dibulatkan; pemahamannya: p50 = pengalaman tipikal, p99 = kasus
terburuk yang dilihat user sabar.
"""
from .data import HARGA
from .tokenizer import token_estimasi


def estimasi_biaya(pertanyaan, jawaban, model="api_mini"):
    """Biaya SATU permintaan (USD) = (token_in/1e6)*harga_input +
    (token_out/1e6)*harga_output. Token = token_estimasi (ceil(len/4)).
    Model tak dikenal → KeyError biar ketahuan (jangan diam-diam 0)."""
    # TODO
    raise NotImplementedError


def biaya_run(hasil_list, model="api_mini"):
    """Total biaya run (jumlah estimasi_biaya tiap kasus) → USD float."""
    # TODO
    raise NotImplementedError


def latensi_persentil(latensi_ms, persen=(50, 90, 99)):
    """List latensi (ms) → {'p50', 'p90', 'p99'} sesuai `persen`, dibulatkan
    round(x, 1).

    Metode: urutkan naik; rank = ceil(persen/100 * n); ambil urut[rank-1]
    (nearest-rank). List kosong → {'p50': 0.0, 'p90': 0.0, 'p99': 0.0}.
    """
    # TODO
    raise NotImplementedError
