"""Bagian 5 — Streaming: merakit chunk & mengukur TTFT.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `konsumsi_stream(chunks)` menerima iterable dict {"delta": str, "t_ms": int}
  - delta kosong = penanda/padding -> TIDAK dihitung sebagai chunk konten
  - ttft_ms = t_ms delta PERTAMA yang tidak kosong (None bila tidak ada konten)
  - total_ms = t_ms chunk TERAKHIR (termasuk padding)
  - teks = gabungan seluruh delta, apa adanya (jangan menambah spasi)
"""


def konsumsi_stream(chunks):
    """Return {'teks', 'jumlah_chunk', 'ttft_ms', 'total_ms'}."""
    # TODO
    raise NotImplementedError
