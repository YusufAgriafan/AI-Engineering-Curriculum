"""TODO Bagian 2 — Validasi permintaan + rate limit per klien (pydantic-mini).

Layer API adalah MESIN: validasi & rate limit bekerja terhadap model apa pun
(Bab 14: mesin vs otak). Return format standar:
    (status, body) — status: 200 | 422 | 429
"""
from .data import MAKS_KARAKTER, MAKS_PESAN


def validasi_chat(body):
    """Body permintaan /chat → (status, body_balasan).

    Aturan:
    - body bukan dict                       → (422, {'error': 'body harus dict'})
    - 'pesan' hilang / bukan list / kosong  → (422, {'error': 'pesan harus list non-kosong'})
    - ada elemen pesan bukan dict, tanpa 'peran'+'teks',
      atau 'teks' kosong setelah strip     → (422, {'error': 'format pesan salah'})
    - jumlah pesan > MAKS_PESAN             → (422, {'error': 'terlalu banyak pesan'})
    - teks pesan terakhir > MAKS_KARAKTER   → (422, {'error': 'pesan terlalu panjang'})
    - else                                  → (200, body)
    """
    # TODO
    raise NotImplementedError


class RateLimiter:
    """Rate limit fixed-window per klien: jendela = [mulai, mulai + 60).

    - RateLimiter(batas_per_menit=60) — batas > 0, else ValueError.
    - izinkan(klien, waktu_detik=0.0):
        * klien baru → jendela 60 detik dimulai di waktu itu, kuota penuh.
        * waktu masih < mulai + 60 dan kuota > 0 → (True, sisa_kuota).
        * kuota habis → (False, 0).
        * waktu >= mulai + 60 → jendela baru dimulai di waktu itu (kuota penuh).
    - reset_detik(klien, waktu_detik) → ceil((mulai + 60) − waktu); ≤ 0 atau
      klien tak dikenal → 0 (detik menuju reset jendela berjalan).
    """

    def __init__(self, batas_per_menit=60):
        # TODO
        raise NotImplementedError

    def izinkan(self, klien, waktu_detik=0.0):
        # TODO
        raise NotImplementedError

    def reset_detik(self, klien, waktu_detik=0.0):
        # TODO
        raise NotImplementedError
