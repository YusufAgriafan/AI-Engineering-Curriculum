"""Dataset & konfigurasi terkunci project Bab 16 — JANGAN DIUBAH.

TANPA GPU, TANPA jaringan, TANPA API key — pola yang sama dengan Bab 11-15:

- "Layanan" yang di-deploy adalah RAG toko yang SAMA dengan Bab 15 (retrieval
  kata kunci + generator ekstraktif, sistem v3): sudah punya eval gate, jadi
  bab ini fokus ke lapisan OPERASI — konfigurasi 12-factor, validasi permintaan,
  rate limit, router 2-tier, fallback, canary rollout, monitoring & drift.

- Semua angka (biaya per permintaan, breakeven router, keputusan canary,
  health score) TERKUNCI SEED 16 dan diverifikasi `tests/` + `_calib.py` +
  `_verify_project.py`. Input yang sama → hasil yang sama di komputer siapa
  pun — syarat latihan operasi yang adil dan bisa diaudit.
"""
import math

SEED = 16

# ---------------------------------------------------------------------------
# 1. Konfigurasi layanan (12-factor): SEMUA dari env, tidak ada hardcode
# ---------------------------------------------------------------------------
SCHEMA_ENV = {
    "PORT": ("int", 8000),                    # (tipe, default)
    "AMBAT_ABSTAIN": ("float", 0.20),         # ambang abstain (Bab 12/15)
    "SKOR_MIN": ("float", 0.75),              # gate rilis eval (Bab 15)
    "RATE_LIMIT_PER_MENIT": ("int", 60),      # rate limit per klien
    "MODEL_API": ("str", "api_mini"),         # tier default router
    "CANARY_PERSEN": ("int", 10),             # persen traffic ke versi baru
    "BIAYA_HARIAN_MAKS": ("float", 5.0),      # alarm biaya (USD)
}

# Env "produksi" yang dipakai angka terkunci (urutan dict Python 3.7+ terjaga)
ENV_PRODUKSI = {
    "PORT": "9000",
    "AMBAT_ABSTAIN": "0.25",
    "SKOR_MIN": "0.8",
    "RATE_LIMIT_PER_MENIT": "120",
    "MODEL_API": "api_besar",
    "CANARY_PERSEN": "20",
    "BIAYA_HARIAN_MAKS": "3.5",
}

# ---------------------------------------------------------------------------
# 2. Permintaan chat & validasi (pydantic-mini)
# ---------------------------------------------------------------------------
MAKS_PESAN = 20           # riwayat terlalu panjang → 422
MAKS_KARAKTER = 2000      # satu pesan terlalu panjang → 422

# Header rate limit terkunci (format standar HTTP)
# X-RateLimit-Limit / X-RateLimit-Remaining / Retry-After (detik, bulat ke atas)

# ---------------------------------------------------------------------------
# 3. Router 2-tier: pertanyaan mudah → model mini, sulit → model besar
# ---------------------------------------------------------------------------
KATA_SULIT = ("kenapa", "mengapa", "bandingkan", "jelaskan", "analisis",
              "langkah", "rekomendasi")

# Tarif API (USD per 1 JUTA token) — konsisten Bab 11/13/15
HARGA = {
    "api_besar": {"input": 2.50, "output": 10.0},
    "api_mini": {"input": 0.15, "output": 0.60},
    "lokal": {"input": 0.0, "output": 0.0},
}

# Profil permintaan tipikal (token) untuk hitungan biaya & breakeven
PROFIL_TOKEN = {"input": 1000, "output": 300}

# Latensi ilustratif (ms, nearest-rank p50) per tier — dipakai canary SLO
LATENSI_P50 = {"api_besar": 900.0, "api_mini": 480.0, "lokal": 210.0}

# SLO canary: p50 baru harus <= 1.5 × p50 lama (ratro 1.5x)
SLO_LATENSI_RASIO = 1.5

# ---------------------------------------------------------------------------
# 4. Canary rollout & keputusan rilis
# ---------------------------------------------------------------------------
# Kriteria GAGAL rollout (cek berurutan; pertama yang cocok jadi alasan):
#   1) error_rate  > 0.02                     → 'error rate'
#   2) skor_eval   < SKOR_MIN (dari env!)     → 'skor eval'
#   3) p50_latenis > SLO_LATENSI_RASIO × p50_lama → 'latensi'
#   4) biaya_per_req > 2 × biaya_per_req_lama → 'biaya'
CANARY_KASUS = {
    #                error_rate, skor_eval, p50_baru_ms, biaya_baru, biaya_lama
    "sehat":        (0.001, 0.90, 500.0, 0.00033, 0.00033),
    "error":        (0.05,  0.90, 500.0, 0.00033, 0.00033),
    "skor_turun":   (0.001, 0.60, 500.0, 0.00033, 0.00033),
    "lambat":       (0.001, 0.90, 1500.0, 0.00033, 0.00033),
    "mahal":        (0.001, 0.90, 500.0, 0.00100, 0.00033),
}

# ---------------------------------------------------------------------------
# 5. Monitoring & drift (per jam — data produksi mini)
# ---------------------------------------------------------------------------
DRIFT = [
    # (jam, kueri) — jam 0-9 produksi toko
    (0, "Berapa lama pengiriman reguler?"),
    (1, "Berapa lama pengiriman kilat?"),
    (2, "Kapan cashback e-wallet diberikan?"),
    (3, "Bagaimana status pesanan INV-001?"),
    (4, "Berapa lama pengiriman reguler?"),
    (5, "Apakah toko punya cabang di Surabaya?"),
    (6, "Boleh bayar pakai QRIS?"),            # ← NEW: topik baru mulai jam 6
    (7, "QRIS bisa untuk semua bank?"),
    (8, "Cara bayar QRIS dari e-wallet lain?"),
    (9, "QRIS kadaluarsa berapa lama?"),
]
DRIFT_JAM_PENUH = 6          # mulai jam ini topik baru muncul
AMBAT_DRIFT = 0.4            # proporsi topik baru ≥ 0.4 → drift terdeteksi

HEALTH_METRIK = {
    # metrik: (nilai, target, arah)  arah: 'maks' (nilai ≤ target) | 'min' (nilai ≥ target)
    "error_rate": (0.004, 0.01, "maks"),
    "p95_ms": (1500.0, 2000.0, "maks"),
    "skor_eval": (0.86, 0.75, "min"),
    "biaya_usd_per_1000req": (1.20, 2.00, "maks"),
}

# Kata fungsi/tanya yang dibuang saat mengekstrak topik (monitor.deteksi_drift)
STOPWORD = ("apakah", "berapa", "bagaimana", "kapan", "yang", "apa",
            "untuk", "dari", "dengan", "pada", "dan", "atau")


def token_estimasi(teks):
    """Kontrak Bab 11/15: token = ceil(len/4)."""
    return math.ceil(len(str(teks)) / 4.0)
