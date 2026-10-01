"""Dataset, konfigurasi tools, & tarif terkunci project Bab 14 — JANGAN DIUBAH.

TANPA GPU, TANPA jaringan, TANPA API key — pola yang sama dengan Bab 9-13:

- "LLM agent" di sini adalah policy DETERMINISTIK (`nlu.py`): aturan intent →
  pilihan tool → Thought/Action/Observation. Yang kamu bangun adalah MESIN
  agent-nya: registry + validasi skema, loop ReAct, guardrail, memory, planner,
  dan eval trace-nya. Itulah bagian yang di produksi harus benar, apa pun LLM-nya.

- Angka-angka (jumlah langkah, token, hit-rate cache, skor eval) TERKUNCI
  SEED 14 dan diverifikasi `tests/` + `_calib.py` + `_verify_project.py`.
"""
SEED = 14

# ---------------------------------------------------------------------------
# 1. Knowledge base agent — dokumen mini + riwayat pesanan (sumber tool RAG & SQL-ish)
# ---------------------------------------------------------------------------
DOKUMEN = {
    "kb": [
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
}

PESANAN = {
    "INV-001": {"id": "INV-001", "user": "wati", "status": "dikirim",
                "produk": "Keyboard mini", "total": 250000, "tanggal": "2026-09-20"},
    "INV-002": {"id": "INV-002", "user": "wati", "status": "diproses",
                "produk": "Mouse gaming", "total": 350000, "tanggal": "2026-09-25"},
    "INV-003": {"id": "INV-003", "user": "budi", "status": "selesai",
                "produk": "Kabel USB", "total": 25000, "tanggal": "2026-09-01"},
    "INV-004": {"id": "INV-004", "user": "budi", "status": "dibatalkan",
                "produk": "Charger cepat", "total": 90000, "tanggal": "2026-09-12"},
}

# Aksi berisiko → wajib konfirmasi manusia (human-in-the-loop)
AKSI_BERISIKO = ("refund", "batalkan_pesanan")

# ---------------------------------------------------------------------------
# 2. Skema tools (JSON Schema subset) — validasi argumen sebelum eksekusi
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = {
    "cari_dokumen": {
        "deskripsi": "Cari potongan kebijakan di KB dengan pencarian kata kunci.",
        "argumen": {
            "type": "object",
            "properties": {
                "kueri": {"type": "string", "min_panjang": 3},
            },
            "required": ["kueri"],
        },
    },
    "cek_pesanan": {
        "deskripsi": "Cek status pesanan berdasarkan ID invoice.",
        "argumen": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "pola": r"INV-\d{3}"},
            },
            "required": ["invoice_id"],
        },
    },
    "refund": {
        "deskripsi": "Ajukan refund untuk invoice (AKSI BERISIKO — butuh konfirmasi).",
        "argumen": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "pola": r"INV-\d{3}"},
                "alasan": {"type": "string", "min_panjang": 5},
            },
            "required": ["invoice_id", "alasan"],
        },
    },
}

# Pemetaan intent → tool + argumen default (policy deterministik di nlu.py)
INTENT_TOOL = {
    "kebijakan": "cari_dokumen",
    "status_pesanan": "cek_pesanan",
    "refund": "refund",
}

# ---------------------------------------------------------------------------
# 3. Tarif API (USD per 1 JUTA token) — konsisten dengan Bab 11 & 13.
# ---------------------------------------------------------------------------
HARGA = {
    "api_besar": {"input": 2.50, "output": 10.0},
    "api_mini": {"input": 0.15, "output": 0.60},
    "lokal": {"input": 0.0, "output": 0.0},
}

# Iterasi ReAct per permintaan
MAKS_ITER = 6          # guardrail loop tak berujung
AMBAT_SKOR_RAG = 0.30  # di bawah ini → abstain (sambungan Bab 12)
