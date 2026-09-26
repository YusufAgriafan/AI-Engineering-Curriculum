"""Konfigurasi & dataset terkunci project Bab 11 — JANGAN DIUBAH.

Sama seperti Bab 10: TANPA API key dan TANPA jaringan. "Provider" di sini adalah
`llmkit/transport.py`, yaitu provider palsu deterministik yang bisa disuruh
GAGAL dengan pola tertentu. Justru itu intinya — pola produksi (retry, fallback,
cache, budget) hanya bisa dilatih kalau kegagalan bisa dijadwalkan.
"""
SEED = 11

# --------------------------------------------------------------------------
# 1. Tarif (USD per 1 juta token) — angka ILUSTRATIF, bukan harga aktual
# --------------------------------------------------------------------------
HARGA = {
    "mini":   {"input": 0.15, "output": 0.60},
    "sedang": {"input": 1.25, "output": 5.00},
    "besar":  {"input": 5.00, "output": 15.00},
}

HARGA_CADANGAN = {
    "murah":  {"input": 0.10, "output": 0.30},
    "mini":   {"input": 0.15, "output": 0.60},
}

# --------------------------------------------------------------------------
# 2. Rencana kegagalan: model -> status per panggilan ke-1, ke-2, ...
#    Status: ok | rate_limit | server_error | timeout | bad_request
#    Habis daftar -> "ok" (pulih sendiri, seperti nyata).
# --------------------------------------------------------------------------
RENCANA = {
    "mini":   ["ok"],
    "sedang": ["rate_limit", "ok"],
    "besar":  ["rate_limit", "server_error", "ok"],
}

RENCANA_CADANGAN = {
    "murah":  ["ok"],
    "mini":   ["ok"],
}

# Latensi dasar per model (ms); panggilan menambah (len(prompt) % 50)
BASIS_LATENSI = {"murah": 90, "mini": 180, "sedang": 240, "besar": 400}

# Streaming palsu: chunk pertama muncul setelah TTFT_MS, lalu MS_PER_CHUNK per kata
TTFT_MS = 120
MS_PER_CHUNK = 20

# --------------------------------------------------------------------------
# 3. Tools (function calling) — kata kunci memicu tool, argumen terkunci
# --------------------------------------------------------------------------
TOOLS = {
    "get_cuaca": {
        "deskripsi": "Ambil cuaca sebuah kota",
        "kata_kunci": ["cuaca", "suhu"],
        "argumen": {"kota": "Jakarta", "satuan": "celsius"},
        "handler": lambda args: {
            "kota": args.get("kota", "Jakarta"),
            "suhu": 28,
            "kondisi": "cerah",
            "satuan": args.get("satuan", "celsius"),
        },
    },
    "hitung": {
        "deskripsi": "Hitung ekspresi aritmetika sederhana",
        "kata_kunci": ["hitung", "kalkulasi", "berapa hasil"],
        "argumen": {"ekspresi": "15 * 8 + 20"},
        "handler": lambda args: {"hasil": 140},
    },
    "cari_catatan": {
        "deskripsi": "Cari catatan internal",
        "kata_kunci": ["cari catatan", "cari dokumen"],
        "argumen": {"kata_kunci": "machine learning", "batas": 3},
        "handler": lambda args: {
            "hasil": [
                f"Catatan A tentang {args.get('kata_kunci', 'x')}",
                f"Catatan B tentang {args.get('kata_kunci', 'x')}",
            ]
        },
    },
}

# --------------------------------------------------------------------------
# 4. Pipeline orkestrasi (prompt chaining) — 3 langkah terkunci
# --------------------------------------------------------------------------
LANGKAH_PIPELINE = [
    {"nama": "klasifikasi", "model": "mini", "suhu": 0.0,
     "template": "Klasifikasi intent tiket berikut, jawab satu kata: {teks}"},
    {"nama": "ekstraksi", "model": "mini", "suhu": 0.0,
     "template": "Ekstrak data pesanan sebagai JSON dari tiket berikut: {teks}"},
    {"nama": "balasan", "model": "sedang", "suhu": 0.4,
     "template": "Tulis balasan pelanggan yang sopan untuk tiket berikut: {teks}"},
]

TIKET_CONTOH = "Pesanan saya belum sampai, order #1234, sudah 5 hari."

# --------------------------------------------------------------------------
# 5. Rantai fallback multi-provider
# --------------------------------------------------------------------------
RANTAI_FALLBACK = [
    {"provider": "utama", "model": "besar"},
    {"provider": "utama", "model": "mini"},
    {"provider": "cadangan", "model": "murah"},
]
