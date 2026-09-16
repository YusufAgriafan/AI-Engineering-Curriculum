"""Data transaksi contoh — JANGAN DIUBAH (dipakai oleh tests/)."""

TRANSAKSI = [
    {"id": 1, "produk": "Kopi",   "kota": "Bandung", "unit": 10, "harga": 25000},
    {"id": 2, "produk": "Teh",    "kota": "Bandung", "unit": 5,  "harga": 15000},
    {"id": 3, "produk": "Kopi",   "kota": "Jakarta", "unit": 8,  "harga": 27000},
    {"id": 4, "produk": "Susu",   "kota": "Jakarta", "unit": 3,  "harga": 18000},
    {"id": 5, "produk": "Teh",    "kota": "Bandung", "unit": 7,  "harga": 15500},
    {"id": 6, "produk": "Kopi",   "kota": "Bandung", "unit": 12, "harga": 26000},
    {"id": 7, "produk": "Susu",   "kota": "Jakarta", "unit": 4,  "harga": 17500},
    {"id": 8, "produk": "Teh",    "kota": "Jakarta", "unit": 6,  "harga": 15000},
]

# Data untuk uji edge case di tests/
TRANSAKSI_KOSONG: list = []
TRANSAKSI_SATU = [{"id": 99, "produk": "Gula", "kota": "Solo", "unit": 2, "harga": 12000}]
