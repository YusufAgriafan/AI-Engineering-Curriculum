# 🏗️ Project Starter — Analisis Penjualan (Bab 1)

> Proyek mini **mode TDD**: test sudah ditulis duluan, tugasmu mengimplementasikan kodenya sampai semua test hijau. Ini meniru cara kerja profesional: spesifikasi jelas → implementasi → verifikasi otomatis.

## Struktur

```
project-starter-analisis-penjualan/
├── README.md            ← kamu di sini
├── RUBRIK.md            ← penilaian mandiri (isi setelah selesai)
├── penjualan/
│   ├── __init__.py
│   ├── data.py          ← data transaksi (jangan diubah)
│   └── analisis.py      ← TUGASMU: 6 fungsi dengan TODO
└── tests/
    └── test_analisis.py ← spesifikasi dalam bentuk test (jangan diubah)
```

## Cara Kerja (Mode TDD)

1. Baca `tests/test_analisis.py` — itulah **spesifikasi** fungsi-fungsimu.
2. Jalankan test, lihat semuanya merah:
   ```bash
   cd AI-Engineering-Curriculum/01-Python-Fondasi/project-starter-analisis-penjualan
   python -m unittest discover -s tests -v
   ```
3. Implementasikan **satu fungsi** di `penjualan/analisis.py`.
4. Jalankan ulang test → buat test fungsi itu hijau.
5. Ulangi untuk fungsi berikutnya. **Satu fungsi = satu commit kecil** (kalau repo ini pakai git).
6. Semua hijau? Isi `RUBRIK.md`, lalu bandingkan pendekatanmu dengan solusi di lab/kunci jawaban.

## Aturan

- ✅ Boleh pakai stdlib (`collections`, dll.) — pandas **tidak diperlukan** di sini.
- ✅ Docstring + type hint untuk setiap fungsi.
- ❌ Jangan mengubah `data.py` atau `tests/` agar test "lolos" — itu kekalahan, bukan kemenangan.
- 🧪 Edge case adalah bagian dari penilaian: list kosong, seri/tie, format angka Indonesia.

## Fungsi yang Harus Diimplementasikan

| # | Fungsi | Return | Kesulitan |
|---|---|---|---|
| 1 | `total_pendapatan(transaksi)` | `float` | ⭐ |
| 2 | `pendapatan_per_produk(transaksi)` | `dict[str, float]` | ⭐⭐ |
| 3 | `produk_terlaris(transaksi)` | `str` (berdasar total unit; tie → produk pertama) | ⭐⭐ |
| 4 | `transaksi_di_atas(transaksi, batas)` | `list[dict]` urut pendapatan menurun | ⭐⭐ |
| 5 | `rata_rata_harga_per_kota(transaksi)` | `dict[str, float]` dibulatkan 2 desimal | ⭐⭐ |
| 6 | `bersihkan_harga(teks)` | `float` — handle `"Rp 2.000"`, `"Rp1.250,75"`, `"abc"` → `0.0` | ⭐⭐⭐ |

> Fungsi #6 melatih error handling + edge case — pola yang dipakai terus di pipeline data nyata.

## Bonus (opsional, +10 poin)

- Tambahkan 1 fungsi analisis buatanmu sendiri **beserta testnya** di file test baru `tests/test_bonus.py`.
- Atau: buat notebook `laporan.ipynb` yang memvisualisasikan pendapatan per produk (matplotlib).
