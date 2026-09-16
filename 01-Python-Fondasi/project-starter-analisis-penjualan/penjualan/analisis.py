"""Implementasikan 6 fungsi di bawah sampai semua test hijau.

Aturan:
- Jangan ubah tests/ atau data.py.
- Docstring sudah ada; lengkapi sesuai kebutuhan.
- Edge case adalah bagian dari spesifikasi (lihat tests/test_analisis.py).

Cara menjalankan test (dari folder project ini):
    python -m unittest discover -s tests -v
"""

from collections import defaultdict


def total_pendapatan(transaksi: list[dict]) -> float:
    """Total pendapatan = sum(unit * harga) untuk semua transaksi.

    List kosong -> 0.0.
    """
    # TODO 1: implementasikan
    raise NotImplementedError


def pendapatan_per_produk(transaksi: list[dict]) -> dict[str, float]:
    """Pendapatan (unit * harga) per produk, dikumpulkan dengan dict.

    List kosong -> {}.
    """
    # TODO 2: implementasikan (hint: defaultdict(float) atau .get())
    raise NotImplementedError


def produk_terlaris(transaksi: list[dict]) -> str | None:
    """Produk dengan total UNIT terbanyak.

    Tie -> produk yang pertama mencapai total tersebut.
    List kosong -> None.
    """
    # TODO 3: implementasikan (hint: max dengan key=)
    raise NotImplementedError


def transaksi_di_atas(transaksi: list[dict], batas: float) -> list[dict]:
    """Semua transaksi dengan pendapatan (unit * harga) > batas,
    diurutkan dari pendapatan TERBESAR.

    Tidak ada yang lolos -> [].
    """
    # TODO 4: implementasikan (hint: sorted dengan key= dan reverse=True)
    raise NotImplementedError


def rata_rata_harga_per_kota(transaksi: list[dict]) -> dict[str, float]:
    """Rata-rata harga per kota, dibulatkan 2 desimal.

    List kosong -> {}.
    """
    # TODO 5: implementasikan (hint: group manual per kota, lalu round(x, 2))
    raise NotImplementedError


def bersihkan_harga(teks: str) -> float:
    """Ubah teks harga menjadi float.

    Aturan:
    - "Rp 2.000" -> 2000.0   (titik = pemisah ribuan)
    - "Rp1.250,75" -> 1250.75 (koma = desimal)
    - "  15000 " -> 15000.0
    - "3,14" -> 3.14
    - Tidak valid / string kosong -> 0.0 (JANGAN raise)
    """
    # TODO 6: implementasikan (hint: strip -> buang 'Rp' -> titik ribuan ->
    #         koma desimal -> float; bungkus try/except ValueError)
    raise NotImplementedError
