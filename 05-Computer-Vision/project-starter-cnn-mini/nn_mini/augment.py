"""Augmentasi data gambar — flip, brightness, shift, pipeline.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_augment.py.

Prinsip penting: augmentasi HANYA mengubah X — label y tidak pernah berubah.
Flip horizontal pada gambar garis horizontal tetap garis horizontal. Kalau
augmentasimu mengubah makna label (mis. flip atas-bawah di data anatomi),
augmentasi itu salah untuk domain itu.

Semua fungsi menerima X bentuk (n, h, w) dan mengembalikan (n, h, w) —
bentuk dan jumlah sampel TIDAK boleh berubah.
"""

import numpy as np

__all__ = ["flip_horizontal", "flip_vertical", "random_noise", "random_shift",
           "augment_batch"]


def flip_horizontal(X):
    """Balik kiri-kanan per gambar: kolom j dibaca sebagai kolom (w-1-j).

    Return array BARU (jangan mengubah X di tempat).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    # TODO: X[:, :, ::-1]
    raise NotImplementedError


def flip_vertical(X):
    """Balik atas-bawah per gambar: baris i dibaca sebagai baris (h-1-i).

    Return array BARU.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    # TODO: X[:, ::-1, :]
    raise NotImplementedError


def random_noise(X, rng, p=0.1, strength=0.5):
    """Tampilkan "noise kasar" seperti di generator data: pilih ~p*n gambar,
    tiap gambar terpilih dapat SATU titik acak yang dinaikkan nilainya.

    X : (n, h, w) — akan disalin dulu (jangan ubah aslinya)
    rng : np.random.Generator (dari luar, supaya deterministik dengan seed)
    p : proporsi gambar yang diberi noise
    strength : penguatan titik, rng.uniform(0.4, 0.4 + strength)

    Return salinan X yang sudah diberi noise.
    """
    X = np.asarray(X, dtype=float).copy()
    n, h, w = X.shape
    n_kacau = int(p * n)
    if n_kacau == 0:
        return X
    idx = rng.choice(n, size=n_kacau, replace=False)
    for i in idx:
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        # TODO: X[i, r, c] += rng.uniform(0.4, 0.4 + strength)
        raise NotImplementedError
    return X


def random_shift(X, rng, max_shift=2):
    """Geser tiap gambar secara acak (dx, dy) ∈ [-max_shift, max_shift].

    Bagian yang keluar frame dibuang; area kosong DIISI NOL (bukan wrap-around).
    Return salinan baru. Hint: pakai np.zeros_like + slicing, atau np.roll
    LALU nolkan tepi yang wrap (np.roll saja TIDAK cukup — dia wrap-around).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    n, h, w = X.shape
    out = np.zeros_like(X)
    for i in range(n):
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        # TODO: hitung potongan sumber & tujuan yang valid (jika |dx|<w dst.),
        #       salin X[i] ke out[i] pada posisi yang digeser
        raise NotImplementedError
    return out


def augment_batch(X, y, rng, p_noise=0.1):
    """Pipeline latihan: flip_horizontal acak per gambar + random_noise.

    Return (X_aug, y) — y dikembalikan TIDAK DIUBAH (bentuk sama).
    """
    X = np.asarray(X, dtype=float).copy()
    n = len(X)
    # TODO 1: pilih ~setengah gambar acak (rng.choice(n, size=n//2, replace=False))
    #         lalu flip horizontal subset itu saja
    # TODO 2: X = random_noise(X, rng, p=p_noise)
    # TODO 3: return X, y
    raise NotImplementedError
