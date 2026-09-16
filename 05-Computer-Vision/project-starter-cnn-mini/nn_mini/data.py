"""Dataset "bars" — gambar kecil 8x8 dengan garis terang horizontal/vertikal.

JANGAN DIUBAH — dataset ini "terkunci" lewat seed. Semua angka di tests dan
starter.ipynb mengacu pada data yang dihasilkan file ini.

Kenapa task ini? Klasifikasi orientasi garis adalah masalah yang sangat mudah
bagi SATU filter konvolusi + pooling, tapi rapuh bagi logistic regression di
piksel mentah (posisi garis berubah-ubah). Ini demonstrasi paling murni kenapa
CNN unggul untuk gambar: weight sharing + invarian translasi.

Struktur:
    x : (N, 8, 8) float32, nilai piksel ~[0, 1]
    y : (N,) int, 0 = garis horizontal, 1 = garis vertikal

Split: 420 train / 90 val / 90 test (seed 42).
"""

import numpy as np

__all__ = ["buat_dataset", "muat_semua", "N_KELAS", "IMG", "NILAI_KELAS"]

N_KELAS = 2
IMG = 8
NILAI_KELAS = {0: "horizontal", 1: "vertikal"}


def buat_dataset(n, rng):
    """n gambar 8x8; tiap gambar = satu garis terang horizontal ATAU vertikal.

    - Latar: noise halus ~ U(0, 0.25)
    - Garis: baris/kolom acak, nilai ~ U(0.75, 1.0) + noise kecil
    - Label: 0 = horizontal, 1 = vertikal
    - 15% kasus: label sengaja "kacau" (garis di latar, bukan noise halus) —
      supaya 100% akurasi mustahil dan model tidak overfit ke pola sempurna.
    """
    x = rng.uniform(0.0, 0.25, size=(n, IMG, IMG)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if rng.random() < 0.5:
            # vertikal: satu kolom terang
            kol = int(rng.integers(0, IMG))
            x[i, :, kol] += rng.uniform(0.5, 0.75)
            y[i] = 1
        else:
            # horizontal: satu baris terang
            bar = int(rng.integers(0, IMG))
            x[i, bar, :] += rng.uniform(0.5, 0.75)
            y[i] = 0

    # 15% "noise kasar" pada posisi acak (bukan garis) — naikkan kesulitan
    n_kacau = int(0.15 * n)
    idx = rng.choice(n, size=n_kacau, replace=False)
    for i in idx:
        r, c = int(rng.integers(0, IMG)), int(rng.integers(0, IMG))
        x[i, r, c] += rng.uniform(0.4, 0.9)

    return x, y


def muat_semua(seed=42):
    """Return (x_train, y_train, x_val, y_val, x_test, y_test)."""
    rng = np.random.default_rng(seed)
    x_tr, y_tr = buat_dataset(420, rng)
    x_va, y_va = buat_dataset(90, rng)
    x_te, y_te = buat_dataset(90, rng)
    return x_tr, y_tr, x_va, y_va, x_te, y_te


if __name__ == "__main__":
    X, y, *_ = muat_semua()
    print("contoh gambar 'horizontal':")
    print(np.round(X[y == 0][0], 2))
    print("contoh gambar 'vertikal':")
    print(np.round(X[y == 1][0], 2))
