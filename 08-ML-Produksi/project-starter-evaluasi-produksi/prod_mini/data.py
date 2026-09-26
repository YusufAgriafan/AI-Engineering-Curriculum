"""Dataset evaluasi terkunci untuk project Bab 8.

JANGAN DIUBAH — semua angka di tests, starter.ipynb, dan RUBRIK.md mengacu pada
dataset ini (seed terkunci 8, resep identik dengan lab & kalibrasi).

Resep:
    y = 2.0·x1 − 1.5·x2 + 1.0 + N(0, 1.2²)     n = 800, 2 fitur
Split standar: train 80% (640) | calib 10% (80) | test 10% (80), potongan berurutan.
Label keputusan (untuk bagian threshold/kalibrasi): 1 jika y > median(train).
"""

import numpy as np

__all__ = ["N", "TRUE_W", "TRUE_B", "SIGMA", "CUT_TRAIN", "CUT_CALIB",
           "make_dataset", "split_tiga_arah", "make_labels", "make_gan_data",
           "GAN_MEAN", "GAN_STD"]

N = 800
TRUE_W = np.array([2.0, -1.5])
TRUE_B = 1.0
SIGMA = 1.2
CUT_TRAIN = int(N * 0.8)   # 640
CUT_CALIB = int(N * 0.9)   # 720

GAN_MEAN, GAN_STD = 2.0, 0.5


def make_dataset(seed=8, n=N):
    """Dataset regresi terkunci. Return (X (n,2), y (n,))."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.5, size=(n, 2))
    noise = rng.normal(0.0, SIGMA, n)
    y = X @ TRUE_W + TRUE_B + noise
    return X, y


def split_tiga_arah(X, y):
    """Split BERURUTAN train/calib/test sesuai CUT_TRAIN dan CUT_CALIB.

    Return (Xtr, ytr, Xca, yca, Xte, yte).
    """
    return (X[:CUT_TRAIN], y[:CUT_TRAIN],
            X[CUT_TRAIN:CUT_CALIB], y[CUT_TRAIN:CUT_CALIB],
            X[CUT_CALIB:], y[CUT_CALIB:])


def make_labels(y_train, *parts):
    """Label keputusan: 1 jika y > median(train). Diterapkan ke setiap part.

    Return list of arrays 0/1 sejajar dengan parts.
    """
    med = float(np.median(y_train))
    return [(np.asarray(p) > med).astype(int) for p in parts]


def make_gan_data(seed=8, n=512):
    """Data 'nyata' untuk bagian GAN: N(2.0, 0.5²). Return array (n,)."""
    rng = np.random.default_rng(seed)
    return rng.normal(GAN_MEAN, GAN_STD, n)


if __name__ == "__main__":
    X, y = make_dataset()
    Xtr, ytr, Xca, yca, Xte, yte = split_tiga_arah(X, y)
    print(f"n={len(X)}  train={len(Xtr)} calib={len(Xca)} test={len(Xte)}")
    print(f"y mean={y.mean():.3f} std={y.std():.3f}")
    print(f"split={CUT_TRAIN}/{CUT_CALIB}")
