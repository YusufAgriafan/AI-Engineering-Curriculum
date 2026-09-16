"""K-means dari nol — numpy murni.

Tugasmu: isi fungsi bertanda TODO. Kontrak ada di docstring dan di
tests/test_clustering.py — baca test-nya dulu (TDD).
"""

import numpy as np


def standardize(X):
    """Z-score per kolom: (X - mean) / std.

    Kolom dengan std == 0 (konstan) di-return apa adanya (dikurangi mean,
    tanpa dibagi) supaya tidak menghasilkan NaN.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # TODO: amankan kolom konstan, lalu return hasil z-score
    raise NotImplementedError


def kmeans(X, k, n_init=10, max_iter=300, tol=1e-6, seed=0):
    """K-means dengan beberapa percobaan (n_init), ambil inertia terkecil.

    Algoritma per percobaan:
      1. Pilih k titik acak dari X sebagai centroid awal (rng di bawah).
      2. Ulangi:
         a. assign: label[i] = centroid terdekat (jarak kuadrat)
         b. update: centroid baru = mean anggota klaster
            (klaster kosong: pertahankan centroid lamanya)
      3. Berhenti jika centroid bergerak < tol, atau max_iter tercapai.

    Return (labels, centroids, inertia) — inertia = Σ jarak² titik ke
    centroid klaster-nya. labels: array int 0..k-1.
    """
    X = np.asarray(X, dtype=float)
    if k < 1:
        raise ValueError("k harus >= 1")
    if len(X) < k:
        raise ValueError("jumlah titik tidak boleh kurang dari k")

    rng = np.random.default_rng(seed)
    best = None  # (inertia, labels, centroids)

    for _ in range(n_init):
        idx = rng.choice(len(X), size=k, replace=False)
        centroids = X[idx].copy()

        for _ in range(max_iter):
            # TODO a: assign tiap titik ke centroid terdekat
            #         (hint: broadcasting, argmin atas sum of squared diffs)
            # TODO b: hitung centroid baru; klaster kosong -> pertahankan lama
            # TODO c: cek konvergensi (pergeseran centroid < tol) -> break
            raise NotImplementedError

        # TODO d: hitung inertia percobaan ini; kalau lebih baik dari
        #         `best`, simpan (inertia, labels, centroids)
        raise NotImplementedError

    inertia, labels, centroids = best
    return labels, centroids, float(inertia)
