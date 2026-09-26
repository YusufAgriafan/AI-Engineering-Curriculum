"""Cari demo leakage yang dramatis."""
import numpy as np

rng = np.random.default_rng(42)
t = np.arange(365 * 3, dtype=float)
seri = 100.0 + 0.05 * t + 20.0 * np.sin(2 * np.pi * t / 365.0) + rng.normal(0.0, 5.0, len(t))

def buat_window(series, W):
    s = np.asarray(series, dtype=float)
    X = np.lib.stride_tricks.sliding_window_view(s, W)[:-1]
    return X, s[W:]

W = 35
Xw, yw = buat_window(seri, W)
n_s = len(Xw)
cut = int(n_s * 0.8)

# ---- 1-NN pada window (model "penghafal") ----
def knn1_pred(Xtr, ytr, Xte):
    pred = np.empty(len(Xte))
    for i, x in enumerate(Xte):
        d = np.mean(np.abs(Xtr - x), axis=1)
        pred[i] = ytr[int(np.argmin(d))]
    return pred

# temporal
Xtr, ytr, Xva, yva = Xw[:cut], yw[:cut], Xw[cut:], yw[cut:]
pred_t = knn1_pred(Xtr, ytr, Xva)
mae_t = float(np.mean(np.abs(yva - pred_t)))

# acak (leakage: tetangga window hampir identik ada di train)
rng_s = np.random.default_rng(42)
perm = rng_s.permutation(n_s)
itr, iva = perm[:cut], perm[cut:]
pred_r = knn1_pred(Xw[itr], yw[itr], Xw[iva])
mae_r = float(np.mean(np.abs(yw[iva] - pred_r)))

print(f"1-NN window W={W}:")
print(f"  MAE split ACAK    : {mae_r:.4f}")
print(f"  MAE split TEMPORAL: {mae_t:.4f}")
print(f"  rasio acak/temporal: {mae_r / mae_t:.3f}")

# ---- alternatif: durasi seri sintetis cepat berubah (regime change) ----
# seri dengan tren naik TAJAM di bagian akhir
trend2 = 100.0 + 0.0001 * t ** 1.6
seri2 = trend2 + 20.0 * np.sin(2 * np.pi * t / 365.0) + rng.normal(0.0, 5.0, len(t))
Xw2, yw2 = buat_window(seri2, W)
cut2 = int(len(Xw2) * 0.8)
Xtr2, ytr2, Xva2, yva2 = Xw2[:cut2], yw2[:cut2], Xw2[cut2:], yw2[cut2:]
itr2, iva2 = perm[:cut2], perm[cut2:]

# linreg
A = np.column_stack([np.ones(len(Xtr2)), Xtr2])
b2, *_ = np.linalg.lstsq(A, ytr2, rcond=None)
pred2_t = b2[0] + Xva2 @ b2[1:]
mae2_t = float(np.mean(np.abs(yva2 - pred2_t)))
b3, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(itr2)), Xw2[itr2]]), yw2[itr2], rcond=None)
pred2_r = b3[0] + Xw2[iva2] @ b3[1:]
mae2_r = float(np.mean(np.abs(yw2[iva2] - pred2_r)))
print(f"\nlinreg window (seri tren melengkung):")
print(f"  MAE ACAK    : {mae2_r:.4f}")
print(f"  MAE TEMPORAL: {mae2_t:.4f}")
print(f"  rasio: {mae2_r / mae2_t:.3f}")

# ---- alternatif: scaler bocor (std dihitung dari SEMUA data) ----
# fitur = (window - mean_lengkap) / std_lengkap vs fit hanya train
# efeknya kecil untuk linreg; skip.
