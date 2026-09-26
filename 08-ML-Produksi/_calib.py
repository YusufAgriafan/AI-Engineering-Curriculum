"""Kalibrasi angka untuk assert di lab/kuis/project Bab 8 (jalankan sekali, catat hasilnya).

Dataset & seed terkunci — angka di bawah harus bisa direproduksi persis:
  - RNG utama: np.random.default_rng(8)
  - Regresi 2 fitur, n=800 (true w=[2.0, -1.5], b=1.0, sigma=1.2)
  - Split: train 80% (640), calib 10% (80), test 10% (80) — potongan pertama->terakhir
Semua fungsi identik dengan implementasi referensi (prod_mini_ref.py).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

# ============================================================
# Dataset evaluasi (identik dengan prod_mini/data.py)
# ============================================================
N = 800
TRUE_W = np.array([2.0, -1.5])
TRUE_B = 1.0
SIGMA = 1.2

rng = np.random.default_rng(8)
X = rng.normal(0.0, 1.5, size=(N, 2))
noise = rng.normal(0.0, SIGMA, N)
y = X @ TRUE_W + TRUE_B + noise

cut1 = int(N * 0.8)          # 640
cut2 = int(N * 0.9)          # 720
Xtr, ytr = X[:cut1], y[:cut1]
Xca, yca = X[cut1:cut2], y[cut1:cut2]
Xte, yte = X[cut2:], y[cut2:]
print(f"n={N}  train={len(Xtr)} calib={len(Xca)} test={len(Xte)}")
print(f"y mean={y.mean():.3f} std={y.std():.3f}")
print(f"corr(x1,y)={np.corrcoef(X[:,0],y)[0,1]:.4f}  corr(x2,y)={np.corrcoef(X[:,1],y)[0,1]:.4f}")

# ============================================================
# Model (identik dengan prod_mini/models.py)
# ============================================================
def fit_ols(Xa, ya):
    Xm = np.column_stack([np.asarray(Xa, dtype=float), np.ones(len(Xa))])
    coef, *_ = np.linalg.lstsq(Xm, ya, rcond=None)
    return coef[:-1], float(coef[-1])

def predict(w, b, Xa):
    return np.asarray(Xa, dtype=float) @ w + b

def fit_neg(Xa, ya):
    """Model 'negatif': hanya fitur kedua, koefisien DIBALIK — overfit eksperimen."""
    X1 = np.asarray(Xa, dtype=float)[:, 1:2]
    neg = np.column_stack([-X1, np.ones(len(X1))])
    coef, *_ = np.linalg.lstsq(neg, ya, rcond=None)
    return np.array([-coef[0]]), float(coef[1])

def predict_neg(w, b, Xa):
    X1 = np.asarray(Xa, dtype=float)[:, 1:2]
    return -X1 @ w + b  # w sudah dibalik di fit; prediksi = -w*x1 + b

def fit_ham(ya):
    return float(np.mean(ya))

def predict_ham(mean_y, Xa):
    return np.full(len(Xa), mean_y)

# ============================================================
# Metrik (identik dengan prod_mini/metrics.py)
# ============================================================
def rmse(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot

def skill_score(rmse_model, rmse_ref):
    return 1.0 - rmse_model / rmse_ref

w_ols, b_ols = fit_ols(Xtr, ytr)
w_neg, b_neg = fit_neg(Xtr, ytr)
mean_y = fit_ham(ytr)

rmse_ols = rmse(yte, predict(w_ols, b_ols, Xte))
rmse_neg = rmse(yte, predict_neg(w_neg, b_neg, Xte))
rmse_ham = rmse(yte, predict_ham(mean_y, Xte))
print(f"\nw_ols={np.round(w_ols, 4).tolist()}  b_ols={b_ols:.4f}  (true w={TRUE_W.tolist()} b={TRUE_B})")
print(f"w_neg={w_neg.tolist()}  b_neg={b_neg:.4f}")
print(f"RMSE ols={rmse_ols:.4f}  neg={rmse_neg:.4f}  ham={rmse_ham:.4f}")
print(f"R2  ols={r2(yte, predict(w_ols, b_ols, Xte)):.4f}")
print(f"MAE ols={mae(yte, predict(w_ols, b_ols, Xte)):.4f}")
print(f"skill ols vs HAM = {skill_score(rmse_ols, rmse_ham):.4f}")
print(f"skill neg vs HAM = {skill_score(rmse_neg, rmse_ham):.4f}")

# ============================================================
# Threshold & confusion (identik dengan prod_mini/threshold.py)
# ============================================================
# Skor = model terhadap y (regresi) dipakai sebagai "skor keputusan".
# Label = y > median(train). Threshold dipilih di CALIB, dievaluasi di TEST.
def confusion(y_true01, y_score, thr):
    tp = int(np.sum((y_true01 == 1) & (y_score >= thr)))
    fp = int(np.sum((y_true01 == 0) & (y_score >= thr)))
    fn = int(np.sum((y_true01 == 1) & (y_score < thr)))
    tn = int(np.sum((y_true01 == 0) & (y_score < thr)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def prf(cm):
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}

med = float(np.median(ytr))
ytr01 = (ytr > med).astype(int)
yca01 = (yca > med).astype(int)
yte01 = (yte > med).astype(int)

score_ols_te = predict(w_ols, b_ols, Xte)
score_ham_te = np.full(len(Xte), mean_y)

print("\n=== Threshold tuning di CALIB (target F1) ===")
print(f"median(train) = {med:.4f}")
print(f"prevalence: calib {yca01.mean():.3f}  test {yte01.mean():.3f}")

best = None
for thr in np.arange(0.5, 3.01, 0.1):
    cmc = confusion(yca01, predict(w_ols, b_ols, Xca), thr)
    m = prf(cmc)
    if best is None or m["f1"] > best[2]["f1"]:
        best = (round(float(thr), 2), cmc, m)
thr_star, cm_c, m_c = best
print(f"thr*={thr_star}  calib: cm={cm_c}  P={m_c['precision']:.3f} R={m_c['recall']:.3f} F1={m_c['f1']:.3f}")

cm_t = confusion(yte01, score_ols_te, thr_star)
m_t = prf(cm_t)
print(f"test @thr*={thr_star}: cm={cm_t}  P={m_t['precision']:.3f} R={m_t['recall']:.3f} F1={m_t['f1']:.3f}")

# baseline HAM (prediksi konstan) → semua di satu sisi
cm_ham = confusion(yte01, score_ham_te, thr_star)
m_ham = prf(cm_ham)
print(f"HAM  @thr*={thr_star}: cm={cm_ham}  P={m_ham['precision']:.3f} R={m_ham['recall']:.3f} F1={m_ham['f1']:.3f}")

# sweep F1 vs threshold di test (untuk narasi lab)
print("\nF1 di TEST per threshold:")
for thr in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    cmx = confusion(yte01, score_ols_te, thr)
    mx = prf(cmx)
    print(f"  thr={thr:.1f}: P={mx['precision']:.3f} R={mx['recall']:.3f} F1={mx['f1']:.3f}  cm={cmx}")

# ============================================================
# Persistensi & hash deterministik (prod_mini/persist.py)
# ============================================================
def save_model(path, w, b):
    np.savez(path, w=np.asarray(w, dtype=float), b=np.array(float(b)))

def load_model(path):
    with np.load(path) as z:
        return z["w"], float(z["b"])

def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()

tmpdir = tempfile.mkdtemp()
p1, p2 = os.path.join(tmpdir, "a.npz"), os.path.join(tmpdir, "b.npz")
save_model(p1, w_ols, b_ols)
save_model(p2, w_ols, b_ols)
h1, h2 = sha256_file(p1), sha256_file(p2)
print(f"\nhash sama (deterministik): {h1 == h2}  sha256[:16]={h1[:16]}")

w2, b2 = load_model(p1)
pred_before = predict(w_ols, b_ols, Xte)
pred_after = predict(w2, b2, Xte)
print(f"load == sebelum save: {np.array_equal(pred_before, pred_after)}")
pred_a3 = predict(w_ols, b_ols, Xte)
print(f"prediksi ulang identik: {np.array_equal(pred_before, pred_a3)}")

# ============================================================
# GAN pipeline numpy (seed terkunci 8) — resep Bab 8 lab
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

N_GAN = 512
MEAN_G, STD_G = 2.0, 0.5
rg = np.random.default_rng(8)
real_data = rg.normal(MEAN_G, STD_G, N_GAN)
print(f"\n=== GAN 1D ===  real: mean={real_data.mean():.4f} std={real_data.std():.4f}")

G_MU, G_LOGS = -2.0, np.log(0.5)
# D di-init ACAK kecil — init nol = saddle point (semua gradien nol, D tidak pernah belajar)
D_W1, D_B1 = rg.normal(0.0, 0.5, 8), np.zeros(8)
D_W2, D_B2 = rg.normal(0.0, 0.5, 8), 0.0
LR = 0.05

def gen_sample(bs, rng_g):
    z = rng_g.normal(0.0, 1.0, bs)
    return z * np.exp(G_LOGS) + G_MU, z

def disc_forward(x):
    h = np.tanh(x[:, None] * D_W1 + D_B1)
    o = h @ D_W2 + D_B2
    return o, h

def dscore(x):
    return sigmoid(disc_forward(x)[0])

def d_loss(x_real, x_fake):
    return float(-(np.log(dscore(x_real) + 1e-12).mean()
                   + np.log(1.0 - dscore(x_fake) + 1e-12).mean()))

def g_loss(x_fake):
    return float(-np.log(dscore(x_fake) + 1e-12).mean())

rng_t = np.random.default_rng(8)
BATCH = 128
EPOCHS = 300
hist = []
for ep in range(EPOCHS):
    xr = rng_t.choice(real_data, BATCH, replace=False)
    xf, _ = gen_sample(BATCH, rng_t)
    for x, label in ((xr, 1.0), (xf, 0.0)):
        o, h = disc_forward(x)
        p = sigmoid(o)
        dO = (p - label) / len(x)
        dW2 = h.T @ dO
        dB2 = dO.sum()
        dH = dO[:, None] * D_W2[None, :] * (1.0 - h ** 2)
        dW1 = (x[:, None] * dH).sum(0)
        dB1 = dH.sum(0)
        D_W1 -= LR * dW1; D_B1 -= LR * dB1
        D_W2 -= LR * dW2; D_B2 -= LR * dB2
    # gradien G: non-saturating loss -log D(G(z)), backprop MELEWATI D ke parameter G
    o_g, h_g = disc_forward(xf)
    p_g = sigmoid(o_g)
    dO_g = (p_g - 1.0) / len(xf)                       # d(-log p)/do = p - 1
    dx_g = (D_W2[None, :] * (1.0 - h_g ** 2)) @ D_W1   # do/dx dijumlah over hidden
    dmu = float(np.sum(dO_g * dx_g))                   # d xf/dmu = 1
    dlogs = float(np.sum(dO_g * dx_g * (xf - G_MU)))   # d xf/dlogs = z*sigma = xf - mu
    G_MU -= LR * dmu
    G_LOGS -= LR * dlogs
    if ep % 50 == 0 or ep == EPOCHS - 1:
        xf_, _ = gen_sample(2000, rng_t)
        hist.append((ep, d_loss(xr, xf_), g_loss(xf_),
                     float(xf_.mean()), float(xf_.std())))
        print(f"ep {ep:4d}: D={hist[-1][1]:.3f} G={hist[-1][2]:.3f} "
              f"fake mean={hist[-1][3]:+.3f} std={hist[-1][4]:.3f}")

print(f"final: G_MU={G_MU:.4f}  D(real)={dscore(real_data).mean():.3f}  "
      f"D(fake)={dscore(gen_sample(2000, rng_t)[0]).mean():.3f}")
print(f"hist pertama: {hist[0]}")
print(f"hist terakhir: {hist[-1]}")

# ============================================================
# Kuis: konstanta yang dirujuk
# ============================================================
print("\n=== KUIS ===")
print(f"rmse_ols={rmse_ols:.4f}  rmse_ham={rmse_ham:.4f}  rmse_neg={rmse_neg:.4f}")
print(f"skill_ols={skill_score(rmse_ols, rmse_ham):.4f}  skill_neg={skill_score(rmse_neg, rmse_ham):.4f}")
print(f"thr*={thr_star}  F1 test @thr* = {m_t['f1']:.4f}  F1 HAM = {m_ham['f1']:.4f}")
print(f"prevalence test = {yte01.mean():.4f}")
print(f"confusion test = {cm_t}")
