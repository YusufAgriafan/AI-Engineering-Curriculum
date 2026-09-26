"""Kalibrasi tambahan untuk lab Bab 7 (angka yang tidak ada di _calib.py)."""
import numpy as np

rng = np.random.default_rng(42)

# ===== seri sintetis yang sama dengan _calib.py =====
t = np.arange(365 * 3, dtype=float)
trend = 100.0 + 0.05 * t
seasonality = 20.0 * np.sin(2 * np.pi * t / 365.0)
noise = rng.normal(0.0, 5.0, len(t))
seri = trend + seasonality + noise

split_idx = int(len(seri) * 0.8)
train, val = seri[:split_idx], seri[split_idx:]
H = len(val)

# ===== 1. ACF noise murni (untuk cek 1.1b) =====
def acf(x, lag):
    x = x - x.mean()
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x))

rng_n = np.random.default_rng(7)
noise_murni = rng_n.normal(0, 1, 1000)
print(f"acf noise murni lag1 = {acf(noise_murni, 1):.4f} (harus |.| < 0.1)")
print(f"acf noise murni lag5 = {acf(noise_murni, 5):.4f}")

# ===== 2. Leakage demo: regresi linear pada window, split acak vs temporal =====
def buat_window(series, W):
    s = np.asarray(series, dtype=float)
    X = np.lib.stride_tricks.sliding_window_view(s, W)[:-1]
    return X, s[W:]

W = 35
Xw, yw = buat_window(seri, W)
n_s = len(Xw)
cut = int(n_s * 0.8)

# temporal
Xtr, ytr = Xw[:cut], yw[:cut]
Xva, yva = Xw[cut:], yw[cut:]
b_temp, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
pred_va = b_temp[0] + Xva @ b_temp[1:]
mae_temp = float(np.mean(np.abs(yva - pred_va)))
pred_tr = b_temp[0] + Xtr @ b_temp[1:]
mae_in = float(np.mean(np.abs(ytr - pred_tr)))

# acak (leakage)
rng_s = np.random.default_rng(42)
perm = rng_s.permutation(n_s)
cut2 = int(n_s * 0.8)
itr, iva = perm[:cut2], perm[cut2:]
b_rand, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(itr)), Xw[itr]]), yw[itr], rcond=None)
pred_r = b_rand[0] + Xw[iva] @ b_rand[1:]
mae_rand = float(np.mean(np.abs(yw[iva] - pred_r)))

print(f"\nleakage demo (linreg window W={W}):")
print(f"  MAE in-sample        : {mae_in:.4f}")
print(f"  MAE split ACAK       : {mae_rand:.4f}")
print(f"  MAE split TEMPORAL   : {mae_temp:.4f}")
print(f"  rasio acak/temporal  : {mae_rand / mae_temp:.3f}")

# ===== 3. AR(1) recovery =====
def fit_ar_ols(s, p):
    s = np.asarray(s, dtype=float)
    X = np.column_stack([s[p - k - 1:len(s) - k - 1] for k in range(p)])
    y = s[p:]
    return np.linalg.lstsq(X, y, rcond=None)[0]

rng_a = np.random.default_rng(42)
n_ar = 2000
z = np.empty(n_ar)
z[0] = 0.0
for i in range(1, n_ar):
    z[i] = 0.7 * z[i - 1] + rng_a.normal(0, 1.0)
phi = fit_ar_ols(z, 1)
print(f"\nAR(1) phi-hat = {phi[0]:.4f} (target 0.7, toleransi +-0.05)")

# ===== 4. pdq in-sample MAE =====
def pdq_fit(tr, period):
    m = period
    y = np.asarray(tr, dtype=float)
    n = len(y)
    X = np.zeros((n, 2 + m))
    X[:, 0] = 1.0
    X[:, 1] = np.arange(n, dtype=float)
    idx = np.arange(n)
    X[idx >= m, 2 + (idx[idx >= m] % m)] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

beta = pdq_fit(train, 365)
pred_in = np.array([beta[0] + beta[1] * i + beta[2 + (i % 365)] if i >= 365 else beta[0] + beta[1] * i
                    for i in range(len(train))])
mae_pdq_in = float(np.mean(np.abs(train - pred_in)))
print(f"pdq in-sample MAE = {mae_pdq_in:.4f} (vs out-of-sample 11.76)")

# ===== 5. seri multiplikatif + HW add vs mult =====
def holt_winters_add(tr, period, alpha, beta_, gamma, h):
    n, m = len(tr), period
    level0 = float(tr[:m].mean())
    level, trend_b = level0, 0.0
    seas = list(tr[:m] - level0)
    for i in range(m, n):
        lv_old = level
        level = alpha * (tr[i] - seas[i - m]) + (1 - alpha) * (level + trend_b)
        trend_b = beta_ * (level - lv_old) + (1 - beta_) * trend_b
        seas.append(gamma * (tr[i] - level) + (1 - gamma) * seas[i - m])
    f = [level + (j + 1) * trend_b + seas[n - m + (j % m)] for j in range(h)]
    return np.array(f), level, trend_b, np.array(seas[-m:])

def holt_winters_mult(tr, period, alpha, beta_, gamma, h):
    n, m = len(tr), period
    eps = 1e-8
    level0 = float(tr[:m].mean())
    level, trend_b = level0, 0.0
    seas = list(tr[:m] / (level0 + eps))
    for i in range(m, n):
        lv_old = level
        level = alpha * (tr[i] / (seas[i - m] + eps)) + (1 - alpha) * (level + trend_b)
        trend_b = beta_ * (level - lv_old) + (1 - beta_) * trend_b
        seas.append(gamma * (tr[i] / (level + eps)) + (1 - gamma) * seas[i - m])
    f = [(level + (j + 1) * trend_b) * seas[n - m + (j % m)] for j in range(h)]
    return np.array(f)

# seri multiplikatif: amplitudo musiman membesar bersama level
rng_m = np.random.default_rng(42)
t2 = np.arange(365 * 3, dtype=float)
level2 = 100.0 + 0.05 * t2
seri_mult = level2 * (1.0 + 0.15 * np.sin(2 * np.pi * t2 / 365.0)) + rng_m.normal(0, 4.0, len(t2))
sp2 = int(len(seri_mult) * 0.8)
tr2, va2 = seri_mult[:sp2], seri_mult[sp2:]
H2 = len(va2)
fc_add2 = holt_winters_add(tr2, 365, 0.3, 0.05, 0.3, H2)[0]
fc_mul2 = holt_winters_mult(tr2, 365, 0.3, 0.05, 0.3, H2)
mae_add2 = float(np.mean(np.abs(fc_add2 - va2)))
mae_mul2 = float(np.mean(np.abs(fc_mul2 - va2)))
print(f"\nseri MULTIPLIKATIF: HW add MAE = {mae_add2:.4f} | HW mult MAE = {mae_mul2:.4f}")
print(f"seri ADITIF (dari calib): HW add MAE = 20.9744 | HW mult MAE = 16.3961")

# ===== 6. backtest mean MAE =====
def rolling_backtest(tr, h, build_forecast, n_folds=3):
    metrik = []
    n = len(tr)
    for k in range(n_folds):
        cut_k = n - (k + 1) * h
        trk, tek = tr[:cut_k], tr[cut_k:cut_k + h]
        fc = build_forecast(trk, h)
        metrik.append((float(np.mean(np.abs(tek - fc))), float(np.sqrt(np.mean((tek - fc) ** 2)))))
    return metrik

def buat_naive(trk, h):
    return np.full(h, float(trk[-1]))

def buat_sn365(trk, h):
    return trk[len(trk) - 365:len(trk) - 365 + h]

met_naive = rolling_backtest(train, 90, buat_naive, 3)
met_sn = rolling_backtest(train, 90, buat_sn365, 3)
print(f"\nbacktest mean MAE naive      : {np.mean([m[0] for m in met_naive]):.4f}")
print(f"backtest mean MAE seas-naive : {np.mean([m[0] for m in met_sn]):.4f}")
