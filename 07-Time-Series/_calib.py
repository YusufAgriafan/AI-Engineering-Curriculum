"""Kalibrasi angka untuk assert di lab/kuis/project Bab 7 (jalankan sekali, catat hasilnya)."""
import numpy as np

rng = np.random.default_rng(42)

# ===== series sintetis (materi inti 1.1, rng 42) =====
t = np.arange(365 * 3, dtype=float)
trend = 100.0 + 0.05 * t
seasonality = 20.0 * np.sin(2 * np.pi * t / 365.0)
noise = rng.normal(0.0, 5.0, len(t))
seri = trend + seasonality + noise
print(f"seri: n={len(seri)} mean={seri.mean():.3f} std={seri.std():.3f}")
print(f"corr(seri, trend): {np.corrcoef(seri, trend)[0, 1]:.4f}")
tc = t - t.mean()
slope = float(np.sum(tc * (seri - seri.mean())) / np.sum(tc * tc))
print(f"slope OLS (x dicenter): {slope:.4f}  (target 0.05)")

split_idx = int(len(seri) * 0.8)
train, val = seri[:split_idx], seri[split_idx:]
H = len(val)
print(f"split: {split_idx}, |val|={H}")

# ===== Bagian 3/5: baselines (train-only compliant) =====
mae_naive1 = float(np.mean(np.abs(val[1:] - val[:-1])))          # 1-step naive di val
pred_sn = seri[split_idx - 365:split_idx - 365 + H]              # seasonal naive 365
mae_sn365 = float(np.mean(np.abs(pred_sn - val)))
mae_ham = float(np.mean(np.abs(val - train.mean())))
print(f"\nMAE naive-1step : {mae_naive1:.4f}")
print(f"MAE seas-naive365: {mae_sn365:.4f}")
print(f"MAE HAM          : {mae_ham:.4f}")

# ===== AR(p) recursive multi-step =====
def ar_fit(tr, p):
    X = np.column_stack([tr[p - k - 1:len(tr) - k - 1] for k in range(p)])
    y = tr[p:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef

def ar_forecast(coef, last_values, h):
    hist = list(last_values)[::-1]  # hist[0] = t-1, hist[1] = t-2, ...
    out = []
    for _ in range(h):
        yhat = float(coef @ np.array(hist[:len(coef)]))
        out.append(yhat)
        hist.insert(0, yhat)
    return np.array(out)

coef7 = ar_fit(train, 7)
fc7 = ar_forecast(coef7, train[-7:], H)
mae_ar7 = float(np.mean(np.abs(fc7 - val)))
print(f"AR(7) coefs     : {np.round(coef7, 4).tolist()}")
print(f"MAE AR(7) recurs: {mae_ar7:.4f}")

# ===== Cek struktural windowing (bebas noise) =====
W = seri[100:135]
y = seri[135]
print(f"\ncek 2.1 struktural: len(W)={len(W)} y==seri[135] -> {y == seri[135]}")
print(f"cek 2.1 toleransi: W.mean()={W.mean():.3f} (harus di (117,125))")

# ===== Cek 1.1: autocorr =====
def acf(x, lag):
    x = x - x.mean()
    return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x))

print(f"cek 1.1: autocorr(seri,1)={acf(seri, 1):.4f} (harus > 0.9)")

# ===== Cek 4.1: seasonal naive h=7 dari t=1000 =====
print(f"cek 4.1: pred = seri[1000] = {seri[1000]:.3f}")

# ===== Cek 5.1: APE 7-langkah dari ujung train =====
fc7_train = ar_forecast(ar_fit(train, 7), train[-7:], 7)
akt7 = seri[split_idx:split_idx + 7]
ape7 = float(np.mean(np.abs(fc7_train - akt7) / akt7))
print(f"cek 5.1: APE 7-langkah AR(7) = {ape7:.4f}")

# ===== Bagian 6: log1p + back-transform =====
y_log = np.log1p(seri)
sp = int(len(y_log) * 0.8)
tr, va = y_log[:sp], y_log[sp:]
mae_naive_log = float(np.mean(np.abs(va[1:] - va[:-1])))
naive_log = np.concatenate([[tr[-1]], va[:-1]])
mae_back = float(np.mean(np.abs(np.expm1(va) - np.expm1(naive_log))))
print(f"\nMAE naive ruang log      : {mae_naive_log:.4f}")
print(f"MAE naive back-transform : {mae_back:.4f}")

# ===== Bagian 7: Holt-Winters =====
def holt_winters_add(tr, period, alpha, beta, gamma, h):
    n, m = len(tr), period
    level0 = float(tr[:m].mean())
    level, trend_b = level0, 0.0
    seas = list(tr[:m] - level0)
    for i in range(m, n):
        lv_old = level
        level = alpha * (tr[i] - seas[i - m]) + (1 - alpha) * (level + trend_b)
        trend_b = beta * (level - lv_old) + (1 - beta) * trend_b
        seas.append(gamma * (tr[i] - level) + (1 - gamma) * seas[i - m])
    f = [level + (j + 1) * trend_b + seas[n - m + (j % m)] for j in range(h)]
    return np.array(f), level, trend_b, np.array(seas[-m:])

def holt_winters_mult(tr, period, alpha, beta, gamma, h):
    n, m = len(tr), period
    eps = 1e-8
    level0 = float(tr[:m].mean())
    level, trend_b = level0, 0.0
    seas = list(tr[:m] / (level0 + eps))
    for i in range(m, n):
        lv_old = level
        level = alpha * (tr[i] / (seas[i - m] + eps)) + (1 - alpha) * (level + trend_b)
        trend_b = beta * (level - lv_old) + (1 - beta) * trend_b
        seas.append(gamma * (tr[i] / (level + eps)) + (1 - gamma) * seas[i - m])
    f = [(level + (j + 1) * trend_b) * seas[n - m + (j % m)] for j in range(h)]
    return np.array(f)

fc_add, lv, tb, ss = holt_winters_add(train, 365, 0.3, 0.05, 0.3, H)
mae_add = float(np.mean(np.abs(fc_add - val)))
fc_mul = holt_winters_mult(train, 365, 0.3, 0.05, 0.3, H)
mae_mul = float(np.mean(np.abs(fc_mul - val)))
print(f"\nHW add  MAE: {mae_add:.4f}  level={lv:.2f} trend={tb:.4f}")
print(f"HW add  season[:4]: {np.round(ss[:4], 2).tolist()}")
print(f"HW mult MAE: {mae_mul:.4f}")

# ===== pdq: regresi OLS dengan dummies musiman (modulo benar) =====
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

def pdq_forecast(beta, n_train, period, h):
    m = period
    b0, b1, seas = beta[0], beta[1], beta[2:]
    return np.array([b0 + b1 * (n_train + j) + seas[(n_train + j) % m] for j in range(h)])

beta = pdq_fit(train, 365)
fc_pdq = pdq_forecast(beta, len(train), 365, H)
mae_pdq = float(np.mean(np.abs(fc_pdq - val)))
print(f"\npdq MAE: {mae_pdq:.4f}  a0={beta[0]:.3f} b1={beta[1]:.4f} seas[:3]={np.round(beta[2:5], 3).tolist()}")

# ===== Bagian 8: backtesting =====
def rolling_backtest(tr, h, build_forecast, n_folds=3):
    metrik = []
    n = len(tr)
    for k in range(n_folds):
        cut = n - (k + 1) * h
        trk, tek = tr[:cut], tr[cut:cut + h]
        fc = build_forecast(trk, h)
        metrik.append((float(np.mean(np.abs(tek - fc))), float(np.sqrt(np.mean((tek - fc) ** 2)))))
    return metrik

def buat_naive(trk, h):
    return np.full(h, float(trk[-1]))

def buat_sn365(trk, h):
    return trk[len(trk) - 365:len(trk) - 365 + h]

def buat_hw(trk, h):
    f, *_ = holt_winters_add(trk, 365, 0.3, 0.05, 0.3, h)
    return f

for nama, fn in [("naive", buat_naive), ("seas-naive365", buat_sn365), ("HW-add", buat_hw)]:
    met = rolling_backtest(train, 90, fn, 3)
    print(f"backtest {nama:14s}: {[(round(a, 3), round(b, 3)) for a, b in met]}")

# ===== kuis: konstanta =====
print(f"\nKUIS: APE7={ape7:.4f}")
print(f"KUIS: sMAPE [1,3,5,7] vs [2,2,4,4] = 0.4586 (rmse sqrt(3)=1.732)")
print(f"KUIS: r_final = 0.5000 (analitik)")
lo, hi = min(fc7.min(), fc_add.min()), max(fc7.max(), fc_add.max())
print(f"KUIS: walk_forward range pred = ({lo:.3f}, {hi:.3f})")
print(f"KUIS: mae_add range = ({mae_add - 0.25:.3f}, {mae_add + 0.25:.3f})")
print(f"KUIS: mae_mul range = ({mae_mul - 0.25:.3f}, {mae_mul + 0.25:.3f})")
print(f"KUIS: mae_pdq range = ({mae_pdq - 0.25:.3f}, {mae_pdq + 0.25:.3f})")

# ===== sunspots (butuh internet; opsional) =====
try:
    sunspots = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv',
                          delimiter=',', skiprows=1)[:, 1]
    print(f"\nsunspots: n={len(sunspots)} min={sunspots.min():.0f} max={sunspots.max():.0f}")
    print(f"sunspots ACF lag 132: {acf(sunspots, 132):.4f}")
    print(f"sunspots ACF lag 12 : {acf(sunspots, 12):.4f}")
    acf_all = {L: acf(sunspots, L) for L in range(1, 200)}
    below = [L for L, v in acf_all.items() if v < 0.2]
    print(f"lag pertama ACF < 0.2: {below[0] if below else 'tidak ada'}")
    top3 = sorted(acf_all, key=acf_all.get, reverse=True)[:5]
    print(f"5 lag ACF tertinggi: {[(L, round(acf_all[L], 3)) for L in top3]}")
except Exception as e:
    print(f"\nsunspots GAGAL diunduh: {type(e).__name__}: {e}")
