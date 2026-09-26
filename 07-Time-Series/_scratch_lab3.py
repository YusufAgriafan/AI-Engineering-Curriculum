"""Angka pending: naive konstan di val, sum season HW, sunspots via urllib."""
import numpy as np

rng = np.random.default_rng(42)
t = np.arange(365 * 3, dtype=float)
seri = 100.0 + 0.05 * t + 20.0 * np.sin(2 * np.pi * t / 365.0) + rng.normal(0.0, 5.0, len(t))
split_idx = int(len(seri) * 0.8)
train, val = seri[:split_idx], seri[split_idx:]
H = len(val)

# naive konstan
pred_naive = np.full(H, float(train[-1]))
mae_naive_c = float(np.mean(np.abs(val - pred_naive)))
print(f"MAE naive KONSTAN di val ({H} hari): {mae_naive_c:.4f}  (train[-1]={train[-1]:.2f})")

# HAM
mae_ham = float(np.mean(np.abs(val - train.mean())))
print(f"MAE HAM: {mae_ham:.4f}")

# HW add season sum & max (kode sama dengan calib)
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

fc_add, lv, tb, ss = holt_winters_add(train, 365, 0.3, 0.05, 0.3, H)
print(f"sum(season) = {np.sum(ss):.4f} | max|season| = {np.max(np.abs(ss)):.3f}")

# sunspots via urllib
try:
    import urllib.request
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    vals = [float(ln.split(',')[1]) for ln in raw.strip().splitlines()[1:] if ln.strip()]
    sunspots = np.array(vals)
    print(f"sunspots: n={len(sunspots)} min={sunspots.min():.0f} max={sunspots.max():.1f}")

    def acf(x, lag):
        x = np.asarray(x, dtype=float)
        xc = x - x.mean()
        return float(np.sum(xc[:-lag] * xc[lag:]) / np.sum(xc * xc))

    print(f"ACF raw lag 132: {acf(sunspots, 132):.4f} | lag 12: {acf(sunspots, 12):.4f}")
    w = 132
    kernel = np.ones(w) / w
    ma = np.convolve(sunspots, kernel, mode='same')
    det = sunspots - ma
    acf_all = {L: acf(det[w:-w], L) for L in range(1, 300)}
    top5 = sorted(acf_all, key=acf_all.get, reverse=True)[:5]
    print(f"top-5 lag setelah detrend: {[(L, round(acf_all[L], 3)) for L in top5]}")
    print(f"ACF detrended lag 132: {acf_all[132]:.4f}")
except Exception as e:
    print(f"sunspots gagal: {type(e).__name__}: {e}")
