"""Sunspots: argmax ACF di rentang lag panjang."""
import numpy as np
import urllib.request

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
raw = urllib.request.urlopen(url, timeout=30).read().decode()
vals = [float(ln.split(',')[1]) for ln in raw.strip().splitlines()[1:] if ln.strip()]
sunspots = np.array(vals)

def acf(x, lag):
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    return float(np.sum(xc[:-lag] * xc[lag:]) / np.sum(xc * xc))

w = 132
kernel = np.ones(w) / w
ma = np.convolve(sunspots, kernel, mode='same')
det = sunspots - ma
d = det[w:-w]

acf_all = {L: acf(d, L) for L in range(1, 400)}
best_long = max((L for L in acf_all if 60 <= L <= 300), key=acf_all.get)
print(f"argmax ACF (lag 60-300): {best_long} nilai {acf_all[best_long]:.4f}")
# sekitar puncak
sekitar = [(L, round(acf_all[L], 3)) for L in range(best_long - 20, best_long + 21, 4)]
print("sekitar puncak:", sekitar)
print(f"ACF det lag 132: {acf_all[132]:.4f} | lag 264: {acf_all[264]:.4f}")

# alternatif: ACF pada seri asli (tanpa detrend) di rentang panjang
acf_raw = {L: acf(sunspots, L) for L in range(1, 400)}
best_raw = max((L for L in acf_raw if 60 <= L <= 300), key=acf_raw.get)
print(f"argmax ACF RAW (60-300): {best_raw} nilai {acf_raw[best_raw]:.4f}")
