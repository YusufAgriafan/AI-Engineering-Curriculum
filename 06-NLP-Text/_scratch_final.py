"""Final config check: seed stability, generation, vanishing gradient."""
import numpy as np
from collections import Counter

# ---------- corpus ----------
KALIMAT = [
    "adi makan nasi", "budi minum air", "citra membaca buku",
    "dewi menulis surat", "eka bermain bola", "fajar melihat kaca",
    "gita menyanyi lagu", "hadi membuka kelas", "indah menanam bunga",
    "joko menghitung uang", "kartika menjahit baju", "lina menjual buah",
    "maya menyapu lantai", "nanda mendengar kabar", "okta membeli roti",
    "putri menyiram tanaman", "rizky mengejar bus", "sari mencuci piring",
    "tono memperbaiki motor", "utami mengaduk adonan", "vina mengambil paket",
    "wawan mengangkat meja", "yuni menjahit kerudung", "zahra mengeringkan baju",
    "bagus membawa kasur", "candra mengisi gelas", "dian menyusun kartu",
    "fitri menata piring", "gilang menyalakan lampu", "husni menutup pintu",
    "irma menggulai sayur", "krisna meniup terompet",
    "makan nasi itu enak", "minum air itu menyegarkan",
    "membaca buku itu bagus", "menulis surat itu mudah",
    "bermain bola itu seru", "melihat kaca itu aneh",
    "menyanyi lagu itu indah", "membuka kelas itu pagi",
    "adi minum teh", "budi makan roti", "citra menjahit baju",
    "dewi menyiram bunga", "eka membaca koran", "fajar mengejar kucing",
    "gita membeli buku", "hadi menyapu halaman", "indah menjahit gaun",
    "joko menanam padi", "kartika menyusun bunga", "lina menghitung uang",
    "maya membuka pintu", "nanda memperbaiki sepeda", "okta mengambil air",
    "putri menulis surat", "rizky menyiram tanaman", "sari mengupas kentang",
    "tono membawa tas", "utami meniup peluit", "vina menjahit sarung",
    "wawan mengaduk kopi", "yuni menata meja", "zahra mengeringkan rambut",
    "makan roti itu enak", "minum teh itu nikmat", "membaca koran itu bagus",
    "menulis surat itu mudah", "bermain bola itu seru", "melihat kaca itu aneh",
]
teks_full = "\n".join(KALIMAT) + "\n"
chars = sorted(set(teks_full))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
Csz = len(chars)
seq = [c2i[c] for c in teks_full]
CTX = 4

Xr, yr = [], []
for i in range(len(seq) - CTX):
    Xr.append(seq[i:i + CTX])
    yr.append(seq[i + CTX])
Xr, yr = np.array(Xr), np.array(yr)
perm = np.random.default_rng(0).permutation(len(Xr))
Xr, yr = Xr[perm], yr[perm]
n_tr = int(0.75 * len(Xr))
Xtr, ytr, Xva, yva = Xr[:n_tr], yr[:n_tr], Xr[n_tr:], yr[n_tr:]


def onehot_batch(ids, width):
    z = np.zeros((len(ids), width))
    z[np.arange(len(ids)), ids] = 1.0
    return z


def train_rnn(seed=0, hidden=32, epochs=150, lr=0.2, init=0.5):
    rng = np.random.default_rng(seed)
    Wx = rng.normal(0, init, (Csz, hidden))
    Wh = rng.normal(0, init, (hidden, hidden))
    Wy = rng.normal(0, 0.1, (hidden, Csz))
    by = np.zeros(Csz)
    bh = np.zeros(hidden)
    first_loss = last_loss = None
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        tot, cnt = 0.0, 0
        for s in range(0, len(order), 32):
            bidx = order[s:s + 32]
            Xb, yb = Xtr[bidx], ytr[bidx]
            n = len(bidx)
            hs = [np.zeros((n, hidden))]
            Hs = []
            for t in range(CTX):
                h_pre = hs[t] @ Wh.T + onehot_batch(Xb[:, t], Csz) @ Wx + bh
                hs.append(np.tanh(h_pre))
                Hs.append(h_pre)
            logits = hs[-1] @ Wy + by
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs = e / e.sum(1, keepdims=True)
            tot += -np.mean(np.log(probs[np.arange(n), yb] + 1e-12)); cnt += 1
            dlogits = probs.copy()
            dlogits[np.arange(n), yb] -= 1.0
            dlogits /= n
            dWy = hs[-1].T @ dlogits
            dby = dlogits.sum(0)
            dh = dlogits @ Wy.T
            dWx = np.zeros_like(Wx); dWh = np.zeros_like(Wh); dbh = np.zeros_like(bh)
            for t in reversed(range(CTX)):
                da = dh * (1 - np.tanh(Hs[t]) ** 2)
                dWx += onehot_batch(Xb[:, t], Csz).T @ da
                dWh += da.T @ hs[t]
                dbh += da.sum(0)
                dh = da @ Wh
            for g in (dWx, dWh, dWy, dbh, dby):
                np.clip(g, -5, 5, out=g)
            Wx -= lr * dWx; Wh -= lr * dWh; Wy -= lr * dWy; bh -= lr * dbh; by -= lr * dby
        if ep == 0:
            first_loss = tot / cnt
        last_loss = tot / cnt
    hs = [np.zeros((len(Xva), hidden))]
    for t in range(CTX):
        h_pre = hs[t] @ Wh.T + onehot_batch(Xva[:, t], Csz) @ Wx + bh
        hs.append(np.tanh(h_pre))
    logits = hs[-1] @ Wy + by
    e = np.exp(logits - logits.max(1, keepdims=True))
    pr = e / e.sum(1, keepdims=True)
    acc = float((logits.argmax(1) == yva).mean())

    def generate(seed_str, n=150, temp=1.0):
        out = seed_str
        ctx_ids = [c2i[c] for c in seed_str[-CTX:]]
        for _ in range(n):
            h = np.zeros(hidden)
            for cid in ctx_ids:
                h = np.tanh(onehot_batch([cid], Csz) @ Wx + h @ Wh.T + bh)
            logits = (h @ Wy + by).ravel()
            p = np.exp((logits - logits.max()) / temp)
            p = p / p.sum()
            nxt = int(rng.choice(Csz, p=p))
            out += i2c[nxt]
            ctx_ids = ctx_ids[1:] + [nxt]
        return out

    return first_loss, last_loss, acc, Wx, Wh, Wy, by, bh, generate


counts = np.ones((Csz, Csz))
for a, b in zip(seq[:n_tr + CTX], seq[1:n_tr + CTX]):
    counts[a, b] += 1
Pn = counts / counts.sum(1, keepdims=True)
acc_b = float((np.array([Pn[x[-1]].argmax() for x in Xva]) == yva).mean())
print(f"bigram acc: {acc_b:.3f}")

for seed in (0, 1):
    fl, ll, acc, *rest = train_rnn(seed=seed)
    print(f"seed={seed}: loss {fl:.3f}->{ll:.3f} val_acc={acc:.3f}")

# generation demo with the seed-0 model
res = train_rnn(seed=0)
gen = res[8]
for t in (0.3, 0.7, 1.2):
    out = gen("mak", 130, t)
    print(f"\n--- T={t} ---\n{out[:160]}")

# vanishing gradient (skala gradien rata-rata per langkah waktu)
def bptt_norm(act, T=30, seed=0):
    r = np.random.default_rng(seed)
    Wh = np.abs(r.normal(0, 0.35, (20, 20)))
    g = np.full(20, 0.5)
    out = []
    for _ in range(T):
        if act == "tanh":
            deriv = 1 - np.tanh(np.linspace(-1.5, 1.5, 20)) ** 2
        else:
            s = 1 / (1 + np.exp(-np.linspace(-2.5, 2.5, 20)))
            deriv = s * (1 - s)
        g = Wh @ (g * deriv)
        out.append(g.mean())
    return out

nt, ns = bptt_norm("tanh"), bptt_norm("sigmoid")
print(f"\ngrad T=30: tanh {nt[-1]:.2e} vs sigmoid {ns[-1]:.2e}  (T=10: {nt[9]:.2e} vs {ns[9]:.2e})")
