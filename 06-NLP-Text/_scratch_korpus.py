"""Verifikasi TERPADU: satu korpus 6 kategori -> skip-gram -> purity + geometri sentimen."""
import numpy as np
from collections import Counter


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


TOPIK = {
    "hewan": ["kucing", "anjing", "burung", "kelinci"],
    "kendaraan": ["mobil", "motor", "sepeda", "kereta"],
    "makanan": ["nasi", "roti", "bakso", "soto"],
}
POS = ["bagus", "enak", "senang", "mantap"]
NEG = ["jelek", "payah", "kecewa", "buruk"]
FUNGSI = ["dan", "itu", "di"]
KAT3 = list(TOPIK)

kat_of = {}
for k, ws in TOPIK.items():
    for w in ws:
        kat_of[w] = k
for w in POS:
    kat_of[w] = "sifat_pos"
for w in NEG:
    kat_of[w] = "sifat_neg"
for w in FUNGSI:
    kat_of[w] = "fungsional"


def buat_korpus(rng, n=400):
    kalimat = []
    for _ in range(n):
        # --- klausa topik ---
        kat = rng.choice(KAT3)
        toks = [str(x) for x in rng.choice(TOPIK[kat], size=int(rng.integers(2, 5)), replace=True)]
        for _f in range(int(rng.integers(0, 3))):
            toks.insert(int(rng.integers(0, len(toks) + 1)), str(rng.choice(FUNGSI)))
        # --- klausa sentimen (70% kalimat) ---
        if rng.random() < 0.70:
            label_klausa = rng.random() < 0.5
            toks.append("itu")
            pool = POS if label_klausa else NEG
            for _s in range(int(rng.integers(1, 3))):
                toks.append(str(rng.choice(pool)))
            if rng.random() < 0.3:
                toks.append(str(rng.choice(pool)))
        kalimat.append(toks)
    return kalimat


korpus = buat_korpus(np.random.default_rng(42))
print("contoh:", [" ".join(s) for s in korpus[:4]])
cnt = Counter(w for s in korpus for w in s)
vocab = sorted(cnt)
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

pasangan = []
for s in korpus:
    ids = [w2i[w] for w in s]
    for i, t in enumerate(ids):
        for j in range(max(0, i - 2), min(len(ids), i + 3)):
            if j != i:
                pasangan.append((t, ids[j]))
print(f"pairs={len(pasangan)}")

prob = np.array([cnt[w] for w in vocab], dtype=float)
prob /= prob.sum()
prob[-1] = 1.0 - prob[:-1].sum()

DIM = 16
r = np.random.default_rng(0)
W_in = r.normal(0, 0.1, (V, DIM))
W_out = r.normal(0, 0.1, (V, DIM))
noise = prob ** 0.75
noise /= noise.sum()
for ep in range(5):
    lr = 0.05 * (1.0 - 0.5 * ep / 5)
    for (t, c) in pasangan:
        v, u = W_in[t], W_out[c]
        s = sigmoid(float(v @ u))
        gv = (s - 1.0) * u
        gu = (s - 1.0) * v
        for n_id in r.choice(V, size=3, p=noise):
            if n_id == c:
                continue
            un = W_out[n_id]
            sn = sigmoid(float(v @ un))
            gv += sn * un
            W_out[n_id] -= lr * (sn * v)
        W_in[t] -= lr * gv
        W_out[c] -= lr * gu

# --- purity topik ---
CATS = set(KAT3)
p = tot = 0
for i, w in enumerate(vocab):
    if kat_of[w] not in CATS:
        continue
    tot += 1
    bj = max((cos(W_in[i], W_in[j]), j) for j in range(V) if j != i)[1]
    p += kat_of[vocab[bj]] == kat_of[w]
print(f"NN purity topik: {p / tot:.2f} ({p}/{tot})")

# --- geometri sentimen ---
ip = [w2i[w] for w in POS]
inn = [w2i[w] for w in NEG]
spp = [cos(W_in[a], W_in[b]) for a in ip for b in ip if a < b]
snn = [cos(W_in[a], W_in[b]) for a in inn for b in inn if a < b]
spn = [cos(W_in[a], W_in[b]) for a in ip for b in inn]
print(f"pos-pos {np.mean(spp):+.3f} | neg-neg {np.mean(snn):+.3f} | pos-neg {np.mean(spn):+.3f}")
assert np.mean(spp) > np.mean(spn) + 0.15 and np.mean(snn) > np.mean(spn) + 0.15

# --- klasifikasi sentimen dari W_in ---
def train_sent(E, seed=0, epochs=200, lr=0.5):
    rng = np.random.default_rng(seed)
    kal = [s for s in korpus if any(t in POS or t in NEG for t in s)]
    idx = rng.permutation(len(kal))
    n_tr = int(0.75 * len(kal))
    X = []
    Y = []
    for s in kal:
        ids = [w2i[t] for t in s]
        npol = sum(1 for t in s if t in POS)
        nng = sum(1 for t in s if t in NEG)
        X.append(E[ids].mean(0))
        Y.append(1.0 if npol >= nng else 0.0)
    X, Y = np.array(X), np.array(Y)
    Xtr, ytr, Xva, yva = X[idx[:n_tr]], Y[idx[:n_tr]], X[idx[n_tr:]], Y[idx[n_tr:]]
    w = np.zeros(E.shape[1])
    b = 0.0
    for ep in range(epochs):
        z = Xtr @ w + b
        pr = sigmoid(z)
        w -= lr * (Xtr.T @ (pr - ytr) / len(ytr))
        b -= lr * (pr - ytr).mean()
    pv = sigmoid(Xva @ w + b)
    ce = -np.mean(yva * np.log(pv + 1e-12) + (1 - yva) * np.log(1 - pv + 1e-12))
    return float(((pv >= 0.5) == yva).mean()), float(ce)


acc_t, ce_t = train_sent(W_in)
rs = np.random.default_rng(7)
acc_r, ce_r = train_sent(rs.normal(0, 0.1, (V, DIM)))
print(f"sentimen: trained={acc_t:.3f} (ce {ce_t:.3f}) | random={acc_r:.3f} (ce {ce_r:.3f})")

# --- seeds ---
for seed_k, seed_i in [(43, 0), (44, 0)]:
    korp2 = buat_korpus(np.random.default_rng(seed_k))
    cnt2 = Counter(w for s in korp2 for w in s)
    pas2 = []
    for s in korp2:
        ids = [w2i[w] for w in s]
        for i, t in enumerate(ids):
            for j in range(max(0, i - 2), min(len(ids), i + 3)):
                if j != i:
                    pas2.append((t, ids[j]))
    pr2 = np.array([cnt2[w] for w in vocab], dtype=float)
    pr2 /= pr2.sum()
    pr2[-1] = 1.0 - pr2[:-1].sum()
    Wi = np.random.default_rng(seed_i).normal(0, 0.1, (V, DIM))
    Wo = np.random.default_rng(seed_i + 1000).normal(0, 0.1, (V, DIM))
    nz2 = pr2 ** 0.75
    nz2 /= nz2.sum()
    for ep in range(5):
        lr = 0.05 * (1.0 - 0.5 * ep / 5)
        for (t, c) in pas2:
            v, u = Wi[t], Wo[c]
            s = sigmoid(float(v @ u))
            gv = (s - 1.0) * u
            gu = (s - 1.0) * v
            for n_id in np.random.default_rng(seed_i + ep).choice(V, size=3, p=nz2):
                if n_id == c:
                    continue
                un = Wo[n_id]
                sn = sigmoid(float(v @ un))
                gv += sn * un
                Wo[n_id] -= lr * (sn * v)
            Wi[t] -= lr * gv
            Wo[c] -= lr * gu
    pp = tt = 0
    for i, w in enumerate(vocab):
        if kat_of[w] not in CATS:
            continue
        tt += 1
        bj = max((cos(Wi[i], Wi[j]), j) for j in range(V) if j != i)[1]
        pp += kat_of[vocab[bj]] == kat_of[w]
    print(f"seed korpus {seed_k}: purity {pp / tt:.2f}")
