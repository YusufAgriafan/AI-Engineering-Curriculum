"""Tune RNN (next-char) + verifikasi sentiment avg-embedding dari W_in part 1."""
import numpy as np
from collections import Counter

# ============ bagian 1: toy corpus + skip-gram (config terkunci) ============
KATA = {
    "hewan":     ["kucing", "anjing", "burung", "kelinci"],
    "kendaraan": ["mobil", "motor", "sepeda", "kereta"],
    "makanan":   ["nasi", "roti", "bakso", "soto"],
    "sifat_pos": ["bagus", "enak", "senang", "mantap"],
    "sifat_neg": ["jelek", "payah", "kecewa", "buruk"],
    "fungsional": ["saya", "dan", "itu", "di", "tapi"],
}
freq = {"hewan": 20, "kendaraan": 20, "makanan": 20, "sifat_pos": 16,
        "sifat_neg": 16, "fungsional": 72}
kata_list, kat_list = [], []
for kat, daftar in KATA.items():
    kata_list.extend(daftar)
    kat_list.extend([kat] * len(daftar))
V = len(kata_list)
kat_of = dict(zip(kata_list, kat_list))
prob = np.array([freq[kat_of[w]] for w in kata_list], dtype=float)
prob /= prob.sum()
prob[-1] = 1.0 - prob[:-1].sum()
CATS = ("hewan", "kendaraan", "makanan")


def dist_kat(a, b):
    if a == b:
        return 0
    if a in CATS and b in CATS:
        return 2
    return 3


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


rng = np.random.default_rng(42)
pairs = []
while len(pairs) < 8000:
    t, c = rng.choice(V, p=prob, size=2)
    if t == c:
        continue
    d = dist_kat(kat_of[kata_list[t]], kat_of[kata_list[c]])
    if d == 0:
        pairs.append((int(t), int(c)))
    elif d == 2 and rng.random() < 0.10:
        pairs.append((int(t), int(c)))

rng_t = np.random.default_rng(0)
DIM = 16
W_in = rng_t.normal(0, 0.1, (V, DIM))
W_out = rng_t.normal(0, 0.1, (V, DIM))
noise = prob ** 0.75
noise /= noise.sum()
for ep in range(5):
    lr_ep = 0.05 * (1.0 - 0.5 * ep / 5)
    for (t, c) in pairs:
        v = W_in[t]
        u = W_out[c]
        s = sigmoid(float(v @ u))
        grad_v = (s - 1.0) * u
        grad_u = (s - 1.0) * v
        for n_id in rng_t.choice(V, size=3, p=noise):
            if n_id == c:
                continue
            un = W_out[n_id]
            sn = sigmoid(float(v @ un))
            grad_v += sn * un
            W_out[n_id] -= lr_ep * (sn * v)
        W_in[t] -= lr_ep * grad_v
        W_out[c] -= lr_ep * grad_u

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

p_full = 0
for i, w in enumerate(kata_list):
    if kat_of[w] not in CATS:
        continue
    bj = max((cos(W_in[i], W_in[j]), j) for j in range(V) if j != i)[1]
    p_full += kat_of[kata_list[bj]] == kat_of[w]
print(f"sanity purity: {p_full / 12:.2f}")

# ============ sentiment pakai W_in (korpus toy) ============
TOPIK = ["kucing", "mobil", "nasi", "burung", "motor", "bakso"]
POS = ["bagus", "enak", "senang", "mantap"]
NEG = ["jelek", "payah", "kecewa", "buruk"]
w2i = {w: i for i, w in enumerate(kata_list)}

rng_s = np.random.default_rng(7)
kalimat = []
for _ in range(800):
    label = 1.0 if rng_s.random() < 0.5 else 0.0
    sifat = POS if label == 1.0 else NEG
    toks = ["saya"]
    for _t in range(int(rng_s.integers(1, 3))):
        toks.append(rng_s.choice(TOPIK))
    toks.append("itu")
    k = int(rng_s.integers(1, 4))
    for j in range(k):
        toks.append(rng_s.choice(sifat))
        if j < k - 1 and rng_s.random() < 0.4:
            toks.append("dan")
    kalimat.append((toks, label))

cnt_lab = Counter(l for _, l in kalimat)
print("label balance:", dict(cnt_lab))


def fitur_avg(ids, E):
    return E[ids].mean(0)


def train_sent(E, seed=0, epochs=200, lr=0.5):
    rs = np.random.default_rng(seed)
    idx = rs.permutation(len(kalimat))
    n_tr = int(0.75 * len(kalimat))
    tr, va = idx[:n_tr], idx[n_tr:]
    Xtr = np.array([fitur_avg([w2i[t] for t in kalimat[i][0]], E) for i in tr])
    ytr = np.array([kalimat[i][1] for i in tr])
    Xva = np.array([fitur_avg([w2i[t] for t in kalimat[i][0]], E) for i in va])
    yva = np.array([kalimat[i][1] for i in va])
    w = np.zeros(E.shape[1])
    b = 0.0
    for ep in range(epochs):
        z = Xtr @ w + b
        p = sigmoid(z)
        w -= lr * (Xtr.T @ (p - ytr) / len(ytr))
        b -= lr * (p - ytr).mean()
    return float((((sigmoid(Xva @ w + b)) >= 0.5) == yva).mean())


acc_t = train_sent(W_in)
acc_r = train_sent(rng_s.normal(0, 0.1, (V, DIM)))
print(f"sentiment: trained={acc_t:.3f} random={acc_r:.3f}")

# ============ RNN next-char: tuning ============
NAMA = ["adi", "budi", "citra", "dewi", "eka", "fajar", "gita", "hadi", "indah",
        "joko", "kartika", "lina", "maya", "nanda", "okta", "putri", "rizky",
        "sari", "tono", "utami", "vina", "wawan", "yuni", "zahra"]
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
Csz = len(chars)
seq = [c2i[c] for c in teks_full]


def onehot_batch(ids, width):
    z = np.zeros((len(ids), width))
    z[np.arange(len(ids)), ids] = 1.0
    return z


CTX = 8
Xr, yr = [], []
for i in range(len(seq) - CTX):
    Xr.append(seq[i:i + CTX])
    yr.append(seq[i + CTX])
Xr, yr = np.array(Xr), np.array(yr)
perm = np.random.default_rng(0).permutation(len(Xr))
Xr, yr = Xr[perm], yr[perm]
n_tr = int(0.75 * len(Xr))
Xtr, ytr, Xva, yva = Xr[:n_tr], yr[:n_tr], Xr[n_tr:], yr[n_tr:]


def train_rnn(hidden, epochs, lr, seed=0, init=0.5):
    rng = np.random.default_rng(seed)
    Wx = rng.normal(0, init, (Csz, hidden))
    Wh = rng.normal(0, init, (hidden, hidden))
    Wy = rng.normal(0, 0.1, (hidden, Csz))
    by = np.zeros(Csz)
    bh = np.zeros(hidden)
    hist = []
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
            tot += -np.mean(np.log(probs[np.arange(n), yb] + 1e-12))
            cnt += 1
            dlogits = probs.copy()
            dlogits[np.arange(n), yb] -= 1.0
            dlogits /= n
            dWy = hs[-1].T @ dlogits
            dby = dlogits.sum(0)
            dh = dlogits @ Wy.T
            dWx = np.zeros_like(Wx)
            dWh = np.zeros_like(Wh)
            dbh = np.zeros_like(bh)
            for t in reversed(range(CTX)):
                da = dh * (1 - np.tanh(Hs[t]) ** 2)
                dWx += onehot_batch(Xb[:, t], Csz).T @ da
                dWh += da.T @ hs[t]
                dbh += da.sum(0)
                dh = da @ Wh
            for g in (dWx, dWh, dWy, dbh, dby):
                np.clip(g, -5, 5, out=g)
            Wx -= lr * dWx
            Wh -= lr * dWh
            Wy -= lr * dWy
            bh -= lr * dbh
            by -= lr * dby
        hs = [np.zeros((len(Xva), hidden))]
        for t in range(CTX):
            h_pre = hs[t] @ Wh.T + onehot_batch(Xva[:, t], Csz) @ Wx + bh
            hs.append(np.tanh(h_pre))
        acc = float(((hs[-1] @ Wy + by).argmax(1) == yva).mean())
        hist.append((tot / cnt, acc))
    return hist


counts = np.ones((Csz, Csz))
for a, b in zip(seq[:n_tr + CTX], seq[1:n_tr + CTX]):
    counts[a, b] += 1
Pn = counts / counts.sum(1, keepdims=True)
acc_bigram = float((np.array([Pn[x[-1]].argmax() for x in Xva]) == yva).mean())
print(f"bigram val acc: {acc_bigram:.3f}")

for hidden, epochs, lr, init in [(24, 80, 0.3, 0.5), (32, 150, 0.1, 0.5), (32, 150, 0.2, 0.5),
                                 (48, 150, 0.1, 0.5), (32, 300, 0.1, 0.5), (64, 200, 0.1, 0.3)]:
    h = train_rnn(hidden, epochs, lr, init=init)
    print(f"h={hidden} ep={epochs} lr={lr} init={init}: loss {h[0][0]:.3f}->{h[-1][0]:.3f} val_acc={h[-1][1]:.3f}")
