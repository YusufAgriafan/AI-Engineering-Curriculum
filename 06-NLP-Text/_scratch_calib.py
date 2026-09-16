"""Calibration for Bab 6 (NLP & Text) — everything numpy-only.

Locks down:
1. Corpus ko-okurensi: struktur kategori via jendela.
2. Skip-gram + negative sampling: cluster semantik di PCA 2D (stabil seed).
3. Korpus review: skip-gram belajar geometri sentimen TANPA label.
4. Klasifikasi sentimen avg-embedding: trained-emb >> random-emb.
5. Char-LM bigram: konvergen, sampling temperature berperilaku benar.
6. RNN char-level > bigram pada next-char prediction.
7. Vanishing gradient: sigmoid vs tanh pada ||dW_rec||.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

RNG = np.random.default_rng(42)

# ============================================================ 1-2. CORPUS + SKIP-GRAM
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
kata_list, kategori_kata = [], []
for kat, daftar in KATA.items():
    kata_list.extend(daftar)
    kategori_kata.extend([kat] * len(daftar))
V = len(kata_list)
idx2w = dict(enumerate(kata_list))
w2idx = {w: i for i, w in enumerate(kata_list)}
kat_of = dict(zip(kata_list, kategori_kata))
prob = np.array([freq[kat_of[w]] for w in kata_list], dtype=float)
prob = prob / prob.sum()
prob[-1] = 1.0 - prob[:-1].sum()   # paksa jumlah tepat 1.0 (presisi float)
print(f"vocab = {V} kata")


def dist_kat(a, b):
    if a == b:
        return 0
    content = ("hewan", "kendaraan", "makanan")
    if a in content and b in content:
        return 2
    return 3


GROUPS = np.array([[dist_kat(kat_of[a], kat_of[b]) for b in kata_list] for a in kata_list])


def sample_pairs(n_pairs, rng, win=2):
    """Jarak 0 (satu kategori) selalu diterima; jarak 2 (lintas topik) jarang —
    seperti teks nyata: kalimat biasanya satu topik, kadang mencampur."""
    pairs = []
    while len(pairs) < n_pairs:
        t, c = rng.choice(V, p=prob, size=2)
        if t == c:
            continue
        d = dist_kat(kat_of[kata_list[t]], kat_of[kata_list[c]])
        if d == 0:
            pairs.append((int(t), int(c)))
        elif d == 2 and rng.random() < 0.25:
            pairs.append((int(t), int(c)))
    return pairs


pairs = sample_pairs(8000, RNG)
c_same = sum(1 for t, c in pairs if GROUPS[t, c] == 0) / len(pairs)
c_d2 = sum(1 for t, c in pairs if GROUPS[t, c] == 2) / len(pairs)
print(f"pair stats: satu-kategori={c_same:.2f}, lintas-topik={c_d2:.2f}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_sg(dim=32, epochs=5, lr=0.05, k_neg=3, seed=0):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0, 0.1, (V, dim))
    W_out = rng.normal(0, 0.1, (V, dim))
    noise = prob ** 0.75
    noise = noise / noise.sum()
    loss_hist = []
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for (t, c) in pairs:
            v = W_in[t]
            u = W_out[c]
            s = sigmoid(float(v @ u))
            grad_v = (s - 1.0) * u
            grad_u = (s - 1.0) * v
            tot += -np.log(max(s, 1e-12))
            for n_id in rng.choice(V, size=k_neg, p=noise):
                if n_id == c:
                    continue
                un = W_out[n_id]
                sn = sigmoid(float(v @ un))
                grad_v += sn * un
                W_out[n_id] -= lr * (sn * v)
                tot += -np.log(max(1 - sn, 1e-12))
            W_in[t] -= lr * grad_v
            W_out[c] -= lr * grad_u
            cnt += 1
        loss_hist.append(tot / cnt)
    return W_in, loss_hist


W_in, loss_hist = train_sg()
print("sg loss:", [f"{x:.3f}" for x in loss_hist])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


sims_same, sims_cross = [], []
for i in range(V):
    for j in range(i + 1, V):
        s = cos(W_in[i], W_in[j])
        if kat_of[kata_list[i]] == kat_of[kata_list[j]]:
            sims_same.append(s)
        elif kat_of[kata_list[i]] != "fungsional" and kat_of[kata_list[j]] != "fungsional":
            sims_cross.append(s)
m_same, m_cross = np.mean(sims_same), np.mean(sims_cross)
print(f"cosine: same-cat {m_same:+.3f}, cross-content {m_cross:+.3f}")
assert m_same > m_cross + 0.15

# manual PCA to 2D
mu = W_in.mean(0)
Xc = W_in - mu
Cm = (Xc.T @ Xc) / V
evals, evecs = np.linalg.eigh(Cm)
W2 = Xc @ evecs[:, ::-1][:, :2]
cats3 = ["hewan", "kendaraan", "makanan"]
centers = {c: W2[[i for i, w in enumerate(kata_list) if kat_of[w] == c]].mean(0) for c in cats3}
# kriteria utama: nearest-neighbor purity di RUANG PENUH (cosine) — klaim inti
# "kata yang mirip = tetangga terdekat". PCA 2D hanya untuk visual.
content_idx_full = [i for i, w in enumerate(kata_list) if kat_of[w] in cats3]
nn_purity = 0
for i in content_idx_full:
    best_j = max((cos(W_in[i], W_in[j]), j) for j in range(V) if j != i)[1]
    if kat_of[kata_list[best_j]] == kat_of[kata_list[i]]:
        nn_purity += 1
nn_purity /= len(content_idx_full)
print(f"nearest-neighbor purity (full dim): {nn_purity:.2f}")
# kriteria 2D (sekunder, untuk plot): purity proyeksi
content_idx = [i for i, w in enumerate(kata_list) if kat_of[w] in cats3]
purity = 0
for i in content_idx:
    ds = {c: np.linalg.norm(W2[i] - centers[c]) for c in cats3}
    if min(ds, key=ds.get) == kat_of[kata_list[i]]:
        purity += 1
purity /= len(content_idx)
print(f"2D cluster purity: {purity:.2f}")


def mean_dist(idxs):
    tot, cnt = 0.0, 0
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            tot += np.linalg.norm(W2[idxs[a]] - W2[idxs[b]])
            cnt += 1
    return tot / max(cnt, 1)


grp = {c: [i for i, w in enumerate(kata_list) if kat_of[w] == c] for c in cats3}
intra = np.mean([mean_dist(grp[c]) for c in cats3])
inter = np.mean([np.linalg.norm(W2[a] - W2[b]) for c1 in cats3 for c2 in cats3 if c1 < c2
                 for a in grp[c1] for b in grp[c2]])
print(f"intra {intra:.2f} vs inter {inter:.2f}")
assert nn_purity >= 0.9, "tetangga terdekat harus satu kategori"
assert purity >= 0.6 and intra < inter, "2D harus menunjukkan kecenderungan klaster"

# ============================================================ 3-4. REVIEW CORPUS → SENTIMENT
TOPIK = ["kucing", "mobil", "nasi", "burung", "motor", "bakso"]
POS = ["bagus", "enak", "senang", "mantap"]
NEG = ["jelek", "payah", "kecewa", "buruk"]
FUNC = ["saya", "ini", "dan", "sekali"]


def korpus_review(rng, n=600):
    """Review sintetis: kata positif hanya dekat kata positif, negatif dengan negatif."""
    kalimat = []
    for _ in range(n):
        label = 1.0 if rng.random() < 0.5 else 0.0
        topik = rng.choice(TOPIK)
        sifat = POS if label == 1.0 else NEG
        k = int(rng.integers(1, 3))
        toks = ["saya", topik, "ini"]
        toks.append(sifat[0]) if False else None
        for j in range(k):
            toks.append(rng.choice(sifat))
            if j < k - 1 and rng.random() < 0.5:
                toks.append("dan")
        if rng.random() < 0.4:
            toks.append("sekali")
        kalimat.append((" ".join(toks), label))
    return kalimat


RNG2 = np.random.default_rng(7)
kalimat_review = korpus_review(RNG2)
print("contoh:", kalimat_review[0], "|", kalimat_review[1])
VOCAB_R = sorted({w for s, _ in kalimat_review for w in s.split()})
w2i_r = {w: i for i, w in enumerate(VOCAB_R)}
VR = len(VOCAB_R)
print(f"vocab review = {VR}")


def pasangan_dari_korpus(kalimat_list, w2i, win=2):
    pairs = []
    for s, _lab in kalimat_list:
        ids = [w2i[w] for w in s.split()]
        for i, t in enumerate(ids):
            for j in range(max(0, i - win), min(len(ids), i + win + 1)):
                if j != i:
                    pairs.append((t, ids[j]))
    return pairs


pairs_r = pasangan_dari_korpus(kalimat_review, w2i_r)
print(f"pairs dari korpus review: {len(pairs_r)}")

# frekuensi token untuk negative sampling
cnt_r = np.zeros(VR)
for s, _ in kalimat_review:
    for w in s.split():
        cnt_r[w2i_r[w]] += 1
prob_r = cnt_r / cnt_r.sum()


def train_sg_pairs(pairs, Vsz, prob_uni, dim=24, epochs=4, lr=0.05, k_neg=3, seed=0):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0, 0.1, (Vsz, dim))
    W_out = rng.normal(0, 0.1, (Vsz, dim))
    noise = prob_uni ** 0.75
    noise = noise / noise.sum()
    loss_hist = []
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for (t, c) in pairs:
            v = W_in[t]
            u = W_out[c]
            s = sigmoid(float(v @ u))
            grad_v = (s - 1.0) * u
            grad_u = (s - 1.0) * v
            tot += -np.log(max(s, 1e-12))
            for n_id in rng.choice(Vsz, size=k_neg, p=noise):
                if n_id == c:
                    continue
                un = W_out[n_id]
                sn = sigmoid(float(v @ un))
                grad_v += sn * un
                W_out[n_id] -= lr * (sn * v)
                tot += -np.log(max(1 - sn, 1e-12))
            W_in[t] -= lr * grad_v
            W_out[c] -= lr * grad_u
            cnt += 1
        loss_hist.append(tot / cnt)
    return W_in, loss_hist


E_r, loss_r = train_sg_pairs(pairs_r, VR, prob_r)
print("sg review loss:", [f"{x:.3f}" for x in loss_r])

# geometri: pos words should cluster, neg words cluster, both apart
i_pos = [w2i_r[w] for w in POS]
i_neg = [w2i_r[w] for w in NEG]
c_pos = E_r[i_pos].mean(0)
c_neg = E_r[i_neg].mean(0)
sims_pos = [cos(E_r[a], E_r[b]) for a in i_pos for b in i_pos if a < b]
sims_neg = [cos(E_r[a], E_r[b]) for a in i_neg for b in i_neg if a < b]
sims_pn = [cos(E_r[a], E_r[b]) for a in i_pos for b in i_neg]
print(f"cos pos-pos {np.mean(sims_pos):+.3f} | neg-neg {np.mean(sims_neg):+.3f} | pos-neg {np.mean(sims_pn):+.3f}")
assert np.mean(sims_pos) > np.mean(sims_pn) + 0.15
assert np.mean(sims_neg) > np.mean(sims_pn) + 0.15


def fitur_avg(seq_ids, E):
    return E[seq_ids].mean(0)


def train_sentiment(E, seed=0, epochs=200, lr=0.5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(kalimat_review))
    n_tr = int(0.75 * len(kalimat_review))
    tr, va = idx[:n_tr], idx[n_tr:]
    Xtr = np.array([fitur_avg([w2i_r[w] for w in kalimat_review[i][0].split()], E) for i in tr])
    ytr = np.array([kalimat_review[i][1] for i in tr])
    Xva = np.array([fitur_avg([w2i_r[w] for w in kalimat_review[i][0].split()], E) for i in va])
    yva = np.array([kalimat_review[i][1] for i in va])
    w = np.zeros(E.shape[1])
    b = 0.0
    for ep in range(epochs):
        z = Xtr @ w + b
        p = sigmoid(z)
        gw = Xtr.T @ (p - ytr) / len(ytr)
        gb = (p - ytr).mean()
        w -= lr * gw
        b -= lr * gb
    acc = float((((sigmoid(Xva @ w + b)) >= 0.5) == yva).mean())
    return acc, yva.mean()


acc_r_trained, bal = train_sentiment(E_r)
acc_r_rand, _ = train_sentiment(RNG.normal(0, 0.1, (VR, 24)))
print(f"sentiment acc: trained-emb={acc_r_trained:.3f}, random-emb={acc_r_rand:.3f} (label balance {bal:.2f})")
assert bal > 0.45 and bal < 0.55, "label harus balanced"
assert acc_r_trained >= 0.9
assert acc_r_trained > acc_r_rand + 0.15

# ============================================================ 5. CHAR-LM
NAMA = [
    "adi", "budi", "citra", "dewi", "eka", "fajar", "gita", "hadi", "indah",
    "joko", "kartika", "lina", "maya", "nanda", "okta", "putri", "rizky",
    "sari", "tono", "utami", "vina", "wawan", "yuni", "zahra",
]
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
    "putri menulis surat", "rizky menyiram tanaman", "sari mengupas Kentang",
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
print(f"corpus chars: {len(teks_full)}, vocab char: {Csz}")


def eval_bigram_ee(seq_in, alpha=0.5):
    counts = np.ones((Csz, Csz)) * alpha
    for a, b in zip(seq_in[:-1], seq_in[1:]):
        counts[a, b] += 1
    P = counts / counts.sum(1, keepdims=True)
    ee = -np.mean(np.log(P[seq_in[:-1], seq_in[1:]] + 1e-12))
    return ee, P


ee_bigram, P_bigram = eval_bigram_ee(seq)
unigram = np.bincount(seq, minlength=Csz) / len(seq)
ee_unigram = -np.mean(np.log(unigram[seq[1:]] + 1e-12))
print(f"EE unigram: {ee_unigram:.3f}, bigram: {ee_bigram:.3f}")
assert ee_bigram < ee_unigram - 0.2


def sample_bigram(P, seed_str="m", n=120, temp=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    out = seed_str
    cur = c2i[seed_str[-1]]
    for _ in range(n):
        p = P[cur] ** (1.0 / temp)
        p = p / p.sum()
        nxt = rng.choice(Csz, p=p)
        out += i2c[nxt]
        cur = nxt
    return out


print("--- bigram samples ---")
for t in (0.3, 1.0, 1.8):
    print(f"[T={t}]", repr(sample_bigram(P_bigram, "m", 90, t, np.random.default_rng(1))[:90]))

# ---- RNN char-level (next-char classification) ----
def onehot_batch(ids, width):
    z = np.zeros((len(ids), width))
    z[np.arange(len(ids)), ids] = 1.0
    return z


def build_data(seq_in, ctx=8):
    X, y = [], []
    for i in range(len(seq_in) - ctx):
        X.append(seq_in[i:i + ctx])
        y.append(seq_in[i + ctx])
    return np.array(X), np.array(y)


CTX = 8
Xr, yr = build_data(seq, CTX)
perm = np.random.default_rng(0).permutation(len(Xr))
Xr, yr = Xr[perm], yr[perm]
n_tr = int(0.75 * len(Xr))
Xtr, ytr, Xva, yva = Xr[:n_tr], yr[:n_tr], Xr[n_tr:], yr[n_tr:]
print(f"RNN data: train {len(Xtr)}, val {len(Xva)}")


def train_rnn(hidden=24, epochs=80, lr=0.3, seed=0, act="tanh", clip=5.0):
    rng = np.random.default_rng(seed)
    Wx = rng.normal(0, 0.3, (Csz, hidden))
    Wh = rng.normal(0, 0.3, (hidden, hidden))
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
                np.clip(g, -clip, clip, out=g)
            Wx -= lr * dWx
            Wh -= lr * dWh
            Wy -= lr * dWy
            bh -= lr * dbh
            by -= lr * dby
        hs = [np.zeros((len(Xva), hidden))]
        for t in range(CTX):
            h_pre = hs[t] @ Wh.T + onehot_batch(Xva[:, t], Csz) @ Wx + bh
            hs.append(np.tanh(h_pre))
        logits = hs[-1] @ Wy + by
        acc = float((logits.argmax(1) == yva).mean())
        hist.append((tot / cnt, acc))
    return hist


hist_tanh = train_rnn()
print(f"RNN tanh: loss {hist_tanh[0][0]:.3f} -> {hist_tanh[-1][0]:.3f}, val acc {hist_tanh[-1][1]:.3f}")


def bigram_val_acc():
    counts = np.ones((Csz, Csz))
    for a, b in zip(seq[:n_tr + CTX], seq[1:n_tr + CTX]):
        counts[a, b] += 1
    Pn = counts / counts.sum(1, keepdims=True)
    preds = np.array([Pn[x[-1]].argmax() for x in Xva])
    return float((preds == yva).mean())


acc_bigram = bigram_val_acc()
print(f"bigram val acc: {acc_bigram:.3f}")
assert hist_tanh[-1][1] > acc_bigram, "RNN harus menang atas bigram"

# ============================================================ 7. VANISHING GRADIENT
def bptt_norm(act, T=30, seed=0):
    rng = np.random.default_rng(seed)
    Wh = np.abs(rng.normal(0, 0.35, (20, 20)))
    g = np.full(20, 0.5)
    norms = []
    for _t in range(T):
        if act == "tanh":
            deriv = 1 - np.tanh(np.linspace(-1.5, 1.5, 20)) ** 2
        else:
            s = 1.0 / (1.0 + np.exp(-np.linspace(-2.5, 2.5, 20)))
            deriv = s * (1 - s)
        g = Wh @ (g * deriv)
        norms.append(g.mean())
    return norms


n_tanh = bptt_norm("tanh")
n_sig = bptt_norm("sigmoid")
print(f"grad norm T=30: tanh={n_tanh[-1]:.2e}, sigmoid={n_sig[-1]:.2e}")
assert n_sig[-1] < n_tanh[-1]

# ============================================================ TOKENIZER / BPE calib
def bpe_train(kalimat_list, n_merges=6):
    vocab = Counter()
    for s in kalimat_list:
        for w in s.split():
            vocab[" ".join(w)] += 1
    merges = []
    for _m in range(n_merges):
        pair_counts = Counter()
        for tok, cnt in vocab.items():
            parts = tok.split()
            for a, b in zip(parts[:-1], parts[1:]):
                pair_counts[(a, b)] += cnt
        if not pair_counts:
            break
        best, _cb = pair_counts.most_common(1)[0]
        merges.append(best)
        new_vocab = Counter()
        for tok, cnt in vocab.items():
            parts = tok.split()
            i = 0
            out = []
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                    out.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            new_vocab[" ".join(out)] += cnt
        vocab = new_vocab
    return merges, vocab


merges, _vb = bpe_train(KALIMAT, 6)
print("BPE merges:", merges)

print("\n=== SEMUA KALIBRASI LOLOS ===")
