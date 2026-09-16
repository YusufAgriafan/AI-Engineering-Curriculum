"""Debug NN purity: print nearest-neighbor table for the toy corpus."""
import numpy as np

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
kat_of = dict(zip(kata_list, kategori_kata))
prob = np.array([freq[kat_of[w]] for w in kata_list], dtype=float)
prob = prob / prob.sum()
prob[-1] = 1.0 - prob[:-1].sum()


def dist_kat(a, b):
    if a == b:
        return 0
    content = ("hewan", "kendaraan", "makanan")
    if a in content and b in content:
        return 2
    return 3


def sample_pairs(n_pairs, rng, win=2):
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_sg(pairs, dim=32, epochs=5, lr=0.05, k_neg=3, seed=0):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0, 0.1, (V, dim))
    W_out = rng.normal(0, 0.1, (V, dim))
    noise = prob ** 0.75
    noise = noise / noise.sum()
    for ep in range(epochs):
        for (t, c) in pairs:
            v = W_in[t]
            u = W_out[c]
            s = sigmoid(float(v @ u))
            grad_v = (s - 1.0) * u
            grad_u = (s - 1.0) * v
            for n_id in rng.choice(V, size=k_neg, p=noise):
                if n_id == c:
                    continue
                un = W_out[n_id]
                sn = sigmoid(float(v @ un))
                grad_v += sn * un
                W_out[n_id] -= lr * (sn * v)
            W_in[t] -= lr * grad_v
            W_out[c] -= lr * grad_u
    return W_in, W_out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


RNG = np.random.default_rng(42)
pairs = sample_pairs(8000, RNG)
W_in, W_out = train_sg(pairs)

# ---- EKSPERIMEN: mean-centering (trik 'all-but-the-top') ----
Wc = W_in - W_in.mean(0)
print("=== SEBELUM mean-centering ===")
print(f"{'kata':<10} {'kategori':<11} NN(full-dim W_in)     NN(W_out)")
for i, w in enumerate(kata_list):
    if kat_of[w] not in ("hewan", "kendaraan", "makanan"):
        continue
    sims_in = sorted(((cos(W_in[i], W_in[j]), j) for j in range(V) if j != i), reverse=True)
    nn_in = kata_list[sims_in[0][1]]
    print(f"{w:<10} {kat_of[w]:<11} {nn_in:<12} {sims_in[0][0]:+.3f}")

purity_raw = 0
for i, w in enumerate(kata_list):
    if kat_of[w] not in ("hewan", "kendaraan", "makanan"):
        continue
    best_j = max((cos(W_in[i], W_in[j]), j) for j in range(V) if j != i)[1]
    purity_raw += kat_of[kata_list[best_j]] == kat_of[w]
print(f"purity raw     : {purity_raw / 12:.2f}")

purity_c = 0
nn_table = []
for i, w in enumerate(kata_list):
    if kat_of[w] not in ("hewan", "kendaraan", "makanan"):
        continue
    best_j = max((cos(Wc[i], Wc[j]), j) for j in range(V) if j != i)[1]
    purity_c += kat_of[kata_list[best_j]] == kat_of[w]
    nn_table.append((w, kat_of[w], kata_list[best_j], cos(Wc[i], Wc[best_j])))
print(f"\n=== SETELAH mean-centering ===")
print(f"purity centered: {purity_c / 12:.2f}")
for w, k, nn, s in nn_table:
    print(f"  {w:<10} {k:<11} -> {nn:<12} {s:+.3f}")

# PCA 2D dari EMBEDDING TER-CENTER
def pca2(X):
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / len(X)
    ev, evec = np.linalg.eigh(C)
    return Xc @ evec[:, ::-1][:, :2]

P2 = pca2(Wc)
cats3 = ("hewan", "kendaraan", "makanan")
centers = {c: P2[[i for i, w in enumerate(kata_list) if kat_of[w] == c]].mean(0) for c in cats3}
purity2d = 0
for i, w in enumerate(kata_list):
    if kat_of[w] not in cats3:
        continue
    ds = {c: np.linalg.norm(P2[i] - centers[c]) for c in cats3}
    purity2d += min(ds, key=ds.get) == kat_of[w]
print(f"2D purity (centered): {purity2d / 12:.2f}")

# cek norm tiap vektor
print("\nnorms W_in:", np.round(np.linalg.norm(W_in, axis=1), 3))
print("mean cos per kategori (W_in):")
for kat in set(kategori_kata):
    idxs = [i for i, k in enumerate(kategori_kata) if k == kat]
    m = np.mean([cos(W_in[a], W_in[b]) for a in idxs for b in idxs if a < b])
    print(f"  {kat:<11} intra={m:+.3f}")
