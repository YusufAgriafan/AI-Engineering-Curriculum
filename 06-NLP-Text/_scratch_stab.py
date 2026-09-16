"""Stabilitas konfigurasi terpilih + efek mean-centering."""
import numpy as np


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


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


def train(seed_pair, seed_init, cross_p=0.10, dim=16, epochs=5, lr=0.05, k_neg=3, n_pairs=8000):
    rng = np.random.default_rng(seed_pair)
    pairs = []
    while len(pairs) < n_pairs:
        t, c = rng.choice(V, p=prob, size=2)
        if t == c:
            continue
        d = dist_kat(kat_of[kata_list[t]], kat_of[kata_list[c]])
        if d == 0:
            pairs.append((int(t), int(c)))
        elif d == 2 and rng.random() < cross_p:
            pairs.append((int(t), int(c)))
    rng_t = np.random.default_rng(seed_init)
    W_in = rng_t.normal(0, 0.1, (V, dim))
    W_out = rng_t.normal(0, 0.1, (V, dim))
    noise = prob ** 0.75
    noise /= noise.sum()
    for ep in range(epochs):
        lr_ep = lr * (1.0 - 0.5 * ep / epochs)
        for (t, c) in pairs:
            v = W_in[t]
            u = W_out[c]
            s = sigmoid(float(v @ u))
            grad_v = (s - 1.0) * u
            grad_u = (s - 1.0) * v
            for n_id in rng_t.choice(V, size=k_neg, p=noise):
                if n_id == c:
                    continue
                un = W_out[n_id]
                sn = sigmoid(float(v @ un))
                grad_v += sn * un
                W_out[n_id] -= lr_ep * (sn * v)
            W_in[t] -= lr_ep * grad_v
            W_out[c] -= lr_ep * grad_u
    return W_in


def purity(W, centered):
    Wc = W - W.mean(0) if centered else W
    p = 0
    for i, w in enumerate(kata_list):
        if kat_of[w] not in CATS:
            continue
        bj = max((cos(Wc[i], Wc[j]), j) for j in range(V) if j != i)[1]
        p += kat_of[kata_list[bj]] == kat_of[w]
    return p / 12


for sp in (42, 43, 44):
    W = train(sp, 0)
    print(f"seed_pairs={sp}: purity raw={purity(W, False):.2f}  centered={purity(W, True):.2f}")
