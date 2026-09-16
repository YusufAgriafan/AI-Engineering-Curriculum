"""Grid: tunjuk konfigurasi yang bikin NN purity & 2D purity tinggi."""
import numpy as np


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run(func_freq, cross_p, dim, epochs, lr, k_neg, seed=0, n_pairs=8000):
    KATA = {
        "hewan":     ["kucing", "anjing", "burung", "kelinci"],
        "kendaraan": ["mobil", "motor", "sepeda", "kereta"],
        "makanan":   ["nasi", "roti", "bakso", "soto"],
        "sifat_pos": ["bagus", "enak", "senang", "mantap"],
        "sifat_neg": ["jelek", "payah", "kecewa", "buruk"],
        "fungsional": ["saya", "dan", "itu", "di", "tapi"],
    }
    freq = {"hewan": 20, "kendaraan": 20, "makanan": 20, "sifat_pos": 16,
            "sifat_neg": 16, "fungsional": func_freq}
    kata_list, kat_list = [], []
    for kat, daftar in KATA.items():
        kata_list.extend(daftar)
        kat_list.extend([kat] * len(daftar))
    V = len(kata_list)
    kat_of = dict(zip(kata_list, kat_list))
    prob = np.array([freq[kat_of[w]] for w in kata_list], dtype=float)
    prob /= prob.sum()
    prob[-1] = 1.0 - prob[:-1].sum()

    def dist_kat(a, b):
        if a == b:
            return 0
        content = ("hewan", "kendaraan", "makanan")
        if a in content and b in content:
            return 2
        return 3

    rng = np.random.default_rng(42)
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

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    rng_t = np.random.default_rng(seed)
    W_in = rng_t.normal(0, 0.1, (V, dim))
    W_out = rng_t.normal(0, 0.1, (V, dim))
    noise = prob ** 0.75
    noise /= noise.sum()
    for ep in range(epochs):
        lr_ep = lr * (1.0 - 0.5 * ep / epochs)  # decay linear
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

    Wc = W_in - W_in.mean(0)
    # NN purity full-dim (centered)
    cats = ("hewan", "kendaraan", "makanan")
    p_full = 0
    for i, w in enumerate(kata_list):
        if kat_of[w] not in cats:
            continue
        bj = max((cos(Wc[i], Wc[j]), j) for j in range(V) if j != i)[1]
        p_full += kat_of[kata_list[bj]] == kat_of[w]
    p_full /= 12
    # 2D purity
    mu = Wc.mean(0)
    Xc = Wc - mu
    Cm = (Xc.T @ Xc) / V
    ev, evec = np.linalg.eigh(Cm)
    P2 = Xc @ evec[:, ::-1][:, :2]
    centers = {c: P2[[i for i, w in enumerate(kata_list) if kat_of[w] == c]].mean(0) for c in cats}
    p2d = 0
    for i, w in enumerate(kata_list):
        if kat_of[w] not in cats:
            continue
        ds = {c: np.linalg.norm(P2[i] - centers[c]) for c in cats}
        p2d += min(ds, key=ds.get) == kat_of[w]
    p2d /= 12
    return p_full, p2d


print(f"{'func':>5} {'cross':>6} {'dim':>4} {'ep':>3} {'lr':>5} {'kneg':>4} | {'p_full':>7} {'p_2d':>5}")
for func_freq in (72, 36):
    for cross_p in (0.25, 0.10):
        for dim in (32, 16):
            for epochs in (5, 12):
                pf, p2 = run(func_freq, cross_p, dim, epochs, 0.05, 3)
                print(f"{func_freq:>5} {cross_p:>6} {dim:>4} {epochs:>3} {0.05:>5} {3:>4} | {pf:>7.2f} {p2:>5.2f}")
