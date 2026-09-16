"""Vanishing gradient: Wh dinormalisasi spectral norm 0.9 -> keduanya vanish, sigmoid jauh lebih cepat."""
import numpy as np


def bptt_norm(act, T=30, seed=0, radius=0.9):
    r = np.random.default_rng(seed)
    Wh = np.abs(r.normal(0, 0.35, (20, 20)))
    Wh = Wh / np.linalg.norm(Wh, 2) * radius
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


nt = bptt_norm("tanh")
ns = bptt_norm("sigmoid")
print("tanh   :", " ".join(f"{v:.2e}" for v in nt[::5]))
print("sigmoid:", " ".join(f"{v:.2e}" for v in ns[::5]))
print(f"T=30: tanh {nt[-1]:.2e} vs sigmoid {ns[-1]:.2e}  rasio {nt[-1] / ns[-1]:.0f}x")
assert ns[-1] < nt[-1] / 10, "sigmoid harus vanish jauh lebih cepat"
assert nt[-1] < 1.0, "tanh juga vanish (hanya lebih lambat)"
print("OK")
