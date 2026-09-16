"""Scratch: verify XOR training demo + blob demos converge deterministically."""
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

Xxor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
yxor = np.array([0., 1., 1., 0.])

def latih_xor(lr=1.0, epochs=3000, n_hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, (2, n_hidden)); b1 = np.zeros(n_hidden)
    W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1)); b2 = np.zeros(1)
    losses = []
    m = len(yxor)
    for ep in range(epochs):
        h = np.maximum(Xxor @ W1 + b1, 0)
        o = sigmoid(h @ W2 + b2)
        p = np.clip(o.ravel(), 1e-12, 1 - 1e-12)
        losses.append(float(-np.mean(yxor * np.log(p) + (1 - yxor) * np.log(1 - p))))
        dz2 = (o - yxor.reshape(-1, 1)) / m
        dW2 = h.T @ dz2; db2 = dz2.sum(0)
        dh = dz2 @ W2.T
        dz1 = dh * (h > 0)
        dW1 = Xxor.T @ dz1; db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    h = np.maximum(Xxor @ W1 + b1, 0)
    o = sigmoid(h @ W2 + b2).ravel()
    acc = float(((o >= 0.5) == yxor).mean())
    return acc, losses[-1], losses

for lr in (0.5, 1.0, 2.0):
    for seed in (0, 1, 2):
        acc, loss, _ = latih_xor(lr=lr, seed=seed)
        print(f"lr={lr} seed={seed}: acc={acc} final_loss={loss:.5f}")

# blob demo sanity (Bagian 4/5)
rng = np.random.default_rng(3)
n_per = 150
blob0 = rng.normal(-1, 0.6, (n_per, 2))
blob1 = rng.normal(+1, 0.6, (n_per, 2))
X = np.vstack([blob0, blob1])
y = np.array([0] * n_per + [1] * n_per)

def acc_of(model_w, model_b):
    return float(((sigmoid(X @ model_w + model_b).ravel() >= 0.5) == y).mean())

# full-batch logistic 100 epochs
w = np.zeros(2); b = 0.0
for _ in range(100):
    p = sigmoid(X @ w + b)
    w -= 0.5 * (X.T @ (p - y)) / len(y)
    b -= 0.5 * np.sum(p - y) / len(y)
print("blob logistic full-batch acc:", acc_of(w, b))
