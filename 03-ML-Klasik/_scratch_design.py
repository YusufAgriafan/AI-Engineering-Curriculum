"""Scratch: freeze dataset + verify all reference numbers for Bab 3 project tests (numpy-only)."""
import json

import numpy as np

# ---------------------------------------------------------------
# Generator mirip make_classification (numpy murni, seed 42)
# ---------------------------------------------------------------
def buat_dataset(n_samples=1200, n_informative=6, n_redundant=2, weights=(0.9, 0.1),
                 flip_y=0.01, seed=42):
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_samples * weights[1]))
    n_neg = n_samples - n_pos
    y = np.array([0] * n_neg + [1] * n_pos)

    # Fitur 0 ("jam_pakai") sangat membedakan kelas; sisanya overlap moderat
    shift = np.full(n_informative, 0.4)
    shift[0] = 2.0
    X_inf = np.vstack([
        rng.normal(-shift / 2, 1.0, size=(n_neg, n_informative)),
        rng.normal(+shift / 2, 1.0, size=(n_pos, n_informative)),
    ])
    # fitur redundant = kombinasi linear fitur informatif + noise kecil
    W = rng.normal(0, 1, size=(n_informative, n_redundant))
    X_red = X_inf @ W * 0.5 + rng.normal(0, 0.3, size=(n_samples, n_redundant))
    X = np.hstack([X_inf, X_red])

    # label noise (flip_y)
    flip = rng.random(n_samples) < flip_y
    y = np.where(flip, 1 - y, y)

    # shuffle
    perm = rng.permutation(n_samples)
    return X[perm].astype(np.float64), y[perm]

def split_stratified(X, y, test_size, seed):
    rng = np.random.default_rng(seed)
    idx_test = []
    for k in np.unique(y):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        n_test = int(round(len(idx_k) * test_size))
        idx_test.extend(idx_k[:n_test])
    idx_test = np.array(sorted(idx_test))
    mask = np.ones(len(y), dtype=bool)
    mask[idx_test] = False
    return X[mask], X[idx_test], y[mask], y[idx_test]

X, y = buat_dataset()
print("X", X.shape, "positif:", int(y.sum()), "rasio:", y.mean().round(4))

X_train, X_temp, y_train, y_temp = split_stratified(X, y, 0.40, 42)
X_val, X_test, y_val, y_test = split_stratified(X_temp, y_temp, 0.50, 42)
print("train/val/test:", len(y_train), len(y_val), len(y_test),
      "| positif:", y_train.sum(), y_val.sum(), y_test.sum())

np.savez("AI-Engineering-Curriculum/03-ML-Klasik/project-starter-kampanye-marketing/ml_sederhana/data/data.npz",
         X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
print("saved npz")

# --- all numbers used in the notebook/tests (verify each) ---
out = {}

# scratch confusion matrix (fixed predictions)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
TP = int(((y_true == 1) & (y_pred == 1)).sum())
TN = int(((y_true == 0) & (y_pred == 0)).sum())
FP = int(((y_true == 0) & (y_pred == 1)).sum())
FN = int(((y_true == 1) & (y_pred == 0)).sum())
out["cm"] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
out["acc"] = (TP + TN) / 10
out["prec"] = TP / (TP + FP)
out["rec"] = TP / (TP + FN)
out["f1"] = 2 * out["prec"] * out["rec"] / (out["prec"] + out["rec"])
print("scratch cm:", out["cm"], "acc", out["acc"], "P", round(out["prec"], 4),
      "R", round(out["rec"], 4), "F1", round(out["f1"], 4))

# from-scratch logistic regression, full-batch GD (matches solusi impl)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bce(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def fit_scratch(X, yy, lr=0.1, n_iter=2000):
    m, n = X.shape
    w = np.zeros(n); b = 0.0
    for _ in range(n_iter):
        p = sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - yy)) / m
        b -= lr * float(np.sum(p - yy)) / m
    return w, b

w, b = fit_scratch(X_train, y_train)
pv = sigmoid(X_val @ w + b)
pv_bin = (pv >= 0.5).astype(int)
out["val_acc_t0.5"] = float((pv_bin == y_val).mean())
TPv = int(((y_val == 1) & (pv_bin == 1)).sum()); FPv = int(((y_val == 0) & (pv_bin == 1)).sum())
FNv = int(((y_val == 1) & (pv_bin == 0)).sum())
prec_v = TPv / (TPv + FPv) if TPv + FPv else 0.0
rec_v = TPv / (TPv + FNv) if TPv + FNv else 0.0
out["val_prec@0.5"] = prec_v
out["val_rec@0.5"] = rec_v
out["val_bce@0.5"] = bce(y_val, pv)

# ROC-AUC from scratch (rank statistic + tie handling)
def roc_auc(yy, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # rata-rata rank 1-based untuk tie
        i = j + 1
    n_pos, n_neg = int(yy.sum()), int((yy == 0).sum())
    return (ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

auc_v = roc_auc(y_val, pv)
out["val_auc"] = auc_v

th_best, f1_best = 0.5, 0.0
for th in np.linspace(0.05, 0.95, 181):
    pb = (pv >= th).astype(int)
    tp = int(((y_val == 1) & (pb == 1)).sum()); fp = int(((y_val == 0) & (pb == 1)).sum())
    fn = int(((y_val == 1) & (pb == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    if f1 > f1_best:
        th_best, f1_best = float(th), f1
out["val_best_th"] = th_best
out["val_best_f1"] = f1_best
pt = sigmoid(X_test @ w + b)
out["test_acc@bestth"] = float(((pt >= th_best).astype(int) == y_test).mean())
print("val acc@0.5", round(out["val_acc_t0.5"], 4), "| prec", round(prec_v, 4),
      "| rec", round(rec_v, 4), "| AUC", round(auc_v, 4))
print("best th on val:", th_best, "F1:", round(f1_best, 4),
      "| test acc @best_th:", round(out["test_acc@bestth"], 4))

# K-means reference (feature-wise standardized), best of 10 seeds
Xall = np.vstack([X_train, X_val, X_test])
Xmean, Xstd = Xall.mean(0), Xall.std(0)
Xstd_data = (Xall - Xmean) / Xstd
best_inertia = None
for seed in range(10):
    rng = np.random.default_rng(seed)
    cent = Xstd_data[rng.choice(len(Xstd_data), 2, replace=False)]
    for _ in range(100):
        d2 = ((Xstd_data[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = d2.argmin(1)
        newc = np.array([Xstd_data[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(2)])
        if np.allclose(newc, cent):
            break
        cent = newc
    inertia = float(((Xstd_data - cent[lab]) ** 2).sum())
    if best_inertia is None or inertia < best_inertia:
        best_inertia = inertia
        best_lab = lab
out["kmeans_inertia"] = best_inertia
sizes = np.bincount(best_lab, minlength=2)
out["kmeans_sizes"] = sizes.tolist()
out["kmeans_pos_rate"] = [float(y[best_lab == k].mean()) for k in range(2)]
print("kmeans inertia", round(best_inertia, 4), "sizes", sizes.tolist(),
      "pos-rate", [round(r, 4) for r in out["kmeans_pos_rate"]])

with open("AI-Engineering-Curriculum/03-ML-Klasik/_scratch_reference.json", "w") as f:
    json.dump(out, f, indent=1)
print("reference json saved")
