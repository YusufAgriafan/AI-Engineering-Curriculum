"""Verify 01_lab_nlp_text.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari root repo):
    python AI-Engineering-Curriculum/06-NLP-Text/_verify_lab.py
"""
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from collections import Counter  # noqa: E402

# console Windows (cp1252) tak bisa mencetak emoji notebook -> paksa UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NB = Path(__file__).resolve().parent / "01_lab_nlp_text.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "tokenize_kata": (
        "def tokenize_kata(teks):\n"
        "    return teks.lower().split()\n"
    ),
    "bangun_vocab": (
        "def bangun_vocab(kalimat_list):\n"
        "    kata = sorted({t for s in kalimat_list for t in s.lower().split()})\n"
        "    w2i = {w: i for i, w in enumerate(kata)}\n"
        "    i2w = {i: w for w, i in w2i.items()}\n"
        "    return kata, w2i, i2w\n"
    ),
    "encode": (
        "def encode(kalimat, w2i):\n"
        "    return [w2i[t] for t in kalimat.lower().split()]\n"
    ),
    "pad_sequences": (
        "def pad_sequences(seqs, maxlen):\n"
        "    out = np.zeros((len(seqs), maxlen), dtype=int)\n"
        "    for i, s in enumerate(seqs):\n"
        "        potong = list(s)[:maxlen]\n"
        "        out[i, :len(potong)] = potong\n"
        "    return out\n"
    ),
    "encode_aman": (
        "def encode_aman(kalimat, w2i):\n"
        "    return [w2i.get(t, OOV_ID) for t in kalimat.lower().split()]\n"
    ),
    "pasangan_skipgram": (
        "def pasangan_skipgram(kalimat_list, w2i, window=2):\n"
        "    pairs = []\n"
        "    for s in kalimat_list:\n"
        "        ids = [w2i[w] for w in s]\n"
        "        for i, t in enumerate(ids):\n"
        "            for j in range(max(0, i - window), min(len(ids), i + window + 1)):\n"
        "                if j != i:\n"
        "                    pairs.append((t, ids[j]))\n"
        "    return pairs\n"
    ),
    "train_skipgram": (
        "def train_skipgram(pairs, vocab_size, freq_unigram,\n"
        "                   dim=16, epochs=5, lr=0.05, k_neg=3, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    W_in = rng.normal(0, 0.1, (vocab_size, dim))\n"
        "    W_out = rng.normal(0, 0.1, (vocab_size, dim))\n"
        "    noise = np.asarray(freq_unigram, dtype=float) ** 0.75\n"
        "    noise = noise / noise.sum()\n"
        "    loss_hist = []\n"
        "    for ep in range(epochs):\n"
        "        lr_ep = lr * (1.0 - 0.5 * ep / epochs)\n"
        "        tot, cnt = 0.0, 0\n"
        "        for (t, c) in pairs:\n"
        "            v, u = W_in[t], W_out[c]\n"
        "            s = sigmoid(float(v @ u))\n"
        "            grad_v = (s - 1.0) * u\n"
        "            grad_u = (s - 1.0) * v\n"
        "            L = -np.log(max(s, 1e-12))\n"
        "            for n_id in rng.choice(vocab_size, size=k_neg, p=noise):\n"
        "                if n_id == c:\n"
        "                    continue\n"
        "                un = W_out[n_id]\n"
        "                sn = sigmoid(float(v @ un))\n"
        "                grad_v += sn * un\n"
        "                W_out[n_id] -= lr_ep * (sn * v)\n"
        "                L += -np.log(max(1 - sn, 1e-12))\n"
        "            W_in[t] -= lr_ep * grad_v\n"
        "            W_out[c] -= lr_ep * grad_u\n"
        "            tot += L\n"
        "            cnt += 1\n"
        "        loss_hist.append(tot / cnt)\n"
        "    return W_in, loss_hist\n"
    ),
    "nn_purity": (
        "def nn_purity(E, vocab, kat_map, kategori_set):\n"
        "    benar = total = 0\n"
        "    for i, w in enumerate(vocab):\n"
        "        if kat_map[w] not in kategori_set:\n"
        "            continue\n"
        "        total += 1\n"
        "        best_j = max((cos_sim(E[i], E[j]), j) for j in range(len(vocab)) if j != i)[1]\n"
        "        benar += kat_map[vocab[best_j]] == kat_map[w]\n"
        "    return benar / total\n"
    ),
    "pca_2d": (
        "def pca_2d(X):\n"
        "    Xc = X - X.mean(0)\n"
        "    C = (Xc.T @ Xc) / len(X)\n"
        "    ev, evec = np.linalg.eigh(C)\n"
        "    return Xc @ evec[:, ::-1][:, :2]\n"
    ),
    "fitur_avg": (
        "def fitur_avg(kalimat_toks, E, w2i_):\n"
        "    return E[[w2i_[t] for t in kalimat_toks]].mean(0)\n"
    ),
    "fitur_onehot_mean": (
        "def fitur_onehot_mean(kalimat_toks, w2i_):\n"
        "    out = np.zeros(len(w2i_))\n"
        "    for t in kalimat_toks:\n"
        "        out[w2i_[t]] += 1.0\n"
        "    return out / len(kalimat_toks)\n"
    ),
    "label_sentimen": (
        "def label_sentimen(toks):\n"
        "    np_ = sum(1 for t in toks if t in POS)\n"
        "    ng = sum(1 for t in toks if t in NEG)\n"
        "    if np_ == 0 and ng == 0:\n"
        "        return 0\n"
        "    if np_ == ng:\n"
        "        return 0\n"
        "    return 1 if np_ > ng else -1\n"
    ),
    "bigram_matrix": (
        "def bigram_matrix(seq, vocab_size, alpha=0.5):\n"
        "    counts = np.ones((vocab_size, vocab_size)) * (1 + alpha)\n"
        "    for a, b in zip(seq[:-1], seq[1:]):\n"
        "        counts[a, b] += 1\n"
        "    return counts / counts.sum(axis=1, keepdims=True)\n"
    ),
    "ee_bigram": (
        "def ee_bigram(seq, P):\n"
        "    return -np.mean(np.log(P[seq[:-1], seq[1:]] + 1e-12))\n"
    ),
    "sample_bigram": (
        "def sample_bigram(P, seed_str, n_char, temp, rng):\n"
        "    out = seed_str\n"
        "    cur = c2i_lm[seed_str[-1]]\n"
        "    for _ in range(n_char):\n"
        "        p = P[cur] ** (1.0 / temp)\n"
        "        p = p / p.sum()\n"
        "        nxt = int(rng.choice(len(P), p=p))\n"
        "        out += i2c_lm[nxt]\n"
        "        cur = nxt\n"
        "    return out\n"
    ),
    "rnn_forward": (
        "def rnn_forward(X, params, ctx):\n"
        "    Wx, Wh, Wy, bh, by = params\n"
        "    n = len(X)\n"
        "    hidden = Wh.shape[0]\n"
        "    hs = [np.zeros((n, hidden))]\n"
        "    Hs = []\n"
        "    for t in range(ctx):\n"
        "        h_pre = hs[t] @ Wh.T + onehot_batch(X[:, t], C_lm) @ Wx + bh\n"
        "        hs.append(np.tanh(h_pre))\n"
        "        Hs.append(h_pre)\n"
        "    logits = hs[-1] @ Wy + by\n"
        "    e = np.exp(logits - logits.max(1, keepdims=True))\n"
        "    probs = e / e.sum(1, keepdims=True)\n"
        "    return probs, (hs, Hs)\n"
    ),
    "rnn_backward": (
        "def rnn_backward(probs, y, X, params, cache, ctx):\n"
        "    Wx, Wh, Wy, bh, by = params\n"
        "    hs, Hs = cache\n"
        "    n = len(X)\n"
        "    dlogits = probs.copy()\n"
        "    dlogits[np.arange(n), y] -= 1.0\n"
        "    dlogits /= n\n"
        "    dWy = hs[-1].T @ dlogits\n"
        "    dby = dlogits.sum(0)\n"
        "    dh = dlogits @ Wy.T\n"
        "    dWx = np.zeros_like(Wx)\n"
        "    dWh = np.zeros_like(Wh)\n"
        "    dbh = np.zeros_like(bh)\n"
        "    for t in reversed(range(ctx)):\n"
        "        da = dh * (1 - np.tanh(Hs[t]) ** 2)\n"
        "        dWx += onehot_batch(X[:, t], C_lm).T @ da\n"
        "        dWh += da.T @ hs[t]\n"
        "        dbh += da.sum(0)\n"
        "        dh = da @ Wh\n"
        "    return dWx, dWh, dWy, dbh, dby\n"
    ),
}

ctx = {"__name__": "__main__"}
ctx["np"] = np
ctx["plt"] = plt
ctx["Counter"] = Counter
ctx["PAD_ID"] = 0
ctx["OOV_ID"] = 1
errors = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src and "raise NotImplementedError" in src:
        injected = False
        for name, refsrc in REF.items():
            if f"def {name}(" in src or f"class {name}" in src:
                try:
                    exec(refsrc, ctx)
                    print(f"[cell {i}] injected ref: {name}")
                except Exception:
                    errors.append((i, f"inject {name}", traceback.format_exc()))
                injected = True
        if not injected:
            errors.append((i, "todo-cell-without-ref", src[:80]))
        continue
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ctx)
        print(f"[cell {i}] ok")
    except Exception:
        errors.append((i, "exec", traceback.format_exc()))

print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for i, kind, tb in errors:
        print(f"\n--- cell {i} ({kind}) ---\n{tb}")
    sys.exit(1)
print("LAB OK: semua cell (dengan implementasi referensi) lolos.")
