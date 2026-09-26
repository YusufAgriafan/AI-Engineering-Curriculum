"""Verify 01_lab_llm_fondasi.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import contextlib  # noqa: E402
import io  # noqa: E402
import json
import os
import sys
import time  # noqa: E402
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")


def contextlib_redirect_stdout(buf):
    return contextlib.redirect_stdout(buf)

# console Windows (cp1252) tak bisa mencetak emoji notebook -> paksa UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NB = Path(__file__).resolve().parent / "01_lab_llm_fondasi.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "hitung_pasangan": (
        "def hitung_pasangan(simbol_per_kata):\n"
        "    c = Counter()\n"
        "    for s in simbol_per_kata:\n"
        "        for a, b in zip(s, s[1:]):\n"
        "            c[(a, b)] += 1\n"
        "    return c\n"
    ),
    "merge_pasangan": (
        "def merge_pasangan(simbol_per_kata, pasangan):\n"
        "    out = []\n"
        "    for s in simbol_per_kata:\n"
        "        t, i = [], 0\n"
        "        while i < len(s):\n"
        "            if i < len(s) - 1 and (s[i], s[i + 1]) == pasangan:\n"
        "                t.append(pasangan[0] + pasangan[1]); i += 2\n"
        "            else:\n"
        "                t.append(s[i]); i += 1\n"
        "        out.append(tuple(t))\n"
        "    return out\n"
    ),
    "best_pair": (
        "def best_pair(counter):\n"
        "    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]\n"
    ),
    "latih_bpe": (
        "def latih_bpe(corpus, n_merge):\n"
        "    work = [ke_simbol(k) for k in corpus]\n"
        "    merges = []\n"
        "    for _ in range(n_merge):\n"
        "        c = hitung_pasangan(work)\n"
        "        if not c:\n"
        "            break\n"
        "        p = best_pair(c)\n"
        "        merges.append((p, c[p]))\n"
        "        work = merge_pasangan(work, p)\n"
        "    return merges\n"
    ),
    "encode_kata": (
        "def encode_kata(kata, merges):\n"
        "    s = ke_simbol(kata)\n"
        "    for p, _ in merges:\n"
        "        s = merge_pasangan([s], p)[0]\n"
        "    return s\n"
    ),
    "cos_sim": (
        "def cos_sim(a, b):\n"
        "    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)\n"
        "    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))\n"
    ),
    "analogi": (
        "def analogi(a, b, c, W):\n"
        "    target = W[b] - W[a] + W[c]\n"
        "    best, best_s = None, -2.0\n"
        "    for i in range(len(W)):\n"
        "        if i in (a, b, c):\n"
        "            continue\n"
        "        s = cos_sim(W[i], target)\n"
        "        if s > best_s:\n"
        "            best, best_s = i, s\n"
        "    return int(best)\n"
    ),
    "konteks_window": (
        "def konteks_window(X, k):\n"
        "    n, d = X.shape\n"
        "    out = np.zeros((n, (2 * k + 1) * d))\n"
        "    for i in range(n):\n"
        "        cols = []\n"
        "        for off in range(-k, k + 1):\n"
        "            j = min(max(i + off, 0), n - 1)\n"
        "            cols.append(X[j])\n"
        "        out[i] = np.concatenate(cols)\n"
        "    return out\n"
    ),
    "mean_pool": (
        "def mean_pool(X):\n"
        "    return np.asarray(X, dtype=float).mean(axis=0)\n"
    ),
    "vektor_kalimat": (
        "def vektor_kalimat(ids, W):\n"
        "    X = W[np.asarray(ids)]\n"
        "    return mean_pool(X)\n"
    ),
    "positional_encoding": (
        "def positional_encoding(max_len, d):\n"
        "    PE = np.zeros((max_len, d))\n"
        "    pos = np.arange(max_len)[:, None]\n"
        "    i = np.arange(0, d, 2)\n"
        "    div = np.exp(-np.log(10000.0) * i / d)\n"
        "    PE[:, 0::2] = np.sin(pos * div)\n"
        "    PE[:, 1::2] = np.cos(pos * div)\n"
        "    return PE\n"
    ),
    "rpe": (
        "def rpe(pos, d):\n"
        "    return positional_encoding(pos + 1, d)[pos]\n"
    ),
    "embedding_dengan_posisi": (
        "def embedding_dengan_posisi(ids, W, PE_tab):\n"
        "    X = W[np.asarray(ids)]\n"
        "    return X + PE_tab[:len(ids)]\n"
    ),
    "softmax_stabil": (
        "def softmax_stabil(x, axis=-1):\n"
        "    z = x - np.max(x, axis=axis, keepdims=True)\n"
        "    e = np.exp(z)\n"
        "    return e / np.sum(e, axis=axis, keepdims=True)\n"
    ),
    "self_attention": (
        "def self_attention(Q, K, V, causal=False):\n"
        "    n, d_k = Q.shape\n"
        "    scores = Q @ K.T / np.sqrt(d_k)\n"
        "    if causal:\n"
        "        scores = scores + np.triu(np.full((n, n), -1e9), k=1)\n"
        "    attn = softmax_stabil(scores, axis=-1)\n"
        "    return attn @ V, attn\n"
    ),
    "one_hot": (
        "def one_hot(ids, vocab):\n"
        "    out = np.zeros((len(ids), vocab))\n"
        "    out[np.arange(len(ids)), ids] = 1.0\n"
        "    return out\n"
    ),
    "cross_entropy": (
        "def cross_entropy(logits, targets):\n"
        "    logits = np.asarray(logits, dtype=float)\n"
        "    targets = np.asarray(targets, dtype=int)\n"
        "    z = logits - np.max(logits, axis=1, keepdims=True)\n"
        "    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))\n"
        "    n = len(targets)\n"
        "    return float(-np.sum(logp[np.arange(n), targets]) / n)\n"
    ),
    "softmax_rows": (
        "def softmax_rows(x):\n"
        "    return softmax_stabil(x, axis=-1)\n"
    ),
    "prediksi_berikut": (
        "def prediksi_berikut(params, ids):\n"
        "    logits = forward(params, ids)\n"
        "    return int(np.argmax(logits[-1]))\n"
    ),
    "sample_greedy": (
        "def sample_greedy(logits):\n"
        "    return int(np.argmax(logits))\n"
    ),
    "sample_topk": (
        "def sample_topk(logits, k, rng, T=1.0):\n"
        "    idx = np.argsort(logits)[::-1][:k]\n"
        "    p = softmax_stabil(np.asarray(logits)[idx] / T)\n"
        "    return int(rng.choice(idx, p=p))\n"
    ),
    "sample_topp": (
        "def sample_topp(logits, p_min, rng, T=1.0):\n"
        "    order = np.argsort(logits)[::-1]\n"
        "    ps = softmax_stabil(np.asarray(logits)[order] / T)\n"
        "    cum = np.cumsum(ps)\n"
        "    kc = 1 + int(np.searchsorted(cum, p_min, side='left'))\n"
        "    kc = min(kc, len(order))\n"
        "    sub = order[:kc]\n"
        "    pp = softmax_stabil(np.asarray(logits)[sub] / T)\n"
        "    return int(rng.choice(sub, p=pp))\n"
    ),
}

ctx = {"__name__": "__main__", "__builtins__": __builtins__}

t0 = time.time()
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src:
        names = [n for n in REF if f"def {n}(" in src]
        if not names:
            print(f"[cell {i:>2}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        for n in names:
            exec(REF[n], ctx)
        print(f"[cell {i:>2}] injected: {', '.join(names)}")
        continue
    buf_out = sys.stdout
    cap = io.StringIO()
    try:
        with contextlib_redirect_stdout(cap):
            exec(compile(src, f"<lab cell {i}>", "exec"), ctx)
    except Exception as e:
        print(f"FAILED di cell {i}: {type(e).__name__}: {e}")
        print(cap.getvalue()[-800:])
        sys.exit(1)
    out = cap.getvalue()
    if "✅" in out:
        last = [l for l in out.splitlines() if "✅" in l]
        print(f"[cell {i:>2}] OK  {last[0][:90] if last else ''}")
    else:
        print(f"[cell {i:>2}] OK ({len(out)} chars)")

print(f"\nSemua cell lab dieksekusi dalam {time.time() - t0:.1f} detik.")
print("LAB VERIFY OK")
