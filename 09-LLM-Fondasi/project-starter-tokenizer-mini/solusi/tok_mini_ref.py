"""tok_mini_ref — implementasi referensi project Bab 9.

Untuk cek mandiri SETELAH selesai. Verifier juga memakai modul ini untuk
mem-patch tok_mini sebelum menjalankan test suite & notebook.
"""
import math
from collections import Counter

import numpy as np

# ============================ BAGIAN 1: BPE ============================

UNK_PENANDA = "</w>"


def ke_simbol(kata):
    """'low' -> ('l', 'o', 'w', '</w>')."""
    return tuple(kata) + (UNK_PENANDA,)


def normalisasi(teks):
    """Lowercase + pisah spasi. Kontrak: output = teks.lower().split()."""
    return teks.lower().split()


def hitung_pasangan(simbol_per_kata):
    c = Counter()
    for s in simbol_per_kata:
        for a, b in zip(s, s[1:]):
            c[(a, b)] += 1
    return c


def best_pair(counter):
    """Terfrekuensi tertinggi; seri -> leksikografis terkecil (deterministik)."""
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def merge_pasangan(simbol_per_kata, pasangan):
    out = []
    for s in simbol_per_kata:
        t, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i + 1]) == pasangan:
                t.append(pasangan[0] + pasangan[1])
                i += 2
            else:
                t.append(s[i])
                i += 1
        out.append(tuple(t))
    return out


def latih_bpe(corpus, n_merge):
    """Return (list merges berurutan [(pair, freq), ...], list state simbol)."""
    work = [ke_simbol(k) for k in corpus]
    merges = []
    states = [list(work)]
    for _ in range(n_merge):
        c = hitung_pasangan(work)
        if not c:
            break
        p = best_pair(c)
        merges.append((p, c[p]))
        work = merge_pasangan(work, p)
        states.append(list(work))
    return merges, states


def apply_merges(simbol, merges):
    s = tuple(simbol)
    for p, _ in merges:
        s = merge_pasangan([s], p)[0]
    return s


def buat_vocab(corpus, n_merge):
    """Vocab = semua simbol awal + hasil merge, terurut leksikografis, index 0..V-1."""
    merges, _ = latih_bpe(corpus, n_merge)
    sym = set()
    for kata in corpus:
        sym.update(ke_simbol(kata))
    for p, _ in merges:
        sym.add(p[0] + p[1])
    return {s: i for i, s in enumerate(sorted(sym))}


def encode(kata, merges, vocab):
    """Kata -> tuple token. Token yang tidak ada di vocab di-skip (fallback karakter
    dijamin ada karena vocab memuat semua karakter awal)."""
    s = apply_merges(ke_simbol(kata), merges)
    return tuple(tok for tok in s if tok in vocab)


def decode(token_tuple):
    """Tuple token -> string, tanpa penanda </w>."""
    return "".join(token_tuple).replace(UNK_PENANDA, "")


def encode_teks(teks, merges, vocab):
    """Teks multi-kata -> list of list token (tiap kata satu list)."""
    return [encode(k, merges, vocab) for k in normalisasi(teks)]


def decode_teks(tokens_per_kata):
    """List of list token -> teks ber-spasi, round-trip dari encode_teks."""
    return " ".join(decode(t) for t in tokens_per_kata)


# ============================ BAGIAN 2: EMBEDDING ============================

def cos_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def cari_mirip(W, vektor, exclude=(), top_n=3):
    """Index token paling mirip (cosine tertinggi), skip yang di-exclude."""
    skor = [(cos_sim(W[i], vektor), i) for i in range(len(W)) if i not in exclude]
    skor.sort(reverse=True)
    return [i for _, i in skor[:top_n]]


def analogi(a, b, c, W):
    """a : b :: c : ? — token dengan cos tertinggi ke W[b] - W[a] + W[c]."""
    target = W[b] - W[a] + W[c]
    hasil = cari_mirip(W, target, exclude=(a, b, c), top_n=1)
    return int(hasil[0])


def latih_embedding(idx_vokal=(0, 4, 6), vocab=8, d=8, epochs=200, seed=9):
    """Fake task: token 'vokal' digeser +signal tiap epoch, semua didecay 0.99."""
    signal = np.zeros(d)
    signal[:2] = 1.5
    r = np.random.default_rng(seed)
    W = r.normal(0, 1, (vocab, d))
    for _ in range(epochs):
        W[list(idx_vokal)] += signal
        W *= 0.99
    return W


# ============================ BAGIAN 3: POSITIONAL ENCODING ============================

def positional_encoding(max_len, d):
    PE = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d, 2)
    div = np.exp(-math.log(10000.0) * i / d)
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE


def rpe(pos, d):
    """PE untuk satu posisi (baris tabel)."""
    return positional_encoding(pos + 1, d)[pos]


# ============================ BAGIAN 4: ATTENTION ============================

def softmax_stabil(x, axis=-1):
    x = np.asarray(x, dtype=float)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def self_attention(Q, K, V, causal=False):
    """Return (output, attn_weights); Q, K, V: (n, d_k)."""
    n, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)
    if causal:
        scores = scores + np.triu(np.full((n, n), -1e9), k=1)
    attn = softmax_stabil(scores, axis=-1)
    return attn @ V, attn


# ============================ BAGIAN 5: MINI LANGUAGE MODEL ============================

def one_hot(ids, vocab):
    out = np.zeros((len(ids), vocab))
    out[np.arange(len(ids)), ids] = 1.0
    return out


def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    z = logits - np.max(logits, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    n = len(targets)
    return float(-np.sum(logp[np.arange(n), targets]) / n)


def init_params(seed, vocab, d):
    r = np.random.default_rng(seed)
    return {
        "We": r.normal(0, 0.5, (vocab, d)),
        "PE": positional_encoding(32, d) * 0.1,
        "W1": r.normal(0, 0.5, (d, d)), "b1": np.zeros(d),
        "W2": r.normal(0, 0.5, (d, d)), "b2": np.zeros(d),
        "Wo": r.normal(0, 0.1, (d, vocab)), "bo": np.zeros(vocab),
    }


def forward(params, ids):
    n = len(ids)
    X = params["We"][np.asarray(ids)] + params["PE"][:n]
    A1 = np.maximum(X @ params["W1"] + params["b1"], 0.0)
    A2 = np.maximum(A1 @ params["W2"] + params["b2"], 0.0)
    return A2 @ params["Wo"] + params["bo"]


def loss_fn(params, ids):
    logits = forward(params, ids)
    return cross_entropy(logits[:-1], np.asarray(ids[1:]))


def numeric_grads(f, params, seq, eps=1e-5):
    grads = {}
    for k, arr in params.items():
        g = np.zeros_like(arr)
        flat, gflat = arr.ravel(), g.ravel()
        for i in range(len(flat)):
            orig = flat[i]
            flat[i] = orig + eps
            fp = f(params, seq)
            flat[i] = orig - eps
            fm = f(params, seq)
            flat[i] = orig
            gflat[i] = (fp - fm) / (2 * eps)
        grads[k] = g
    return grads


def train_lm(params, seq, epochs, lr):
    losses = []
    trainables = [k for k in params if k != "PE"]
    for _ in range(epochs):
        grads = numeric_grads(loss_fn, params, seq)
        for k in trainables:
            params[k] = params[k] - lr * grads[k]
        losses.append(loss_fn(params, seq))
    return params, losses


def prediksi_berikut(params, ids):
    return int(np.argmax(forward(params, ids)[-1]))


# ============================ BAGIAN 6: SAMPLING ============================

def sample_greedy(logits):
    return int(np.argmax(logits))


def sample_topk(logits, k, rng, T=1.0):
    logits = np.asarray(logits, dtype=float)
    idx = np.argsort(logits)[::-1][:k]
    p = softmax_stabil(logits[idx] / T)
    return int(rng.choice(idx, p=p))


def sample_topp(logits, p_min, rng, T=1.0):
    logits = np.asarray(logits, dtype=float)
    order = np.argsort(logits)[::-1]
    ps = softmax_stabil(logits[order] / T)
    kc = 1 + int(np.searchsorted(np.cumsum(ps), p_min, side="left"))
    kc = min(kc, len(order))
    sub = order[:kc]
    pp = softmax_stabil(logits[sub] / T)
    return int(rng.choice(sub, p=pp))
