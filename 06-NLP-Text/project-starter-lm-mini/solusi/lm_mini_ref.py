"""Implementasi referensi LM mini — untuk cek mandiri SETELAH selesai.

Self-contained: hanya butuh numpy. Dipakai oleh _verify_project.py untuk mengisi
semua TODO lalu menjalankan seluruh test suite. Jangan dibuka sebelum mencoba —
sama seperti kunci jawaban kuis.
"""

import numpy as np

__all__ = ["PAD_ID", "OOV_ID", "tokenize_kata", "bangun_vocab", "encode", "encode_aman",
           "pad_sequences", "sigmoid", "pasangan_skipgram", "frekuensi_unigram",
           "train_skipgram", "bigram_matrix", "ee_bigram", "prob_baris_dengan_temp",
           "sample_bigram", "onehot_batch", "softmax", "rnn_step_forward",
           "rnn_step_backward", "rnn_forward", "rnn_backward"]

PAD_ID = 0
OOV_ID = 1


# ------------------------------------------------------------------ text ----
def tokenize_kata(teks):
    return teks.lower().split()


def bangun_vocab(kalimat_list):
    tokens = []
    for k in kalimat_list:
        if isinstance(k, str):
            tokens.extend(tokenize_kata(k))
        else:
            tokens.extend(str(t).lower() for t in k)
    vocab = sorted(set(tokens))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w


def encode(kalimat, w2i):
    return [w2i.get(t, OOV_ID) for t in tokenize_kata(kalimat)]


def encode_aman(kalimat, w2i):
    return encode(kalimat, w2i)


def pad_sequences(seqs, maxlen):
    out = np.full((len(seqs), maxlen), PAD_ID, dtype=int)
    for i, s in enumerate(seqs):
        potong = s[:maxlen]
        out[i, :len(potong)] = potong
    return out


# ------------------------------------------------------------ embeddings ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def pasangan_skipgram(kalimat_list, w2i, window=2):
    pairs = []
    for toks in kalimat_list:
        ids = [w2i[t] for t in toks]
        for i, t in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((t, ids[j]))
    return pairs


def frekuensi_unigram(kalimat_list, w2i):
    cnt = np.zeros(len(w2i))
    for s in kalimat_list:
        for t in s:
            cnt[w2i[t]] += 1.0
    prob = cnt / cnt.sum()
    prob[-1] = 1.0 - prob[:-1].sum()   # paksa jumlah tepat 1.0 (presisi float)
    return prob


def train_skipgram(pairs, vocab_size, freq_unigram,
                   dim=16, epochs=5, lr=0.05, k_neg=3, seed=0):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0, 0.1, (vocab_size, dim))
    W_out = rng.normal(0, 0.1, (vocab_size, dim))
    noise = np.asarray(freq_unigram, dtype=float) ** 0.75
    noise = noise / noise.sum()
    loss_hist = []
    for ep in range(epochs):
        lr_ep = lr * (1.0 - 0.5 * ep / epochs)
        tot, cnt = 0.0, 0
        for (t, c) in pairs:
            v, u = W_in[t], W_out[c]
            s = sigmoid(float(v @ u))
            grad_v = (s - 1.0) * u
            grad_u = (s - 1.0) * v
            loss = -np.log(max(s, 1e-12))
            for n_id in rng.choice(vocab_size, size=k_neg, p=noise):
                if n_id == c:
                    continue
                un = W_out[n_id]
                sn = sigmoid(float(v @ un))
                grad_v += sn * un
                W_out[n_id] -= lr_ep * (sn * v)
                loss += -np.log(max(1 - sn, 1e-12))
            W_in[t] -= lr_ep * grad_v
            W_out[c] -= lr_ep * grad_u
            tot += loss
            cnt += 1
        loss_hist.append(tot / cnt)
    return W_in, loss_hist


# -------------------------------------------------------- languagemodel ----
def bigram_matrix(seq, vocab_size, alpha=0.5):
    counts = np.full((vocab_size, vocab_size), 1.0 + alpha)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def ee_bigram(seq, P):
    tot = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        tot += -np.log(P[a, b] + 1e-12)
    return tot / max(len(seq) - 1, 1)


def prob_baris_dengan_temp(baris, temp):
    baris = np.asarray(baris, dtype=float)
    if not np.any(baris > 0):
        return baris
    p = baris ** (1.0 / temp)
    return p / p.sum()


def sample_bigram(P, seed_id, n_token, temp, rng):
    out = []
    cur = int(seed_id)
    V = P.shape[0]
    for _ in range(n_token):
        baris = prob_baris_dengan_temp(P[cur], temp)
        nxt = int(rng.choice(V, p=baris))
        out.append(nxt)
        cur = nxt
    return out


# ------------------------------------------------------------------ rnn ----
def onehot_batch(ids, width):
    z = np.zeros((len(ids), width))
    z[np.arange(len(ids)), ids] = 1.0
    return z


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def rnn_step_forward(x_t, h_prev, Wx, Wh, bh):
    h_pre = x_t @ Wx + h_prev @ Wh.T + bh
    h = np.tanh(h_pre)
    return (h, (h_prev, h_pre))


def rnn_step_backward(dh, cache, x_t, Wx, Wh):
    h_prev, h_pre = cache
    da = dh * (1 - np.tanh(h_pre) ** 2)
    dWx = x_t.T @ da
    dWh = da.T @ h_prev      # h_pre = x@Wx + h_prev@Wh.T -> dWh = da.T @ h_prev
    dbh = da.sum(axis=0)
    dx_t = da @ Wx.T
    dh_prev = da @ Wh
    return (dx_t, dh_prev, dWx, dWh, dbh)


def rnn_forward(X, params, ctx):
    Wx, Wh, Wy, bh, by = params
    n = len(X)
    hidden = Wh.shape[0]
    vocab = Wy.shape[1]
    hs = [np.zeros((n, hidden))]
    Hs = []                      # pre-aktivasi per langkah (untuk backward tanh)
    for t in range(ctx):
        x_t = onehot_batch(X[:, t], vocab)
        h_pre = x_t @ Wx + hs[t] @ Wh.T + bh
        hs.append(np.tanh(h_pre))
        Hs.append(h_pre)
    logits = hs[-1] @ Wy + by
    return softmax(logits), (hs, Hs)


def rnn_backward(probs, y, X, params, cache, ctx):
    Wx, Wh, Wy, bh, by = params
    hs, Hs = cache
    n = len(X)
    vocab = Wy.shape[1]
    dlogits = (probs - onehot_batch(y, vocab)) / n
    dWy = hs[-1].T @ dlogits
    dby = dlogits.sum(axis=0)
    dh = dlogits @ Wy.T
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbh = np.zeros_like(bh)
    for t in reversed(range(ctx)):
        x_t = onehot_batch(X[:, t], vocab)
        _, dh_prev, dWx_s, dWh_s, dbh_s = rnn_step_backward(
            dh, (hs[t], Hs[t]), x_t, Wx, Wh)
        dWx += dWx_s
        dWh += dWh_s
        dbh += dbh_s
        dh = dh_prev
    return (dWx, dWh, dWy, dbh, dby)
