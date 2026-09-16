# 📄 Cheatsheet — NLP & Text (Bab 6)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Pipeline Teks → Angka — Fondasi Semua NLP

```
teks → tokenize (lowercase, split) → vocabulary (kata → id) → sequence of ids → padding
```

- **Vocabulary harus dari data TRAIN saja** — membangun vocab dari test = data leakage.
- Konvensi id: `0 = PAD`, `1 = OOV`. Kata tak dikenal → `OOV_ID`, **jangan crash**.
- Tiga granularitas: per **kata** (vocab besar, sequence pendek), per **karakter**
  (vocab kecil, sequence panjang), **subword/BPE** (kompromi — dipakai semua LLM modern).

```python
def pad_sequences(seqs, maxlen):          # pad & truncate 'post' sekaligus
    out = np.full((len(seqs), maxlen), 0, dtype=int)
    for i, s in enumerate(seqs):
        potong = s[:maxlen]
        out[i, :len(potong)] = potong
    return out
```

- Banyaknya PAD mempengaruhi rata-rata pooling — di Keras dipadukan dengan **masking**.

---

## 2. Distributional Hypothesis — Dari Mana "Makna" Datang

> Firth (1957): *"You shall know a word by the company it keeps."*

- Korpus ko-okurensi: pasangan `(target, konteks)` dalam **jendela dua arah**.
- Makna tidak diberi tahu — ia **muncul** sebagai efek samping dari statistik ko-okurensi.
- Bukti di lab: `nn_purity ≥ 0.9` (tetangga terdekat cosine = kategori sama) + PCA 2D
  memperlihatkan 3 gugus topik + kutub sentimen, **tanpa label saat training**.

```python
def pasangan_skipgram(kalimat_list, w2i, window=1):
    pairs = []
    for toks in kalimat_list:
        ids = [w2i[t] for t in toks]
        for i, t in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((t, ids[j]))
    return pairs
```

---

## 3. Skip-gram + Negative Sampling — Rumus & Gradien Wajib Hafal

```
loss  = −log σ(v_t · u_c)  +  Σ_neg −log σ(−v_t · u_neg)
grad  = (σ(z) − y) · konteks        dengan y = 1 (asli) / 0 (noise)
```

- `grad_v = (s − 1)·u` dan `grad_u = (s − 1)·v` — **keduanya dihitung pada v, u asli**
  (sebelum update). 90% bug: memakai nilai yang sudah ter-update.
- Negative sampling: `k` kata acak didorong menjauh, distribusi noise = `unigram^0.75`
  (trik word2vec: meredam dominasi kata frekuensi tinggi).
- lr decay linear per epoch; gradien diverifikasi **gradient check** numerik.

```python
s = sigmoid(float(v @ u))       # clip z ke [-30, 30] biar stabil
grad_v = (s - 1.0) * u
grad_u = (s - 1.0) * v
v -= lr * grad_v ; u -= lr * grad_u
```

- **Avg-embedding + regresi logistik** mengalahkan one-hot mean & random embedding:
  one-hot hanya identitas kata (hafalan), embedding membawa **makna yang
  generalisasi**. Inilah alasan RAG (Bab 12) memakai embedding dokumen + cosine.

---

## 4. Char-LM I: Bigram Count-Based & Temperature

```python
counts = np.full((V, V), 1.0 + alpha)          # smoothing: TANPA probabilitas nol
for a, b in zip(seq[:-1], seq[1:]):
    counts[a, b] += 1.0
P = counts / counts.sum(axis=1, keepdims=True)  # normalisasi per baris
```

- Bigram hampir selalu menang telak dari unigram (EE turun > 0.2 nats/char di lab) —
  konteks 1 karakter saja sudah sangat informatif.

**Temperature** — kontrol randomness sampling:

```python
z = logits - logits.max()        # stabilisasi (konstanta hilang saat normalisasi)
p = np.exp(z / temp) ; p /= p.sum()
next_id = rng.choice(len(logits), p=p)
```

| T | Efek | Kapan |
|---|---|---|
| → 0 | deterministik (argmax) | parsing, ekstraksi |
| 1.0 | sesuai distribusi model | default |
| > 1 | meratakan → acak & beragam | brainstorming |

---

## 5. RNN Elman — Forward, BPTT, Sampling

```
forward : h_t   = tanh(x_t @ Wx + h_{t-1} @ Whᵀ + bh)
          logits = h_T @ Wy + by
backward: dlogits = (probs − onehot(y)) / n          (softmax + CE)
          da_t   = dh_t * (1 − tanh(H_t)²)           (turunan tanh)
          dWx += x_tᵀ @ da_t ; dWh += da_tᵀ ... ; dh_{t-1} = da_t @ Wh   ← gradien mengalir ke waktu
```

- **Akumulasi (`+=`) sepanjang waktu** — sama seperti backward konvolusi; lupa `+=` =
  bug #1.
- Gradient clipping `np.clip(g, -5, 5)` menahan **exploding** gradient.
- Hidden state = ringkasan seluruh konteks → RNN mengalahkan bigram (hanya 1 char)
  pada next-char prediction.
- Sampling generator: "hangatkan" state dengan seed, lalu loop prediksi → feed-back.

---

## 6. Vanishing Gradient — Kenapa Lahir LSTM & Transformer

```
gradien ∝ (W_hᵀ · diag(aktivasi'))^T        # produk berulang sepanjang waktu
```

- `|faktor| < 1` → meluruh **eksponensial**. Lab Bagian 9: T=30, tanh ~1e-4,
  sigmoid jauh lebih parah (turunan maks 0.25).
- Kedua RNN polos kehilangan konteks jauh → **LSTM/GRU** (gate menjaga aliran memori)
  → **Transformer** (Bab 9): self-attention memberi akses langsung ke semua token.
- Dua alasan Transformer menggantikan RNN: (1) **paralel** penuh sepanjang sequence,
  (2) **jarak konteks = O(1)**, bukan O(T).

---

## 7. Pola Kode: Uji Sebelum Klaim

| Yang dibangun | Uji yang menahan |
|---|---|
| gradien skip-gram / BPTT | gradient check numerik `(L(w+h)−L(w−h))/2h` |
| LM | EE/cross-entropy harus < baseline (unigram/bigram) |
| sampling temperature | entropy karakter T tinggi > T rendah |
| embedding bermakna | nn_purity + akurasi sentimen > random embedding |
| RNN | val acc > bigram pada split yang sama |

---

## 8. Koneksi ke Bab Lain

- **Bab 3**: klasifikasi sentimen = regresi logistik di atas fitur avg-embedding.
- **Bab 4**: RNN = Dense berulang dengan state; BPTT = backprop dengan `+=` sepanjang waktu.
- **Bab 9**: semua yang manual di sini (token → embedding → sequence model) adalah
  separuh awal dari Transformer; separuh lainnya adalah self-attention.
- **Bab 11**: `temperature`, `top_p` di LLM API = persis sampling yang kamu tulis di Soal 9.
- **Bab 12**: pencarian RAG = cosine similarity di ruang embedding — lihat Bagian 5 lab.
