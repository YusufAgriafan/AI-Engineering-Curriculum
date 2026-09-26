# 📄 Cheatsheet — Fondasi LLM (Bab 9)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Tokenisasi & BPE

```
vocab = karakter + </w> + hasil merge (urutan merge = bagian dari MODEL)
melatih  : hitung pasangan bersebelahan -> merge terfrekuensi (seri: leksikografis)
           -> ulangi n_merge kali
encoding : apply merges BERURUTAN ke kata baru — fallback selalu ada (karakter)
```

- Kata sering cepat jadi 1 token; kata langka pecah per karakter — teks apa pun
  bisa di-encode tanpa `<UNK>`.
- **Biaya API dihitung per token, bukan per kata.** Bahasa Indonesia sering
  1.5–2x token dibanding Inggris untuk teks yang sama (vocabulary condong Inggris).
- Tokenizer BAGIAN dari model: tokenizer beda = token id beda = tidak kompatibel.
- Konvensi deterministik (tie-break leksikografis) = bisa diaudit & direproduksi.

```python
def best_pair(counter):                      # deterministik saat seri
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]

def merge_pasangan(spk, pasangan):           # gabung semua kemunculan bersebelahan
    out = []
    for s in spk:
        t, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i+1]) == pasangan:
                t.append(pasangan[0] + pasangan[1]); i += 2
            else:
                t.append(s[i]); i += 1
        out.append(tuple(t))
    return out
```

---

## 2. Embedding — Makna sebagai Geometri

```
embedding = tabel lookup V x D yang DIPETAHARI
cosine(a, b) = a·b / (|a| |b|)      1 = searah, 0 = ortogonal
```

- Token "satu keluarga" berkerumun (lab: cos dalam-kelompok 0.55 vs antar 0.24).
- Analogi vektor: `king - man + woman ≈ queen` — arah = fitur yang bisa di-query.
- Dimensi tinggi: vektor acak hampir ortogonal → model besar butuh BANYAK data
  agar struktur cluster tetap terbaca.
- Ini prasyarat RAG (Bab 12): `query·dokumen` tertinggi = kandidat paling relevan.

---

## 3. Masalah Urutan → Positional Encoding

```
Mean pooling = bag of words : "A menyerang B" == "B menyerang A"   (SALAH!)
Self-attention permutation-invariant → butuh info posisi dari luar
```

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))       nilai bounded [-1, 1]
```

- **Properti emas:** `PE[p] · PE[q]` = fungsi murni dari `(p − q)`
  (karena sin·sin + cos·cos = cos(p−q)) → relative position terbaca dari dot product.
- `PE[0] = [0, 1, 0, 1, ...]` (sin 0 = 0, cos 0 = 1).
- Tanpa PE: `H(flip(kalimat)) = flip(H(kalimat))` — simetri penuh; DENGAN PE
  simetri pecah → subjek vs objek terbedakan.

```python
def positional_encoding(max_len, d):
    PE = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d, 2)
    div = np.exp(-np.log(10000.0) * i / d)   # 10000^(-2i/d)
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE
```

---

## 4. Self-Attention — Jantung Transformer

```
Q = X @ Wq   "apa yang kucari?"      K = X @ Wk   "apa yang kutawarkan?"
V = X @ Wv   "informasi yang kubawa"

Scores  = Q @ K^T / sqrt(d_k)        kecocokan tiap pasangan token
Attn    = softmax per baris          tiap baris jumlah 1, semua > 0
Output  = Attn @ V                   campuran berbobot informasi tetangga
```

- **Kenapa `sqrt(d_k)`:** varians skor ~ d_k → softmax satu-hot → gradien mati.
  Scaling menjaga varians ~1.
- **Causal mask (decoder):** posisi i dilarang melihat j > i —
  `scores += triu(full(-1e9), k=1)` sebelum softmax. Tanpa ini, training
  next-token curang dan generasi autoregressif hancur.
- Encoder (BERT): tanpa mask, bidirectional. Decoder (GPT): causal.
  Encoder-decoder (T5): cross-attention menghubungkan keduanya.
- **Multi-head:** beberapa Q/K/V paralel → beberapa "perspektif" (sintaksis,
  semantik, posisi) → concat → proyeksi.

```python
def softmax_stabil(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def self_attention(Q, K, V, causal=False):
    n, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)
    if causal:
        scores = scores + np.triu(np.full((n, n), -1e9), k=1)
    attn = softmax_stabil(scores, axis=-1)
    return attn @ V, attn
```

---

## 5. Mini-LM — Next-Token Prediction & Training

```
ids → We[ids] + PE → blok (attention + dense ReLU) × L → Wo → logits
loss = cross-entropy(logits[:-1], ids[1:])    ← SHIFT: prediksi token BERIKUTNYA
```

- **Baseline teoretis:** model buta → CE = ln(vocab). Loss awal harus dekat angka itu.
- CE(uniform, target apa pun) = ln(V) — ingat untuk sanity check.
- Backprop dari Bab 4 dipakai apa adanya; lab pakai finite-difference agar fokus
  ke arsitektur.
- Loss turun ≠ jawaban bagus (Bab 15) — tapi loss TIDAK turun = pasti salah
  (cek shift target, normalisasi, learning rate).

```python
def cross_entropy(logits, targets):          # versi stabil (log-softmax)
    z = logits - np.max(logits, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    n = len(targets)
    return float(-np.sum(logp[np.arange(n), targets]) / n)
```

---

## 6. Sampling — Dari Logits ke Teks

```
greedy        : argmax — deterministik, repetitif
temperature T : softmax(logits / T)   T kecil = tajam (konsisten), besar = datar (liar)
top-k         : sampling hanya dari k token teratas (softmax LOKAL di kandidat)
top-p         : nucleus — kandidat terkecil yang kumulatif prob >= p (min. 1 token)
```

- Lab: T=0.5 → max prob 0.742; T=2.0 → 0.246. T→0 ≈ greedy; T→∞ ≈ uniform.
- Nucleus itu nonlinier: p=0.3 bisa {1 token}, p=0.9 bisa hampir seluruh vocab —
  p RENDAH yang memotong agresif; p tinggi hampir tak menyaring.
- Kombinasi umum: `temperature + top_p` (mis. T=0.7, p=0.9 untuk chat).
- Parameter ini yang kamu set di API LLM (Bab 10–11) — sekarang kamu tahu isi mesinnya.

```python
def sample_topk(logits, k, rng, T=1.0):
    idx = np.argsort(logits)[::-1][:k]
    p = softmax_stabil(logits[idx] / T)
    return int(rng.choice(idx, p=p))

def sample_topp(logits, p_min, rng, T=1.0):
    order = np.argsort(logits)[::-1]
    ps = softmax_stabil(logits[order] / T)
    kc = 1 + int(np.searchsorted(np.cumsum(ps), p_min, side='left'))
    kc = min(kc, len(order))
    sub = order[:kc]
    return int(rng.choice(sub, p=softmax_stabil(logits[sub] / T)))
```

---

## 7. Skala & Ekonomi (dari materi inti)

| Konsep | Ingat |
|---|---|
| Parameter | lebih banyak = lebih capable = lebih mahal |
| Context window | token yang "terlihat" sekali forward; makin besar makin mahal |
| Biaya API | per token; input lebih murah dari output |
| Bahasa Indonesia | ~1.5–2x token Inggris → RAG berbahasa Indonesia lebih mahal |
| Scaling laws | performa naik dengan skala, tapi diminishing returns |

---

## 8. Koneksi ke Bab Lain

- **Bab 4:** training loop & backprop — sama, hanya targetnya token berikutnya.
- **Bab 5:** konteks window = Conv1D; attention = convolution dengan bobot dinamis.
- **Bab 6:** next-token prediction = language modeling dari Bab 6, skala raksasa.
- **Bab 10:** sampling strategy = parameter API prompt.
- **Bab 12:** cosine embedding = vector search RAG.
- **Bab 13:** fine-tuning = training loop ini di atas model open-weight.
- **Bab 15:** loss bukan ukuran kualitas jawaban — butuh evaluasi khusus LLM.

---

## 9. Kesalahan Umum (90% Bug)

1. **Lupa `sqrt(d_k)`** — softmax satu-hot, gradien mati, training macet.
2. **Softmax tidak stabil** — `exp` overflow pada skor besar; kurangi max dulu.
3. **Mask diarahkan terbalik** — yang di-mask segitiga ATAS (`j > i`), k=1
   (diagonal tetap boleh dilihat).
4. **Lupa shift target** — prediksi token SEKARANG dari input sekarang = model
   cuma belajar copy. Target adalah `ids[1:]` untuk logits `[:-1]`.
5. **Softmax di seluruh vocab saat top-k** — normalisasi di dalam kandidat.
6. **Merge BPE tidak berurutan saat encode** — urutan merges bagian dari model.
7. **Menghitung token = kata** — biaya & context window dihitung per token.
8. **CE dihitung tanpa log-softmax stabil** — `log(softmax(x))` pada logits
   ekstrem menghasilkan NaN.
9. **Klaim "model paham urutan" tanpa tes pembalikan** — uji dengan
   "A menyerang B" vs "B menyerang A".
