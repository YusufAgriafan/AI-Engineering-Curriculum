# Bab 9 — Fondasi LLM: Transformer & Tokenisasi

> Bab pertama jalur AI Engineering. Semua yang Anda pelajari di Bab 4 & 6 (backprop, embedding, next-token prediction) menyatu di sini menjadi satu arsitektur: **Transformer**.

## 🗺️ Urutan Belajar Bab Ini

1. Baca materi inti di bawah (±2 jam) — fokus pada intuisi Q/K/V dan BPE.
2. Kerjakan **lab** — semuanya dibangun dari nol dengan numpy (±3 jam).
3. Uji diri dengan **kuis** (10 soal, 22 poin, skor otomatis).
4. Bandingkan jawaban dengan **kunci** — baca bagian *kenapa*.
5. Bangun **project starter** (TDD, 76 test) — tokenizer + mini-LM end-to-end.
6. Review cepat dengan **cheatsheet** sebelum lanjut ke Bab 10.

## 🎯 Tujuan Belajar
- Memahami self-attention secara intuisi dan matematis ringan.
- Tokenisasi subword (BPE) dan kenapa "strawberry" punya 3 huruf r.
- Perbedaan encoder-only / decoder-only / encoder-decoder.
- Konteks, parameter, dan trade-off (kualitas vs biaya vs latensi).

## 1. Materi Inti

> Bab pertama jalur AI Engineering. Semua yang Anda pelajari di Bab 4 & 6 (backprop, embedding, next-token prediction) menyatu di sini menjadi satu arsitektur: **Transformer**.

---

### 1.1 Arsitektur Transformer — Intuisi

#### Apa Itu Transformer?

```
Transformer adalah arsitektur neural network yang:
1. Mengerti SEQUENSI (seperti RNN/LSTM)
2. Tapi PARALELIZABLE (berbeda dengan RNN yang sekuensial)
3. Menggunakan SELF-ATTENTION: tiap token "memperhatikan" semua token lain

 → Ini memungkinkan training massive dan kualitas tinggi
```

```
┌─────────────────────────────────────────────────┐
│  SEQUENSI INPUT: "Saya suka machine learning"   │
│  Token: [Saya][suka][machine][learning]         │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  SELF-ATTENTION LAYER                           │
│  Tiap token lihat semua token lain:            │
│  - "Saya" → perhatikan "suka", "machine"     │
│  - "suka" → perhatikan "Saya", "machine"      │
│  - "machine" → perhatikan "learning"          │
│  - "learning" → perhatikan "machine"          │
│                                                 │
│  Hasil: representasi konteks untuk tiap token  │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  OUTPUT: distribusi probabilitas token berikut  │
│  misal: "machine" → 40%, "deep" → 25%, ...    │
└─────────────────────────────────────────────────┘
```

#### Self-Attention — Inti dari Transformer

> **Inti cerita:** Setiap token bertanya: "Token lain yang mana yang penting bagi saya?"

```
Untuk tiap token, hitung:
  Q (Query)   = "Apa yang saya cari?"
  K (Key)     = "Apa yang saya tawarkan?"
  V (Value)   = "Informasi apa yang aku miliki?"

  Score = Q · K^T / √d   (berapa mirip query dengan key?)
  Attention Weight = softmax(Score)
  Output = Σ(attention_weight · V)   (kombinasi informasi dari semua token)

→ Token yang relevan dapat weight tinggi, yang tidak relevan rendah
```

**Contoh visual:**

```
Kalimat: "Bank berdiri di tepi SUNGI."

Token "SUNGAI" harus memperhatikan:
  - "tepi" (posisi relatif) → weight tinggi
  - "Bank" (apakah berhubungan?) → weight sedang
  - "berdiri" → weight rendah
  - "di" → weight rendah

Token "Bank" harus memperhatikan:
  - "berdiri" (aksi) → weight tinggi
  - "SUNGAI" (lokasi) → weight tinggi
  - kata lain → weight rendah
```

#### Multi-Head Attention

```
Daripada 1 perspektif attention, pakai beberapa sekaligus:

Head 1: hubungan sintaksis (subyek - predikat)
Head 2: hubungan semantik (kata terkait makna)
Head 3: hubungan posisi (jarak antar kata)
Head 4: hubungan lainnya...

 → Gabungkan semua perspektif → representasi lebih kaya
```

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(Q, K, V, mask=None):
    """
    Self-attention manual.
    Q, K, V: (batch, seq_len, d_model)
    """
    d_k = Q.shape[-1]
    
    # Score = Q · K^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Mask (opsional: untuk causal language modeling)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Attention weights = softmax(scores)
    attn_weights = softmax(scores)
    
    # Output = attention_weights · V
    output = np.matmul(attn_weights, V)
    
    return output, attn_weights

# Contoh kecil
seq_len = 4
d_model = 8
batch = 1

Q = np.random.randn(batch, seq_len, d_model)
K = np.random.randn(batch, seq_len, d_model)
V = np.random.randn(batch, seq_len, d_model)

output, attn_weights = self_attention(Q, K, V)

print(f"Output shape: {output.shape}  ← (batch, seq_len, d_model)")
print(f"Attention shape: {attn_weights.shape}  ← (batch, seq_len, seq_len)")

# Visualisasi attention matrix untuk 1 sampel
print("\nAttention matrix (token ke-berapa yang diperhatikan):")
print("Baris: token yang mau fokus")
print("Kolom: token yang diperhatikan")
print(attn_weights[0].round(3))
```

> **Koneksi ke Bab 4:** Self-attention mirip dengan convolution tapi lebih fleksibel — bisa melihat "jauh" tanpa bertingkat. Dan seperti ResNet, transformer pakai skip connection + layer normalization di setiap block.

---

### 1.2 Positional Encoding — Urutan itu Penting

```
Problem: Self-attention itu PERMUTATION-INVARIANT
  → "Saya suka AI" = "AI suka saya" untuk attention purba

Solusi: Tambah informasi posisi ke embedding:
  Positional Encoding = f(posisi, dimensi)
  Sinusoidal: PE(pos, 2i) = sin(pos / 10000^(2i/d model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d model))

 → Token di posisi berbeda dapat encoding berbeda
 → Model bisa belajar dependensi posisi
```

---

### 1.3 Jenis Transformer

| Tipe | Arsitektur | Penggunaan |
|---|---|---|
| **Encoder-only** (BERT, RoBERTa) | Self-attention bidirectional (lihat semua token sekaligus) | Classification, understanding, NER, extractive QA |
| **Decoder-only** (GPT, LLaMA, Claude) | Causal attention (lihat hanya token sebelumnya) — next-token prediction | Text generation, chat, code generation, instruction following |
| **Encoder-Decoder** (T5, BART) | Encoder lihat semua input; decoder generate output dengan cross-attention ke encoder | Translation, summarization, text-to-text task |

> **Pesan kunci:** Hampir semua LLM modern (GPT-4, Claude, Gemini, LLaMA) adalah decoder-only. Encoder-only (BERT) dipakai untuk task memahami teks, bukan menghasilkan.

---

### 1.4 Tokenisasi — Lebih Dalam

```
Tokenisasi = mengubah teks ke ID numerik yang dimengerti model

Teks: "Machine learning is fascinating."

Word-based tokenizer:
  → ["Machine", "learning", "is", "fascinating", "."]
  → [4521, 1893, 23, 8976, 5]

Subword tokenizer (BPE/WordPiece — GPT/LLaMA pakai ini):
  → ["Machine", " learning", " is", " fascin", "ating", "."]
  → [4521, 8923, 23, 4511, 7823, 5]

→ Kata jarang dipecah jadi subword (mis. "fascinating" → "fascin" + "ating")
→ Vocabulary terbatas (mis. 50.000 subword) tapi bisa encode semua kata
```

#### Implikasi Praktis

| Aspek | Dampak |
|---|---|
| **Biaya** | Dihitung per token, bukan per kata. 1000 kata ≠ 1000 token |
| **Bahasa Indonesia** | Kadang lebih banyak token karena vocabulary lebih kecil untuk bahasa non-Inggris |
| **Offset mapping** | Perlu mapping dari token → posisi karakter asli (untuk highlight, RAG citation) |
| **Case sensitivity** | Tergantung tokenizer: "Apple" ≠ "apple" bisa jadi token berbeda |

```python
# Contoh penggunaan tokenizer (HuggingFace)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Machine learning is fascinating."

encoded = tokenizer(text)
print(f"Input IDs: {encoded['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
print(f"Number of tokens: {len(encoded['input_ids'])}")
print(f"Number of words: {len(text.split())}")
print(f"Ratio (token/word): {len(encoded['input_ids']) / len(text.split()):.2f}")

# Decode balik
decoded = tokenizer.decode(encoded['input_ids'])
print(f"\nDecoded: {decoded}")
```

---

### 1.5 Next-Token Prediction & Sampling

```
Tugas utama LLM: prediksi token berikutnya

Input:  "Machine learning is"
Model → output distribusi:
  "a"     : 0.45  ← paling mungkin
  "the"   : 0.30
  "one"   : 0.12
  "an"    : 0.08
  ...       : ...

Sampling:
  - Greedy: pilih yang probabilitas tertinggi (selalu "a")
  - Temperature: scaling distribusi → kontrol randomness
  - Top-k: ambil K token dengan prob tertinggi, sampling dari mereka
  - Top-p (nucleus): sampling dari token yang cumulative prob > p (mis. 0.9)
  - Greedy + Top-p kombinasi umum
```

---

### 1.6 Skala & Ekonomi LLM

| Konsep | Penjelasan | Dampak |
|---|---|---|
| **Parameter** | Jumlah bobot dalam model | Lebih banyak → lebih capable, tapi lebih mahal compute |
| **Context window** | Berapa token yang bisa "dilihat" model sekaligus | Lebih besar → lebih banyak konteks, tapi lebih mahal inference |
| **Training cost** | Komputasi untuk melatih model | BERT-base ~ $5k-$10k; GPT-3 ~ $4.6M; GPT-4 ~ ratusan juta+ (estimasi) |
| **Inference cost** | Biaya berjalan tiap request | API: per token; Self-host: GPU + listrik + maintenance |
| **Scaling laws** | Hubungan statistik: lebih banyak parameter + data → performa lebih baik | Kualitas bisa ditebak dari scale, tapi dengan diminishing returns |

#### Estimasi Biaya Sederhana

```python
# Estimator biaya API
def estimate_cost(input_tokens, output_tokens, model="gpt-4o-mini"):
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M token, USD
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }
    
    rates = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    total = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total, 6),
    }

# Contoh: sistem RAG
# 1 user query + retrieval 5 dokumen + generate answer
query_tokens = 50
retrieved_tokens = 5 * 500  # 5 dokumen, rata-rata 500 token
answer_tokens = 200

input_total = query_tokens + retrieved_tokens
result = estimate_cost(input_total, answer_tokens, "gpt-4o-mini")

print("Kost sistem RAG per request:")
for key, value in result.items():
    print(f"  {key}: {value}")

# Estimasi harian (1000 request/hari)
daily_requests = 1000
daily_cost = result["total_cost_usd"] * daily_requests
print(f"\nEstimasi biaya harian ({daily_requests} request): ${daily_cost:.2f}")
print(f"Estimasi biaya bulanan: ${daily_cost * 30:.2f}")
```

---

### 1.7 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 9:
  1. Transformer: arsitektur berbasis self-attention, paralelizable
  2. Self-attention: Q·K^T·V, tiap token melihat semua token lain
  3. Multi-head: beberapa perspektif attention sekaligus
  4. Positional encoding: menambahkan informasi urutan ke embedding
  5. Tipe: Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5)
  6. Tokenisasi: teks → subword IDs; biaya dihitung per token
  7. Next-token prediction: distribusi probabilitas → sampling
  8. Sampling: temperature, top-k, top-p
  9. Skala: parameter, context window, biaya, scaling laws
```

---

## 2. Latihan Praktis

> Latihan penuh (bertahap, dengan cek otomatis) ada di `01_lab_llm_fondasi.ipynb`.
> Ringkasannya:

### Latihan 1: BPE dari Nol

**Tujuan:** Pahami apa yang terjadi di balik self-attention dengan menghitung manual.

```python
import numpy as np

def softmax(x, axis=-1):
    """Softmax yang numerically stable."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def self_attention_manual(query, key, value, d_k=None, causal_mask=False):
    """
    Implementasi self-attention manual step-by-step.
    
    Parameters:
    -----------
    query: (seq_len, d_model)
    key: (seq_len, d_model)
    value: (seq_len, d_model)
    d_k: dimension of key (default: d_model)
    causal_mask: jika True, mask token masa depan (decoder style)
    
    Returns:
    --------
    output: (seq_len, d_model)
    attention_weights: (seq_len, seq_len)
    """
    seq_len = query.shape[0]
    
    if d_k is None:
        d_k = query.shape[1]
    
    # Step 1: Compute scores = Q · K^T / sqrt(d_k)
    # Q: (seq_len, d_k), K^T: (d_k, seq_len)
    # scores: (seq_len, seq_len)
    scores = np.dot(query, key.T) / np.sqrt(d_k)
    
    print(f"Step 1 - Scores shape: {scores.shape}")
    print("Scores (Q·K^T / √d_k):")
    print(scores.round(3))
    print()
    
    # Step 2: Apply causal mask (jika perlu)
    if causal_mask:
        # Mask yang membuat token hanya bisa melihat token sebelumnya
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores - 1e9 * mask  # -infinity untuk posisi yang di-mask
        
        print("Step 2 - Setelah causal mask (diagonal & bawah):")
        print(scores.round(3))
        print()
    
    # Step 3: Softmax → attention weights
    attention_weights = softmax(scores, axis=-1)
    
    print("Step 3 - Attention weights (softmax(scores)):")
    print(attention_weights.round(3))
    print()
    
    # Step 4: Weighted sum of values
    # output[i] = Σ_j attention_weights[i,j] * value[j]
    output = np.dot(attention_weights, value)
    
    print("Step 4 - Output:")
    print(output.round(3))
    print()
    
    return output, attention_weights

# Contoh: 4 token, masing-masing dimensi 8
np.random.seed(42)
seq_len = 4
d_model = 8

print("="*60)
print("SELF-ATTENTION MANUAL - CONTOH")
print("="*60)

query = np.random.randn(seq_len, d_model)
key = np.random.randn(seq_len, d_model)
value = np.random.randn(seq_len, d_model)

# Tanpa causal mask (encoder style - BERT)
print("\n--- KASUS 1: Encoder-style (tanpa causal mask) ---")
output_enc, attn_enc = self_attention_manual(query, key, value)

print("Interpretasi attention matrix:")
print("Baris i, kolom j = seberapa besar token i memperhatikan token j")
print()

# Dengan causal mask (decoder style - GPT)
print("\n--- KASUS 2: Decoder-style (dengan causal mask) ---")
print("Token hanya bisa melihat token SEBELUMNYA (termasuk dirinya sendiri)")
print()
output_dec, attn_dec = self_attention_manual(query, key, value, causal_mask=True)

# Perbedaan attention pattern
print("\n" + "="*60)
print("PERBANDINGAN ATTENTION PATTERN")
print("="*60)
print("\nEncoder (tanpa mask):")
print("Token 0 bisa lihat semua token: ", attn_enc[0].round(3))
print("Token 1 bisa lihat semua token: ", attn_enc[1].round(3))

print("\nDecoder (dengan causal mask):")
print("Token 0 hanya lihat token 0:       ", attn_dec[0].round(3))
print("Token 1 hanya lihat token 0-1:     ", attn_dec[1].round(3))
print("Token 2 hanya lihat token 0-2:     ", attn_dec[2].round(3))
print("Token 3 hanya lihat token 0-3:     ", attn_dec[3].round(3))
```

---

### Latihan 2: Tokenization Comparison

> 💡 Jalankan lab Bagian 1 dulu — setelah membangun BPE sendiri, tabel rasio
> token/kata di latihan ini terasa jauh lebih bermakna.

**Tujuan:** Pahami perbedaan tokenization dan dampaknya terhadap biaya.

```python
from transformers import AutoTokenizer
import numpy as np

# Load beberapa tokenizer berbeda
tokenizers = {
    "GPT-2 (English)": "gpt2",
    "LLaMA (multilingual)": "mock-eleuther-ai-gpt-neox-20b",  # akan fallback
}

texts = [
    "Machine learning adalah bidang yang menarik.",  # Inggris
    "Machine learning is a fascinating field.",       # Indonesia
    "Saya suka belajar deep learning setiap hari.",  # Campuran
    "Transformers have revolutionized NLP.",          # Inggris teknis
    "Namaku Andi dan saya suka coding.",             # Indonesia santai
]

print("="*70)
print("KOMPARASI TOKENIZASI ANTAR BAHASA & MODEL")
print("="*70)

for model_name, model_id in tokenizers.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"\n{model_name}:")
        print(f"  Gagal load: {e}")
        continue
    
    print(f"\n{model_name} ({model_id}):")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print()
    
    for text in texts:
        tokens = tokenizer.encode(text)
        words = text.split()
        ratio = len(tokens) / len(words)
        
        print(f"  Teks: '{text}'")
        print(f"    Words: {len(words)}, Tokens: {len(tokens)}, Ratio: {ratio:.2f}")
        print(f"    Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
        print()

# Perbandingan biaya
print("\n" + "="*70)
print("ESTIMASI BIAYA PER 1000 KALIMAT (asumsi rata-rata 10 kata/kalimat)")
print("="*70)

# Asumsi: 1000 kalimat, rata-rata 10 kata/kalimat = 10.000 kata
total_words = 10_000

# Estimasi token per kata untuk bahasa berbeda
estimations = {
    "Inggris (GPT-2)": 1.3,   # ~1.3 token per word
    "Indonesia (GPT-2)": 1.8,  # Lebih banyak karena vocabulary lebih kecil
    "Campuran": 1.5,
}

pricing_per_1m = {
    "gpt-4o-mini": 0.15,  # USD per 1M input tokens
    "gpt-4o": 5.00,
}

for lang, ratio in estimations.items():
    total_tokens = total_words * ratio
    cost_mini = (total_tokens / 1_000_000) * pricing_per_1m["gpt-4o-mini"]
    cost_4o = (total_tokens / 1_000_000) * pricing_per_1m["gpt-4o"]
    
    print(f"\n{lang} (ratio={ratio} token/word):")
    print(f"  Total tokens untuk 10.000 kata: {total_tokens:,.0f}")
    print(f"  Estimasi biaya (gpt-4o-mini): ${cost_mini:.4f}")
    print(f"  Estimasi biaya (gpt-4o): ${cost_4o:.4f}")
```

> **Tulis:** Mengapa bahasa Indonesia lebih "mahal" dalam hal token? Apa implikasinya untuk sistem RAG berbahasa Indonesia?

---

### Latihan 3: Mini Next-Token Prediction Model

> 💡 Versi numpy murni dari model ini ada di lab Bagian 6 (tanpa PyTorch, tanpa
> GPU) — kerjakan di sana kalau environment-mu belum terpasang PyTorch.

**Tujuan:** Implementasi forward pass sederhana GPT-style.

```python
import numpy as np
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    """Mini GPT untuk pemahaman konsep."""
    
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=d_model*4
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        x: (batch, seq_len) - token IDs
        Returns: (batch, seq_len, vocab_size) - logits untuk tiap posisi
        """
        batch, seq_len = x.shape
        
        # Embed tokens + posisi
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).to(x.device)
        
        x = self.token_embed(x) + self.pos_embed(positions)
        
        # Causal mask: jangan biarkan melihat future
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Transformer decoder
        x = self.transformer(x, memory=None, tgt_mask=causal_mask)
        
        # Output projection
        logits = self.output(x)
        
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Generate teks dari prompt."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(prompt_ids)  # (batch, seq_len, vocab_size)
            
            # Ambil logits untuk posisi terakhir
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
            
            # Top-k filtering
            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, top_k)
                logits = torch.where(logits >= top_k_logits[:, -1:], logits, torch.tensor(float('-inf')).to(logits.device))
            
            # Softmax → probabilitas
            probs = torch.softmax(logits, dim=-1)
            
            # Sample dari distribusi
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Append ke sequence
            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
        
        return prompt_ids

# Tes konsep
vocab_size = 1000
model = MiniGPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

# Dummy input: 2 batch, sequence length 10
x = torch.randint(0, vocab_size, (2, 10))

print("="*60)
print("MINI GPT FORWARD PASS")
print("="*60)

logits = model(x)
print(f"Input shape: {x.shape}  ← (batch, seq_len)")
print(f"Output shape: {logits.shape}  ← (batch, seq_len, vocab_size)")

# Loss calculation (next token prediction)
# Target: token di posisi t+1 adalah target untuk input di posisi t
logits_shifted = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
targets = x[:, 1:]  # (batch, seq_len-1)

loss_fn = nn.CrossEntropyLoss()
# Flatten untuk cross entropy
loss = loss_fn(logits_shifted.reshape(-1, vocab_size), targets.reshape(-1))

print(f"\nLoss (next-token prediction): {loss.item():.4f}")
print(f"\nPenjelasan:")
print(f"  - Untuk tiap posisi, model memprediksi token berikutnya")
print(f"  - Loss dihitung antara prediksi dan token asli berikutnya")
print(f"  - Semakin rendah loss → semakin baik model memprediksi")

# Contoh generation (dengan random weights - hasil akan random!)
print("\n" + "="*60)
print("CONTOH GENERATION (dengan random weights - hasil random!)")
print("="*60)

# Prompt awal (random IDs)
prompt = torch.randint(0, vocab_size, (1, 5))
print(f"Prompt IDs: {prompt.numpy()}")

generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
print(f"Generated IDs: {generated.numpy()}")
print(f"Generated length: {generated.shape[1]} token")
```

---

## 📚 Referensi Online (Prioritas Urut)

| Sumber | Tipe | Kenapa Baca |
|---|---|---|
| [Neural Networks: Zero to Hero — Karpathy](https://karpathy.ai/zero-to-hero.html) | Video (wajib!) | "Let's build GPT" — pemahaman paling intuitif dari nol sampai GPT |
| [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Visual | Visualisasi terbaik untuk memahami arsitektur |
| [Attention Is All You Need (paper asli)](https://arxiv.org/abs/1706.03762) | Paper | Baca sekilas untuk paham original architecture |
| [Hugging Face NLP Course — Transformer](https://huggingface.co/learn/nlp-course) | Kursus interaktif | Jembatan dari NLP klasik ke Transformer/LLM |
| [Karpathy — Intro to LLMs (1 jam)](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Video singkat | Ringkasan bagus untuk pemula LLM |
| [Blog post: transformer circuits](https://transformer-circuits.pub/) | Analisis mendalam | Untuk yang ingin paham "apa yang terjadi di dalam" |

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_llm_fondasi.ipynb` | Lab praktikum: 8 bagian — BPE dari nol (merge + encode kata baru), embedding yang belajar struktur (cosine), konteks window & mean pooling, positional encoding (properti `PE_p·PE_q = f(p−q)`), self-attention + causal mask, mini-LM (loss 1.386 → <0.5), sampling greedy/top-k/top-p | Kerjakan setelah baca materi inti |
| `02_kuis_llm_fondasi.ipynb` | 10 soal (6 PG + 4 coding), 22 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_llm_fondasi.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-llm-fondasi.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-tokenizer-mini/` | Proyek end-to-end: BPE trainer + tokenizer (vocab, round-trip) + embedding + PE + attention + mini-LM + sampling (TDD, 76 test) + starter/solusi + RUBRIK | Setelah kuis ≥ 17/22 |

> File `_gen_*.py`, `_verify_*.py` = skrip generator/verifier internal (angka di
> notebook terkunci seed & terverifikasi otomatis). Tidak perlu dibuka untuk
> belajar, tapi silakan baca kalau ingin melihat cara materinya dikalibrasi.

---

## ✅ Checklist Kompetensi

- [ ] Menjelaskan self-attention dengan contoh kalimat konkret
- [ ] Membedakan BERT vs GPT dan use case-nya
- [ ] Menghitung estimasi biaya permintaan dari jumlah token
- [ ] Melatih BPE mini dan meng-encode kata yang tidak ada di corpus
- [ ] Menjelaskan kenapa `sqrt(d_k)` wajib (varians skor → softmax satu-hot)
- [ ] Bisa jelaskan mengapa causal mask penting untuk decoder
- [ ] Menunjukkan properti `PE_p · PE_q = f(p−q)` dari positional encoding buatanmu
- [ ] Melatih mini-LM: loss turun dari ln(vocab) dan memprediksi token berikutnya
- [ ] Menjelaskan efek temperature & memilih top-k/top-p untuk kasus tertentu
- [ ] Implementasi self-attention manual dengan NumPy (tanpa melihat catatan)
