# Bab 6 — NLP & Text

> Dari teks mentah ke model. Bab ini adalah jembatan langsung ke LLM: tokenisasi, embedding, dan sequence model semuanya bermuara ke Transformer.

## 🎯 Tujuan Belajar

- Tokenisasi teks & sequence padding.
- Embedding: kata → vektor bermakna.
- RNN, LSTM, GRU, Conv1D untuk teks.
- Text generation klasik (fondasi memahami LLM).

## 1. Materi Inti

> Bab ini adalah jembatan langsung ke LLM: tokenisasi, embedding, dan sequence model semuanya bermuara ke Transformer (Bab 9).

---

### 1.1 Tokenisasi & Vocabulary — Fondasi Semua NLP

#### Apa itu Tokenisasi?

```
Teks: "Saya suka AI"

Tokenisasi                 Hasil
─────────────────────────────────────────
Per kata:                 ["Saya", "suka", "AI"]
Per karakter:             ["S", "a", "y", "a", " ", "s", "u", ...]
Subword (BPE/LLM):        [" Saya", " su", "ka", " AI"]  ← LLM pakai ini
```

**Kenapa penting:** Model tidak mengerti "kata" secara langsung — mereka butuh angka. Tokenisasi adalah terjemahan dari teks → angka.

#### Tokenizer dengan TensorFlow/Keras

```python
from tensorflow.keras.preprocessing import text, sequence
import numpy as np

# Contoh teks
texts = [
    "Saya sangat suka machine learning",
    "Deep learning itu menarik",
    "Machine learning dan deep learning berbeda",
    "Saya ingin belajar NLP",
]

# 1. Belajar vocabulary dari teks
tokenizer = text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Contoh word_index: {list(tokenizer.word_index.items())[:5]}")

# 2. Konversi teks ke sequence of integers
sequences = tokenizer.texts_to_sequences(texts)
print(f"\nSequences:")
for text, seq in zip(texts, sequences):
    print(f"  '{text}' → {seq}")

# 3. Padding: samakan panjang sequence
padded = sequence.pad_sequences(sequences, maxlen=10, padding='post')
print(f"\nPadded shape: {padded.shape}")
print(f"Contoh: {padded[0]}")
```

#### Out-of-Vocabulary (OOV)

```python
# Tes dengan kata yang tidak ada di vocabulary
test_text = "Machine learning itu keren sekali"
test_seq = tokenizer.texts_to_sequences([test_text])
print(f"Test: '{test_text}' → {test_seq}")

# Kata "keren" dan "sekali" tidak ada → diwakili oleh oov_token
```

> **Koneksi ke LLM:** LLM modern (GPT, LLaMA) pakai subword tokenization (BPE, WordPiece) — bukan word-based seperti di atas. Tapi prinsipnya sama: teks → tokens → IDs.

---

### 1.2 Embedding — Kata ke Vektor Bermakna

#### Konsep

```
Teks                          Vektor (embedding)
─────────────────────────────────────────────────────────
"Kucing"  →  [0.12, -0.34, 0.89, 0.01, ...]  ←  representasi numerik
"Anjing"  →  [0.15, -0.29, 0.85, -0.02, ...]  ←  mirip dengan "kucing"
"Mobil"   →  [-0.72, 0.45, 0.11, 0.33, ...]  ←  berbeda domain
```

**Pola kunci:** Kata-kata yang sering muncul di konteks serupa → vektornya berdekatan di ruang embedding.

#### Embedding Layer di Keras

```python
from tensorflow.keras import layers
import tensorflow as tf

# Misal: 10.000 kata dalam vocabulary, embedding dimensi 128
vocab_size = 10000
embedding_dim = 128
max_length = 100

model = tf.keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# Embedding layer bisa dilatih bersama model:
# - Awal: random weights
# - Saat training: embedding di-update agar makna lebih baik

# Akses embedding matrix setelah training
# embeddings = model.layers[0].get_weights()[0]  # shape: (vocab_size, embedding_dim)
```

#### Visualisasi Embedding (PCA)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Simulasi embedding untuk beberapa kata (setelah training)
words = ['kucing', 'anjing', 'mobil', 'motor', 'makanan', 'minuman', 'AI', 'robot']
# Embedding acak sebagai placeholder (di praktik, ini dari model yang sudah dilatih)
np.random.seed(42)
embeddings = {word: np.random.randn(128) * 0.1 for word in words}

# PCA ke 2 dimensi untuk visualisasi
X = np.array([embeddings[w] for w in words])
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(X_2d[i, 0], X_2d[i, 1], s=200)
    plt.annotate(word, (X_2d[i, 0], X_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Embedding Visualization (PCA)')
plt.grid(True, alpha=0.3)
plt.show()

# Kata yang mirip dalam konteks seharusnya berdekatan
# "kucing" dekat "anjing", "mobil" dekat "motor", dll.
```

> **Koneksi ke Bab 12 (RAG):** Cosine similarity antara embedding dokumen = cara RAG mencari dokumen relevan.

---

### 1.3 Arsitektur Sequence Model

#### RNN: State yang Berulang

```
 RNN "mengingat" apa yang sudah dibaca:
 
  x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → output
         ↑        ↑        ↑
         │        │        └─── hidden state membawa "memori"
         └────────────────────
```

- Pada tiap langkah, RNN membaca input baru + state sebelumnya → update state
- State = "ringkasan" apa yang sudah dibaca sejauh ini

**Masalah RNN:**

- Proses sekuensial → tidak bisa paralel → lambat training
- Gradient vanishing/exploding di sequence panjang → konteks jauh "terlupa"

#### LSTM & GRU: Solusi untuk Long-Term Dependency

| Aspek                 | RNN Dasar | LSTM                           | GRU                    |
| --------------------- | --------- | ------------------------------ | ---------------------- |
| Gating mechanism      | Tidak ada | 3 gate (input, forget, output) | 2 gate (update, reset) |
| Memori jangka panjang | Lemah     | Kuat                           | Cukup kuat             |
| Kompleksitas          | Rendah    | Lebih tinggi                   | Sedang                 |
| Kecepatan             | Cepat     | Lebih lambat                   | Sedang                 |

```python
from tensorflow.keras import layers

# LSTM untuk klasifikasi teks
model_lstm = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# GRU — lebih ringkas dari LSTM
model_gru = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.GRU(64, return_sequences=True),
    layers.GRU(32),
    layers.Dense(1, activation='sigmoid')
])

# Conv1D — pendekatan berbeda: pola lokal di teks
model_conv1d = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation='sigmoid')
])
```

> **Kapan pakai:** LSTM/GRU untuk sequence yang panjang & penting konteks panjang; Conv1D untuk pola lokal; keduanya bisa jadi baseline sebelum pakai Transformer.

---

### 1.4 Text Generation Klasik

#### Next-Token Prediction — Fondasi LLM

```
Prinsip yang sama dengan GPT, hanya skalanya berbeda:

  Input: "Machine learning is"
  Model → distribusi probabilitas token berikutnya:
    "a"     : 0.45
    "the"   : 0.30
    "one"   : 0.12
    "an"    : 0.08
    ...     : ...
  Pilih satu (sampling) → lanjutkan → ulangi
```

```python
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Contoh sederhana text generation dengan LSTM
# (Versi lengkap lebih kompleks — ini hanya konsep)

vocab_size = 10000
embedding_dim = 256
rnn_units = 1024

class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)
  
    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.lstm.get_initial_state(x)
        x, states = self.lstm(x, initial_state=states, training=training)
        logits = self.dense(x, training=training)
      
        if return_state:
            return logits, states
        return logits

# Sampling dengan temperature
def sample_token(probabilities, temperature=1.0):
    # Temperature: lebih tinggi = lebih random, lebih rendah = lebih deterministik
    preds = np.log(probabilities) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

print("Contoh temperature:")
probs = np.array([0.5, 0.3, 0.2])
for temp in [0.2, 1.0, 2.0]:
    print(f"  Temperature {temp}: token yang dipilih cenderung...")
    print(f"    0.2 = sangat deterministik (selalu pilih yang probabilitas tertinggi)")
    print(f"    1.0 = sesuai distribusi")
    print(f"    2.0 = sangat random (semua punya kesempatan cukup)")
```

---

### 1.5 Mengapa Transformer Menggantikan RNN

| Aspek               | RNN/LSTM                | Transformer                    |
| ------------------- | ----------------------- | ------------------------------ |
| Parallelization     | ❌ Sekuensial (lambat)  | ✅ Seluruh sequence sekaligus  |
| Long-range context  | ⚠️ Menurun di panjang | ✅ Self-attention langsung     |
| Training speed      | Lambat                  | Cepat (setelah di-parallelize) |
| Kualitas (saat ini) | Kurang kompetitif       | State-of-the-art               |

```
RNN:          "Machine" → [h1] → "learning" → [h2] → "is" → [h3] → ...
               (harus tunggu token sebelumnya selesai)

Transformer:  "Machine", "learning", "is"  →  [SEMUA DI-PROSES SEKALIGUS]
               self-attention: tiap token lihat semua token lain
```

> **Koneksi ke Bab 9:** Ini adalah alasan utama mengapa semua LLM modern berbasis Transformer. RNN tidak lagi dipakai untuk task tingkat produksi kecuali ada batasan spesifik.

---

### 1.6 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 6:
  1. Tokenisasi: teks → sequence → padding (fondasi semua NLP)
  2. OOV token: kata tidak dikenal ditangani dengan OOV
  3. Embedding: kata → vektor bermakna, dilatih bersama model
  4. RNN: state berulang, masalah gradient vanishing
  5. LSTM/GRU: gate mechanism untuk long-term memory
  6. Conv1D: pendekatan alternatif dengan pola lokal
  7. Text generation: next-token prediction (sama seperti GPT)
  8. Temperature: kontrol randomness sampling
  9. Transformer > RNN untuk hampir semua task modern
```

---

## 2. Latihan Praktis

---

### Latihan 1: Klasifikasi Teks Sederhana

**Tujuan:** Bangun pipeline klasifikasi teks dari nol sampai model siap pakai.

```python
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Dataset contoh: teks + label (1=positif, 0=negatif)
texts = [
    "Saya sangat senang dengan layanan ini",
    "Produknya bagus dan pengiriman cepat",
    "Sangat puas, akan order lagi",
    "Layanan pelanggan sangat membantu",
    "Produk rusak saat tiba",
    "Pengiriman terlalu lambat",
    "Saya kecewa dengan kualitasnya",
    "Tidak akan membeli lagi",
    "Sangat menyenangkan, recommend",
    "Sangat buruk, saya marah",
] * 20  # Duplicate untuk punya lebih banyak data

labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 20

# 1. Tokenisasi
vocab_size = 1000
tokenizer = text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = sequence.pad_sequences(sequences, maxlen=20, padding='post')

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Data shape: {padded.shape}")

# 2. Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# 3. Bangun model dengan berbagai pendekatan

# a. Embedding + GlobalAveragePooling (paling sederhana)
model_simple = tf.keras.Sequential([
    layers.Embedding(vocab_size, 64, input_length=20),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# b. Embedding + LSTM
model_lstm = tf.keras.Sequential([
    layers.Embedding(vocab_size, 64, input_length=20),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# c. Embedding + Conv1D
model_conv = tf.keras.Sequential([
    layers.Embedding(vocab_size, 64, input_length=20),
    layers.Conv1D(64, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile semua model
for name, model in [('Simple', model_simple), ('LSTM', model_lstm), ('Conv1D', model_conv)]:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"\n{name} model summary:")
    model.summary()

# 4. Training & perbandingan
results = {}
for name, model in [('Simple', model_simple), ('LSTM', model_lstm), ('Conv1D', model_conv)]:
    print(f"\nTraining {name} model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_split=0.2,
        verbose=0
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = {'history': history, 'test_acc': test_acc}
    print(f"Test accuracy: {test_acc:.4f}")

# 5. Visualisasi perbandingan
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

for name, result in results.items():
    plt.subplot(1, 3, list(results.keys()).index(name)+1)
    plt.plot(result['history'].history['accuracy'], label='Train')
    plt.plot(result['history'].history['val_accuracy'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{name} - Accuracy')

plt.tight_layout()
plt.show()

print(f"\nPerbandingan Test Accuracy:")
for name, result in results.items():
    print(f"  {name}: {result['test_acc']:.4f}")
```

---

### Latihan 2: Text Generator Sederhana

**Tujuan:** Pahami next-token prediction dengan praktik nyata.

```python
import tensorflow as tf
import numpy as np

# Contoh dataset kecil (lagu atau puisi pendek)
text = """
Hari yang cerah sembari senyum
Matahari bersinar terik disana
Angin sepoi-sepoi bertiup
Memberi rasa yang menenangkan
""".strip()

# Buat karakter vocabulary
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Contoh mapping: {list(char2idx.items())[:5]}")

# Buat sequences: input karakter + target karakter berikutnya
seq_length = 40
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, 3):  # step=3 untuk lebih banyak samples
    sequences.append(text[i:i+seq_length])
    next_chars.append(text[i+seq_length])

print(f"Jumlah sequences: {len(sequences)}")

# Konversi ke tensor
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool_)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)

for i, (seq, next_char) in enumerate(zip(sequences, next_chars)):
    for t, char in enumerate(seq):
        X[i, t, char2idx[char]] = 1
    y[i, char2idx[next_char]] = 1

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Bangun model sederhana
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(seq_length, vocab_size)),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih
print("\nTraining text generator...")
history = model.fit(X, y, epochs=20, batch_size=64, verbose=1)

# Generate teks baru
def generate_text(model, seed_text, gen_length=100, temperature=1.0):
    generated = seed_text
  
    for _ in range(gen_length):
        # Encode seed
        x_pred = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(generated[-seq_length:]):
            if char in char2idx:
                x_pred[0, t, char2idx[char]] = 1
      
        # Prediksi
        preds = model.predict(x_pred, verbose=0)[0]
      
        # Sampling dengan temperature
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = idx2char[next_index]
      
        generated += next_char
  
    return generated

print("\n" + "="*50)
print("GENERATED TEXT (berbagai temperature)")
print("="*50)

seed = "Hari yang cerah"
for temp in [0.2, 0.5, 1.0, 1.5]:
    print(f"\n--- Temperature: {temp} ---")
    generated = generate_text(model, seed_text=seed, gen_length=50, temperature=temp)
    print(generated)
```

> **Tulis:** Apa yang berbeda antara teks dengan temperature rendah vs tinggi? Apa kelebihan dan kekurangannya?

---

### Latihan 3: Visualisasi Embedding + Analisis

**Tujuan:** Pahami apa yang direpresentasikan embedding dan kapan mirip.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Contoh kata-kata dengan embedding acak (simulasi)
# Dalam praktik, ini diambil dari model yang sudah dilatih
np.random.seed(42)

words = [
    # Hewan
    'kucing', 'anjing', 'burung', 'ikan', 'kelinci', 'marem',
    # Kendaraan
    'mobil', 'motor', 'sepeda', 'pesawat', 'kereta', 'truk',
    # Makanan
    'nasi', 'ayam', 'ikan', 'sayur', 'buah', 'minuman',
    # Teknologi
    'AI', 'robot', 'komputer', 'software', 'hardware', 'data',
    # Alam
    'matahari', 'bau', 'hutan', 'sungai', 'gunung', 'laut',
]

# Hubungan semantic (untuk simulasi)
# Kata yang mirip secara makna punya embedding yang mirip
semantic_groups = {
    'Hewan': ['kucing', 'anjing', 'burung', 'ikan', 'kelinci'],
    'Kendaraan': ['mobil', 'motor', 'sepeda', 'pesawat', 'kereta'],
    'Makanan': ['nasi', 'ayam', 'sayur', 'buah'],
    'Teknologi': ['AI', 'robot', 'komputer', 'software'],
}

# Generate embedding dengan kelompok
embeddings = {}
for group_name, group_words in semantic_groups.items():
    # Pusat cluster
    center = np.random.randn(100) * 0.5
    for word in group_words:
        # Embedding = center + noise kecil
        embeddings[word] = center + np.random.randn(100) * 0.1

# Kata yang tidak di-cluster
for word in words:
    if word not in embeddings:
        embeddings[word] = np.random.randn(100) * 0.5

# Visualisasi dengan PCA
X = np.array([embeddings[w] for w in words])
word_names = words

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Tentukan warna per kelompok
colors = []
for word in words:
    found = False
    for group_name, group_words in semantic_groups.items():
        if word in group_words:
            colors.append(group_name)
            found = True
            break
    if not found:
        colors.append('gray')

color_map = {
    'Hewan': 'red',
    'Kendaraan': 'blue',
    'Makanan': 'green',
    'Teknologi': 'purple',
    'gray': 'gray'
}

plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], c=color_map[colors[i]], s=100, alpha=0.7)
    plt.annotate(word, (X_pca[i, 0], X_pca[i, 1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Word Embedding Visualization (PCA)')
plt.grid(True, alpha=0.3)

# Tambah legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=label) for label, c in color_map.items() if label != 'gray']
plt.legend(handles=legend_elements, title='Semantic Groups')

plt.tight_layout()
plt.show()

# Analisis
print("\n" + "="*50)
print("ANALISIS EMBEDDING")
print("="*50)

# Hitung cosine similarity antar beberapa pasangan
from sklearn.metrics.pairwise import cosine_similarity

def compare_words(w1, w2):
    v1 = embeddings[w1].reshape(1, -1)
    v2 = embeddings[w2].reshape(1, -1)
    sim = cosine_similarity(v1, v2)[0, 0]
    print(f"  '{w1}' vs '{w2}': cosine similarity = {sim:.4f}")
    return sim

print("\nKata dari kelompok sama (seharusnya similarity tinggi):")
compare_words('kucing', 'anjing')
compare_words('mobil', 'motor')
compare_words('nasi', 'ayam')

print("\nKata dari kelompok berbeda (seharusnya similarity rendah):")
compare_words('kucing', 'mobil')
compare_words('AI', 'nasi')
compare_words('pesawat', 'sayur')
```

> **Tulis:** Apa yang kamu pelajari dari visualisasi ini? Kapan embedding bisa digunakan untuk RAG?

---

## 🧰 Materi Pendukung (Folder Ini)

| File                                     | Apa                                                                                                                                                                                              | Kapan Dipakai                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| `01_lab_nlp_text.ipynb`                | Lab praktikum: 10 bagian — tokenisasi/padding, skip-gram + negative sampling (gradient check), purity + PCA, klasifikasi sentimen, bigram + temperature, RNN BPTT vs bigram, vanishing gradient | Kerjakan setelah baca materi inti |
| `02_kuis_nlp_text.ipynb`               | 10 soal (PG + coding), 23 poin, skor otomatis                                                                                                                                                    | Setelah lab selesai               |
| `03_kunci_jawaban_kuis_nlp_text.ipynb` | Kunci + intuisi di balik tiap jawaban                                                                                                                                                            | HANYA setelah mencoba kuis        |
| `cheatsheet-nlp-text.md`               | Rumus kunci + pola kode + koneksi ke bab lain                                                                                                                                                    | Review harian / sebelum kuis      |
| `project-starter-lm-mini/`             | Proyek end-to-end: language model mini dari nol — embedding + bigram LM + RNN BPTT (TDD, 40 test) + starter/solusi + RUBRIK                                                                     | Setelah kuis ≥ 18/23             |

---

## 📚 Referensi Online

| Sumber                                                                                                                                      | Topik          | Keterangan                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------- |
| [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)                                             | Tutorial resmi | Praktik langsung dengan LSTM text generation      |
| [Stanford CS224n — NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)                                                           | Kuliah lengkap | Materi paling komprehensif untuk NLP              |
| [The Illustrated Word2vec — Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)                                                  | Visualisasi    | Penjelasan embedding yang sangat mudah dimengerti |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)                                                                           | Kursus gratis  | Jembatan ke Transformer/Lab berikutnya            |
| [Natural Language Processing with Transformers (book)](https://www.oreilly.com/library/view/natural-language-processing-with/9781098136789/) | Buku           | Jika ingin pendalaman transformer-based NLP       |

---

## ✅ Checklist Kompetensi

- [ ] Menjelaskan alur teks mentah → token → id → embedding → prediksi
- [ ] Menjelaskan kenapa Transformer menggantikan RNN
- [ ] Menjelaskan temperature dengan analogi yang tepat
- [ ] Membangun pipeline tokenisasi + classifier teks dari nol
- [ ] Visualisasikan embedding dan jelaskan clustering semantik
- [ ] Implementasi text generator sederhana dengan temperature control
- [ ] Paham perbedaan RNN, LSTM, GRU, Conv1D untuk teks
