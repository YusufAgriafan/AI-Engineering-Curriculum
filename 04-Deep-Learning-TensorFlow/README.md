# Bab 4 — Deep Learning & TensorFlow

> Neural network adalah mesin di balik semua model modern — termasuk LLM. Bab ini membuat Anda paham *isi kap mesin* tersebut.

## 🎯 Tujuan Belajar
- Memahami neuron, layer, aktivasi, backpropagation.
- Membangun & melatih model dengan TensorFlow/Keras: sequential, functional, custom.
- Teknik praktis: callback, callback kustom, batch normalization.

## 1. Materi Inti

> Bab ini adalah jembatan dari "konsep ML klasik" ke "neural network yang jadi basis semua model modern" — termasuk LLM di Bab 9+.

---

### 1.1 Neuron & Jaringan — Intuisi

#### Apa itu Neuron?

```
Pikirkan neuron sebagai "decision maker kecil":

  input1 ── w1 ──┐
                  ├──(+)──→ aktivasi → output
  input2 ── w2 ──┘     ↑
                   bias b

  Rumus: output = aktivasi(w1·x1 + w2·x2 + ... + b)
```

- `w` = bobot: seberapa penting tiap input
- `b` = bias: "baseline" sebelum ada input
- `aktivasi`: fungsi yang memutuskan apakah neuron "menyala"

#### Layer adalah Kumpulan Neuron

```
Layer input → Layer tersembunyi → Layer output
    ↓              ↓              ↓
  4 neuron     8 neuron      3 neuron (kelas)
```

#### Fungsi Aktivasi — Kapan Pakai Apa

| Fungsi | Rumus | Output range | Penggunaan |
|---|---|---|---|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layer (paling umum) |
| **Sigmoid** | 1/(1+e⁻ˣ) | [0, 1] | Output binary classification |
| **Softmax** | eˣᵢ / Σeˣⱼ | [0, 1], Σ=1 | Output multiclass classification |
| **Tanh** | (eˣ - e⁻ˣ)/(eˣ+e⁻ˣ) | [-1, 1] | Jarang dipakai sekarang |
| **Linear** | f(x)=x | (-∞, ∞) | Output regresi |

```python
import tensorflow as tf
import numpy as np

# ReLU contoh
x = np.array([-2, -1, 0, 1, 2])
print(f"ReLU: {tf.nn.relu(x).numpy()}")     # [0, 0, 0, 1, 2]

# Sigmoid contoh
print(f"Sigmoid: {tf.nn.sigmoid(x).numpy()}")  # [0.12, 0.27, 0.5, 0.73, 0.88]

# Softmax contoh (multinomial)
logits = np.array([2.0, 1.0, 0.1])
probs = tf.nn.softmax(logits).numpy()
print(f"Softmax: {probs}")   # [0.66, 0.24, 0.10] — jumlah = 1
print(f"Jumlah probs: {probs.sum():.2f}")  # 1.00
```

---

### 1.2 Backpropagation — Konsep Utama

> **Inti cerita:** Backpropagation adalah cara menghitung "seberapa bertanggung jawab" tiap bobot terhadap error total — sehingga kita bisa memperbaikinya.

#### Analogi Sederhana

```
Bayangkan tim yang membuat kue:
- Bapak mengambil telur → membuat adonan → memanggang → kue jadi

Setelah kue jadi, ternyata terlalu asam.
Siapa yang harus disalahkan?

- Rasa asam → mungkin garpu takaran gula salah
- Atau pilih telur yang salah
- Atau oven terlalu panas

Backpropagation = proses "menelusuri mundur" dari error akhir
untuk menghitung kontribusi tiap langkah.
```

#### Chain Rule dalam 1 Menit

```
Kalau y = f(g(x)), maka:
dy/dx = dy/dg · dg/dx

Contoh: y = (2x + 3)²
  g(x) = 2x + 3
  f(g) = g²
  
dy/dx = dy/dg · dg/dx
       = 2g · 2
       = 2(2x+3) · 2
       = 4(2x+3)
```

Ini sama persis yang terjadi di neural network — hanya lebih banyak fungsi bersarang.

---

### 1.3 Membangun Model dengan Keras

#### Sequential API — Untuk Alur Lurus

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

#### Functional API — Untuk Arsitektur Lebih Kompleks

```python
from tensorflow.keras import Model
from tensorflow.keras import Input

# Multi-input: user profile + item features
user_input = Input(shape=(50,), name='user_features')
item_input = Input(shape=(30,), name='item_features')

user_encoded = layers.Dense(32, activation='relu')(user_input)
item_encoded = layers.Dense(32, activation='relu')(item_input)

# Gabungkan
combined = layers.Concatenate()([user_encoded, item_encoded])
combined = layers.Dense(16, activation='relu')(combined)
output = layers.Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

#### Custom Layer — Buat Sendiri

```python
class NoisyDense(layers.Layer):
    """Dense layer yang tambah noise ke weights saat training."""
    
    def __init__(self, units, noise_std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.noise_std = noise_std
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs, training=False):
        if training:
            # Tambah noise hanya saat training, bukan inference
            noise = tf.random.normal(shape=tf.shape(self.w), 
                                     mean=0, stddev=self.noise_std)
            w_noisy = self.w + noise
        else:
            w_noisy = self.w
        return tf.matmul(inputs, w_noisy) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'noise_std': self.noise_std})
        return config
```

> **Kenapa penting:** Custom layer/loss/model adalah kemampuan yang membedakan engineer dari pengguna library. Di Bab 13 (fine-tuning), kamu bakal pakai konsep serupa.

---

### 1.4 Custom Training Loop dengan GradientTape

> Ini adalah pondasi untuk fine-tuning model di Bab 13.

```python
import tensorflow as tf

# Persiapkan data
X = tf.random.normal((1000, 20))
y = tf.cast(X[:, 0] > 0, tf.float32)  # binary label dari fitur pertama

# Buat model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.BinaryCrossentropy()

# Training loop
for epoch in range(10):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Hitung metrik
    predictions_binary = (predictions > 0.5).astype(tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_binary, y), tf.float32))
    
    print(f"Epoch {epoch+1}: loss={loss:.4f}, accuracy={accuracy:.4f}")
```

**Perbedaan dengan `model.fit()`:**

| `model.fit()` | Custom training loop |
|---|---|
| Cepat, sederhana | Lebih verbose tapi fleksibel |
| Cukup untuk 90% kasus | Perlu untuk: custom loss dynamic, multi-model training, log kustom |
| Abstraksi tinggi | Kontrol penuh |

---

### 1.5 Callbacks & Kontrol Training

```python
callbacks = [
    # Berhenti kalau val_loss tidak membaik 10 epoch berturut-turut
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    # Simpan model terbaik secara otomatis
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    # Kurangi learning rate kalau stagnan
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=callbacks
)
```

---

### 1.6 Resnet & Skip Connection — Pola yang Muncul Lagi di Transformer

```python
# Resnet block dasar
inputs = layers.Input(shape=(64,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
# Skip connection: tambahkan input asli ke output
outputs = layers.Add()([inputs, x])
outputs = layers.BatchNormalization()(outputs)

resnet_block = keras.Model(inputs, outputs)
```

> **Kenapa skip connection penting:** Tanpa skip connection, neural network yang sangat dalam sulit dilatih (vanishing gradient). Transformer (Bab 9) juga pakai pola yang sama di dalam attention block.

---

### 1.7 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 4:
  1. Neuron = weighted sum + activation
  2. ReLU (hidden), Sigmoid (binary output), Softmax (multiclass)
  3. Sequential API: alur lurus; Functional API: multi-input/output
  4. Backprop = chain rule + gradient descent untuk bobot
  5. GradientTape = training loop dari nol (fondasi fine-tuning)
  6. Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  7. Skip connection & BatchNorm: pola yang dipakai berulang
  8. Custom layer/loss = kemampuan engineer premium
```


## 2. Latihan Praktis

---

### Latihan 1: Classifier dengan Berbagai Konfigurasi

**Tujuan:** Pahami pengaruh arsitektur terhadap performa.

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Baseline: 1 hidden layer
model1 = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Lebih dalam: 3 hidden layers
model2 = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model 1 (baseline)...")
hist1 = model1.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=0)

print("Training model 2 (deeper)...")
hist2 = model2.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=0)

# Plot perbandingan
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(hist1.history['val_loss'], label='Model 1 (baseline)')
plt.plot(hist2.history['val_loss'], label='Model 2 (deeper)')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Validation Loss Comparison')

plt.subplot(1, 2, 2)
plt.plot(hist1.history['val_accuracy'], label='Model 1 (baseline)')
plt.plot(hist2.history['val_accuracy'], label='Model 2 (deeper)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy Comparison')

plt.tight_layout()
plt.show()

# Evaluasi akhir
print(f"\nModel 1 test accuracy: {model1.evaluate(x_test, y_test, verbose=0)[1]:.4f}")
print(f"Model 2 test accuracy: {model2.evaluate(x_test, y_test, verbose=0)[1]:.4f}")
```

> **Tulis:** Mana yang lebih baik? Mengapa? Kapan model "deeper" bisa jadi lebih buruk?

---

### Latihan 2: Custom Dense Layer dari Nol

**Tujuan:** Pahami apa yang terjadi di dalam layer.Dense.

```python
import tensorflow as tf

class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # Simpan nama aktivasi untuk di-restore
        self.activation_name = activation
        if activation is None:
            self.activation_fn = lambda x: x
        elif activation == 'relu':
            self.activation_fn = tf.nn.relu
        elif activation == 'sigmoid':
            self.activation_fn = tf.nn.sigmoid
        elif activation == 'softmax':
            self.activation_fn = tf.nn.softmax
        else:
            self.activation_fn = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # Inisialisasi weights (mirip GlorotUniform untuk Dense)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return self.activation_fn(z)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation_name})
        return config

# Test: buat model dengan layer kustom, bandingkan dengan layer Dense
model_custom = keras.Sequential([
    layers.Dense(64, activation='relu'),  # built-in
    MyDense(32, activation='relu'),        # kustom
    MyDense(10, activation='softmax')      # kustom
])

model_builtin = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Inisialisasi sama
model_custom.set_weights(model_builtin.get_weights())

# Test dengan data acak
x_test = tf.random.normal((10, 20))
out_custom = model_custom(x_test)
out_builtin = model_builtin(x_test)

print(f"Output custom shape: {out_custom.shape}")
print(f"Output builtin shape: {out_builtin.shape}")
print(f"Output sama? {tf.reduce_all(tf.abs(out_custom - out_builtin) < 1e-6).numpy()}")
```

---

### Latihan 3: Custom Training Loop Lengkap

**Tujuan:** Pahami tiap langkah di balik `model.fit()`, termasuk logging kustom.

```python
import tensorflow as tf
import time

# Data sederhana
X_train = tf.random.normal((5000, 30))
y_train = tf.cast(X_train[:, 0] + X_train[:, 1] > 0, tf.float32)

X_val = tf.random.normal((1000, 30))
y_val = tf.cast(X_val[:, 0] + X_val[:, 1] > 0, tf.float32)

model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.BinaryCrossentropy()
metric = keras.metrics.BinaryAccuracy()

EPOCHS = 10
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # Training
    train_loss.reset_states()
    train_metric.reset_states()
    
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss.update_state(y_batch, predictions)
        train_metric.update_state(y_batch, predictions)
    
    # Validation
    val_loss.reset_states()
    val_metric.reset_states()
    
    for x_batch, y_batch in val_dataset:
        predictions = model(x_batch, training=False)
        loss = loss_fn(y_batch, predictions)
        val_loss.update_state(y_batch, predictions)
        val_metric.update_state(y_batch, predictions)
    
    epoch_time = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss.result():.4f} | "
          f"Train Acc: {train_metric.result():.4f} | "
          f"Val Loss: {val_loss.result():.4f} | "
          f"Val Acc: {val_metric.result():.4f} | "
          f"Time: {epoch_time:.2f}s")
```

> **Tulis:** Bandingkan dengan `model.fit()` — apa yang lebih sulit? Apa yang lebih mudah?

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_deep_learning.ipynb` | Lab praktikum: 10 bagian — NN dibangun dari nol (numpy) lalu dipetakan ke TensorFlow/Keras | Kerjakan setelah baca materi inti |
| `02_kuis_deep_learning.ipynb` | 10 soal (PG + coding), 19 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_deep_learning.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-deep-learning.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-numpy-nn/` | Proyek end-to-end: keras mini dari nol — Dense + backprop + SGD + early stopping (TDD, 48 test) + starter/solusi + RUBRIK | Setelah kuis ≥ 15/19 |

---

## 📚 Referensi Online

| Sumber | Topik | Keterangan |
|---|---|---|
| [TensorFlow Official Tutorial](https://www.tensorflow.org/tutorials) | Semua dasar | Dokumentasi resmi avec contoh |
| [Neural Networks: Zero to Hero — Karpathy](https://karpathy.ai/zero-to-hero.html) | Video | **Wajib** — dari neuron ke GPT dengan penjelasan intuitif |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Transformer visual | Lebih mudah paham di Bab 9 setelah baca ini |
| [DeepLearning.AI TensorFlow Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice) | Kursus lengkap | Jika mau depth lebih | 

---

## ✅ Checklist Kompetensi

- [ ] Menjelaskan apa yang terjadi pada bobot saat `model.fit()` berjalan
- [ ] Membuat model functional API dengan 2 output berbeda
- [ ] Menulis custom training loop dengan GradientTape
- [ ] Implementasi custom layer/class sendiri
- [ ] Paham kapan pakai ReLU vs Sigmoid vs Softmax
- [ ] Bisa jelaskan fungsi callbacks dan kapan pakai masing-masing
- [ ] Bisa jelaskan dengan kata sendiri apa itu backpropagation

