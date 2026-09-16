# Bab 8 — ML Produksi: Evaluasi, GAN, Struktur Proyek

> Bab transisi: dari "model jupyter yang jalan" ke "model siap dipakai". Plus generative model (GAN) sebagai pengantar dunia generative AI sebelum LLM.

## 🎯 Tujuan Belajar
- Struktur ML project & menulis laporan/eksperimen yang layak baca.
- Generative model: GAN, WGAN-GP.
- Menyusun decision: model klasik vs deep vs "cukup rules".

## 1. Materi Inti

> Bab transisi: dari "model Jupyter yang jalan" ke "model siap dipakai". Bab ini menyiapkan mentalitas production engineer.

---

### 1.1 Struktur Project yang Benar

#### Dari Notebook Menjadi Software

```
Notebook memungkinkan eksplorasi cepat, tapi:
  - Tidak testable
  - Tidak versionable dengan rapi
  - Tidak reusable
  - Output terancam kalau library berubah

Solusi: pisahkan eksplorasi (notebook) dari produksi (module).
```

#### Template Struktur Proyek

```
my-ml-project/
├── README.md                 # Apa, mengapa, instalasi, cara pakai
├── requirements.txt          # Dependencies (atau pyproject.toml/poetry.lock)
├── setup.py / pyproject.toml # Agar bisa di-install: pip install -e .
│
├── .gitignore                # JANGAN commit: data/, __pycache__/, .env, *.pyc
│
├── data/
│   ├── raw/                  # Data asli (tidak dimodifikasi) — JANGAN commit!
│   ├── processed/            # Data setelah preprocessing — bisa commit kalau kecil
│   └── external/             # Data dari sumber eksternal (API, download)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Eksplorasi data
│   ├── 02_baseline.ipynb     # Baseline model
│   ├── 03_experiment_X.ipynb # Eksperimen spesifik
│   └── README.md             # Penjelasan tiap notebook dan urutan
│
├── src/                      # Kode produksi (importable, testable)
│   ├── __init__.py           # Package init
│   ├── data/                 # Data loading & preprocessing
│   │   ├── load.py
│   │   ├── preprocess.py
│   │   └── __init__.py
│   ├── features/             # Feature engineering
│   │   ├── build_features.py
│   │   └── __init__.py
│   ├── models/               # Model definition & training
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── __init__.py
│   └── visualization/        # Plot & visualisasi
│       └── visualize.py
│
├── tests/                    # Unit test (Bab 1)
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── models/                   # Model artifacts (checkpoint, serialized model)
│   ├── best_model.pkl        # Model yang sudah dilatih
│   └── latest/               # Versi terbaru
│
├── reports/                  # Experiment log & laporan
│   ├── experiment_log.md     # Log eksperimen (lihat template di bawah)
│   └── figures/              # Plot yang di-save
│
├── configs/                  # Konfigurasi eksperimen
│   ├── experiment_1.yaml     # Hiperparameter & setup
│   └── default.yaml
│
├── Makefile                  # Command shortcuts: make train, make test, dll
│
└── .env.example              # Template environment variables (commit ini!)
    # .env file sendiri JANGAN commit (berisi API key, path, dll)
    
```

#### Contoh: Memisahkan Notebook dari Code

```python
# notebooks/03_experiment.ipynb
# Di notebook, panggil fungsi dari src/:

from src.data.load import load_dataset
from src.features.build_features import create_features
from src.models.train import train_model

# Load data
df = load_dataset('data/raw/dataset.csv')

# Buat features
X, y = create_features(df)

# Train model
model, metrics = train_model(X, y, config={...})

# Notebook ini hanya "script" yang menjalankan pipeline → bisa dipanggil ulang
```

---

### 1.2 Eksperimen & Dokumentasi yang Terstruktur

#### Template Experiment Log

Setiap eksperimen harus terdokumentasi agar bisa:
- Di-reproduce 6 bulan kemudian
- Dibandingkan dengan eksperimen lain
- Menjelaskan KEPUTUSAN, bukan hanya hasil

```markdown
# Eksperimen: [Judul singkat]

**Tanggal:** YYYY-MM-DD
**Dilakukan oleh:** [nama]
**Branch/Commit:** [git reference]

## 1. Hipotesis
> [Apa yang kita coba? Misalnya: "Menambah layer akan meningkatkan akurasi karena..."]

## 2. Setup
- Dataset: [nama, versi, split]
- Model arsitektur: [deskripsi]
- Hyperparameters:
  - Learning rate: X
  - Batch size: Y
  - Epoch: Z
  - Optimizer: [nama]
- Baseline untuk dibandingkan: [nama eksperimen sebelumnya]

## 3. Hasil
| Metric | Baseline | Eksperimen Ini | Perubahan |
|--------|----------|----------------|----------|
| Train Accuracy | X% | Y% | +Z% |
| Val Accuracy | X% | Y% | +Z% |
| Train Loss | X | Y | +Z |
| Val Loss | X | Y | +Z |
| Training Time | X mnt | Y mnt | +Z% |

## 4. Analisis
- [Apa yang Unexpected?]
- [Apakah hipotesis terbukti?]
- [Apa yang mau diulang/diuji selanjutnya?]

## 5. Keputusan
- [ ] 멜라닌 adopt perubahan ini ke branch utama
- [ ] 멜 melakukan eksperimen lanjutan (sebutkan apa)
- [ ] 멜 revert / coba pendekatan lain

## 6. File Artifact
- Model checkpoint: [path]
- Plot: [path]
- Log training: [path]
```

#### Alasan Pentingnya Dokumentasi

```
Tanpa dokumentasi:
  "Eksperimen 3 minggu lalu: akurasi naik 2%."  → TIDAK BERMEANAPA-APA
  Anda tidak tahu:
  - Apa yang diubah?
  - data apa yang dipakai?
  - hyperparameter apa?
  - Kenapa naik?

Dengan dokumentasi:
  "Eksperimen 3 minggu lalu: menambah dropout 0.3 → val accuracy naik 2% 
   tapi train accuracy turun 1% → indikasi overfitting berkurang.
   Data: versi 2.1. Hyperparameter: lr=0.001, bs=32. 
   Keputusan: adopt ke main, karena val metric memang itu yang kita targetkan."
  → AKHIRNYA BERMEANAPA-APA
```

---

### 1.3 GAN — Generative Adversarial Networks

#### Konsep Dasar: Duel Dua Jaringan

```
GAN = Generator (G) vs Discriminator (D) dalam "permainan"

                    ┌─────────────────────────────────────┐
                    │         PERMAINAN GANs              │
                    │                                     │
                    │  Generator (G):                     │
                    │    - Mulai dari noise random        │
                    │    - Coba buat gambar "nyata"      │
                    │    - Tujuannya: DEPOK D            │
                    │                                     │
                    │  Discriminator (D):                 │
                    │    - Lihat gambar (asli atau palsu) │
                    │    - Tebak: asli atau palsu?        │
                    │    - Tujuannya: TEPAT MEMBEDA       │
                    │                                     │
                    │  Proses:                            │
                    │    1. G buat gambar palsu           │
                    │    2. D lihat → tebak               │
                    │    3. D dikoreksi (jika salah)      │
                    │    4. G dikoreksi (jika D menang)   │
                    │    5. Ulangi → G makin bagus        │
                    └─────────────────────────────────────┘
```

> **Intuisi:** Seperti pemalsu vs detektif pemalsu. Seiring waktu, pemalsu semakin jenius (G), dan detektif semakin piawai (D). Hasilnya: G bisa membuat gambar yang hampir tidak bisa dibedakan dari asli.

#### Diagram Arsitektur

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)

# Generator: noise → gambar
def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), 
                                      padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), 
                                      padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), 
                                      padding='same', use_bias=False, 
                                      activation='tanh'))
    return model

# Discriminator: gambar → real/fake prediction
def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

# Visualisasi arsitektur
print("Generator:")
generator.summary()

print("\nDiscriminator:")
discriminator.summary()

# Latih GAN (simplified training loop)
-cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Generate gambar sintetis
def generate_and_save_images(model, epoch, noise):
    predictions = model(noise, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

# Contoh: lihat generator sebelum vs sesudah training
noise = tf.random.normal([16, 100])  # 16 noise vectors

print("\nGambar awal (generator belum trained):")
generated_before = generator(noise, training=False)
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_before[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.suptitle('Generator Output - Before Training')
plt.show()
```

---

### 1.4 Kapan Pakai Apa — Decision Guide Terstruktur

| Situasi | Pertanyaan Kunci | Pilihan | Alasan |
|---|---|---|---|
| **Data tabular, rows > 10rb** | Perlu interpretability? | XGBoost/LightGBM | Performa bagus, feature importance tersedia |
| **Data tabular, butuh kepastian** | Prediksi probabilistic? | Logistic Regression | Interpretable, kalibrasi probabilitas bagus |
| **Gambar, data sedikit (< 1000/kelas)** | Ada pre-trained model? | Transfer Learning (CNN) | Mulailah dari feature extractor yang sudah belajar |
| **Gambar, data banyak (> 100rb)** | Compute tersedia? | Latih dari nol / Fine-tune besar | Custom architecture bisa dioptimasi untuk task spesifik |
| **Teks pendek, klasifikasi volume tinggi** | Latency sensitive? | Embedding + Logistic Regression | Cepat inference, cukup untuk banyak task |
| **Teks, kualitas tinggi butuh reasoning** | Budget cukup? | LLM (Bab 9+) | Kualitas terbaik untuk task kompleks |
| **Teks, privasi/offline wajib** | Bisa pakai open-weight? | Open-weight model (Bab 13) | Data tidak keluar dari infrastructure |
| **Anomaly detection, tanpa label** | Unsupervised yang mana? | Isolation Forest / Autoencoder | Tidak butuh label; deteksi nilai ekstrem |
| **Rekomendasi, ada interaksi user** | Collaborative atau content-based? | Hybrid approach | Studi kasus spesifik; sering perlu custom |

---

### 1.5 Mentalitas Production

```
┌─────────────────────────────────────────────────────────────┐
│  MENTALITAS REASEARCHER                  MENTALITAS ENGINEER │
│                                                             │
│  "Model ini akurasi 98%!"                  "Model ini     │
│   → Senang dengan angka                     98% di test,  │
│                                              inference < 50ms,│
│                                              memory < 100MB,  │
│                                              bisa monitor,    │
│                                              punya fallback,  │
│                                              dokumentasi ada" │
└─────────────────────────────────────────────────────────────┘
```

> **Pesan kunci:** Di production, akurasi adalah SATU dari banyak metrik. Latensi, biaya, reliabilitas, maintainability, dan observability sama pentingnya.

---

### 1.6 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 8:
  1. Struktur project: data/, notebooks/, src/, tests/, reports/
  2. Pisahkan eksplorasi (notebook) dari produksi (module/code)
  3. Experiment log: hipotesis → setup → hasil → keputusan
  4. GAN: Generator vs Discriminator duel, generator membuat data sintetis
  5. Decision guide: kapan pakai model/classifier/what-not
  6. Mentalitas production: lebih dari sekadar akurasi
```

---

## 2. Latihan Praktis

---

### Latihan 1: Refactor Notebook ke Project Structure

**Tujuan:** Pindahkan kode dari notebook ke struktur project yang rapi.

**Langkah 1 — Buat struktur project:**

```bash
# Buat struktur project
mkdir -p src/data src/features src/models tests reports
touch src/__init__.py src/data/__init__.py src/features/__init__.py src/models/__init__.py
```

**Langkah 2 — Ekstrak fungsi dari notebook:**

```python
# src/data/load.py
import pandas as pd
from pathlib import Path

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset dari CSV."""
    return pd.read_csv(path)

def get_train_val_split(df, val_size=0.2, random_state=42):
    """Split data dengan time-based atau random."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=val_size, random_state=random_state)
```

```python
# src/features/build_features.py
import numpy as np
import pandas as pd

def create_features(df: pd.DataFrame) -> tuple:
    """Buat features dari dataframe."""
    # Contoh: drop missing, encode categorical, dll
    df = df.dropna()
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    return X, y

def normalize_features(X_train, X_val=None, X_test=None):
    """Normalize fitur dengan StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    results = {'X_train': X_train_scaled, 'scaler': scaler}
    
    if X_val is not None:
        results['X_val'] = scaler.transform(X_val)
    if X_test is not None:
        results['X_test'] = scaler.transform(X_test)
    
    return results
```

```python
# src/models/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train, X_val, y_val, config=None):
    """Train model dan return model + metrics."""
    if config is None:
        config = {}
    
    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', None),
        random_state=config.get('random_state', 42)
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'classification_report': classification_report(y_val, y_pred)
    }
    
    return model, metrics

def save_model(model, path: str):
    """Simpan model ke file."""
    import joblib
    joblib.dump(model, path)

def load_model(path: str):
    """Load model dari file."""
    import joblib
    return joblib.load(path)
```

**Langkah 3 — Buat test untuk fungsi penting:**

```python
# tests/test_features.py
import pytest
import numpy as np
from src.features.build_features import normalize_features

class TestNormalizeFeatures:
    def test_output_shape(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_features(X_train)
        assert result['X_train'].shape == X_train.shape
    
    def test_mean_zero(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_features(X_train)
        # Mean of normalized features should be ~0
        assert np.allclose(result['X_train'].mean(axis=0), 0, atol=1e-10)
    
    def test_std_one(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_features(X_train)
        # Std of normalized features should be ~1
        assert np.allclose(result['X_train'].std(axis=0), 1, atol=1e-10)
```

**Langkah 4 — Buat script utama:**

```python
# src/run_experiment.py
import argparse
from pathlib import Path

from src.data.load import load_dataset, get_train_val_split
from src.features.build_features import create_features, normalize_features
from src.models.train import train_model, save_model

def main(data_path, output_model_path):
    # Load data
    df = load_dataset(data_path)
    
    # Split
    train_df, val_df = get_train_val_split(df)
    
    # Features
    X_train, y_train = create_features(train_df)
    X_val, y_val = create_features(val_df)
    
    # Normalize
    normalized = normalize_features(X_train, X_val=X_val)
    X_train_scaled = normalized['X_train']
    X_val_scaled = normalized['X_val']
    
    # Train
    model, metrics = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics['classification_report'])
    
    # Save
    save_model(model, output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='models/best_model.pkl')
    args = parser.parse_args()
    
    main(args.data, args.output)
```

---

### Latihan 2: GAN Sederhana untuk MNIST

**Tujuan:** Latih GAN dan lihat perkembangan kualitas gambar sintetis.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Dataset pipeline
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator
def make_generator():
    model = keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'),
    ])
    return model

# Discriminator
def make_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ])
    return model

generator = make_generator()
discriminator = make_discriminator()

# Loss & optimizer
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss

def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)

gen_optimizer = keras.optimizers.Adam(1e-4)
disc_optimizer = keras.optimizers.Adam(1e-4)

# Training
EPOCHS = 50
noise_dim = 100

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise, training=True)
        real_out = discriminator(images, training=True)
        fake_out = discriminator(generated, training=True)
        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)
    
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Training loop
history_gen_loss = []
history_disc_loss = []

for epoch in range(EPOCHS):
    for batch, images in enumerate(train_dataset):
        gen_loss, disc_loss = train_step(images)
        
        if batch % 100 == 0:
            history_gen_loss.append(gen_loss.numpy())
            history_disc_loss.append(disc_loss.numpy())
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Gen Loss={gen_loss.numpy():.4f}, Disc Loss={disc_loss.numpy():.4f}")
        
        # Generate & show images
        test_noise = tf.random.normal([16, noise_dim])
        preds = generator(test_noise, training=False)
        
        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Generator Output - Epoch {epoch+1}')
        plt.show()

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.plot(history_gen_loss, label='Generator Loss', alpha=0.7)
plt.plot(history_disc_loss, label='Discriminator Loss', alpha=0.7)
plt.xlabel('Training Step (every 100 batches)')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Loss')
plt.grid(True, alpha=0.3)
plt.show()
```

> **Tulis:** Apa yang terjadi dengan generator loss dan discriminator loss selama training? Kenapa?

---

### Latihan 3: Template Experiment Log

**Tujuan:** Dokumentasikan 3 eksperimen terakhir dengan template yang rapi.

```markdown
# Template Experiment Log

Salin template di bawah ini untuk setiap eksperimen:

---

# Eksperimen: [Judul singkat]

**Tanggal:** [YYYY-MM-DD]
**Dilakukan oleh:** [nama]
**Branch/Commit:** [git reference]

## 1. Hipotesis
> [Hipotesis singkat]

## 2. Setup
- Dataset: [nama, versi, jumlah sampel, split]
- Model arsitektur: [deskripsi singkat]
- Hyperparameters:
  - Learning rate: [nilai]
  - Batch size: [nilai]
  - Epoch: [nilai]
  - Optimizer: [nama]
- Baseline: [eksperimen yang dibandingkan]

## 3. Hasil
| Metric | Baseline | Eksperimen | Δ |
|--------|----------|------------|---|
| Train Accuracy | X% | Y% | +Z% |
| Val Accuracy | X% | Y% | +Z% |
| Train Loss | X | Y | +Z |
| Val Loss | X | Y | +Z |

## 4. Analisis
- [Temuan menarik]
- [Apakah hipotesis terbukti?]
- [Apa yang unexpected?]

## 5. Keputusan
- [ ] Adopt ke branch utama
- [ ] Eksperimen lanjutan: [deskripsi]
- [ ] Revert / coba pendekatan lain

---
```

---

## 📚 Referensi Online

| Sumber | Topik | Keterangan |
|---|---|---|
| [Structuring ML Projects (Andrew Ng)](https://www.coursera.org/learn/machine-learning-projects) | Kursus | Wajib baca untuk mentalitas production |
| [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) | Template project | Template terkenal untuk struktur project data science |
| [How to organize deep learning project](https://neptune.ai/blog/how-to-organize-deep-learning-projects) | Blog | Tips praktis organization |
| [Papers: GAN](https://arxiv.org/abs/1406.2661) | Paper asli | Read skimming untuk paham konsep |
| [WGAN-GP paper](https://arxiv.org/abs/1704.00028) | Paper lanjutan | Training GAN yang lebih stabil |

---

## ✅ Checklist Kompetensi

- [ ] Struktur project rapi + teruji, bukan notebook liar
- [ ] Bisa menjelaskan cara kerja GAN dalam 3 menit
- [ ] Punya template experiment log pribadi
- [ ] Tahu cara pisahkan notebook dari module
- [ ] Paham minimal 3 alasan kenapa struktur project itu penting
- [ ] Implementasi GAN sederhana dari nol
- [ ] Bisa tulis experiment log yang lengkap & bermanfaat
