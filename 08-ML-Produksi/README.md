# Bab 8 — ML Produksi: Evaluasi, GAN, Struktur Proyek

> Bab transisi: dari "model jupyter yang jalan" ke "model siap dipakai". Plus generative model (GAN) sebagai pengantar dunia generative AI sebelum LLM.

## 🎯 Tujuan Belajar
- Struktur ML project & menulis laporan/eksperimen yang layak baca.
- Metrik produksi: RMSE/MAE/R², **skill score** untuk keputusan ADOPT/TOLAK.
- Metrik keputusan: confusion matrix, precision/recall/F1, **threshold tuning** di calib.
- **Kalibrasi probability**: ECE & Platt scaling.
- Generative model: GAN dari nol (dan peta WGAN-GP/diffusion).
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

### 1.2 Reproduksibilitas & Persistensi Model

> Kode lengkap dari nol ada di `01_lab_ml_produksi.ipynb` Bagian 1–2 dan
> `project-starter-evaluasi-produksi/` (modul `persist.py`).

#### Split Tiga Arah — Fondasi Evaluasi Jujur

```
train  (±80%)  : fit parameter model
calib  (±10%)  : ambil KEPUTUSAN — threshold, Platt scaling, pilih model
test   (±10%)  : lapor hasil — SEKALI. Setelah itu test "habis".

Menyetel apa pun di test = test berubah jadi calib kedua.
Angkanya optimis, tidak sah, dan tidak bisa dikoreksi setelah dilaporkan.
```

Normalisasi/scaler ikut aturan yang sama: `fit` di train saja, `transform` ke
calib/test. Statistik scaler dari seluruh data = data leakage halus tapi sistematis.

#### Artefak yang Bisa Dibukti

```
1. Deterministik : input sama → file byte sama   (uji: sha256 dua kali save)
2. Round-trip    : save → load → prediksi identik bit-per-bit (array_equal, bukan allclose)
3. Ber-hash      : serving memverifikasi hash artefak sebelum dipakai

Klaim tanpa bukti = kebalikan mentalitas produksi.
```

---

### 1.3 Metrik Produksi & Skill Score

#### Regresi: Tiga Metrik Wajib Bisa Dihitung dari Nol

```
RMSE = sqrt(mean(e²))       # menghukum error besar; RMSE >= MAE selalu
MAE  = mean(|e|)            # satuan asli, robust
R²   = 1 − SS_res/SS_tot    # vs baseline mean; BOLEH NEGATIF
```

#### Skill Score — Keputusan ADOPT/TOLAK

```
skill = 1 − RMSE_model / RMSE_baseline        # baseline = HAM (prediksi konstan mean train)

skill > 0   → ADOPT  (model menambah nilai)
skill ≈ 0   → setara baseline (tidak menambah apa-apa)
skill < 0   → TOLAK  (model MENAMBAH noise ke proses)
```

> **Contoh nyata di lab:** model "negatif" (koefisien salah arah) punya RMSE 6.68
> vs HAM 4.83 → skill −0.38. Keputusan yang benar: TOLAK, pakai baseline dulu.
> RMSE absolut tanpa pembanding = iklan, bukan evaluasi.

---

### 1.4 Metrik Keputusan: Confusion Matrix, P/R/F1, Threshold

```
                aktual 1    aktual 0
pred ≥ thr   |   TP     |    FP    |    precision = TP/(TP+FP)
pred < thr   |   FN     |    TN    |    recall    = TP/(TP+FN)
                                        F1 = 2PR/(P+R)   (harmonic mean)
```

- Konvensi wajib: `>= thr` = positif; pembagi nol → metrik 0.0 (bukan crash di serving).
- Accuracy menipu di kelas timpang: prevalence 1% → "selalu negatif" akurasi 99%, recall 0%.

#### Threshold = Kenop Bisnis

| Biaya dominan | Geser threshold | Contoh |
|---|---|---|
| FN mahal (kasus terlewat) | turunkan → recall naik | screening medis, fraud |
| FP mahal (alarm palsu) | naikkan → precision naik | spam filter, auto-reject |

**Protokol:** nilai threshold dipilih di **CALIB** (mis. maksimasi F1), evaluasi
SEKALI di **TEST**. Di lab: F1 "bocor" (tune di test) 0.944 vs jujur 0.932 —
selisih kecil, tapi arahnya selalu optimis dan mekanismenya salah.

---

### 1.5 Kalibrasi Probability: ECE & Platt Scaling

```
Model bilang "90% yakin" — apakah benar-benar 90% benar?

bin   : idx = min(int(p · n_bins), n_bins − 1)     # p=1.0 masuk bin terakhir
ECE   = Σ (n_bin / n) · |acc − conf|               # bin kosong dilewati
```

- ECE tinggi = over/under-confidence sistematis → keputusan berbasis risiko salah.
- **Platt scaling** = regresi logistik 1D `p = sigmoid(a·s + b)`, belajar dari
  CALIB — memperbaiki kalibrasi tanpa menyentuh model asli. Di lab: ECE 0.074 → 0.048.
- Kapan penting: harga risiko, triase, alokasi, keputusan manusia.
  Kapan cukup tanpa kalibrasi: ranking murni (top-k) — transformasi monoton tak mengubah urutan.

---

### 1.6 GAN — Generative Adversarial Networks

#### Konsep Dasar: Duel Dua Jaringan

```
GAN = Generator (G) vs Discriminator (D) dalam "permainan"

  Generator (G):
    - Mulai dari noise random z ~ N(0,1)
    - Coba buat sampel "nyata"
    - Tujuannya: MENIPU D

  Discriminator (D):
    - Lihat sampel (asli atau palsu)
    - Tebak: asli atau palsu?
    - Tujuannya: TEPAT MEMBEDA

  Proses:
    1. G buat sampel palsu
    2. D menilai
    3. D di-update dari kesalahannya
    4. G di-update lewat gradien yang MELEWATI D (dibekukan)
    5. Ulangi → G makin bagus → keduanya berakhir di EQUILIBRIUM
```

> **Intuisi:** pemalsu vs detektif pemalsu. Berbeda dengan supervised learning:
> tidak ada loss yang monoton turun — dua pemain saling menjaga, keduanya
> konvergen ke titik di mana D tak lagi bisa membedakan (D(real) ≈ D(fake) ≈ 0.5).

#### Loss & Gradien (inti matematikanya)

```
loss D = −mean log D(real) − mean log(1−D(fake))     (max teoretis = 2·log 2)
loss G = −mean log D(fake)                           NON-saturating:
                                                     gradien tetap kuat saat D kecil

Update bergantian: D dulu (real, fake), lalu G.
Gradien G melewati jaringan D: logit D → hidden D → x_fake → parameter G.
Init D nol = saddle point (semua gradien nol) → init acak kecil.
```

#### GAN 1D dari Nol (di lab) vs GAN Penuh (di bawah)

Lab Bab 8 membangun GAN **paling kecil yang masih mengajarkan semua konsep**:
G hanya punya 2 parameter (mu, sigma) dan D 1→8→1. Kebut-kebutannya terlihat
persis: G menemukan mean data (−2.0 → 2.1, target 2.0). Versi penuh di bawah
(arsitektur DCGAN-style untuk gambar) memakai prinsip yang sama persis.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0 * 2.0 - 1.0   # ke [-1, 1] agar cocok tanh
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)

BATCH_SIZE = 256
noise_dim = 100

# Generator: noise → gambar
def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))
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
generator.summary()
discriminator.summary()

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    # non-saturating: beri "hadiah" kepada G saat berhasil menipu D
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

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

def generate_and_save_images(model, epoch, noise):
    predictions = model(noise, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

# Contoh: lihat generator sebelum training
noise = tf.random.normal([16, noise_dim])
generated_before = generator(noise, training=False)
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_before[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.suptitle('Generator Output - Before Training')
plt.show()
```

#### Mode Collapse & Diagnostik

```
Mode collapse : G berkapasitas besar memetakan banyak z ke output yang SAMA
                (yang paling menipu D) → keragaman hilang.

Deteksi       : histogram fake vs real per epoch; std fake ≪ std real;
                D(fake) turun terus sementara D(real) naik.
Remedi        : kurangi kapasitas G, minibatch diversity, atau arsitektur lain
                (VAE / diffusion) untuk task yang butuh keragaman.
```

---

### 1.7 Kapan Pakai Apa — Decision Guide Terstruktur

| Situasi | Pertanyaan Kunci | Pilihan | Alasan |
|---|---|---|---|
| **Data tabular, rows > 10rb** | Perlu interpretability? | XGBoost/LightGBM | Performa bagus, feature importance tersedia |
| **Data tabular, butuh kepastian** | Prediksi probabilistic? | Logistic Regression (+ Platt scaling) | Interpretable, kalibrasi probabilitas bagus |
| **Gambar, data sedikit (< 1000/kelas)** | Ada pre-trained model? | Transfer Learning (CNN) | Mulailah dari feature extractor yang sudah belajar |
| **Gambar, data banyak (> 100rb)** | Compute tersedia? | Latih dari nol / Fine-tune besar | Custom architecture bisa dioptimasi untuk task spesifik |
| **Teks pendek, klasifikasi volume tinggi** | Latency sensitive? | Embedding + Logistic Regression | Cepat inference, cukup untuk banyak task |
| **Teks, kualitas tinggi butuh reasoning** | Budget cukup? | LLM (Bab 9+) | Kualitas terbaik untuk task kompleks |
| **Teks, privasi/offline wajib** | Bisa pakai open-weight? | Open-weight model (Bab 13) | Data tidak keluar dari infrastructure |
| **Anomaly detection, tanpa label** | Unsupervised yang mana? | Isolation Forest / Autoencoder | Tidak butuh label; deteksi nilai ekstrem |
| **Rekomendasi, ada interaksi user** | Collaborative atau content-based? | Hybrid approach | Studi kasus spesifik; sering perlu custom |

---

### 1.8 Mentalitas Production

```
┌─────────────────────────────────────────────────────────────┐
│  MENTALITAS RESEARCHER                  MENTALITAS ENGINEER │
│                                                             │
│  "Model ini akurasi 98%!"               "Model ini 98% di   │
│   → Senang dengan angka                 test yang JUJUR,    │
│                                         skill > 0 vs base-  │
│                                         line, inference     │
│                                         < 50ms, memory      │
│                                         < 100MB, probabilit-│
│                                         asnya terkalibrasi, │
│                                         artefak ber-hash,   │
│                                         bisa monitor, punya │
│                                         fallback, dokumentasi ada" │
└─────────────────────────────────────────────────────────────┘
```

> **Pesan kunci:** Di production, akurasi adalah SATU dari banyak metrik. Latensi, biaya, reliabilitas, maintainability, dan observability sama pentingnya.

---

### 1.9 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 8:
  1. Struktur project: data/, notebooks/, src/, tests/, reports/
  2. Pisahkan eksplorasi (notebook) dari produksi (module/code)
  3. Split tiga arah: train fit | calib putuskan | test lapor SEKALI
  4. Persistensi terbukti: hash deterministik + round-trip test
  5. Skill score: keputusan ADOPT/TOLAK vs baseline (HAM), bukan angka absolut
  6. P/R/F1 + threshold = kenop bisnis, dipilih di CALIB
  7. Kalibrasi probability: ECE + Platt scaling dari calib
  8. GAN: dua pemain, non-saturating loss, equilibrium (bukan "menang")
  9. Experiment log: hipotesis → setup → hasil → keputusan
```

---

## 2. Latihan Praktis

> Latihan penuh (bertahap, dengan cek otomatis) ada di `01_lab_ml_produksi.ipynb`.
> Ringkasannya:

### Latihan 1: Refactor Notebook ke Project Structure

**Tujuan:** Pindahkan kode dari notebook ke struktur project yang rapi.

**Langkah 1 — Buat struktur project:**

```bash
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
    df = df.dropna()
    X = df.drop('target', axis=1).values
    y = df['target'].values
    return X, y

def normalize_features(X_train, X_val=None, X_test=None):
    """Normalize fitur dengan StandardScaler — fit di TRAIN saja!"""
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
    import joblib
    joblib.dump(model, path)

def load_model(path: str):
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
        assert np.allclose(result['X_train'].mean(axis=0), 0, atol=1e-10)

    def test_std_one(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_features(X_train)
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
    df = load_dataset(data_path)
    train_df, val_df = get_train_val_split(df)

    X_train, y_train = create_features(train_df)
    X_val, y_val = create_features(val_df)

    normalized = normalize_features(X_train, X_val=X_val)
    X_train_scaled = normalized['X_train']
    X_val_scaled = normalized['X_val']

    model, metrics = train_model(X_train_scaled, y_train, X_val_scaled, y_val)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics['classification_report'])

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
Gunakan arsitektur & training loop dari **1.6** di atas. Tambahkan:

```python
# Training loop (lengkapi dari kode 1.6)
EPOCHS = 50
test_noise = tf.random.normal([16, noise_dim])

history_gen, history_disc = [], []
for epoch in range(EPOCHS):
    for batch, images in enumerate(train_dataset):
        gen_loss, disc_loss = train_step(images)
        if batch % 100 == 0:
            history_gen.append(float(gen_loss))
            history_disc.append(float(disc_loss))
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Gen={float(gen_loss):.4f}, Disc={float(disc_loss):.4f}")
        generate_and_save_images(generator, epoch + 1, test_noise)

# Plot loss curve — PERHATIKAN: tidak ada loss yang monoton turun!
plt.figure(figsize=(10, 4))
plt.plot(history_gen, label='Generator Loss', alpha=0.7)
plt.plot(history_disc, label='Discriminator Loss', alpha=0.7)
plt.xlabel('Training Step (every 100 batches)')
plt.ylabel('Loss')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('GAN Training Loss')
plt.show()
```

> **Tulis:** Apa yang terjadi dengan generator loss dan discriminator loss selama
> training? Kenapa keduanya berayun mengelilingi nilai tertentu, bukan turun ke 0?
> (hint: equilibrium — bandingkan dengan GAN 1D di lab Bagian 8.)

---

### Latihan 3: Template Experiment Log

**Tujuan:** Dokumentasikan 3 eksperimen terakhir dengan template yang rapi.

```markdown
# Eksperimen: [Judul singkat]

**Tanggal:** [YYYY-MM-DD]
**Dilakukan oleh:** [nama]
**Branch/Commit:** [git reference]
**Seed & data:** [seed, versi data, split]      ← tanpa ini, hasil tak bisa direproduksi

## 1. Hipotesis
> [Apa yang kita coba? Misalnya: "Menambah dropout 0.3 akan menurunkan val loss
> karena indikasi overfit di eksperimen sebelumnya."]

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
| Val RMSE | X | Y | −Z |
| Skill vs baseline | +0.X | +0.Y | ... |

## 4. Analisis
- [Temuan menarik]
- [Apakah hipotesis terbukti?]
- [Apa yang unexpected?]

## 5. Keputusan
- [ ] Adopt ke branch utama (skill > 0, threshold dari calib)
- [ ] Eksperimen lanjutan: [deskripsi]
- [ ] Revert / coba pendekatan lain

## 6. File Artifact
- Model checkpoint: [path] + [hash]
- Plot: [path]
- Log training: [path]
```

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_ml_produksi.ipynb` | Lab praktikum: 9 bagian — reproduksibilitas (split 3 arah + hash), persistensi & round-trip, RMSE/MAE/R² + baseline HAM, skill score, confusion + P/R/F1, threshold calib-vs-test (demo bocor), kalibrasi probability (ECE + Platt), GAN 1D dari nol | Kerjakan setelah baca materi inti |
| `02_kuis_ml_produksi.ipynb` | 12 soal (8 PG + 4 coding), 23 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_ml_produksi.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-ml-produksi.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-evaluasi-produksi/` | Proyek end-to-end: pipeline evaluasi produksi dari nol — metrik + skill score + persistensi/hash + threshold + kalibrasi + GAN 1D (TDD, 52 test) + starter/solusi + RUBRIK | Setelah kuis ≥ 18/23 |

> File `_gen_*.py`, `_verify_*.py`, `_calib.py` = skrip generator/verifier internal
> (angka di notebook terkunci seed & terverifikasi otomatis). Tidak perlu dibuka
> untuk belajar, tapi silakan baca kalau ingin melihat cara materinya dikalibrasi.

---

## 📚 Referensi Online

| Sumber | Topik | Keterangan |
|---|---|---|
| [Structuring ML Projects (Andrew Ng)](https://www.coursera.org/learn/machine-learning-projects) | Kursus | Wajib baca untuk mentalitas production |
| [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) | Template project | Template terkenal untuk struktur project data science |
| [How to organize deep learning project](https://neptune.ai/blog/how-to-organize-deep-learning-projects) | Blog | Tips praktis organization |
| [Paper: GAN](https://arxiv.org/abs/1406.2661) | Paper asli | Baca skimming untuk paham konsep |
| [WGAN-GP paper](https://arxiv.org/abs/1704.00028) | Paper lanjutan | Training GAN yang lebih stabil |
| [DTU Deep Learning — GAN notes](https://palash89.github.io/notes/Intro_GANs.html) | Catatan | Ringkasan varian GAN (DCGAN, WGAN-GP, dll) |
| [Scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) | Dokumentasi | Pembanding untuk metrik yang kamu bangun dari nol |

---

## ✅ Checklist Kompetensi

- [ ] Struktur project rapi + teruji, bukan notebook liar
- [ ] Menjelaskan & menerapkan split tiga arah (train/calib/test) dan kenapa test disentuh sekali
- [ ] Bisa membuktikan artefak siap serving: hash deterministik + round-trip test
- [ ] Menghitung RMSE/MAE/R² dan skill score dari nol; keputusan ADOPT/TOLAK dari skill
- [ ] Menghitung confusion matrix, P/R/F1 dari nol; memilih threshold di calib sesuai biaya bisnis
- [ ] Menjelaskan ECE dan memperbaiki kalibrasi dengan Platt scaling
- [ ] Bisa menjelaskan cara kerja GAN dalam 3 menit (dua pemain, non-saturating, equilibrium)
- [ ] Implementasi GAN sederhana dari nol (setidaknya versi 1D di lab)
- [ ] Punya template experiment log pribadi yang lengkap (seed, hash, keputusan)
