# Bab 5 — Computer Vision

> Vision adalah area deep learning paling visual untuk belajar: hasilnya langsung terlihat, dan konsep augmentasi/transfer learning berlaku juga di domain lain.

## 🎯 Tujuan Belajar

- Pipeline data gambar: loading, preprocessing, validasi.
- CNN: konvolusi, pooling, arsitektur klasik.
- **Transfer learning** — skill yang paling sering dipakai di industri.
- Augmentasi data untuk generalisasi.

## 1. Materi Inti

> Computer Vision adalah area yang paling visual untuk belajar — hasilnya langsung terlihat, dan konsep augmentasi/transfer learning berlaku juga di domain lain.

---

### 1.1 Preprocessing Data Gambar

#### Pipeline Dasar

```
Gambar mentah (file JPEG/PNG)
    ↓
  Load → decode → resize → rescale → augment → batch
```

```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

# ImageDataGenerator (cara klasik)
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    'data/train',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)

val_ds = image_dataset_from_directory(
    'data/train',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

# Prefetch untuk performa
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
```

**Mengapa semua step ini penting:**

| Step             | Sebab                                                                    |
| ---------------- | ------------------------------------------------------------------------ |
| Resize           | Model butuh input ukuran sama                                            |
| Rescale (0-1)    | Stabilisasi training; pixel 0-255 terlalu besar untuk gradient           |
| Augment          | Mencegah overfitting; model belajar invarian (posisi, cahaya, orientasi) |
| Cache & prefetch | GPU/D/TPU tidak menunggu I/O disk                                        |

---

### 1.2 CNN dari Nol — Cara Kerja

#### Konvolusi: Filter yang Mendeteksi Pola

```
[Misal: detektor "tepi vertikal"]

Filter 3x3:            Gambar asli:          Hasil konvolusi:
  [[-1, 0, 1],        [[100, 120, 130],    [[  (tersa
   [-1, 0, 1],         [110, 120, 140],      (deteksi
   [-1, 0, 1]]         [100, 120, 130]]      tepi         
                                       ^ vertikal])

Filter dikonvolusi dengan gambar → setiap posisi filter menghasilkan 1 angka.
Area dengan pola yang cocok → output tinggi.
```

**Intuisi:**

- Tujuan konvolusi: ekstrak "fitur lokal" (tepi, tekstur, pola kecil)
- Multiple filter = multiple pola yang dicari sekaligus
- Semakin dalam layer, semakin abstrak fitur (layer awal: tepi; layer akhir: mata, roda, dll)

```python
# Contoh sederhana CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
  
    # Conv layer 1: ekstrak fitur dasar
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
  
    # Conv layer 2: fitur lebih kompleks
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
  
    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

#### Pooling — Mengapa Perlu?

- **MaxPooling**: ambil nilai maks di region kecil → "survior" fitur terkuat
- Tujuan: invariance terjemahan (posisi sedikit berubah → output sama), reduce dimensi, reduce komputasi
- Alternatif: AveragePooling, strided convolution

---

### 1.3 Augmentasi Data

> Augmentasi = cara "memperbanyak data" tanpa pengambilan gambar baru.

```python
# Pipeline augmentasi
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),        # kiri-kanan (jangan pakai 'vertical' untuk wajah)
    layers.RandomRotation(0.1),            # rotasi ringan ±10%
    layers.RandomZoom(0.1),                # zoom in/out ±10%
    layers.RandomContrast(0.1),            # kontras sedikit berubah
], name='data_augmentation')

# Contoh: lihat efek augmentasi
import matplotlib.pyplot as plt

img = next(iter(train_ds))[0][0]  # satu gambar dari dataset

plt.figure(figsize=(15, 5))
for i in range(6):
    augmented = data_augmentation(img, training=True)
    plt.subplot(2, 3, i+1)
    plt.imshow(augmented.numpy().astype('uint8'))
    plt.axis('off')
plt.suptitle('Contoh Augmentasi')
plt.show()
```

**Aturan augmentasi:**

- Pilih transformasi yang masuk akal untuk domain:
  - Gambar medis: jangan flip horizontal kalau anatomi berorientasi
  - Dokumen: jangan rotasi 90°
  - Hewan: flip horizontal OK (kucing bisa menghadap kiri atau kanan)
- Jangan augmentasi berlebihan → model bingung antara data asli dan data "tidak realistis"

---

### 1.4 Transfer Learning ⭐ — Skill Paling Praktis

> **Pola mental terpenting di AI Engineering:** "Jangan latih dari nol jika sudah ada yang melatih."

sama seperti kamu tidak melatih LLM sendiri, tapi pakai pre-trained + adaptasi.

#### Decision Diagram: Kapan Pakai Transfer Learning

```
Datamu banyak? (> 10.000 gambar/kelas, training infrastructure lengkap)
  ├── YA → Bisa coba latih dari nol (hanya kalau punya alasan spesifik)
  └── TIDAK → ➜ Pakai transfer learning
             │
             └── Datamu sedikit? (< 1000 gambar/kelas)
                  ├── Ya → Fine-tune hanya classifier head (base di-freeze)
                  └── Sangat sedikit → Freeze base, train head saja, plus augmentasi agresif
```

#### Implementasi Transfer Learning

```python
# 1. Load base model (yang sudah dilatih di ImageNet)
base_model = keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,           # jangan pakai classifier asli
    weights='imagenet',
    pooling='avg'                 # global average pooling
)

# 2. Freeze base model
base_model.trainable = False

# 3. Build classifier di atas
model = keras.Sequential([
    base_model,
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')   # sesuaikan dengan jumlah kelas
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Latih head dulu (base masih freeze)
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 5. Optional: unfreeze sebagian base & fine-tune
base_model.trainable = True

# Hanya unfreeze layer yang lebih dalam (lower layers = fitur lebih umum, jangan diubah)
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # learning rate lebih kecil
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5)
```

> **Catatan:** Di step 5, learning rate dikurangi karena bobot base sudah di-close ke solusi bagus — langkah besar bisa merusak.

---

### 1.5 CNN Lanjutan

#### Arsitektur Klasik

| Arsitektur                    | Karakteristik                     | Kapan                                |
| ----------------------------- | --------------------------------- | ------------------------------------ |
| **LeNet-5** (1998)      | Sangat sederhana, 7 layer         | Sejarah; tidak dipakai produksi      |
| **AlexNet** (2012)      | 8 layer, relu, dropout            | Sejarah; mulai popularisasi CNN      |
| **VGG-16/19**           | Banyak layer 3x3, konvulasi padat | Transfer learning, tapi berat        |
| **ResNet** (2015)       | Skip connection, sangat dalam     | State-of-the-art lama; masih dipakai |
| **EfficientNet** (2019) | Compound scaling, efisien         | Balance akurasi vs biaya             |
| **MobileNet**           | DioConvs, sangat ringan           | Mobile/edge deployment               |

```python
# Coba berbagai arsitektur untuk lihat perbedaan
architectures = {
    'MobileNetV2': keras.applications.MobileNetV2,
    'EfficientNetB0': keras.applications.EfficientNetB0,
    'ResNet50': keras.applications.ResNet50,
}

for name, arch in architectures.items():
    base = arch(weights='imagenet', include_top=False, pooling='avg')
    base.trainable = False
  
    model = keras.Sequential([
        base,
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"{name}: {model.count_params():,} parameters")
```

#### Visualisasi Feature Map

```python
# Lihat apa yang "dilihat" layer tertentu
layer_outputs = [layer.output for layer in base_model.layers[:6]]
activation_model = keras.Model(inputs=base_model.input, outputs=layer_outputs)

sample_img = tf.random.normal((1, 224, 224, 3))
activations = activation_model(sample_img)

# Plot fitur layer 1 (biasanya edge & tekstur)
first_layer_activation = activations[0]
plt.figure(figsize=(15, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('Feature Maps - Layer 1 (Edge & Texture)')
plt.show()
```

---

### 1.6 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 5:
  1. Pipeline gambar: load → resize → rescale → augment → batch
  2. CNN: konvolusi (fitur lokal) + pooling (invarian & reduce)
  3. Augmentasi: flip, rotasi, zoom, contrast — pilih yang masuk akal
  4. Transfer learning: pakai model pre-trained → freeze → train head
  5. Fine-tune: unfreeze sebagian, learning rate kecil
  6. Arsitektur: VGG, ResNet, EfficientNet, MobileNet — tahu perbedaan
  7. Visualisasi feature map → paham apa yang di-ekstrak CNN
```

---

## 2. Latihan Praktis

---

### Latihan 1: Pipeline Gambar End-to-End

**Tujuan:** Bangun pipeline dari nol sampai model bisa memprediksi.

```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset (gunakan Fashion-MNIST karena ringan)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 2. Preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28) → (N, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# 3. Augmentasi untuk data training
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name='augmentation')

# 4. Bangun model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

# 5. Compile & train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# 6. Evaluasi
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Visualisasi hasil
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# 8. Prediksi sampel
predictions = model.predict(x_test[:5])
for i in range(5):
    pred_class = np.argmax(predictions[i])
    true_class = y_test[i]
    confidence = predictions[i, pred_class]
    status = "✓" if pred_class == true_class else "✗"
    print(f"Sample {i}: prediksi={pred_class}, true={true_class}, confidence={confidence:.2f} {status}")
```

---

### Latihan 2: Transfer Learning vs Dari Nol

**Tujuan:** Buktikan keunggulan transfer learning untuk data kecil.

```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Gunakan subset kecil dari CIFAR-10 (seolah-olah data terbatas)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Gunakan hanya 1000 sampel untuk simulasikan data kecil
x_train_small = x_train[:1000]
y_train_small = y_train[:1000]

# Model dari NUL (tanpa transfer learning)
def create_model_from_scratch(input_shape=(32, 32, 3), num_classes=10):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

# Transfer Learning dengan MobileNetV2
def create_transfer_model(input_shape=(32, 32, 3), num_classes=10):
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base.trainable = False
  
    return keras.Sequential([
        base,
        layers.Dense(num_classes, activation='softmax')
    ])

model_scratch = create_model_from_scratch()
model_transfer = create_transfer_model()

model_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_transfer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("=" * 50)
print("TRAINING MODEL DARI NUL (1000 sampel)")
print("=" * 50)
history_scratch = model_scratch.fit(
    x_train_small, y_train_small,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

print("\n" + "=" * 50)
print("TRAINING MODEL TRANSFER LEARNING (1000 sampel)")
print("=" * 50)
history_transfer = model_transfer.fit(
    x_train_small, y_train_small,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# Perbandingan hasil
scratch_test_acc = model_scratch.evaluate(x_test, y_test, verbose=0)[1]
transfer_test_acc = model_transfer.evaluate(x_test, y_test, verbose=0)[1]

print("\n" + "=" * 50)
print("HASIL AKHIR")
print("=" * 50)
print(f"Model dari nol:     {scratch_test_acc:.4f}")
print(f"Transfer learning:   {transfer_test_acc:.4f}")
print(f"Selisih:            {(transfer_test_acc - scratch_test_acc):.4f} ({'transfer' if transfer_test_acc > scratch_test_acc else 'dari nol'} lebih baik)")
```

> **Tulis:** Apa yang kamu pelajari dari perbandingan ini?

---

### Latihan 3: Diagnosis Overfitting

**Tujuan:** Pahami tanda overfitting dan cara memperbaikinya.

```python
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# SENGaja buat model yang overfitting
model_overfit = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_overfit.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train dengan epochs banyak untuk lihat overfitting
print("Training model yang SENGINT overfitting...")
history_overfit = model_overfit.fit(
    x_train[:5000], y_train[:5000],  # gunakan subset agar lebih cepat overfit
    epochs=30,
    validation_split=0.2,
    verbose=1
)

# Plot kurva untuk diagnosis
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_overfit.history['loss'], label='Train Loss')
plt.plot(history_overfit.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Overfitting: Val Loss naik setelah beberapa epoch')

plt.subplot(1, 2, 2)
plt.plot(history_overfit.history['accuracy'], label='Train Acc')
plt.plot(history_overfit.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Overfitting: Val Accuracy stagnan/turun')

plt.tight_layout()
plt.show()

# Perbaiki dengan:
# 1. Dropout
# 2. Data augmentation
# 3. Early stopping
print("\n" + "=" * 50)
print("MEMPERBAIKI OVERSAMPLING dengan Dropout & Augmentation")
print("=" * 50)

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
], name='aug')

model_fixed = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    data_augmentation,
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),  # ← tambah dropout
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # ← tambah dropout
    layers.Dense(10, activation='softmax')
])

model_fixed.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_fixed = model_fixed.fit(
    x_train[:5000], y_train[:5000],
    epochs=30,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ],
    verbose=1
)

# Bandingkan
print(f"\nModel overfit - val accuracy terakhir: {history_overfit.history['val_accuracy'][-1]:.4f}")
print(f"Model fixed   - val accuracy terakhir: {history_fixed.history['val_accuracy'][-1]:.4f}")
```

> **Tulis:** Apa perbedaan kurva antara model overfit dan model yang sudah diperbaiki?

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_computer_vision.ipynb` | Lab praktikum: 10 bagian — konvolusi (forward + backward + gradient check), max-pool, augmentasi aman-label, CNNMini vs MLP, transfer learning (numpy) | Kerjakan setelah baca materi inti |
| `02_kuis_computer_vision.ipynb` | 10 soal (PG + coding), 23 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_computer_vision.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-computer-vision.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-cnn-mini/` | Proyek end-to-end: CNN mini dari nol — conv/pool/augment + training + evaluasi (TDD, 42 test) + starter/solusi + RUBRIK | Setelah kuis ≥ 18/23 |

---

## 📚 Referensi Online

| Sumber                                                                                  | Topik             | Keterangan                                       |
| --------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------ |
| [TensorFlow Image Tutorials](https://www.tensorflow.org/tutorials/images/classification) | Panduan resmi     | Step-by-step dengan contoh                       |
| [Transfer Learning Guide (Keras)](https://keras.io/guides/transfer_learning/)            | Transfer learning | Penjelasan mendalam + praktik                    |
| [CS231n — Stanford](https://cs231n.github.io/)                                          | Kuliah CNN        | Materi paling komprehensif untuk computer vision |
| [Papers with Code](https://paperswithcode.com/)                                          | State-of-the-art  | Lihat performa terbaru di berbagai task          |

---

## ✅ Checklist Kompetensi

- [ ] Membangun pipeline gambar end-to-end (disk → model → prediksi)
- [ ] Menjelaskan kenapa transfer learning bekerja (fitur umum di layer awal)
- [ ] Mendiagnosis overfitting dari kurva train/val dan memperbaikinya
- [ ] Membandingkan transfer learning vs training dari nol untuk data kecil
- [ ] Pilih augmentasi yang sesuai untuk domain gambar tertentu
- [ ] Tahu kapan pakai MobileNetV2 vs EfficientNet vs ResNet
