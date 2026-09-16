# 📄 Cheatsheet — Deep Learning & TensorFlow (Bab 4)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Neuron & Aktivasi — Tabel Hafalan

```
z = w·x + b  →  a = aktivasi(z)      # neuron = weighted sum + aktivasi
```

| Aktivasi | Rumus | Rentang | Gradien | Dipakai di |
|---|---|---|---|---|
| ReLU | `max(0, z)` | `[0, ∞)` | `1` jika z>0, else `0` | hidden layer (default) |
| Sigmoid | `1/(1+e⁻ᶻ)` | `(0, 1)` | `a(1−a)` | output biner |
| Softmax | `eᶻᵢ/Σeᶻⱼ` | `(0,1)`, Σ=1 | (dipakai bareng CE) | output multikelas |
| Linear | `z` | `(-∞, ∞)` | `1` | output regresi |

```python
# Sigmoid stabil — exp tidak pernah menerima angka positif besar
def sigmoid_stabil(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos]); out[~pos] = ez / (1.0 + ez)
    return out

# Softmax stabil — kurangi max per baris (konstanta saling kancel)
def softmax_stabil(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)
```

---

## 2. Backprop Dense Layer — 4 Baris yang Wajib Lancar

```
Forward :  z = X @ W + b        a = f(z)
Output  :  dz = (a_out − y) / m              # sigmoid + BCE (satu langkah!)
Hidden  :  dz = (delta @ Wᵀ) * f'(z)         # gradien mengalir mundur
Params  :  dW = Xᵀ @ dz        db = dz.sum(0)
Update  :  W −= lr · dW        b −= lr · db
```

- Urutan: hitung `dz` dulu, baru `dW` — 90% bug backprop = urutan/transpos terbalik.
- Gradient check: bandingkan gradien analitik dengan selisih-hingga numerik `(L(w+h)−L(w−h))/2h`.
- Chain rule = "menelusuri mundur siapa yang salah" — tiap layer meneruskan gradien ke belakangnya.

---

## 3. Inisialisasi Bobot

| Init | std | Untuk |
|---|---|---|
| Nol | 0 | ❌ semua neuron simetris → belajar hal yang sama |
| Xavier/Glorot | `sqrt(1/fan_in)` | sigmoid / tanh |
| **He** | `sqrt(2/fan_in)` | **ReLU (default)** |

---

## 4. Training Loop Universal (numpy ↔ Keras)

| Langkah | numpy | Keras |
|---|---|---|
| forward + loss | `loss_fn(y, model(X))` | otomatis di `model.fit()` |
| gradien | `tape.gradient(loss, vars)` | otomatis (backprop) |
| update | `optimizer.apply_gradients(...)` | otomatis |
| batching + shuffle | `split_batches` + shuffle per epoch | `model.fit(..., batch_size=32)` |

```python
# Custom loop (fondasi fine-tuning Bab 13)
for epoch in range(epochs):
    for xb, yb in batches:
        with tf.GradientTape() as tape:
            loss = loss_fn(yb, model(xb, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

- 1 **epoch** = sekali melihat seluruh data; 1 **batch** = sekali update bobot.
- `training=True`/`False` penting: Dropout & BatchNorm berperilaku beda saat inference.

---

## 5. Membaca Kurva Training

| Gejala | Diagnosis | Obat |
|---|---|---|
| train & val loss sama-sama tinggi | underfitting | model lebih besar / epoch lebih lama |
| train turun, val naik | **overfitting** | early stopping, dropout, augmentasi, data lebih banyak |
| loss naik / NaN | lr terlalu besar | kecilkan lr 10× |
| loss turun sangat lambat | lr terlalu kecil | besarkan lr 10× |

```
EarlyStopping(monitor='val_loss', patience=k, restore_best_weights=True)
ModelCheckpoint(...)   # simpan model terbaik otomatis
ReduceLROnPlateau(...) # lr dikali factor kalau stagnan
```

- Pantau **val** loss (sinyal jujur), bukan train loss.
- `patience` terlalu kecil → berhenti sebelum model sempat membaik.

---

## 6. Keras Quick Reference

```
Sequential API    → alur lurus (90% kasus)
Functional API    → multi-input/output, skip connection, branch
Subclassing       → custom forward logic (paling fleksibel, paling verbose)

Dense(64, activation='relu')       # hidden
Dense(1,  activation='sigmoid')    # output biner        + binary_crossentropy
Dense(k, activation='softmax')     # output k kelas      + (sparse_)categorical_crossentropy
Dense(1)                           # output regresi      + mse
```

- Label integer `0..k-1` → `sparse_categorical_crossentropy`; one-hot → `categorical_crossentropy`.
- Custom layer: `__init__` (hyperparams) → `build` (buat weights) → `call` (forward) → `get_config`.

---

## 7. Skip Connection & BatchNorm

```
output = BN(Add([input, F(input)]))     # ResNet block
```

- Skip connection mengatasi vanishing gradient di jaringan dalam — pola yang sama dipakai Transformer (Bab 9).
- BatchNorm menstabilkan distribusi aktivasi → training lebih cepat & tahan terhadap lr besar.

---

## 8. Koneksi ke Bab Lain

- **Bab 3**: logistic regression = NN dengan **nol hidden layer** — model linear, mentok di data non-linear.
- **Bab 4 project**: `nn_mini` = versi mini semua rumus di atas; kalau bisa jelaskan tanpa lihat kode, kamu siap.
- **Bab 5**: CNN = konvolusi menggantikan Dense untuk gambar (bobot dibagi, fitur lokal).
- **Bab 9**: Transformer = attention + skip connection + BatchNorm/LayerNorm — semua blok yang dipelajari di sini.
- **Bab 13**: fine-tuning LLM = custom training loop yang sama, hanya lebih besar.
