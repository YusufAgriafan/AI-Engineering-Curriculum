# 📄 Cheatsheet — Computer Vision (Bab 5)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Ukuran Output Konvolusi — Rumus yang Wajib Refleks

```
out = floor((H + 2·pad − k) / stride) + 1
```

| Input | Kernel | Pad | Stride | Output |
|---|---|---|---|---|
| 32 | 3 | 0 | 1 | 30 |
| 32 | 3 | 1 | 1 | 32 (`'same'` utk kernel ganjil) |
| 32 | 3 | 0 | 2 | 15 |
| 28 | 3 | 1 | 2 | 14 |
| 32 | 5 | 0 | 2 | 14 |

- `//` (pembagian bulat) bukan `/` — stride yang tak membagi habis **membulatkan ke bawah**.
- `pad=(k−1)/2`, stride 1 → ukuran tetap. Itulah `padding='same'` untuk kernel ganjil.
- Konvolusi di ML = **cross-correlation** (kernel tidak dibalik); pembalikan justru muncul di backward.

```python
# Konvolusi valid (tanpa pad, stride 1) — pola inti
def conv_valid(img, K):
    H, W = img.shape; kh, kw = K.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * K)
    return out

# Versi cepat tanpa loop (lab Bagian 4)
win = sliding_window_view(pad_zero(img, pad), K.shape)[::stride, ::stride]
out = np.tensordot(win, K, axes=([2, 3], [0, 1]))
```

---

## 2. Backward Konvolusi — 2 Bagian

```
dK = Σ_posisi dOut[p] · jendela_input[p]     # akumulasi: tiap jendela berkontribusi ke K yang sama
dX = full-conv(dilasi(dOut, stride), K[::-1, ::-1])  # sebar gradien, kernel dibalik
```

- `dK` = konvolusi input dengan dOut; `dX` = konvolusi dOut-yang-didilasi dengan **K dibalik 180°**.
- 90% bug backward = lupa dilasi stride atau lupa membalik kernel.
- Gradient check: bandingkan dengan selisih-hingga `(L(w+h)−L(w−h))/2h` — lab Bagian 4 membuktikannya.

---

## 3. Max-Pool 2x2 — Forward & Backward

```python
# Forward: tanpa loop, max di dua sumbu dalam sekaligus
pooled = x.reshape(H//2, 2, W//2, 2).max(axis=(1, 3))

# Backward: gradien hanya ke posisi ARGMAX tiap jendela
idx = np.unravel_index(np.argmax(win), win.shape)
dX[2*i + idx[0], 2*j + idx[1]] += dOut[i, j]
```

- Pooling **tanpa parameter**; invarian translasi-kecil + memperkecil representasi 4×.
- Konsekuensi argmax: `dX.sum() == dOut.sum()`; seri = subgradien (pilih satu pemenang, deterministik).

---

## 4. Augmentasi — Aturan Emas: Jangan Rusak Label

| Transformasi | Aman untuk | Bahaya untuk |
|---|---|---|
| Flip horizontal | hewan, objek umum | teks, digit, logo |
| Flip vertikal | hampir tidak pernah | MNIST (6↔9), wajah, teks |
| Shift 1–2 px | hampir semua | jika objek mepet tepi (terpotong) |
| Rotasi ±10° | objek umum | dokumen, digit (2↔7 miring) |
| Noise/contrast ringan | hampir semua | gambar medis (memori) |

```python
# Shift dengan zero-fill — slice saling-melengkapi, input TIDAK dimutasi
out = np.zeros_like(img)
ys, xs = slice(max(0, -dy), min(H, H - dy)), slice(max(0, -dx), min(W, W - dx))
yd, xd = slice(max(0, dy), min(H, H + dy)), slice(max(0, dx), min(W, W + dx))
out[yd, xd] = img[ys, xs]
```

- Augmentasi diterapkan **hanya saat training** (`training=True`), bukan saat evaluasi.
- Augmentasi berlebihan = model belajar dari data "tidak realistis".

---

## 5. Kenapa CNN, Bukan MLP, untuk Gambar

| | Dense/MLP | Conv |
|---|---|---|
| Bobot | tiap posisi sendiri | **dibagi** (weight sharing) |
| Fitur | global | **lokal** (tepi → tekstur → bagian → objek) |
| Pergeseran objek | harus belajar ulang | relatif tahan (invarian) |
| Parameter gambar 8x8 | 64×h | hanya k×k per filter |

- Eksperimen lab Bagian 8: CNN mengalahkan MLP di ujian posisi **terlarang** (baris/kolom yang tak pernah dilihat saat training) — bukti generalisasi dari weight sharing.

---

## 6. Transfer Learning — Keputusan Cepat

```
Datamu banyak?  ── YA ──→ boleh latih dari nol (kalau ada alasan spesifik)
      │ TIDAK
      ▼
Pakai base pre-trained ImageNet:
  1. base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
  2. base.trainable = False          # freeze
  3. Dense head + Dropout            # latih head dulu (lr normal)
  4. (opsional) unfreeze sebagian + lr KECIL (10× lebih kecil)   # fine-tune
```

- `< 1k gambar/kelas` → freeze total, augmentasi agresif. Unfreeze semua + lr besar = **catastrophic forgetting**.
- Label integer → `sparse_categorical_crossentropy`; one-hot → `categorical_crossentropy`.
- Kenapa bekerja: layer awal belajar fitur **umum** (tepi, tekstur) — sama untuk hampir semua domain gambar.

---

## 7. Arsitektur Klasik — Pilih yang Mana

| Arsitektur | Ciri | Pakai saat |
|---|---|---|
| VGG-16 | polos, 3x3 bertumpuk, berat | belajar struktur; baseline |
| ResNet | skip connection, sangat dalam | default solid, masih relevan |
| EfficientNet | compound scaling | balance akurasi vs biaya |
| MobileNet | depthwise separable conv, ringan | mobile/edge deployment |

---

## 8. Mendiagnosis Kurva Training (dari Bab 4, berlaku di sini)

| Gejala | Diagnosis | Obat khas CV |
|---|---|---|
| train ↓, val ↑ | overfitting | augmentasi, dropout, early stopping |
| keduanya tinggi | underfitting | model lebih besar / epoch lebih lama |
| val stagnan sejak awal | data/label bermasalah | cek pipeline, cek augmentasi yang merusak label |

---

## 9. Koneksi ke Bab Lain

- **Bab 4**: CNNMini = konvolusi menggantikan Dense untuk input spasial; backward pooling = versi ter-modifikasi dari backward dense.
- **Bab 8**: pipeline & struktur proyek `nn_mini` (modul terpisah + test) = praktik proyek ML yang rapi.
- **Bab 9**: konvolusi 1D ~ sliding window di teks; weight sharing & hierarki fitur juga dipakai Transformer.
- **Bab 13**: fine-tuning LLM = transfer learning yang sama: freeze sebagian, lr kecil, hati-hati forgetting.
