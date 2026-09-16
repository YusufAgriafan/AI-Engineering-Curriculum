# Bab 2 — Matematika untuk Machine Learning

> Tidak perlu jadi ahli matematika, tapi AI engineer yang baik tahu *kenapa* gradient descent bekerja dan apa artinya dot product.

## 🎯 Tujuan Belajar
- Operasi vektor & matriks (aljabar linear) secara intuisi + kode.
- Memahami turunan/gradien sebagai dasar *learning*.
- Statistik dasar: distribusi, mean/variance, korelasi.

## 1. Materi Inti

> **Penting:** Anda tidak perlu jadi ahli matematika. Tujuan Bab 2 adalah **paham konsep dasar** dan **bisa kaitkan dengan kode**, bukan bisa membuktikan teorema.

---

### 1.1 Aljabar Linear — Intuisi, Bukan Hafalan

#### Vektor & Matriks: Apa Sebenarnya?

- **Vektor** = daftar angka yang merepresentasikan satu "benda" (satu data point).
  - Contoh: vektor fitur untuk rumah = `[luas, jumlah_kamar, umur, lokasi_score]`
- **Matriks** = kumpulan vektor (banyak data points).
  - Contoh: 1000 baris rumah × 4 fitur = matriks 1000×4

#### Operasi Kunci & Maknanya

| Operasi | Apa | Makna praktis |
|---|---|---|
| **Dot product** (skalar) | `a · b = Σ(a_i * b_i)` | "Kemiripan arah" — semakin mirip, semakin besar |
| **Perkalian matriks** | `A @ B` | Transformasi + kombinasi linear — dasar MLP & attention |
| **Transpose** | `A.T` — baris↔kolom | Ubah orientasi data — layer di MLP 생각 sebagai perkalian matriks |
| **Norm** (panjang vektor) | `||v||₂ = √(Σx_i²)` | "Besaran" vektor — biasa dipakai normalisasi |

#### Dot Product = Kemiripan: Intuisi Visual

```
Vektor A: [1, 0, 0] (arah sumbu X)
Vektor B: [1, 0, 0] (sama arah)  → dot product = 1 (sama persis!)
Vektor C: [0, 1, 0] (siku-siku)  → dot product = 0 (orthogonal)
Vektor D: [-1, 0, 0] (berlawanan) → dot product = -1
```

**Faktanya:** Semakin besar dot product (dengan asumsi vektor sudah dinormalisasi), semakin mirip arah vektornya.

#### Cosine Similarity — Dari Dot Product ke Ukuran Kemiripan

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Contoh: 3 vektor embedding buatan
v1 = np.array([1, 0, 0])      # minat: kopi
v2 = np.array([0.9, 0.1, 0])  # cukup mirip kopi
v3 = np.array([0, 1, 0])      # minat: teh — beda arah

print(cosine_similarity(v1, v2))  # ~0.995 (sangat mirip)
print(cosine_similarity(v1, v3))  # 0 (completely different)
```

> **Kenapa ini penting di AI Engineering:** RAG (Bab 12) pakai cosine similarity untuk mencari dokumen yang relevan dengan kueri. Classifier intent & reranker juga pakai konsep yang sama.

#### Eigenvalue & Eigenvector — Singkat Saja

- **Eigenvector** = vektor yang arahnya tidak berubah saat ditransformasi (hanya di-scale).
- **Eigenvalue** = berapa kali eigenvector di-scale.
- **Penerapan:** PCA (Bab 3 + unsupervised), attention (self-attention memakai operasi yang mirip konsep ini).

---

### 1.2 Kalkulus: Turunan & Gradien

#### Turunan dalam 1 Kalimat

> **Turunan** = seberapa cepet sesuatu berubah kalau inputnya berubah sedikit.

```
y = x²
dy/dx = 2x   → mis. di x=2, turunan = 4 (artinya: kalau x naik sedikit, y naik 4x lipat)
di x=0 → turunan = 0 → ini titik minimum / maksimum
```

#### Gradien = Vektor Turunan Multi-Variabel

Jika fungsi punya beberapa input (mis. bobot neural network yang 개수는 1 juta), kita butuh **gradien**: vektor yang berisi turunan parsial untuk tiap variabel.

```
f(x, y) = x² + y²
∇f = [df/dx, df/dy] = [2x, 2y]

di titik (1, 1): ∇f = [2, 2]
artinya: kalau kita pindah ke arah [2, 2], nilai fungsi naik paling cepet.
```

#### Gradient Descent — Intuisi Visual

```
Kita di puncak gunung (loss tinggi), ingin ke dasar lembah (loss rendah).

Langkah:
1. Cari gradien di posisi sekarang → arah paling curam ke atas
2. Maju ke arah BERLAWANAN gradien (turun)
3. Ulangi sampai dat-ngat (gradien ≈ 0)

                   o (gradient kecil → langkah kecil)
                  / \
                 /   \
  gradien besar → /     \ ← gradien besar (curam)
                 \     /
                  \   /
                   \ /
                    • (di sini gradient ≈ 0 → minimum!)
```

**Learning Rate = ukuran langkah.**

| Learning rate | Apa terjadi |
|---|---|
| Terlalu besar (mis. 10) | Loncat dari satu sisi ke sisi lain, tidak pernah konvergen (divergen) |
| Terlalu kecil (mis. 0.000001) | Maju sangat lambat, mungkin stuck di local minimum |
| Pas (mis. 0.01 — 0.1) | Konvergen stabil | 

---

### 1.3 Statistik Dasar

#### Mean, Variance, Standar Deviasi

```python
import numpy as np

data = np.array([85, 90, 78, 92, 88])

mean = np.mean(data)        # 86.6
variance = np.var(data)     # 23.04
std = np.std(data)          # 4.8 — standar deviasi adalah akar kuadrat variance

# Interpretasi:
# Mean = "rata-rata"
# Standar deviasi = "seberapa tersebar" data dari rata-rata
# Kalau std kecil → data homogen; std besar → variasi tinggi
```

#### Distribusi Normal (Gaussian / "Bell Curve")

Rajah klasik bell curve muncul di banyak tempat: error distribusi, noise, bahkan embedding distribution.

```
          Peak di mean
         /    |    \
        /     |     \
       /      |      \
      /       |       \
  tail      _mean_     tail
```

> Faktanya: 68% data ada dalam 1 std dari mean, 95% dalam 2 std.

#### Korelasi ≠ Kausalitas

```
Contoh klasik:
- Jumlah es krim yang dijual berkorelasi tinggi dengan jumlah kematian renang.
- Tapi es krim tidak menyebab kematian; keduanya sama-sama naik di musim panas.

Pelajaran: korelasi hanya tahu "muncul bareng", bukan "menyebabkan".
```

#### Sampling & Noise — Mengapa LLM "Tidak Pasti"

- **Sampling random** = ambil contoh dari populasi.
- **Noise** = variasi yang tidak bisa dijelaskan dengan model kita.
- Mixed: Stochastic gradient descent (SGD) pakai *subset* data per langkah → noisy tetapi lebih cepat dan sering generalisasi lebih baik.
- Samppaling LLM (temperature, top-p): saat generate teks, model mengacak pilihan dari distribusi — hasilnya tidak deterministik.

---

### 1.4 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 2:
  1. Vektor = representasi fitur 1 data
  2. Matriks = kumpulan vektor (batch)
  3. Dot product = kemiripan arah → dasar cosine similarity
  4. Perkalian matriks = transformasi + kombinasi linear
  5. Turunan = laju perubahan; gradien = turunan multi-variabel
  6. Gradient descent = turun ke arah berlawanan gradien
  7. Mean/std/distribusi normal = statistik deskriptif
  8. Korelasi ≠ kausalitas
  9. Sampling & noise → SGD, LLM sampling
```


## 2. Latihan Praktis

> **Prinsip:** Matematika dipahami lewat kode dan visualisasi, bukan cuma rumus.

---

### Latihan 1: Cosine Similarity Manual

Hitung cosine similarity untuk 3 vektor embedding buatan, lalu bandingkan dengan sklearn.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 3 vektor embedding buatan (mis. representasi dokumen)
doc_a = np.array([0.8, 0.1, 0.3, 0.2])   # topik "machine learning"
doc_b = np.array([0.7, 0.2, 0.4, 0.1])   # mirip ML
doc_c = np.array([0.1, 0.9, 0.1, 0.8])   # topik "resep masakan"

docs = [doc_a, doc_b, doc_c]

# Hitung manual
def cosine_sim_manual(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Manual:")
print(f"A vs B: {cosine_sim_manual(doc_a, doc_b):.4f}")  # seharusnya ~0.98
print(f"A vs C: {cosine_sim_manual(doc_a, doc_c):.4f}")  # seharusnya ~0.35

# Bandingkan dengan sklearn
print("\nSklearn:")
sim_matrix = cosine_similarity(docs)
print(sim_matrix)
```

> **Tulis catatan:** Apa yang kamu pelajari tentang kemiripan teks dari latihan ini?

---

### Latihan 2: Visualisasi Gradient Descent

Visualisasikan surface loss `J(w) = (w-3)²` dan jalur gradient descent dari titik awal.

```python
import numpy as np
import matplotlib.pyplot as plt

# Definisikan loss function
def loss(w):
    return (w - 3) ** 2

def gradient(w):
    return 2 * (w - 3)   # turunan dari (w-3)^2

# Gradient descent path
w_values = [0.0]  # mulai dari w=0
learning_rate = 0.1
for step in range(20):
    w = w_values[-1]
    grad = gradient(w)
    w_new = w - learning_rate * grad
    w_values.append(w_new)

# Plot
ws = np.linspace(-2, 8, 200)
losses = loss(ws)

plt.figure(figsize=(10, 6))
plt.plot(ws, losses, label='Loss J(w) = (w-3)²')
plt.scatter(w_values, [loss(w) for w in w_values], color='red', zorder=5, label='Gradient descent path')
plt.annotate('', xy=(w_values[-1], loss(w_values[-1])), xytext=(w_values[-1], loss(w_values[-1]) + 1),
            arrowprops=dict(arrowstyle='->', color='green'))
plt.axvline(x=3, color='green', linestyle='--', label='Minimum (w=3)')
plt.xlabel('w')
plt.ylabel('J(w)')
plt.title('Gradient Descent Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mulai dari w=0, setelah 20 step: w = {w_values[-1]:.4f}")
print(f"Loss awal: {loss(0):.2f}, Loss akhir: {loss(w_values[-1]):.4f}")
```

**Eksperimen:**

1. Ubah learning rate ke 0.5 — apa yang terjadi?
2. Ubah learning rate ke 0.01 — berapa step yang dibutuhkan?
3. Mulailah dari w=-5 (posisi jauh dari minimum) — apakah masih konvergen?

---

### Latihan 3: Perkalian Matriks sebagai Kombinasi Linear

```python
import numpy as np

# Matriks 4x4
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Vektor
v = np.array([1, 0, -1, 2])

# A @ v = kombinasi linear dari KOLOM A, dengan bobot dari v
# Kolom 0 di-scale jadi 1x, kolom 2 di-scale jadi -1x, kolom 3 di-scale jadi 2x

result_manual = (
    v[0] * A[:, 0] + 
    v[1] * A[:, 1] + 
    v[2] * A[:, 2] + 
    v[3] * A[:, 3]
)

result_numpy = A @ v

print("Hasil manual (kombinasi linear):")
print(result_manual)
print("\nHasil numpy (A @ v):")
print(result_numpy)
print("\nSama? ", np.allclose(result_manual, result_numpy))
```

> **Intuisi:** Perkalian matriks `A @ v` = "ambil setiap kolom A, scale dengan nilai v yang bersangkutan, lalu jumlahkan semua". Ini cara memahami neural network layer sebagai transformasi ruang vektor.

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_matematika_ml.ipynb` | Lab praktikum: demo + 6 latihan bercek otomatis ✅ (termasuk mini search engine ala RAG) | Kerjakan setelah baca materi inti |
| `02_kuis_matematika.ipynb` | 10 soal (PG + coding), 15 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_matematika.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-matematika-ml.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-gradient-descent/` | Proyek eksperimen: GD studio (starter + solusi) + RUBRIK | Setelah kuis ≥ 12/15 |

> Jadwal belajar intensif yang menyatukan semuanya: [`../MINGGU-01.md`](../MINGGU-01.md).

---

## 📚 Referensi Online (Prioritas Urut)

| Sumber | Tipe | Kenapa Baca |
|---|---|---|
| [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Video visual | **Wajib** — visualisasi terbaik untuk intuisi vektor & matriks |
| [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Video visual | Backprop, gradient, attention — sangat intuitif |
| [Mathematics for Machine Learning (buku gratis)](https://mml-book.github.io/) | Buku | Jika ingin pendalaman matematika yang rapi dan terstruktur |
| [Khan Academy — Statistics](https://www.khanacademy.org/math/statistics-probability) | Latihan interaktif | Jika statistik terasa asing |
| [StatQuest YouTube](https://www.youtube.com/@statquest) | Video singkat | Penjelasan konsep ML/statistik dengan bahasa yang sangat mudah |

---

## ✅ Checklist Kompetensi

- [ ] Menjelaskan dot product sebagai ukuran kemiripan (bukan sekadar rumus)
- [ ] Menjelaskan gradient descent ke teman non-teknis dalam 2 menit
- [ ] Tahu mengapa feature scaling mempengaruhi kecepatan konvergensi
- [ ] Bisa menghitung cosine similarity dari nol dengan NumPy
- [ ] Visualisasikan 1 fungsi loss dan jelaskan mengapa minimum ada di turunan = 0
- [ ] Paham korelasi ≠ kausalitas dan bisa kasih 1 contoh sendiri

