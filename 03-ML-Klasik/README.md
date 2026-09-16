# Bab 3 — Machine Learning Klasik

> Fondasi konsep terpenting seluruh buku: **bagaimana mesin belajar dari data**, dan bagaimana menilai apakah benar-benar belajar (bukan menghafal).

## 🎯 Tujuan Belajar
- Linear & logistic regression dari nol (bukan hanya `sklearn.fit`).
- Konsep train/val/test, overfitting, regularisasi.
- Model berbasis pohon (decision tree, ensemble).
- Unsupervised: k-means, anomaly detection, recommender.

## 1. Materi Inti

> Bab 3 adalah **bab paling penting** di kurikulum ini. Baca dua kali. Ini adalah fondasi intelektual yang membedakan engineer dari pengguna tools.

---

### 1.1 Regresi Linear — Dari Nol

#### Model dan Loss

```
Model:   y_pred = w · x + b
  - x = fitur (input)
  - w = bobot (parameter yang dipelajari, satu per fitur)
  - b = bias (intersep)
  - y_pred = prediksi

Loss (MSE):  J(w,b) = (1/n) Σ (y_pred_i - y_true_i)²
  - Semakin besar error → loss naik kuadratik
  - Kuadrat → penalize error besar lebih parah

Optimasi: Gradient descent → cari w, b yang meminimalkan J
```

#### Kode Sederhana dari Nol

```python
import numpy as np

class LinearRegressor:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iterations):
            y_pred = X @ self.w + self.b          # prediksi
            dw = (2/n_samples) * X.T @ (y_pred - y)  # gradien w
            db = (2/n_samples) * np.sum(y_pred - y)  # gradien b
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self
    
    def predict(self, X):
        return X @ self.w + self.b

# Contoh penggunaan
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)   # y = 4 + 3x + noise

model = LinearRegressor(lr=0.1, n_iterations=1000)
model.fit(X, y)

print(f"w learned: {model.w[0]:.2f} (seharusnya ~3)")
print(f"b learned: {model.b:.2f} (seharusnya ~4)")
```

> **Koneksi ke Bab 2:** Ini adalah implementasi langsung dari gradient descent yang sudah kamu visualisasikan di Bab 2. Kini diterapkan ke data nyata.

#### Feature Engineering & Scaling

```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Feature scaling — WAJIB untuk gradient descent yang stabil
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial features — tambah interaksi, tapi hati-hati overfitting
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Contoh: model dengan polynomial bisa kurva, bukan garis lurus
# Tapi kalau degree terlalu tinggi → overfitting (lihat 1.3)
```

**Aturan praktis:**

- Feature scaling → Wajib (kecuali tree-based model)
- Polynomial degree → Mulai dari 1 (linear), naikkan perlahan jika perlu
- Interaksi features → Hanya jika ada alasan domain yang jelas

---

### 1.2 Klasifikasi & Logistic Regression

#### Logistik Regression vs Regresi Linear

| Aspek | Regresi Linear | Logistic Regression |
|---|---|---|
| Output | Bilangan kontinu (mis. harga) | Probabilitas 0–1 |
| Fungsi aktivasi | Tidak ada (linear) | Sigmoid: σ(z) = 1/(1+e^(-z)) |
| Loss | MSE | Binary cross-entropy |
| Penggunaan | Prediksi nilai | Klasifikasi biner |

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    # y_pred harus di-clip agar tidak log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Contoh: prediksi probabilities
z = np.array([-2, -1, 0, 1, 2])
probs = sigmoid(z)
print(f"Probabilitas: {probs}")   # [0.12, 0.27, 0.50, 0.73, 0.88]
```

#### Decision Boundary

- Titik di mana model "memutuskan" antara kelas 0 dan kelas 1.
- Logistic regression → decision boundary linear (garis lurus / bidang datar).
- Model non-linear (decision tree, neural network) → boundary bisa lebih kompleks.

---

### 1.3 Evaluasi Model — BACA DUA KALI ⚠

> Ini adalah skills yang membedakan engineer profesional dari pemula.

#### Split Data yang Benar

```
TRAIN  →  Gunakan untuk belajar parameter
VAL    →  Gunakan untuk pilih hyperparameter / compare model
TEST   →  Hanya diujikan sekali di akhir, sebelum deploy

⚠ PERINGATAN: Jangan pernah pakai data test untuk training atau pilih model.
   Jika ada, itu "data leakage" dan metrik kamu tidak berarti apa-apa.
```

#### Metrik Klasifikasi

| Metrik | Rumus singkat | Kapan dipakai |
|---|---|---|
| Accuracy | (TP+TN)/Total | 데이터 balanced, semua kelas sama penting |
| Precision | TP/(TP+FP) | False positive mahal (mis. spam detection — jangan hapus email penting) |
| Recall | TP/(TP+FN) | False negative mahal (mis. diagnose penyakit — jangan lewatkan kasus) |
| F1-Score | 2 * (P*R)/(P+R) | Butuh keseimbangan precision & recall |
| ROC-AUC | Luas di bawah kurva TPR vs FPR | Model dibandingkan across threshold, probabilitas output |

> **Konteks imbalanced:** Accuracy bisa menipu! Contoh: 99% data normal, 1% fraud. Model yang menebak "semua normal" punya accuracy 99% tapi inginya useless (recall = 0).

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]   # 0=negatif, 1=positif
y_pred = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]   # prediksi model

print(confusion_matrix(y_true, y_pred))
# [[5 1]    ← TN=5, FP=1
#  [1 3]]   ← FN=1, TP=3

print(classification_report(y_true, y_pred))
```

#### Bias vs Variance — Diagnosis Kurva Pembelajaran

```
                                         Kurva yang Bisa Muncul

1. UNDERFIT (high bias, low variance)   2. OVERFIT (low bias, high variance)
   Train loss: tinggi                      Train loss: rendah
   Val loss: tinggi                        Val loss: tinggi (gap besar)
   → Model terlalu sederhana              → Model terlalu kompleks
   → Solusi: lebih banyak fitur, model      → Solusi: lebih banyak data, regularisasi,
     yang lebih kompleks, training lebih        lebih sedikit fitur, early stopping
     lama

3. GOOD FIT                              
   Train loss: rendah                      
   Val loss: rendah (gap kecil)           
   → Ini target kita!                    
```

---

### 1.4 Decision Tree & Ensemble

#### Decision Tree: Intuisi

```
Model memutuskan seperti flowchart:

      Apakah age > 30?
         /        \
       Ya         Tidak
       /            \
  income>5?      student?
    /    \         /     \
  Ya   Tidak    Ya    Tidak
  /      \      /       \
 <-- Kelas A   <-- Kelas B
```

- **Gini impurity** / **Entropy** → ukuran seberapa "bersih" split (segregated kelas)
- **Information gain** → selisih impurity sebelum dan sesudah split
- Model memilih split yang memaksimalkan informasi gain

#### Ensemble: Lebih Baik Dari Satu Pohon

| Metode | Cara Kerja | Kelemahan |
|---|---|---|
| **Random Forest** | Banyak pohon, masing-masing training di subset data & fitur acak; hasil = voting rata-rata | Lebih lambat predict, butuh memori lebih |
| **Gradient Boosting** (XGBoost, LightGBM, CatBoost) | Pohon satu per satu, setiap pohon memperbaiki error pohon sebelumnya | Sensitif hyperparameter, lebih rentan overfit kalau tidak hati-hati |

> **Pesan kunci:** Pohon ensemble SANGAT kuat di data tabular. Seringkali menang di kompetisi Kaggle. Dan mereka bisa jadi komponen kecil di sistem AI modern (mis. classifier intent di agent, Bab 14).

---

### 1.5 Unsupervised Learning

#### K-Means Clustering

```python
from sklearn.cluster import KMeans
import numpy as np

# Contoh: keluster dokumen berdasarkan embedding (Bab 12 nanti)
embeddings = np.random.randn(200, 10)  # 200 dokumen, 10 dimensi

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

print(f"Cluster size: {np.bincount(labels)}")  # distribusi tiap kluster
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")  # 3 x 10
```

- K-means: partisi data ke K kluster, tiap titik ke kluster terdekat (centroid)
- Centroid = rata-rata titik dalam kluster

#### Anomaly Detection

- Cari titik yang "jauh" dari pola normal → mis. deteksi fraud, defect, outlier.
- Metode: Isolation Forest, One-Class SVM, Autoencoder (Bab 4).

#### Recommender System

- **Collaborative filtering**: rekomendasikan berdasarkan pola pengguna serupa
- **Content-based**: rekomendasikan berdasarkan fitur item yang mirip

---

### 1.6 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 3:
  1. Regresi Linear: model y=wx+b, MSE loss, gradient descent
  2. Logistic Regression: sigmoid, BCE loss, decision boundary
  3. Split: train/val/test — test hanya sekali di akhir
  4. Metrik: accuracy (hati), precision, recall, F1, ROC-AUC
  5. Bias-variance: underfit vs overfit, kurva learning
  6. Decision tree & ensemble: impurity, information gain, RF vs boosting
  7. K-means, anomaly detection, recommender (unsupervised)
  8. Imbalanced data → jangan pakai accuracy
```

---

## 2. Latihan Praktis

---

### Latihan 1: Regresi Linear dari Nol

**Tujuan:** Implementasi gradient descent tanpa sklearn, bandingkan dengan skikit-learn.

**Langkah 1 — Buat data sintetis:**

```python
import numpy as np
np.random.seed(42)

# y = 2 + 3*x + noise
X = 2 * np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.5

# Visualisasi data
import matplotlib.pyplot as plt
plt.scatter(X, y, alpha=0.6)
plt.plot([0, 2], [2, 8], 'r--', label='y = 2+3x (true)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Data Sintetis')
plt.show()
```

**Langkah 2 — Implementasi dari Nol:**

```python
class LinearRegressionScratch:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        for i in range(self.n_iter):
            y_pred = X @ self.w + self.b
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.w + self.b

model = LinearRegressionScratch(lr=0.1, n_iter=1000)
model.fit(X, y)

print(f"w: {model.w[0][0]:.3f} (true: 3)")
print(f"b: {model.b:.3f} (true: 2)")
```

**Langkah 3 — Bandingkan dengan sklearn:**

```python
from sklearn.linear_model import LinearRegression

sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

print(f"\nSklearn w: {sklearn_model.coef_[0][0]:.3f}")
print(f"Sklearn b: {sklearn_model.intercept_[0]:.3f}")

# Visualisasi hasil
plt.scatter(X, y, alpha=0.6, label='data')
plt.plot(X, model.predict(X), 'g-', label=f'scratch (w={model.w[0][0]:.2f})')
plt.plot(X, sklearn_model.predict(X), 'r--', label=f'sklearn (w={sklearn_model.coef_[0][0]:.2f})')
plt.legend()
plt.show()
```

> Tulis: apakah hasilnya mirip? Kalau ada perbedaan, apa penyebabnya?

---

### Latihan 2: Dataset Imbalanced

**Tujuan:** Pahami kenapa accuracy menyesatkan, dan bagaimana memilih metrik yang tepat.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Buat dataset sangat imbalanced (95% negatif, 5% positif)
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=0, n_clusters_per_class=1,
    weights=[0.95, 0.05], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Distribusi: {np.bincount(y)} (0={np.bincount(y)[0]}, 1={np.bincount(y)[1]})")
# Harapan: ~950 negatif, ~50 positif

# Model yang "bodoh": selalu prediksi 0
y_dumb = np.zeros(len(y_test))
print(f"\nModel 'semua nol':")
print(f"  Accuracy: {accuracy_score(y_test, y_dumb):.3f}")
print(f"  Recall:   {recall_score(y_test, y_dumb):.3f} ← TINGGI? TIDAK! (seharusnya 0)")
print(f"  F1:       {f1_score(y_test, y_dumb):.3f}")

# Logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"\nLogistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
print(f"  Recall:   {recall_score(y_test, y_pred):.3f}")
print(f"  F1:       {f1_score(y_test, y_pred):.3f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Latihan lanjutan:**

1. Coba ubah threshold probabilitas (bukan 0.5) → bagaimana precision/recall trade-off?
2. Pakai `class_weight='balanced'` di LogisticRegression → apakah hasilnya lebih baik?

---

### Latihan 3: Clustering Embedding

**Tujuan:** Pahami cara kerja k-means untuk menampilkan topik dokumen.

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Simulasi 150 dokumen dengan 2 topik yang jelas
np.random.seed(42)

# Topik A: machine learning
N_A = 75
emb_A = np.random.randn(N_A, 2) * 0.5 + np.array([2, 2])

# Topik B: cooking
N_B = 75
emb_B = np.random.randn(N_B, 2) * 0.5 + np.array([-2, -2])

all_embeddings = np.vstack([emb_A, emb_B])
true_labels = np.array([0]*N_A + [1]*N_B)

# Plot sebenarnya
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=true_labels, cmap='viridis', alpha=0.6)
plt.title('Ground Truth (2 topik)')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')

# K-means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_embeddings)

plt.subplot(1, 2, 2)
plt.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', s=200, marker='X', label='Centroids')
plt.title(f'K-Means Clustering (n_clusters=2)')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Akurasi kluster (dengan label matching):")
# Perhatikan: k-means tidak tahu urutan label → perlu matching
from scipy.optimize import linear_sum_assignment
cost_matrix = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        cost_matrix[i, j] = -np.sum((true_labels == i) & (cluster_labels == j))
row_ind, col_ind = linear_sum_assignment(cost_matrix)
accuracy = np.sum(true_labels[cluster_labels == col_ind[0]] == row_ind[0]) / len(true_labels)
print(f"  ~{accuracy:.2%} (akumulasi true assignments / total)")
```

> **Tulis:** Apa yang kamu lihat dari visualisasi? Bagaimana k-means menemukan cluster tanpa tahu label?

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_ml_klasik.ipynb` | Lab praktikum: 6 bagian + 8 latihan bercek otomatis ✅ — semua algoritma dibangun dari nol (numpy murni) | Kerjakan setelah baca materi inti |
| `02_kuis_ml_klasik.ipynb` | 10 soal (PG + coding), 15 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_ml_klasik.ipynb` | Kunci + intuisi di balik tiap jawaban | HANYA setelah mencoba kuis |
| `cheatsheet-ml-klasik.md` | Rumus kunci + pola kode + koneksi ke bab lain | Review harian / sebelum kuis |
| `project-starter-kampanye-marketing/` | Proyek end-to-end: prediksi respon kampanye — metrik, logistic regression & k-means dari nol (TDD, 32 test) + starter/solusi + RUBRIK | Setelah kuis ≥ 12/15 |

> Jadwal belajar intensif yang menyatukan semuanya: [`../MINGGU-01.md`](../MINGGU-01.md).

---

## 📚 Referensi Online

| Sumber | Topik | Keterangan |
|---|---|---|
| [Machine Learning Specialization — Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction) | Kursus lengkap | Wajib, bisa audit gratis |
| [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) | Dokumentasi | Selalu jadi referensi pertama kalau pakai library ini |
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | Tutorial cepat | Singkat, visual, interaktif |
| [StatQuest YouTube](https://www.youtube.com/@statquest) | Statistik & ML | Penjelasan yang sangat mudah dicerna |
| [Elements of Statistical Learning (gratis)](https://hastie.su.domains/ElemStatLearn/) | Buku lanjutan | Jika ingin pendalaman matematis |

---

## ✅ Checklist Kompetensi

- [ ] Menjelaskan trade-off bias–variance dengan contoh nyata
- [ ] Memilih metrik evaluasi yang tepat untuk kasus imbalanced
- [ ] Tahu kapan memakai model klasik alih-alih LLM (murah, cepat, deterministic)
- [ ] Implementasi gradient descent dari nol
- [ ] Paham confusion matrix dan bisa baca hasil classification_report
- [ ] Implementasi k-means untuk clustering dokumen
- [ ] Bisa jelaskan perbedaan random forest dan gradient boosting
