# 📄 Cheatsheet — Matematika untuk ML (Bab 2)

> Review cepat. Uji diri: tutup file ini, tulis ulang rumus kunci dari ingatan, baru cek.

---

## 1. Vektor & Operasi Inti

| Konsep | Rumus / Kode | Makna |
|---|---|---|
| Dot product | `np.dot(a, b)` = `Σ aᵢbᵢ` | kesamaan arah (× panjang) |
| Norm | `np.linalg.norm(a)` = `√(Σaᵢ²)` | panjang vektor |
| Jarak Euclidean | `np.linalg.norm(a - b)` | seberapa jauh 2 titik |
| Perkalian matriks | `A @ B` — bentuk `(m,n)@(n,k)→(m,k)` | transformasi + kombinasi linear |
| Transpose | `A.T` | baris ↔ kolom |

**Bentuk wajib hafal:** `(m,n) @ (n,k) → (m,k)` — kolom kiri = baris kanan.

---

## 2. Cosine Similarity — Mesin di Balik RAG

```python
def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)
```

| Nilai | Arti |
|---|---|
| `1` | arah identik |
| `0` | tegak lurus — tak berhubungan |
| `-1` | berlawanan |

- Mengabaikan **panjang**, murni **sudut** → bandingkan dokumen panjang/pendek secara adil.
- Vektor nol → undefined → beri guard, return 0.
- Dipakai di: RAG retrieval (Bab 12), reranker, klasifikasi intent.

---

## 3. Turunan & Gradien

- **Turunan** `f'(x)` = laju perubahan `f` terhadap `x`.
- **Gradien** `∇f` = vektor semua turunan parsial = arah **naik** paling curam.
- Gradient descent jalan **berlawanan** gradien (turun):

```
w_baru = w - lr × ∇J(w)
```

**Cek turunan analitik dengan numerik:**

```python
turunan_numerik = lambda f, x, h=1e-5: (f(x+h) - f(x-h)) / (2*h)
```

| Learning rate | Efek |
|---|---|
| Terlalu besar | divergen / osilasi liar |
| Terlalu kecil | lambat, bisa stuck |
| Pas (0.01–0.1) | konvergen stabil |

---

## 4. Pola Kode yang Berulang

```python
# Guard clause (muncul di hampir semua solusi):
if not data:      return (0, 0)
if norm == 0:     return 0.0
if len(x) <= 1:   return True

# Semua pasangan memenuhi syarat:
all(x[i] >= x[i+1] for i in range(len(x)-1))

# np.asarray untuk input fleksibel:
data = np.asarray(data, dtype=float)
```

---

## 5. Statistik 1 Menit

```python
m  = np.mean(x)          # rata-rata
s  = np.std(x)           # sebaran
md = np.median(x)        # tahan outlier
r  = np.corrcoef(a, b)[0, 1]   # korelasi linear (-1..1)
```

- **Z-score** = `(x - mean) / std` → hasil selalu mean 0, std 1. `|z| > 2` ≈ outlier kandidat.
- **Korelasi ≠ kausalitas** — es krim & kematian renang naik bareng karena musim panas.
- Aturan 68–95: ~68% data dalam ±1std, ~95% dalam ±2std (distribusi normal).

---

## 6. Koneksi ke Bab Lain

| Konsep Bab 2 | Dipakai di |
|---|---|
| Dot product & cosine | RAG retrieval (12), reranker (14) |
| Gradient descent | Training TF (4), fine-tuning (13) |
| Matriks @ vektor | Layer NN (4), attention (9) |
| Z-score / scaling | Preprocessing fitur (3) |
| Korelasi | EDA & feature selection (3) |

> Uji diri terakhir: tulis rumus `w_baru` gradient descent DAN rumus cosine similarity dari ingatan. Berhasil dua-duanya? Siap kuis.
