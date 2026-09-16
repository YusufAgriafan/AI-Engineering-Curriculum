# 📄 Cheatsheet — Python untuk AI (Bab 1)

> Review cepat sebelum kuis / sebelum coding. Jangan cuma dibaca — **tutup file ini, tulis ulang dari ingatan**, baru cek (retrieval practice).

---

## 1. Struktur Data — Pilih yang Mana?

| Struktur | Literal | Ordered? | Mutable? | Pakai untuk |
|---|---|---|---|---|
| `list` | `[1, 2, 3]` | ✅ | ✅ | Koleksi data, vektor fitur |
| `tuple` | `(1, 2)` | ✅ | ❌ | Koordinat, return ganda |
| `dict` | `{"a": 1}` | ✅ (3.7+) | ✅ | Metadata, mapping, config |
| `set` | `{1, 2, 3}` | ❌ | ✅ | Unik, keanggotaan O(1) |

```python
d = {"judul": "Pantun", "halaman": 3}
d.get("penulis", "anonim")     # aman: default jika kunci tak ada
d | {"tahun": 2024}            # merge (3.9+), tak mengubah d
{**d, "tahun": 2024}           # merge versi lama

a = [1, 2, 3]
b = a                          # REFERENSI — b.append ikut mengubah a!
c = a.copy()                   # salinan dangkal
squares = [x**2 for x in a]    # comprehension
pairs  = {k: v for k, v in zip("abc", [1, 2, 3])}
```

---

## 2. String — Operasi Wajib NLP

```python
teks = "  Halo Dunia, AI!  "
teks.strip()                   # buang spasi pinggir
teks.lower() / .upper()        # normalisasi huruf
teks.split()                   # tokenisasi kata (whitespace)
teks.split(",")                # split delimiter
",".join(["a", "b"])           # 'a,b'
teks.replace("AI", "ML")       # ganti substring
"ai" in teks.lower()           # keanggotaan
f"{3.14159:.2f}"               # '3.14' — formatting
"PYTHON"[-4:-1]                # 'THO' — indeks negatif, stop eksklusif
```

---

## 3. Fungsi

```python
def nama(param, default=0, *args, **kwargs) -> float:
    """Docstring singkat: APA yang dilakukan, bukan CARA."""
    return param + default

# Lambda = fungsi 1 ekspresi, untuk key=, map=, filter=
sorted(data, key=lambda x: x["nilai"], reverse=True)
max(counter, key=counter.get)          # kunci dengan nilai terbesar
sum(1 for c in s if c in "aiueo")      # hitung kondisi tanpa list
```

---

## 4. OOP Minimal yang Wajib

```python
class Model:
    def __init__(self, lr=0.01):       # konstruktor
        self.lr = lr
        self.trained = False

    def fit(self, X, y):               # method
        self.trained = True
        return self                    # chaining

    def __call__(self, x):             # objek bisa dipanggil: model(x)
        return x * self.lr

    def __repr__(self):                # debugging-friendly
        return f"Model(lr={self.lr})"
```

---

## 5. Error Handling

```python
try:
    nilai = float(input_str)
except ValueError as e:        # TANGKAP SPESIFIK, jangan Exception
    print("bukan angka:", e)
    nilai = 0.0
else:
    print("sukses")            # hanya jika TIDAK error
finally:
    print("selalu jalan")      # cleanup (tutup file, dll.)

class DataKosongError(Exception): ...   # custom exception
if not data:
    raise DataKosongError("kosong!")
```

---

## 6. NumPy Sekali Pandang

```python
import numpy as np
X = np.array([[1., 2.], [3., 4.]])
X.shape, X.dtype           # (2, 2), float64
X @ v                      # perkalian matriks/vektor
np.dot(a, b)               # dot product
np.linalg.norm(a)          # panjang vektor
X.min(axis=0)              # min per KOLOM (axis=0)
rng = X.max(axis=0) - X.min(axis=0)
rng[rng == 0] = 1.0        # hindari pembagi nol
X_norm = (X - X.min(axis=0)) / rng     # min-max scaling
np.allclose(a, b)          # banding float dengan toleransi
np.arange(n), np.zeros((2,3)), np.random.rand(n)
```

**Aturan emas:** operasi NumPy = tanpa loop Python = cepat 10–100x.

---

## 7. pandas Sekali Pandang

```python
import pandas as pd
df = pd.read_csv("data.csv")
df.head() / df.info() / df.describe()
df.isnull().sum()                          # missing per kolom
df[df["umur"] > 25]                        # filter baris
df.groupby("kota")["pendapatan"].sum().sort_values(ascending=False)
df["kolom_baru"] = df["a"] * df["b"]       # kolom turunan
df["harga"].mean(), .idxmax()              # agregat + lokasi max
df.to_csv("out.csv", index=False)
```

---

## 8. File I/O

```python
import json
with open("f.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
with open("f.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# JSONL — satu objek per baris (log training):
with open("log.jsonl", "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

# CSV manual:
import csv
with open("f.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["nama", "nilai"]); w.writerow(["Budi", 85])
```

---

## 9. Kesalahan yang Harus Kamu Hindari (dari Pengalaman)

| ❌ Salah | ✅ Benar |
|---|---|
| `if x == 5 or 6:` | `if x in (5, 6):` |
| Membandingkan float dengan `==` | `np.isclose(a, b)` |
| `b = a` lalu mengubah `b` | `b = a.copy()` |
| `except Exception:` telanjang | tangkap exception spesifik |
| Mutasi list saat di-loop | buat list baru (comprehension) |
| `open("f")` tanpa `with` | `with open(...) as f:` |

---

## 10. Pola Emas yang Berulang di Semua Bab

1. **Guard clause** — tangani kasus tepi di awal fungsi: `if not x: return ...`
2. **Comprehension** — transformasi koleksi 1 baris.
3. **Counter manual** — `d[k] = d.get(k, 0) + 1` (pahami sebelum pakai `Counter`).
4. **max dengan key** — `max(d, key=d.get)`.
5. **.get() dengan default** — akses dict aman.
6. **int(kondisi)** — bool → 0/1.
7. **with open()** — file selalu ditutup.

> Uji diri: tutup file ini, tulis pola #3 dan #4 dari ingatan. Berhasil? Siap kuis.
