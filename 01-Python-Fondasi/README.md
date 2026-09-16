# Bab 1 — Fondasi Python untuk AI

> Python adalah bahasa utama AI. Bab ini memastikan dasar Python, OOP, dan ekosistem data Anda kuat sebelum masuk ke model.

## 🎯 Tujuan Belajar
- Menguasai Python inti: tipe data, fungsi, OOP, error handling, unit test.
- Lancar NumPy, pandas, matplotlib.
- Membaca & menulis file (dataset).

## 1. Materi Inti

> **Tips:** Jangan hanya baca. Ketik ulang contoh kode, ubah nilainya, lihat apa yang terjadi.

### 1.1 Python Dasar — Ringkas & Padat

#### Tipe Data

| Tipe | Contoh | Kapan Dipakai |
|---|---|---|
| `int`, `float` | `42`, `3.14` | Angka |
| `str` | `"Halo"` | Teks |
| `bool` | `True`, `False` | Logika |
| `list` | `[1, 2, 3]` | Koleksi berurutan, bisa ubah |
| `tuple` | `(1, 2)` | Koleksi berurutan, tidak berubah |
| `dict` | `{"nama": "Andi"}` | Pasangan kunci–nilai |
| `set` | `{1, 2, 3}` | Kumpulan unik (tanpa duplikat) |

**Contoh praktis:**

```python
# Dictionary untuk metadata dokumen (sering dipakai di RAG - Bab 12)
doc = {
    "judul": "Pantun",
    "halaman": 3,
    "timestamp": "2024-01-15"
}
print(doc.get("penerjemah", "Tak diketahui"))  # .get() lebih aman dari [] jika kunci tidak ada
```

#### Control Flow

```python
# if/elif/else
score = 85
if score >= 90:
    grade = "A"
elif score >= 75:
    grade = "B"
else:
    grade = "C"

# for dengan enumerate (sering lupa!)
for i, item in enumerate(["a", "b", "c"]):
    print(f"Index {i}: {item}")   # i = index, item = nilai

# while — hati-hati infinite loop
count = 0
while count < 5:
    print(count)
    count += 1   # jangan lupa increment!
```

#### List Comprehension — Lebih Pythonic

```python
# Daripada ini:
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x ** 2)

# Lebih baik:
result = [x**2 for x in range(10) if x % 2 == 0]
# Hasil: [0, 4, 16, 36, 64]
```

#### Fungsi

```python
def greet(name, greeting="Halo", times=1):
    """Contoh fungsi dengan default argumen."""
    for _ in range(times):
        print(f"{greeting}, {name}!")

greet("Budi")                          # default: "Halo, Budi!"
greet("Budi", "Hi", times=3)          # override: "Hi" 3x

greet("Budi", greeting="Selamat")     # keyword argument

# *args dan **kwargs — untuk fungsi yang jumlah parameternya tidak tetap
def log_all(*args, **kwargs):
    print(f"Positional: {args}")    # tuple semua argumen posisi
    print(f"Keyword: {kwargs}")    # dict semua keyword arguments

log_all(1, 2, 3, x=10, y=20)
# Output:
# Positional: (1, 2, 3)
# Keyword: {'x': 10, 'y': 20}
```

#### Error & Exception Handling

```python
# Pola yang benar: tangkap spesifik, bukan semua

def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError as e:
        print(f"Error: Tidak bisa bagi dengan nol. {e}")
        return None
    except TypeError as e:
        print(f"Error: Tipe tidak cocok. {e}")
        return None
    else:
        # Hanya jalan kalau tidak error
        print(f"Hasil: {result}")
        return result
    finally:
        # Selalu jalan, baik error maupun tidak
        print("--- Pembagian selesai ---")

# raise sendiri — buat error custom
def validate_email(email):
    if "@" not in email:
        raise ValueError(f"Email tidak valid: {email}")
    return True
```

> **Catatan penting:** Di AI Engineering, error handling yang rapi jauh lebih penting daripada "kode yang sempurna". Sistem LLM/product akan error — yang membedakan sistem robust vs fragile adalah bagaimana *penanganannya*.

---

### 1.2 OOP & Sub-program

**Kenapa penting:** Layer TensorFlow (Bab 4) dan tool agent (Bab 14) ditulis sebagai class. Tanpa paham OOP dasar, Anda hanya bisa *pakai* library, tidak bisa *memahami* atau *memodifikasi*.

```python
class TextCleaner:
    """Contoh class sederhana yang sering dipakai di preprocessing NLP."""
    
    def __init__(self, lowercase=True, remove_punct=True):
        self.lowercase = lowercase
        self.remove_punct = remove_punct
    
    def clean(self, text):
        """Return cleaned text."""
        result = text
        if self.lowercase:
            result = result.lower()
        if self.remove_punct:
            import string
            result = result.translate(str.maketrans("", "", string.punctuation))
        return result.strip()
    
    def __repr__(self):
        return f"TextCleaner(lower={self.lowercase}, punct={self.remove_punct})"

# Penggunaan
cleaner = TextCleaner()
print(cleaner.clean("Halo, Dunia! "))
# 'halo dunia'

print(cleaner)   # memakai __repr__ custom
# TextCleaner(lower=True, punct=True)
```

**Dunder methods yang perlu tahu:**

| Method | Arti | Kapan Dipakai |
|---|---|---|
| `__init__(self, ...)` | Inisialisasi objek | Hampir selalu |
| `__repr__(self)` | Representasi debugging | Selalu tambahkan untuk class penting |
| `__call__(self, ...)` | Objek bisa dipanggil seperti fungsi | Untuk model callable, custom layer |
| `__str__(self)` | Tampilan user-friendly | Jika class punya tampilan yang berarti |

---

### 1.3 Unit Testing

> **Prinsip:** Unit test untuk logika deterministik — parsing, chunking, transformasi data. Jangan berharap test bisa memprediksi output LLM.

```python
# Contoh sederhana dengan unittest
import unittest

class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()
    
    def test_lowercase(self):
        self.assertEqual(self.cleaner.clean("HALO"), "halo")
    
    def test_empty_string(self):
        self.assertEqual(self.cleaner.clean(""), "")
    
    def test_punctuation_removed(self):
        self.assertEqual(self.cleaner.clean("Halo, Dunia!"), "halo dunia")
    
    def test_already_clean(self):
        self.assertEqual(self.cleaner.clean("halo dunia"), "halo dunia")
    
    def test_numbers_preserved(self):
        self.assertEqual(self.cleaner.clean("Halo 123"), "halo 123")

if __name__ == "__main__":
    unittest.main()
```

**Edge cases yang sering dilupakan:**

- String kosong / hanya spasi
- Emoji, karakter non-ASCII
- URL, mention, hashtag
- Teks sangat panjang
- Teks multilingual (campuran bahasa)

---

### 1.4 Ekosistem Data

#### NumPy — Array & Vectorization

```python
import numpy as np

# Membuat array
a = np.array([1, 2, 3])          # 1D array
b = np.array([[1, 2], [3, 4]])  # 2D array (matriks)

# Broadcasting — operasi tanpa loop Python
# Ini yang membuat NumPy jauh lebih cepat daripada list Python biasa
c = a * 2            # [2, 4, 6]
d = a + b.flatten()  # broadcasting jika shape cocok

# Dot product — penting! Akan dipakai lagi di cosine similarity (Bab 2 & Bab 12)
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
dot = np.dot(v1, v2)    # 0 (orthogonal)
dot2 = np.dot([1,1,1], [1,1,1])  # 3

# Vectorization: bandingkan kecepatan
import time

# Versi loop Python (lambat)
def sum_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total

# Versi NumPy (cepat)
def sum_numpy(n):
    return np.sum(np.arange(n))

n = 1_000_000
start = time.time()
sum_loop(n)
print(f"Loop: {time.time() - start:.4f}s")

start = time.time()
sum_numpy(n)
print(f"NumPy: {time.time() - start:.4f}s")
# NumPy biasanya 10–100x lebih cepat untuk operasi besar
```

#### pandas — DataFrame

```python
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Eksplorasi cepat
print(df.head())           # 5 baris pertama
print(df.info())           # tipe kolom, missing values
print(df.describe())       # statistik numerik

# Filter & manipulasi
filtered = df[df["age"] > 25]
grouped = df.groupby("category")["price"].mean()

# Missing values
print(df.isnull().sum())           # jumlah missing per kolom
df_clean = df.dropna()             # drop baris dengan missing
df_filled = df.fillna(df.mean())   # isi dengan mean (numerik)
```

#### Matplotlib — Visualisasi Cepat

```python
import matplotlib.pyplot as plt

# Contoh paling dasar
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Grafik Sederhana")
plt.show()

# Dari DataFrame
df["price"].hist(bins=30)
plt.title("Distribusi Harga")
plt.show()

# Scatter plot (bisa pakai seaborn untuk lebih bagus)
df.plot.scatter(x="age", y="income")
plt.show()
```

#### File I/O

```python
import json
import csv

# JSON — format yang sering dipakai untuk LLM API (Bab 11)
data = {"nama": "Andi", "umur": 25, "hobi": [" membaca", " coding"]}

# Tulis
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)   # indent=2 biar mudah dibaca manusia

# Baca
with open("data.json", "r") as f:
    loaded = json.load(f)

# CSV — data tabular klasik
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["nama", "nilai"])
    writer.writerow(["Budi", 85])
    writer.writerow(["Siti", 90])
```

---

### 1.5 Kesalahan Umum Pemula Python

| Salah | Benar | Alasan |
|---|---|---|
| `if x == 5 or 6:` | `if x == 5 or x == 6:` (atau `if x in [5, 6]`) | Python eval `6` = True |
| `my_list = []` kemudian `append` vs `my_list = my_list + [...]` | Lihat konteks | `+=` modifikasi in-place, `+` buat baru |
| Menggunakan `==` untuk membandingkan float | Gunakan `np.isclose()` atau tolerance `abs(a-b) < eps` | Floating point punya presisi terbatas |
| Tidak pakai virtual environment | `python -m venv .venv && source .venv/Scripts/activate` | Konflik dependency antar project |
| Import di dalam fungsi (kecuali perlu lazy) | Import di atas file | Lebih jujur dan mudah di-trace |

---

### 1.6 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 1:
  1. Tipe data dasar → bisa pakai dict, list, tuple dengan nyaman
  2. Fungsi & scope → paham argumen, return, mutable vs immutable
  3. OOP basics → class, method, dunder (semua library AI pakai ini)
  4. NumPy → vectorization, dot product, broadcasting
  5. pandas → load, filter, groupby, handle missing
  6. File I/O → JSON/CSV dasar
  7. Unit test → testable, edge case, pytest/junit
```


## 2. Latihan Praktis

> **Prinsip:** Latihan harus bikin Anda *ketik*, bukan cuma baca.

### Latihan Fondasi

#### ✅ Latihan 1: TextCleaner Class + Unit Test

Buat class `TextCleaner` dengan method `clean(text)` dan unit test untuk 5 edge case.

> Scope edge case: string kosong, teks hanya spasi, teks dengan emoji (mis. "Halo 🎉"), teks sangat panjang (>1000 karakter), teks multilingual (campuran bahasa Indonesia-Inggris).

**Template starter:**

```python
import unittest

class TextCleaner:
    def clean(self, text: str) -> str:
        # TODO: implementasi di sini
        pass

class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()
    
    # TODO: tambah method test untuk tiap edge case
```

#### ✅ Latihan 2: EDA Dataset Baru

Load dataset CSV, hitung statistik deskriptif, buat 3 plot EDA.

1. Pilih dataset: bisa dari Kaggle, atau data sendiri (mis. data transkrip kuliah, data penjualan, data cuaca lokal)
2. Load dengan pandas → info() dan describe()
3. Cek missing values → putuskan apakah drop atau impute
4. 3 plot:
   - Distribusi 1 variabel numerik (histogram)
   - Hubungan 2 variabel (scatter)
   - Perbandingan kategori (bar plot atau boxplot)

#### ✅ Latihan 3: Kecepatan Vectorization

Bandingkan kecepatan loop Python vs NumPy untuk dot product 1 juta elemen.

```python
import numpy as np
import time

def dot_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def dot_numpy(a, b):
    return np.dot(a, b)

n = 1_000_000
a = np.random.rand(n)
b = np.random.rand(n)

# Benchmark loop
start = time.perf_counter()
dot_loop(a, b)
loop_time = time.perf_counter() - start

# Benchmark numpy
start = time.perf_counter()
dot_numpy(a, b)
numpy_time = time.perf_counter() - start

print(f"Loop: {loop_time:.4f}s, NumPy: {numpy_time:.4f}s")
print(f"Kecepatan: {loop_time / numpy_time:.1f}x lebih cepat")
```

> **Tulis catatan:** Apa yang kamu lihat? Kenapa beda begitu besar?

---

### Latihan Lanjutan (Opsional, Kalau Ada Waktu)

4. **Refactor notebook ke module:** Ambil 1 notebook lama, pindahkan kode ke file `.py`, tambah `if __name__ == "__main__"` block.
5. **Context manager custom:** Buat context manager sederhana (mis. `Timer` yang print durasi).
6. **Type hinting:** Tambahkan type hints ke fungsi-fungsi yang sudah ada; verifikasi dengan `mypy`.

---

## 🧰 Materi Pendukung (Folder Ini)

| File | Apa | Kapan Dipakai |
|---|---|---|
| `01_lab_python_fondasi.ipynb` | Lab praktikum: demo + 8 latihan dengan cek otomatis ✅ | Kerjakan setelah baca materi inti |
| `02_kuis_python.ipynb` | 10 soal (PG + coding), 15 poin, skor otomatis | Setelah lab selesai |
| `03_kunci_jawaban_kuis_python.ipynb` | Kunci + penjelasan "kenapa" | HANYA setelah mencoba kuis |
| `cheatsheet-python.md` | Ringkasan + pola emas yang berulang | Review harian / sebelum kuis |
| `project-starter-analisis-penjualan/` | Proyek mini TDD: 6 fungsi + 18 unit test + RUBRIK | Setelah kuis ≥ 12/15 |

> Jadwal belajar intensif yang menyatukan semuanya: [`../MINGGU-01.md`](../MINGGU-01.md).

---

## 📚 Referensi Online

| Sumber | Topik | Keterangan |
|---|---|---|
| [Python Official Tutorial](https://docs.python.org/3/tutorial/) | Semua dasar | Referensi resmi — selalu baca jika ada keyword baru |
| [Real Python](https://realpython.com/) | Tutorial praktikal | Jelaskan dengan contoh nyata |
| [NumPy: The Absolute Basics](https://numpy.org/doc/stable/user/absolute_beginners.html) | NumPy | Panduan dari nol |
| [Kaggle Learn: Python & Pandas](https://www.kaggle.com/learn) | Interaktif | Gratis, latihan langsung di browser |
| [pytest docs](https://docs.pytest.org/) | Testing | Standar industri Python |
| [Hugging Face Python Basics](https://huggingface.co/learn) | AI context | Jika ingin konteks AI sejak awal |

---

## ✅ Checklist Kompetensi

- [ ] Menulis class + unit test tanpa melihat contoh
- [ ] EDA dataset baru dalam < 30 menit (load → clean → plot)
- [ ] Paham kenapa vectorization lebih cepat (bebas loop Python)
- [ ] Bisa jelaskan mutable vs immutable dengan contoh sendiri
- [ ] Error handling: tahu kapan pakai try/except vs validasi preventif

