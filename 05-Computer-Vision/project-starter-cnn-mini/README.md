# 🏗️ Project Starter — CNN Mini dari Nol (Bab 5)

> Kamu membangun **CNN mini** dari nol: zero-padding, konvolusi 2D dengan
> backward sendiri, max-pool, flatten, augmentasi data — lalu merakitnya
> jadi `CNNMini` dan mengalahkan MLP di piksel mentah. Semua numpy murni.

## Struktur

```
project-starter-cnn-mini/
├── README.md              ← kamu di sini
├── RUBRIK.md              ← penilaian mandiri
├── starter.ipynb          ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb           ← notebook SOLUSI (buka setelah selesai / mentok)
├── nn_mini/               ← paket yang kamu isi (numpy murni)
│   ├── data.py            ← dataset "bars" terkunci (JANGAN DIUBAH)
│   ├── nnet.py            ← Dense + BCE, kembaran Bab 4 (SUDAH LENGKAP)
│   ├── train.py           ← SGD + training loop, kembaran Bab 4 (SUDAH LENGKAP)
│   ├── conv.py            ← TODO: padding, konvolusi, pooling, flatten, Conv2D
│   ├── augment.py         ← TODO: flip, noise, shift, pipeline
│   └── model.py           ← TODO: ReLU/MaxPool blok + CNNMini
├── tests/                 ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_conv.py       ← termasuk GRADIENT CHECK numerik
│   ├── test_augment.py
│   └── test_model.py      ← termasuk GRADIENT CHECK end-to-end
└── solusi/
    └── cnn_mini_ref.py    ← implementasi referensi (untuk cek mandiri)
```

## Dataset: Bars

Gambar 8x8: latar noise halus + **satu garis terang** horizontal atau vertikal
di posisi acak, plus sedikit titik noise kasar. Split 420/90/90 (seed 42).

Kenapa task ini? Klasifikasi orientasi garis itu **mudah bagi satu filter
konvolusi + pooling**, tapi rapuh bagi model yang mengahmali posisi piksel.
Di akhir kamu akan membuktikannya sendiri: geser garis beberapa piksel →
MLP jatuh, CNN tetap berdiri. Itulah inti *weight sharing + invarian
translasi*.

## Apa yang Kamu Bangun

1. **Konvolusi 2D + backward** — dK dan dX diverifikasi dengan **gradient check**
   numerik (selisih-hingga), teknik debugging NN yang dipakai sungguhan.
2. **Max-pool 2x2** — forward + backward lewat argmax (gradien hanya ke
   "pemenang" tiap jendela), termasuk pemahaman subgradien di kasus seri.
3. **Augmentasi yang benar** — flip/noise/shift yang TIDAK pernah mengubah label,
   dengan area kosong diisi nol (bukan wrap-around).
4. **CNNMini** — conv → ReLU → pool → flatten → dense, dirakit seperti
   `keras.Sequential` mini, dengan kontrak `parameters()/grads()` yang konsisten.
5. **Eksperimen** — baseline MLP vs CNN, ukuran kernel, augmentasi, dan ujian
   ketahanan terhadap pergeseran.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_conv.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `conv.py` → `augment.py` → `model.py`
   sampai **semua 42 test hijau**.
4. Buka `starter.ipynb` — kerjakan eksperimen & jawab **Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/cnn_mini_ref.py` / `solusi.ipynb`.

> 💡 Kalau gradient check gagal padahal loss turun: cek (1) gradien yang harusnya
> di-akumulasi `+=` tapi malah di-assign `=`, (2) pad yang lupa dibuang dari dXp,
> (3) arah slicing saat shift. Tiga itu penyebab 90% bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. MLP di piksel mentah boleh jadi mencapai val acc tinggi — tapi begitu garis
   digeser ke posisi yang tidak pernah dilihat, performanya jatuh. **Kenapa**,
   dan kenapa CNN tidak ikut jatuh? (hint: apa yang "dihafal" masing-masing model)
2. Jelaskan **weight sharing** dan **invarian translasi** dengan kata-katamu
   sendiri. Kenapa satu filter cukup untuk semua posisi garis?
3. Backward max-pool mengirim gradien hanya ke satu posisi per jendela 2x2.
   Apa yang terjadi kalau dua nilai dalam jendela **sama persis**? (ini namanya
   subgradien — cek juga komentar di `test_model.py`!)
4. Di dataset ini, flip horizontal aman tapi flip vertikal **merusak label**.
   Jelaskan, dan tarik kesimpulan umum tentang memilih augmentasi untuk suatu domain.
5. Kernel 1x1, 3x3, 5x5 — apa trade-off-nya untuk gambar 8x8? Ingat: jumlah
   parameter, jangkauan konteks, dan piksel tepi yang hilang tanpa padding.

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini memakai kembali `Dense` + training loop dari Bab 4 (`nnet.py` &
> `train.py`, sudah diberikan). Kalau lama paham bagian itu, ulangi
> project-starter-numpy-nn Bab 4 dulu — konvolusi hanya mengganti cara
> layer "melihat" input-nya.
