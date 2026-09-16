# 🏗️ Project Starter — Neural Network Mini dari Nol (Bab 4)

> Kamu membangun **keras mini** dari nol: Dense layer, backpropagation, SGD,
> batching, dan early stopping — semuanya numpy murni. Setelah ini, `model.fit()`
> tidak akan pernah jadi kotak hitam lagi.

## Struktur

```
project-starter-numpy-nn/
├── README.md              ← kamu di sini
├── RUBRIK.md              ← penilaian mandiri
├── starter.ipynb          ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb           ← notebook SOLUSI (buka setelah selesai / mentok)
├── nn_mini/               ← paket yang kamu isi (numpy murni)
│   ├── data.npz           ← dataset "dua bulan" terkunci (420/90/90, seed 42)
│   ├── data.py            ← loader (JANGAN DIUBAH)
│   ├── activations.py     ← TODO: relu, sigmoid stabil, softmax
│   ├── losses.py          ← TODO: BCE, MSE + gradiennya
│   ├── layers.py          ← TODO: Dense (forward/backward) + Sequential
│   └── train.py           ← TODO: SGD, batching, training loop + early stopping
├── tests/                 ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_activations.py
│   ├── test_losses.py     ← termasuk GRADIENT CHECK numerik
│   ├── test_layers.py     ← termasuk GRADIENT CHECK end-to-end
│   └── test_train.py
└── solusi/
    └── nn_mini_ref.py     ← implementasi referensi (untuk cek mandiri)
```

## Dataset: Dua Bulan

600 titik dalam bentuk dua setengah-lingkaran yang saling mengaitkan + noise.
Sengaja **non-linear**, supaya:

- **Logistic regression** (model linear) mentok di val accuracy ±0.889 — dibuktikan di notebook.
- **MLP-mu** (16 → 8 → 1, ReLU + sigmoid) harus mencapai ≥ 0.94 di val dan test.

Ini demonstrasi paling jelas kenapa butuh "deep" learning: batas keputusan
melengkung tidak bisa ditemukan model linear, berapa pun lama dilatih.

## Apa yang Kamu Bangun

1. **Aktivasi stabil-numerik** — ReLU, sigmoid dua-cabang, softmax anti-overflow.
2. **Losses + gradien analitik** — diverifikasi dengan **gradient check** (selisih-hingga numerik), teknik yang dipakai sungguhan saat debugging NN.
3. **Dense layer dengan backprop** — forward menyimpan cache, backward menghitung dW/db dan melanjutkan gradien.
4. **Training loop** — mini-batch SGD + shuffle per epoch + early stopping, persis pola `model.fit()` dengan callback.
5. **Eksperimen** — baseline linear vs MLP, pengaruh lebar/depth, learning rate terlalu besar, early stopping.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_activations.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `activations.py` → `losses.py` → `layers.py` → `train.py`
   sampai **semua 48 test hijau**.
4. Buka `starter.ipynb` — kerjakan eksperimen & jawab **Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/nn_mini_ref.py` / `solusi.ipynb`.

> 💡 Kalau gradient check gagal padahal loss turun: cek urutan operasi di backward
> (dz sebelum dW), bentuk transpos (`Xᵀ @ dz` vs `dz @ Wᵀ`), dan pembagian `m`.
> Tiga itu penyebab 90% bug backprop.

## Pertanyaan Analisis (bagian penilaian!)

1. Logistic regression mentok di 0.889, MLP-mu 0.97+. **Kenapa** model linear tidak mampu, padahal keduanya dilatih sama keras? (hint: bentuk data + bentuk batas keputusan)
2. Backpropagation: jelaskan dengan kata-katamu sendiri bagaimana gradien "mengalir mundur" dari output ke layer pertama. Pakai analogi — boleh berbeda dari kue/Bab 4.
3. Saat `lr=50`, apa yang kamu lihat di kurva loss? Jelaskan MEKANISMENYA (langkah update vs bentuk loss surface).
4. Early stopping: kenapa memantau **val** loss, bukan train loss? Apa yang terjadi kalau `patience` terlalu kecil?
5. Init He dipakai untuk ReLU, Xavier untuk sigmoid. Apa yang terjadi di hidden layer kedua kalau semua bobot di-init nol? (pikirkan: simetri gradien)

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini jembatan menuju TensorFlow (notebook lab) dan nanti fine-tuning LLM
> (Bab 13). Loop yang kamu tulis di `train.py` = versi mini dari apa yang
> `GradientTape` lakukan — pahami sampai bisa menjelaskan tanpa melihat kode.
