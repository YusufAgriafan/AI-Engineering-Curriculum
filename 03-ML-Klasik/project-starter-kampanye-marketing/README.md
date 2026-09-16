# 🏗️ Project Starter — Prediksi Respon Kampanye Marketing (Bab 3)

> Proyek mini **end-to-end**: dari data mentah → metrik dari nol → model dari nol → tuning threshold → clustering → laporan. Semua numpy murni (tanpa sklearn) supaya setiap langkah terlihat isi kap mesinnya. Notebook starter + notebook solusi tersedia.

## Struktur

```
project-starter-kampanye-marketing/
├── README.md                 ← kamu di sini
├── RUBRIK.md                 ← penilaian mandiri
├── starter.ipynb             ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb              ← notebook SOLUSI (buka setelah selesai / mentok)
├── ml_sederhana/             ← paket yang kamu isi (numpy murni)
│   ├── data.npz              ← dataset terkunci (720/240/240, seed 42)
│   ├── data.py               ← loader (JANGAN DIUBAH)
│   ├── metrik.py             ← TODO: confusion matrix, P/R/F1, ROC-AUC
│   ├── model.py              ← TODO: sigmoid, BCE, LogisticRegressionGD
│   └── clustering.py         ← TODO: standardize, kmeans
├── tests/                    ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_metrik.py
│   ├── test_model.py
│   └── test_clustering.py
└── solusi/
    └── ml_sederhana_ref.py   ← implementasi referensi (untuk cek mandiri)
```

## Dataset

1.200 pelanggan sintetis, 8 fitur, **11% positif** (membeli setelah kampanye) — imbalanced seperti data nyata. Fitur 0 (`jam_pakai_per_minggu`) paling membedakan kelas; sisanya overlap moderat; 2 fitur redundant; sedikit label noise. Split stratified: **720 train / 240 val / 240 test**.

## Apa yang Kamu Bangun

1. **Metrik klasifikasi dari nol** — confusion matrix, accuracy, precision/recall/F1, ROC-AUC (dengan penanganan tie yang benar).
2. **Logistic regression dari nol** — sigmoid stabil-numerik, binary cross-entropy, full-batch gradient descent dengan kurva loss.
3. **K-means dari nol** — standardize, beberapa `n_init`, inertia; temukan **segmen pelanggan** dan positif-rate tiap segmen.
4. **Eksperimen threshold** — buktikan accuracy 0.9125 bisa menipu, lalu turunkan/menaikkan threshold dan lihat trade-off precision–recall.
5. **Disiplin val/test** — semua keputusan di validation set; test set hanya disentuh **sekali** di akhir.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasi tugasmu.
2. Isi `ml_sederhana/metrik.py` → jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Lanjut `model.py`, lalu `clustering.py` sampai **semua 32 test hijau**.
4. Buka `starter.ipynb` — kerjakan eksperimen & jawab **Pertanyaan Analisis** dengan kata-katamu sendiri.
5. Isi `RUBRIK.md`. Baru kalau mentok / sudah selesai: bandingkan dengan `solusi/` dan `solusi.ipynb`.

> ⚠ Jangan pernah mengubah file test atau data.npz untuk "menang" — rubrik memberi skor nol untuk proses seperti itu.

## Pertanyaan Analisis (bagian penilaian!)

1. Model "selalu prediksi tidak" mendapat accuracy ±0.89. Model kamu 0.9125. **Kenapa angka ini menipu**, dan metrik apa yang memperlihatkan kebenarannya?
2. Threshold default 0.5 memberi precision 0.65 / recall 0.48. Kalau biaya mengirim promo **murah** tapi kehilangan pelanggan **mahal**, threshold mana yang kamu pilih? Jelaskan dengan trade-off.
3. Kenapa ROC-AUC tidak berubah ketika threshold digeser, padahal F1 berubah? (hint: AUC menilai *ranking*, threshold menilai *keputusan*)
4. Dua klaster k-means punya positive-rate 16.9% vs 4.3%. Kalau budget promo hanya cukup untuk satu segmen, mana yang kamu pilih — dan angka apa yang kamu pakai sebagai justifikasi?
5. Angka test vs validation: seberapa beda? Kapan perbedaan ini menjadi tanda bahaya (data leakage / test dipakai berkali-kali)?

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini adalah inti dari seluruh buku. Konsep train/val/test + precision/recall akan muncul lagi di hampir semua bab setelah ini — pahami sampai **bisa menjelaskan ke orang lain**, bukan cuma test hijau.
