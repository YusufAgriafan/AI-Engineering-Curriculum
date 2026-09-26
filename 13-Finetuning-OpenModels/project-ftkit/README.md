# 🏗️ Project Starter — ftkit: Fine-tuning dari Nol (Bab 13)

> Kamu membangun **seluruh siklus fine-tuning tanpa GPU**: dataset SFT JSONL ala
> Alpaca + oversampling, matematika LoRA (ΔW = B @ A) dengan gradient check,
> simulasi kuantisasi, loop SFT dengan early stopping yang menyelamatkan dari
> overfitting, keputusan biaya API vs lokal, dan evaluasi sebelum/sesudah +
> kebijakan OOD. Stdlib murni, TDD **44 test**, tanpa API key dan tanpa VRAM.

## Struktur

```
project-ftkit/
├── README.md               ← kamu di sini
├── RUBRIK.md               ← penilaian mandiri
├── starter.ipynb           ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb            ← notebook SOLUSI (buka setelah selesai / mentok)
├── ftkit/                  ← paket yang kamu isi (stdlib murni)
│   ├── data.py             ← dataset, format SFT, tarif, varian (JANGAN DIUBAH)
│   ├── tokenizer.py        ← GIVEN: tokenizer mini (vocab, OOV, negasi)
│   ├── model.py            ← GIVEN: klasifier tiny + loss BCE + gradien
│   ├── sft.py              ← TODO: dataset JSONL ala Alpaca + oversampling
│   ├── lora.py             ← TODO: matematika LoRA (A@B) + gradient check
│   ├── quantize.py         ← TODO: simulasi kuantisasi & kompresi
│   ├── trainer.py          ← TODO: loop SFT + early stopping + riwayat
│   ├── api.py              ← TODO: biaya API vs lokal + keputusan deployment
│   └── evaluasi.py         ← TODO: eval sebelum/sesudah + kebijakan OOD
├── tests/                  ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_sft.py         ( 9 test)
│   ├── test_lora.py        ( 8 test)
│   ├── test_quantize.py    ( 9 test)
│   ├── test_trainer.py     ( 6 test)
│   └── test_api_eval.py    (12 test)
├── _calib.py               ← skrip audit: mencetak angka terkunci (bukan test)
├── _gen_notebooks.py
├── _verify_project.py      ← verifier otomatis (patch referensi + test + notebook)
└── solusi/
    └── ftkit_ref.py        ← implementasi referensi (untuk cek mandiri)
```

## Kenapa "Model"-nya Tiny, Bukan LLM 7B?

Pola yang sama dengan Bab 11 (`transport.py`) dan Bab 12 (`embed.py`) — tiga
alasan:

1. **Reproducible.** Training LLM 7B membutuhkan GPU yang hasilnya tidak bisa
   diulang persis (CUDA nondeterminisme). Model tiny dengan SGD murni-Python
   menghasilkan **angka yang sama di komputer siapa pun** — syarat evaluasi
   yang adil dan test yang bermakna.
2. **Gratis & cepat.** Satu run training < 0,1 detik. Kamu bisa eksperimen lr,
   epoch, data noise, early stopping puluhan kali — bukan sekali semalam.
3. **Matematikanya SAMA.** Forward → loss → gradien → update → early stopping:
   itulah siklus yang sama dengan SFT LoRA 7B di Unsloth/Colab. Yang berubah
   saat skala naik: jumlah param, VRAM, dan waktu — bukan bentuk loop-nya.

Analogi LoRA: bobot embedding tokenizer dibekukan (base model), yang dilatih
hanya `w` & `b` (kepala kecil = "adapter"). Blok 2 menghitung ΔW = B @ A secara
eksplisit supaya kucing-kucingan "rank r" terlihat matematikanya.

## Angka Terkunci (dari `_calib.py`, SEED 13)

```
DATASET
  24 contoh sentimen (16 train + 8 val) | fitur 17-dim (16 + flag negasi)
  loss awal 0.6882 | FORMAT_SFT 8 baris (6 positif / 2 negatif — sengaja timpang)

SFT
  split 8 contoh rasio 0.25 → (5, 3) | oversample per_label=6 → 12 baris (6/6)

LORA
  4096×4096 rank 16 → 131.072 param dari 16,7 jt (rasio 0.0078125 ≈ 0,78%)
  ΔW awal = 0 (B nol) | gradient check max err 6.4e-12

TRAINER (bersih, lr=0.5, 60 epoch)
  train_loss 0.7396 → 0.0872 | val_acc 1.0 | tidak berhenti dini

TRAINER (2 label train dibalik + lr=3.0 → OVERFIT)
  berhenti dini di epoch 66 | best epoch 61 | val_loss best 0.4184 vs akhir 0.4188

KUANTISASI (7B): fp16 14 GB (skor 0.82) | int8 7 GB (0.81) | q4 3.5 GB (0.78)

BIAYA (1k in/300 out): api_besar $0.0055 | api_mini $0.00033 | lokal $0
  bulanan 2000 req/hari = $19.8 → api | 10000 req/hari = $99 → lokal (GPU $60/bln)

EVALUASI
  akurasi val 0.75 → 1.0 (delta 0.25)
  OOD "Lumayan standar saja sih" → skor 0.4988, yakin 0.0024 → JANGAN tampilkan
```

## Apa yang Kamu Bangun

1. **SFT dataset** — format JSONL ala Alpaca, round-trip, oversampling
   deterministik, split interleaved tanpa `random`.
2. **LoRA matematika** — ΔW = B @ A, B nol → model mulai utuh; param = rank×(in+out);
   gradient check membuktikan gradien analitik benar sebelum dipercaya.
3. **Kuantisasi** — skala simetris, galat kuantisasi, tabel ukuran vs skor.
4. **Trainer** — SGD per contoh, loss dicatat sebelum update, early stopping
   dengan **restore bobot terbaik** (bukan terakhir).
5. **Biaya & keputusan** — tarif per 1jt token, proyeksi bulanan, ambang API vs GPU.
6. **Evaluasi jujur** — sebelum/sesudah + delta; OOD → "kurang yakin" (abstain, Bab 12).

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_sft.py`.
2. Isi satu modul, jalankan:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan: `sft.py` → `lora.py` → `quantize.py` → `trainer.py` → `api.py` →
   `evaluasi.py` sampai **44 test hijau**.
4. Buka `starter.ipynb` — jalankan delapan blok eksperimen (angka terkunci), lalu
   jawab **6 Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/ftkit_ref.py` / `solusi.ipynb`.

Ingin melihat angka terkunci tanpa menjalankan notebook?

```bash
python _calib.py
```

> 💡 Tiga jebakan yang menyumbang kebanyakan bug di project ini:
> (1) `oversample` harus mempertahankan **urutan asli** — menggrup per label
> dulu mengubah urutan data (test gagal, dan di nyata mengubah distribusi batch);
> (2) loss epoch dicatat dari `p` **sebelum** update tiap contoh — pakai
> `loss_batch` setelah epoch selesai memberi angka lain;
> (3) `latih` harus memulihkan bobot **terbaik** — lupa `best = (list(w), b)`
> membuat model akhir lebih buruk dari model terbaik.

## Pertanyaan Analisis (bagian penilaian!)

1. Rasio param LoRA 0,78% — mana yang lebih murah di produksi: 1 adapter besar
   per klien atau 100 adapter kecil? Kapan rank perlu dinaikkan?
2. Early stopping berhenti di epoch 66 dari 100. Apa yang terjadi pada val loss
   setelah epoch 61, dan mengapa bobot TERBAIK (bukan terakhir) yang dipulihkan?
3. Oversampling menduplikasi data minoritas. Risiko apa yang muncul bila data
   diduplikasi terlalu banyak, dan alternatif apa selain duplikasi?
4. Kuantisasi Q4 memangkas 14 GB → 3,5 GB dengan skor turun 0.82 → 0.78.
   Untuk kasus apa kompromi itu layak, dan kapan TIDAK layak?
5. Keputusan API vs lokal berubah di sekitar ~2.400 req/hari (asumsi modul).
   Faktor nyata apa yang bisa menggeser ambang itu?
6. Klasifier ini tahu dia tidak yakin via satu angka skor. LLM open-weight tidak
   punya satu angka 'yakin' — bagaimana mendeteksi ketidakpastian di LLM nyata,
   dan lapisan mana yang tetap wajib (Bab 12 & 15)?

## Prasyarat

```bash
# tidak ada!
```

> Stdlib murni (`math`, `json`). Tidak ada PyTorch, tidak ada transformers,
> tidak ada GPU — supaya *mesinnya* terlihat. Setelah paham, memakai
> Unsloth/HF TRL terasa seperti memakai ulang kode yang kamu tulis sendiri.

> Bab ini memakai kembali pola dari Bab 4 (loss, overfitting, early stopping),
> Bab 11 (biaya per token, keputusan berbasis angka) dan Bab 12 (abstain saat
> tidak yakin). Lanjutan: Bab 14 (agent memakai model lokal ini sebagai otak).
